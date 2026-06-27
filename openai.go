package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// translateAnthropicToOpenAIStream translates Anthropic SSE → OpenAI SSE.
func translateAnthropicToOpenAIStream(ctx context.Context, upstream io.Reader, w io.Writer, flusher http.Flusher, model string) error {
	scanner := newSSEScanner(upstream)

	chunkID := newChatCompletionID()
	var started bool
	toolBlocks := map[string]int{} // Anthropic content_block index → OpenAI tool_call index
	var nextToolIdx int

	for scanner.Scan() {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "" {
			continue
		}

		var event map[string]interface{}
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			continue
		}

		etype, _ := event["type"].(string)

		switch etype {
		case "message_start":
			started = true
			chunk := map[string]interface{}{
				"id": chunkID, "object": "chat.completion.chunk", "created": time.Now().Unix(),
				"model": model, "choices": []interface{}{map[string]interface{}{
					"index": 0, "delta": map[string]interface{}{"role": "assistant", "content": ""},
					"finish_reason": nil,
				}},
			}
			b, _ := json.Marshal(chunk)
			fmt.Fprintf(w, "data: %s\n\n", b)
			flusher.Flush()

		case "content_block_start":
			if !started {
				continue
			}
			cb, _ := event["content_block"].(map[string]interface{})
			cbType, _ := cb["type"].(string)
			if cbType == "tool_use" {
				idx, _ := event["index"].(float64)
			 blockIdx := int(idx)
				toolIdx := nextToolIdx
				nextToolIdx++
				toolBlocks[fmt.Sprintf("%d", blockIdx)] = toolIdx
				toolCallID, _ := cb["id"].(string)
				toolCallName, _ := cb["name"].(string)
				emitOpenAIChunk(w, flusher, chunkID, model, map[string]interface{}{
					"tool_calls": []interface{}{map[string]interface{}{
						"index": toolIdx, "type": "function", "id": toolCallID,
						"function": map[string]interface{}{"name": toolCallName, "arguments": ""},
					}},
				}, nil)
			}

		case "content_block_delta":
			if !started {
				continue
			}
			delta, _ := event["delta"].(map[string]interface{})
			deltaType, _ := delta["type"].(string)

			switch deltaType {
			case "text_delta":
				content, _ := delta["text"].(string)
				if content != "" {
					emitOpenAIChunk(w, flusher, chunkID, model, map[string]interface{}{"content": content}, nil)
				}
			case "thinking_delta":
				thinking, _ := delta["thinking"].(string)
				emitOpenAIChunk(w, flusher, chunkID, model, map[string]interface{}{"reasoning_content": thinking}, nil)
			case "input_json_delta":
				idx, _ := event["index"].(float64)
			 blockIdx := int(idx)
				toolIdx := toolBlocks[fmt.Sprintf("%d", blockIdx)]
				partialJSON, _ := delta["partial_json"].(string)
				emitOpenAIChunk(w, flusher, chunkID, model, map[string]interface{}{
					"tool_calls": []interface{}{map[string]interface{}{
						"index": toolIdx, "type": "function",
						"function": map[string]interface{}{"arguments": partialJSON},
					}},
				}, nil)
			}

		case "message_delta":
			if !started {
				continue
			}
			delta, _ := event["delta"].(map[string]interface{})
			stopReason, _ := delta["stop_reason"].(string)
			finishReason := mapFinishReasonReverse(stopReason)
			emitOpenAIChunk(w, flusher, chunkID, model, map[string]interface{}{}, &finishReason)

		case "message_stop":
			fmt.Fprintf(w, "data: [DONE]\n\n")
			flusher.Flush()
			return nil
		}
	}
	return scanner.Err()
}

// emitOpenAIChunk writes an OpenAI SSE chunk.
func emitOpenAIChunk(w io.Writer, flusher http.Flusher, chunkID, model string, delta map[string]interface{}, finishReason *string) {
	chunk := map[string]interface{}{
		"id": chunkID, "object": "chat.completion.chunk", "created": time.Now().Unix(),
		"model": model, "choices": []interface{}{map[string]interface{}{
			"index": 0, "delta": delta, "finish_reason": finishReason,
		}},
	}
	b, _ := json.Marshal(chunk)
	fmt.Fprintf(w, "data: %s\n\n", b)
	flusher.Flush()
}

// handleOpenAI handles POST /v1/chat/completions.
func handleOpenAI(w http.ResponseWriter, r *http.Request) {
	setCORS(w)

	body, err := readRequestBody(r)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]interface{}{
			"error": map[string]interface{}{"message": err.Error(), "type": "invalid_request_error"},
		})
		return
	}

	var req struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]interface{}{
			"error": map[string]interface{}{"message": "invalid JSON body", "type": "invalid_request_error"},
		})
		return
	}

	p, err := getProvider(req.Model)
	if err != nil {
		writeJSON(w, http.StatusNotFound, map[string]interface{}{
			"error": map[string]interface{}{"message": err.Error(), "type": "invalid_request_error"},
		})
		return
	}

	switch p.Type {
	case ProviderOpenAI:
		// Rewrite model name and forward.
		var m map[string]interface{}
		if err := json.Unmarshal(body, &m); err != nil || m == nil {
			writeJSON(w, http.StatusBadRequest, map[string]interface{}{
				"error": map[string]interface{}{"message": "invalid JSON body", "type": "invalid_request_error"},
			})
			return
		}
		m["model"] = p.Name
		if len(p.ExtraParams) > 0 {
			applyExtraParams(m, p.ExtraParams)
		}
		rewritten, _ := json.Marshal(m)
		forwardUpstream(w, r, p.ChatURL, p.APIKey, rewritten, nil)

	case ProviderAnthropic:
		// Convert OpenAI request → IR → Anthropic format.
		ir, err := chatCompletionsToIRRequest(body)
		if err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]interface{}{
				"error": map[string]interface{}{"message": err.Error(), "type": "invalid_request_error"},
			})
			return
		}
		ir.Model = p.Name
		if p.DropParams {
			ir.Thinking = nil
			ir.ToolChoice = nil
		}
		anthReq := irToAnthropicRequest(ir)
		anthBody, _ := json.Marshal(anthReq)
		if len(p.ExtraParams) > 0 {
			var m map[string]interface{}
			if err := json.Unmarshal(anthBody, &m); err == nil && m != nil {
				applyExtraParams(m, p.ExtraParams)
				anthBody, _ = json.Marshal(m)
			}
		}

		ctx := r.Context()
		req2, err := http.NewRequestWithContext(ctx, "POST", p.MessagesURL, bytes.NewReader(anthBody))
		if err != nil {
			writeProxyError(w, r, err)
			return
		}
		req2.Header.Set(headerContentType, contentTypeJSON)
		req2.Header.Set("x-api-key", p.APIKey)
		req2.Header.Set("anthropic-version", anthropicAPIVersion)

		resp, err := httpClient.Do(req2)
		if err != nil {
			writeProxyError(w, r, err)
			return
		}
		defer resp.Body.Close()

		if resp.StatusCode >= 400 {
			errBody := readResponseBody(resp.Body, "upstream error")
			writeJSON(w, resp.StatusCode, map[string]interface{}{
				"error": map[string]interface{}{"type": "api_error", "message": extractUpstreamError(errBody)},
			})
			return
		}

		if ir.Stream {
			flusher, err := setupSSEStream(w, resp.StatusCode)
			if err != nil {
				writeError(w, http.StatusInternalServerError, err.Error())
				return
			}

			if err := translateAnthropicToOpenAIStream(ctx, resp.Body, w, flusher, req.Model); err != nil {
				return
			}
			return
		}

		// Non-streaming: Anthropic response → IR → OpenAI format.
		resBody, _ := io.ReadAll(resp.Body)
		irResp := anthropicToIR(resBody, req.Model)
		openaiResp := irToChatCompletionsResponse(irResp)
		writeJSON(w, resp.StatusCode, openaiResp)

	case ProviderResponses:
		// Convert OpenAI request → IR → Responses format.
		ir, err := chatCompletionsToIRRequest(body)
		if err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]interface{}{
				"error": map[string]interface{}{"message": err.Error(), "type": "invalid_request_error"},
			})
			return
		}
		ir.Model = p.Name
		if p.DropParams {
			ir.Thinking = nil
			ir.ToolChoice = nil
		}
		responsesReq := irToResponsesRequest(ir)
		applyExtraParams(responsesReq, p.ExtraParams)
		responsesBody, err := json.Marshal(responsesReq)
		if err != nil {
			writeError(w, http.StatusInternalServerError, err.Error())
			return
		}

		ctx := r.Context()
		req2, err := http.NewRequestWithContext(ctx, "POST", p.ResponsesURL, bytes.NewReader(responsesBody))
		if err != nil {
			writeProxyError(w, r, err)
			return
		}
		req2.Header.Set(headerContentType, contentTypeJSON)
		req2.Header.Set(headerAuthorization, authBearerPrefix+p.APIKey)

		resp, err := httpClient.Do(req2)
		if err != nil {
			writeProxyError(w, r, err)
			return
		}
		defer resp.Body.Close()

		if resp.StatusCode >= 400 {
			errBody := readResponseBody(resp.Body, "upstream error")
			writeJSON(w, resp.StatusCode, map[string]interface{}{
				"error": map[string]interface{}{"type": "api_error", "message": extractUpstreamError(errBody)},
			})
			return
		}

		if ir.Stream {
			flusher, err := setupSSEStream(w, resp.StatusCode)
			if err != nil {
				writeError(w, http.StatusInternalServerError, err.Error())
				return
			}

			if err := translateResponsesToOpenAIStream(ctx, resp.Body, w, flusher, req.Model); err != nil {
				return
			}
			return
		}

		// Non-streaming: Responses response → IR → OpenAI format.
		resBody, _ := io.ReadAll(resp.Body)
		irResp := responsesToIRResponse(resBody, req.Model)
		openaiResp := irToChatCompletionsResponse(irResp)
		writeJSON(w, resp.StatusCode, openaiResp)

	default:
		writeError(w, http.StatusInternalServerError, "unsupported provider type")
	}
}

// translateResponsesToOpenAIStream translates Responses SSE → OpenAI Chat Completions SSE.
func translateResponsesToOpenAIStream(ctx context.Context, upstream io.Reader, w io.Writer, flusher http.Flusher, model string) error {
	chunkID := newChatCompletionID()
	var started, isReasoning bool
	var inputTokens, outputTokens int
	var hasToolCalls bool
	toolCallIdx := map[int]int{} // Responses output_index → OpenAI tool_call index
	var nextToolIdx int

	scanner := newSSEScanner(upstream)

	ensureStarted := func() {
		if !started {
			started = true
			emitOpenAIChunk(w, flusher, chunkID, model, map[string]interface{}{"role": "assistant", "content": ""}, nil)
		}
	}

	for scanner.Scan() {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		line := scanner.Text()
		if !strings.HasPrefix(line, "event: ") {
			continue
		}
		eventType := strings.TrimPrefix(line, "event: ")

		if !scanner.Scan() {
			break
		}
		dataLine := scanner.Text()
		if !strings.HasPrefix(dataLine, "data: ") {
			continue
		}
		dataStr := strings.TrimPrefix(dataLine, "data: ")

		var event map[string]interface{}
		if err := json.Unmarshal([]byte(dataStr), &event); err != nil {
			continue
		}

		switch eventType {
		case "response.created":
			ensureStarted()

		case "response.output_item.added":
			ensureStarted()
			item, _ := event["item"].(map[string]interface{})
			itemType, _ := item["type"].(string)
			switch itemType {
			case "reasoning":
				isReasoning = true
			case "message":
				isReasoning = false
			case "function_call":
				hasToolCalls = true
				callID, _ := item["call_id"].(string)
				toolName, _ := item["name"].(string)
				outputIdx, _ := event["output_index"].(float64)
				toolIdx := nextToolIdx
				nextToolIdx++
				toolCallIdx[int(outputIdx)] = toolIdx
				emitOpenAIChunk(w, flusher, chunkID, model, map[string]interface{}{
					"tool_calls": []interface{}{map[string]interface{}{
						"index": toolIdx, "type": "function", "id": callID,
						"function": map[string]interface{}{"name": toolName, "arguments": ""},
					}},
				}, nil)
			}

		case "response.output_text.delta":
			if !started {
				continue
			}
			delta, _ := event["delta"].(string)
			if delta != "" {
				if isReasoning {
					emitOpenAIChunk(w, flusher, chunkID, model, map[string]interface{}{"reasoning_content": delta}, nil)
				} else {
					emitOpenAIChunk(w, flusher, chunkID, model, map[string]interface{}{"content": delta}, nil)
				}
			}

		case "response.function_call_arguments.delta":
			if !started {
				continue
			}
			delta, _ := event["delta"].(string)
			outputIdx, _ := event["output_index"].(float64)
			toolIdx := toolCallIdx[int(outputIdx)]
			if delta != "" {
				emitOpenAIChunk(w, flusher, chunkID, model, map[string]interface{}{
					"tool_calls": []interface{}{map[string]interface{}{
						"index": toolIdx, "type": "function",
						"function": map[string]interface{}{"arguments": delta},
					}},
				}, nil)
			}

		case "response.completed":
			respData, _ := event["response"].(map[string]interface{})
			if usage, ok := respData["usage"].(map[string]interface{}); ok {
				if v, ok := usage["input_tokens"].(float64); ok {
					inputTokens = int(v)
				}
				if v, ok := usage["output_tokens"].(float64); ok {
					outputTokens = int(v)
				}
			}
			ensureStarted()
			finishReason := "stop"
			respStatus, _ := respData["status"].(string)
			if hasToolCalls {
				finishReason = "tool_calls"
			} else if respStatus == "incomplete" {
				finishReason = "length"
			}
			chunk := map[string]interface{}{
				"id": chunkID, "object": "chat.completion.chunk", "created": time.Now().Unix(),
				"model": model, "choices": []interface{}{map[string]interface{}{
					"index": 0, "delta": map[string]interface{}{}, "finish_reason": finishReason,
				}},
			}
			if inputTokens > 0 || outputTokens > 0 {
				chunk["usage"] = map[string]interface{}{
					"prompt_tokens":     inputTokens,
					"completion_tokens": outputTokens,
					"total_tokens":      inputTokens + outputTokens,
				}
			}
			b, _ := json.Marshal(chunk)
			fmt.Fprintf(w, "data: %s\n\n", b)
			fmt.Fprintf(w, "data: [DONE]\n\n")
			flusher.Flush()
			return nil
		}
	}
	return scanner.Err()
}
