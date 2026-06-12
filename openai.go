package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// openaiToAnthropicRequest converts an OpenAI chat request to Anthropic format.
func openaiToAnthropicRequest(body []byte) (*anthropicMsgReq, error) {
	var oa struct {
		Model       string                   `json:"model"`
		Messages    []map[string]interface{} `json:"messages"`
		MaxTokens   int                      `json:"max_tokens"`
		Temperature *float64                 `json:"temperature,omitempty"`
		TopP        *float64                 `json:"top_p,omitempty"`
		Stop        []string                 `json:"stop,omitempty"`
		Stream      bool                     `json:"stream,omitempty"`
		Tools       []struct {
			Type     string `json:"type"`
			Function struct {
				Name        string      `json:"name"`
				Description string      `json:"description,omitempty"`
				Parameters  interface{} `json:"parameters"`
			} `json:"function"`
		} `json:"tools,omitempty"`
		ToolChoice interface{} `json:"tool_choice,omitempty"`
	}
	if err := json.Unmarshal(body, &oa); err != nil {
		return nil, err
	}

	req := &anthropicMsgReq{
		Model:         oa.Model,
		MaxTokens:     oa.MaxTokens,
		Temperature:   oa.Temperature,
		TopP:          oa.TopP,
		StopSequences: oa.Stop,
		Stream:        oa.Stream,
	}

	// Convert system messages to Anthropic system field.
	var systemParts []string
	var msgs []anthropicMsg
	for _, m := range oa.Messages {
		role, _ := m["role"].(string)
		if role == "system" {
			systemParts = append(systemParts, extractText(m["content"]))
			continue
		}
		if role == "tool" {
			toolCallID, _ := m["tool_call_id"].(string)
			content, _ := m["content"].(string)
			msgs = append(msgs, anthropicMsg{
				Role: "user",
				Content: []interface{}{map[string]interface{}{
					"type":        "tool_result",
					"tool_use_id": toolCallID,
					"content":     content,
				}},
			})
			continue
		}
		msgs = append(msgs, anthropicMsg{Role: role, Content: m["content"]})
	}
	if len(systemParts) > 0 {
		req.System = strings.Join(systemParts, "\n\n")
	}
	req.Messages = msgs

	// Convert tools.
	if len(oa.Tools) > 0 {
		for _, t := range oa.Tools {
			req.Tools = append(req.Tools, anthropicTool{
				Name:        t.Function.Name,
				Description: t.Function.Description,
				InputSchema: t.Function.Parameters,
			})
		}
	}

	return req, nil
}

// anthropicResponseToOpenAI converts an Anthropic non-streaming response to OpenAI format.
func anthropicResponseToOpenAI(body []byte, model string) map[string]interface{} {
	var anth struct {
		ID      string `json:"id"`
		Role    string `json:"role"`
		Content []struct {
			Type     string      `json:"type"`
			Text     string      `json:"text,omitempty"`
			Thinking string      `json:"thinking,omitempty"`
			ID       string      `json:"id,omitempty"`
			Name     string      `json:"name,omitempty"`
			Input    interface{} `json:"input,omitempty"`
		} `json:"content"`
		StopReason string `json:"stop_reason"`
		Usage      *struct {
			InputTokens  int `json:"input_tokens"`
			OutputTokens int `json:"output_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(body, &anth); err != nil {
		var errResp struct {
			Error struct {
				Message string `json:"message"`
			} `json:"error"`
		}
		msg := "[upstream response parse error]"
		if json.Unmarshal(body, &errResp) == nil && errResp.Error.Message != "" {
			msg = errResp.Error.Message
		}
		return map[string]interface{}{
			"id": "chatcmpl-" + randomHex(12), "object": "chat.completion", "created": time.Now().Unix(),
			"model": model, "choices": []interface{}{map[string]interface{}{
				"index": 0, "message": map[string]interface{}{"role": "assistant", "content": msg},
				"finish_reason": "stop",
			}},
		}
	}

	var content, reasoningContent string
	var toolCalls []interface{}
	for _, c := range anth.Content {
		switch c.Type {
		case "text":
			content = c.Text
		case "thinking":
			reasoningContent = c.Thinking
		case "tool_use":
			inputJSON, _ := json.Marshal(c.Input)
			toolCalls = append(toolCalls, map[string]interface{}{
				"id": c.ID, "type": "function",
				"function": map[string]interface{}{"name": c.Name, "arguments": string(inputJSON)},
			})
		}
	}

	finishReason := mapFinishReasonReverse(anth.StopReason)

	msg := map[string]interface{}{"role": "assistant"}
	if content != "" {
		msg["content"] = content
	}
	if reasoningContent != "" {
		msg["reasoning_content"] = reasoningContent
	}
	if len(toolCalls) > 0 {
		msg["tool_calls"] = toolCalls
	}

	resp := map[string]interface{}{
		"id": "chatcmpl-" + randomHex(12), "object": "chat.completion", "created": time.Now().Unix(),
		"model": model, "choices": []interface{}{map[string]interface{}{
			"index": 0, "message": msg, "finish_reason": finishReason,
		}},
	}
	if anth.Usage != nil {
		resp["usage"] = map[string]interface{}{
			"prompt_tokens": anth.Usage.InputTokens, "completion_tokens": anth.Usage.OutputTokens,
			"total_tokens": anth.Usage.InputTokens + anth.Usage.OutputTokens,
		}
	}
	return resp
}

// translateAnthropicToOpenAIStream translates Anthropic SSE → OpenAI SSE.
func translateAnthropicToOpenAIStream(ctx context.Context, upstream io.Reader, w io.Writer, flusher http.Flusher, model string) error {
	scanner := bufio.NewScanner(upstream)
	scanner.Buffer(make([]byte, 64*1024), 1024*1024)

	chunkID := fmt.Sprintf("chatcmpl-%s", randomHex(12))
	var started bool

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
				partialJSON, _ := delta["partial_json"].(string)
				emitOpenAIChunk(w, flusher, chunkID, model, map[string]interface{}{
					"tool_calls": []interface{}{map[string]interface{}{
						"index": 0, "type": "function",
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

	body, err := readBody(r)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]interface{}{
			"error": map[string]interface{}{"message": err.Error(), "type": "invalid_request_error"},
		})
		return
	}

	var req struct {
		Model string `json:"model"`
	}
	json.Unmarshal(body, &req)

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
		json.Unmarshal(body, &m)
		m["model"] = p.Name
		rewritten, _ := json.Marshal(m)
		forwardOpenAI(w, r, p, rewritten)

	case ProviderAnthropic:
		// Convert OpenAI request → Anthropic format.
		oaReq, err := openaiToAnthropicRequest(body)
		if err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]interface{}{
				"error": map[string]interface{}{"message": err.Error(), "type": "invalid_request_error"},
			})
			return
		}
		oaReq.Model = p.Name
		anthBody, _ := json.Marshal(oaReq)

		ctx := r.Context()
		req2, err := http.NewRequestWithContext(ctx, "POST", p.MessagesURL, bytes.NewReader(anthBody))
		if err != nil {
			writeProxyError(w, r, err)
			return
		}
		req2.Header.Set("Content-Type", "application/json")
		req2.Header.Set("x-api-key", p.APIKey)
		req2.Header.Set("anthropic-version", "2023-06-01")

		resp, err := httpClient.Do(req2)
		if err != nil {
			writeProxyError(w, r, err)
			return
		}
		defer resp.Body.Close()

		if resp.StatusCode >= 400 {
			errBody, _ := io.ReadAll(resp.Body)
			writeJSON(w, resp.StatusCode, map[string]interface{}{
				"error": map[string]interface{}{"type": "api_error", "message": extractUpstreamError(errBody)},
			})
			return
		}

		if oaReq.Stream {
			flusher, ok := w.(http.Flusher)
			if !ok {
				writeJSON(w, http.StatusInternalServerError, map[string]interface{}{
					"error": map[string]interface{}{"message": "streaming not supported", "type": "server_error"},
				})
				return
			}
			w.Header().Set("Content-Type", "text/event-stream")
			w.Header().Set("Cache-Control", "no-cache")
			http.NewResponseController(w).SetWriteDeadline(time.Time{})
			w.WriteHeader(resp.StatusCode)

			if err := translateAnthropicToOpenAIStream(ctx, resp.Body, w, flusher, req.Model); err != nil {
				return
			}
			return
		}

		// Non-streaming: convert Anthropic response → OpenAI format.
		resBody, _ := io.ReadAll(resp.Body)
		openaiResp := anthropicResponseToOpenAI(resBody, req.Model)
		writeJSON(w, resp.StatusCode, openaiResp)
	}
}
