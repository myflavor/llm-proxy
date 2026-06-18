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
		rewritten, _ := json.Marshal(m)
		forwardOpenAI(w, r, p, rewritten)

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

		if ir.Stream {
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

		// Non-streaming: Anthropic response → IR → OpenAI format.
		resBody, _ := io.ReadAll(resp.Body)
		irResp := anthropicToIR(resBody, req.Model)
		openaiResp := irToChatCompletionsResponse(irResp)
		writeJSON(w, resp.StatusCode, openaiResp)
	}
}
