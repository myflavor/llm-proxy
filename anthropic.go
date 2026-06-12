package main

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"time"
)

// anthropicToOpenAI converts an Anthropic request to OpenAI format.
// If dropParams is true, thinking/tool_choice are dropped.
func anthropicToOpenAI(req anthropicMsgReq, dropParams bool) map[string]interface{} {
	oa := map[string]interface{}{
		"model":    req.Model,
		"messages": []interface{}{},
		"stream":   req.Stream,
	}
	if req.Stream {
		oa["stream_options"] = map[string]interface{}{"include_usage": true}
	}
	if req.MaxTokens > 0 {
		oa["max_tokens"] = req.MaxTokens
	}
	if req.Temperature != nil {
		oa["temperature"] = *req.Temperature
	}
	if req.TopP != nil {
		oa["top_p"] = *req.TopP
	}
	if len(req.StopSequences) > 0 {
		oa["stop"] = req.StopSequences
	}
	if len(req.Tools) > 0 {
		tools := make([]interface{}, 0, len(req.Tools))
		for _, t := range req.Tools {
			tools = append(tools, map[string]interface{}{
				"type": "function",
				"function": map[string]interface{}{
					"name":        t.Name,
					"description": t.Description,
					"parameters":  t.InputSchema,
				},
			})
		}
		oa["tools"] = tools
	}
	if !dropParams {
		if req.ToolChoice != nil {
			oa["tool_choice"] = convertToolChoice(req.ToolChoice)
		}
		if req.Thinking != nil && req.Thinking.Type == "enabled" && req.Thinking.BudgetTokens > 0 {
			oa["thinking_budget"] = req.Thinking.BudgetTokens
		}
	}
	if req.Metadata != nil && req.Metadata.UserID != "" {
		oa["user"] = req.Metadata.UserID
	}

	messages := oa["messages"].([]interface{})

	if sysText := extractText(req.System); sysText != "" {
		messages = append(messages, map[string]interface{}{"role": "system", "content": sysText})
	}

	for _, m := range req.Messages {
		content := m.Content

		// Assistant messages with tool_use / thinking blocks.
		if m.Role == "assistant" {
			if blocks, ok := content.([]interface{}); ok {
				var textParts, reasoningParts []string
				var toolCalls []interface{}
				for _, block := range blocks {
					b, ok := block.(map[string]interface{})
					if !ok {
						continue
					}
					switch b["type"] {
					case "text":
						if txt, _ := b["text"].(string); txt != "" {
							textParts = append(textParts, txt)
						}
					case "thinking":
						if txt, _ := b["thinking"].(string); txt != "" {
							reasoningParts = append(reasoningParts, txt)
						}
					case "tool_use":
						toolCalls = append(toolCalls, map[string]interface{}{
							"id":   b["id"],
							"type": "function",
							"function": map[string]interface{}{
								"name":      b["name"],
								"arguments": mustJSON(b["input"]),
							},
						})
					}
				}
				if len(toolCalls) > 0 || len(reasoningParts) > 0 {
					msg := map[string]interface{}{"role": "assistant"}
					if len(textParts) > 0 {
						msg["content"] = strings.Join(textParts, "")
					}
					if len(reasoningParts) > 0 {
						msg["reasoning_content"] = strings.Join(reasoningParts, "")
					}
					if len(toolCalls) > 0 {
						msg["tool_calls"] = toolCalls
					}
					messages = append(messages, msg)
					continue
				}
			}
		}

		// User messages with tool_result / image blocks.
		if m.Role == "user" {
			if blocks, ok := content.([]interface{}); ok {
				var contentParts []interface{}
				var toolResults []interface{}
				for _, block := range blocks {
					b, ok := block.(map[string]interface{})
					if !ok {
						continue
					}
					switch b["type"] {
					case "text":
						if txt, _ := b["text"].(string); txt != "" {
							contentParts = append(contentParts, map[string]interface{}{"type": "text", "text": txt})
						}
					case "image":
						if url := extractImageURL(b); url != "" {
							contentParts = append(contentParts, map[string]interface{}{
								"type": "image_url",
								"image_url": map[string]interface{}{"url": url},
							})
						}
					case "tool_result":
						resultContent := ""
						if c, ok := b["content"].(string); ok {
							resultContent = c
						} else if blocks2, ok := b["content"].([]interface{}); ok {
							var parts []string
							for _, b2 := range blocks2 {
								if m2, ok := b2.(map[string]interface{}); ok && m2["type"] == "text" {
									parts = append(parts, m2["text"].(string))
								}
							}
							resultContent = strings.Join(parts, "")
						}
						toolResults = append(toolResults, map[string]interface{}{
							"role":         "tool",
							"tool_call_id": b["tool_use_id"],
							"content":      resultContent,
						})
					}
				}
				if len(contentParts) > 0 {
					if len(contentParts) == 1 {
						if p, ok := contentParts[0].(map[string]interface{}); ok && p["type"] == "text" {
							messages = append(messages, map[string]interface{}{"role": "user", "content": p["text"]})
						} else {
							messages = append(messages, map[string]interface{}{"role": "user", "content": contentParts})
						}
					} else {
						messages = append(messages, map[string]interface{}{"role": "user", "content": contentParts})
					}
				}
				for _, tr := range toolResults {
					messages = append(messages, tr)
				}
				continue
			}
		}

		messages = append(messages, map[string]interface{}{"role": m.Role, "content": extractText(content)})
	}
	oa["messages"] = messages
	return oa
}

// openaiToAnthropic converts an OpenAI response to Anthropic format.
func openaiToAnthropic(openaiBody []byte, model string) map[string]interface{} {
	var oa struct {
		ID      string `json:"id"`
		Choices []struct {
			Message struct {
				Role             string `json:"role"`
				Content          string `json:"content"`
				ReasoningContent string `json:"reasoning_content"`
				ToolCalls        []struct {
					ID       string `json:"id"`
					Type     string `json:"type"`
					Function struct {
						Name      string `json:"name"`
						Arguments string `json:"arguments"`
					} `json:"function"`
				} `json:"tool_calls"`
			} `json:"message"`
			FinishReason *string `json:"finish_reason"`
		} `json:"choices"`
		Usage *struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(openaiBody, &oa); err != nil {
		var errResp struct {
			Error struct {
				Message string `json:"message"`
			} `json:"error"`
		}
		msg := "[upstream response parse error]"
		if json.Unmarshal(openaiBody, &errResp) == nil && errResp.Error.Message != "" {
			msg = errResp.Error.Message
		}
		return map[string]interface{}{
			"id": "msg_" + randomHex(16), "type": "message", "role": "assistant",
			"content": []interface{}{map[string]interface{}{"type": "text", "text": msg}},
			"model": model,
		}
	}

	stopReason := "end_turn"
	if len(oa.Choices) > 0 && oa.Choices[0].FinishReason != nil {
		stopReason = mapFinishReason(*oa.Choices[0].FinishReason)
	}

	role := "assistant"
	var contentBlocks []interface{}
	if len(oa.Choices) > 0 {
		msg := oa.Choices[0].Message
		if msg.Role != "" {
			role = msg.Role
		}
		if msg.ReasoningContent != "" {
			contentBlocks = append(contentBlocks, map[string]interface{}{"type": "thinking", "thinking": msg.ReasoningContent})
		}
		if msg.Content != "" {
			contentBlocks = append(contentBlocks, map[string]interface{}{"type": "text", "text": msg.Content})
		}
		for _, tc := range msg.ToolCalls {
			var input map[string]interface{}
			json.Unmarshal([]byte(tc.Function.Arguments), &input)
			contentBlocks = append(contentBlocks, map[string]interface{}{
				"type": "tool_use", "id": tc.ID, "name": tc.Function.Name, "input": input,
			})
		}
	}
	if len(contentBlocks) == 0 {
		contentBlocks = []interface{}{map[string]interface{}{"type": "text", "text": ""}}
	}

	resp := map[string]interface{}{
		"id": "msg_" + randomHex(16), "type": "message", "role": role,
		"content": contentBlocks, "model": model, "stop_reason": stopReason,
	}
	if oa.Usage != nil {
		resp["usage"] = map[string]interface{}{
			"input_tokens": oa.Usage.PromptTokens, "output_tokens": oa.Usage.CompletionTokens,
		}
	}
	return resp
}

// handleAnthropic handles POST /v1/messages.
func handleAnthropic(w http.ResponseWriter, r *http.Request) {
	setCORS(w)

	body, err := readBody(r)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]interface{}{
			"error": map[string]interface{}{"message": err.Error(), "type": "invalid_request_error"},
		})
		return
	}

	var req anthropicMsgReq
	if err := json.Unmarshal(body, &req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]interface{}{
			"error": map[string]interface{}{"message": err.Error(), "type": "invalid_request_error"},
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
	case ProviderAnthropic:
		// Rewrite model name and forward.
		req.Model = p.Name
		rewritten, _ := json.Marshal(req)
		forwardAnthropic(w, r, p, rewritten)

	case ProviderOpenAI:
		// Convert Anthropic request → OpenAI format.
		oaReq := anthropicToOpenAI(req, p.DropParams)
		oaReq["model"] = p.Name
		oaBody, err := json.Marshal(oaReq)
		if err != nil {
			writeJSON(w, http.StatusInternalServerError, map[string]interface{}{
				"error": map[string]interface{}{"message": err.Error(), "type": "server_error"},
			})
			return
		}

		ctx := r.Context()
		req2, err := http.NewRequestWithContext(ctx, "POST", p.ChatURL, bytes.NewReader(oaBody))
		if err != nil {
			writeProxyError(w, r, err)
			return
		}
		req2.Header.Set("Content-Type", "application/json")
		req2.Header.Set("Authorization", "Bearer "+p.APIKey)

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

		if req.Stream {
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

			inputTokens := countTokens(oaReq)
			if err := translateStream(ctx, resp.Body, w, flusher, req.Model, inputTokens); err != nil {
				return
			}
			return
		}

		// Non-streaming: convert OpenAI response → Anthropic format.
		resBody, _ := io.ReadAll(resp.Body)
		anthResp := openaiToAnthropic(resBody, req.Model)
		writeJSON(w, resp.StatusCode, anthResp)
	}
}

// forwardAnthropic forwards an Anthropic-format request to an Anthropic upstream.
func forwardAnthropic(w http.ResponseWriter, r *http.Request, p *Provider, body []byte) {
	req, err := http.NewRequestWithContext(r.Context(), "POST", p.MessagesURL, bytes.NewReader(body))
	if err != nil {
		writeProxyError(w, r, err)
		return
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", p.APIKey)
	req.Header.Set("anthropic-version", "2023-06-01")

	resp, err := httpClient.Do(req)
	if err != nil {
		writeProxyError(w, r, err)
		return
	}
	defer resp.Body.Close()

	streamPassthrough(w, resp)
}
