package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"time"
)

// ============================================================
// OpenAI Responses API 类型定义
// ============================================================

// --- 请求 ---

type responsesReq struct {
	Model              string            `json:"model"`
	Input              interface{}       `json:"input"` // string or []responsesInputItem
	Instructions       string            `json:"instructions,omitempty"`
	Tools              []responsesTool   `json:"tools,omitempty"`
	Stream             bool              `json:"stream,omitempty"`
	Text               *responsesTextCfg `json:"text,omitempty"`
	Reasoning          *responsesReasoningCfg `json:"reasoning,omitempty"`
	MaxOutputTokens    int               `json:"max_output_tokens,omitempty"`
	Temperature        *float64          `json:"temperature,omitempty"`
	TopP               *float64          `json:"top_p,omitempty"`
	PreviousResponseID string            `json:"previous_response_id,omitempty"`
	Include            []string          `json:"include,omitempty"`
	Store              *bool             `json:"store,omitempty"`
	ToolChoice         interface{}       `json:"tool_choice,omitempty"`
}

type responsesInputItem struct {
	Type      string      `json:"type"` // "message", "function_call", "function_call_output"
	Role      string      `json:"role,omitempty"`
	Content   interface{} `json:"content,omitempty"` // string or []responsesContentPart
	CallID    string      `json:"call_id,omitempty"` // for function_call and function_call_output
	Output    string      `json:"output,omitempty"`  // for function_call_output
	Name      string      `json:"name,omitempty"`     // for function_call
	Arguments string      `json:"arguments,omitempty"` // for function_call (JSON string)
}

type responsesContentPart struct {
	Type     string `json:"type"` // "input_text", "input_image", "output_text", "refusal"
	Text     string `json:"text,omitempty"`
	ImageURL string `json:"image_url,omitempty"`
}

type responsesTool struct {
	Type        string      `json:"type"` // "function", "web_search", etc.
	Name        string      `json:"name,omitempty"`
	Description string      `json:"description,omitempty"`
	Parameters  interface{} `json:"parameters,omitempty"`
	Strict      *bool       `json:"strict,omitempty"`
}

type responsesTextCfg struct {
	Format interface{} `json:"format,omitempty"` // structured output schema
}

type responsesReasoningCfg struct {
	Effort  string `json:"effort,omitempty"`  // low/medium/high
	Summary string `json:"summary,omitempty"` // auto/none
}

// --- 响应 ---

type responsesResp struct {
	ID         string              `json:"id"`
	Object     string              `json:"object"` // "response"
	CreatedAt  int64               `json:"created_at"`
	Model      string              `json:"model"`
	Output     []responsesOutputItem `json:"output"`
	Usage      *responsesUsage     `json:"usage,omitempty"`
	Status     string              `json:"status"` // "completed", "failed", etc.
	Error      *responsesError     `json:"error,omitempty"`
}

type responsesOutputItem struct {
	Type      string                   `json:"type"` // "message", "reasoning", "function_call"
	ID        string                   `json:"id,omitempty"`
	Role      string                   `json:"role,omitempty"`       // for message
	Content   []responsesOutputContent `json:"content,omitempty"`    // for message
	// function_call fields
	CallID    string                   `json:"call_id,omitempty"`
	Name      string                   `json:"name,omitempty"`
	Arguments string                   `json:"arguments,omitempty"`
	// reasoning fields
	Summary   []interface{}            `json:"summary,omitempty"`
}

type responsesOutputContent struct {
	Type        string        `json:"type"` // "output_text", "refusal", "reasoning_text"
	Text        string        `json:"text,omitempty"`
	Annotations []interface{} `json:"annotations,omitempty"`
}

type responsesUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

type responsesError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code,omitempty"`
}


// ============================================================
// handleResponses — POST /v1/responses
// ============================================================

func handleResponses(w http.ResponseWriter, r *http.Request) {
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

	// 不支持有状态调用
	var fullReq responsesReq
	json.Unmarshal(body, &fullReq)
	if fullReq.PreviousResponseID != "" {
		writeJSON(w, http.StatusBadRequest, map[string]interface{}{
			"error": map[string]interface{}{
				"message": "previous_response_id is not supported by this proxy",
				"type":    "invalid_request_error",
			},
		})
		return
	}

	// 上游原生支持 Responses API → 直接透传
	if p.Type == ProviderResponses {
		var m map[string]interface{}
		if err := json.Unmarshal(body, &m); err != nil || m == nil {
			writeJSON(w, http.StatusBadRequest, map[string]interface{}{
				"error": map[string]interface{}{"message": "invalid JSON body", "type": "invalid_request_error"},
			})
			return
		}
		m["model"] = p.Name
		rewritten, _ := json.Marshal(m)
		forwardUpstream(w, r, p.ResponsesURL, p.APIKey, rewritten, nil)
		return
	}

	// Responses → IR → 上游格式
	ir, err := responsesToIR(body)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]interface{}{
			"error": map[string]interface{}{"message": err.Error(), "type": "invalid_request_error"},
		})
		return
	}

	switch p.Type {
	case ProviderOpenAI:
		handleResponsesToOpenAI(w, r, p, ir)
	case ProviderAnthropic:
		handleResponsesToAnthropic(w, r, p, ir)
	}
}

func handleResponsesToOpenAI(w http.ResponseWriter, r *http.Request, p *Provider, ir *IRRequest) {
	ir.Model = p.Name
	oaReq := irToChatCompletions(ir)
	applyExtraParams(oaReq, p.ExtraParams)
	oaBody, err := json.Marshal(oaReq)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	ctx := r.Context()
	req2, err := http.NewRequestWithContext(ctx, "POST", p.ChatURL, bytes.NewReader(oaBody))
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

		if err := translateChatCompletionsToResponsesStream(ctx, resp.Body, w, flusher, ir.Model); err != nil {
			return
		}
		return
	}

	// Non-streaming: Chat Completions → IR → Responses
	resBody, _ := io.ReadAll(resp.Body)
	irResp := chatCompletionsToIR(resBody, ir.Model)
	responsesResp := irToResponses(irResp)
	writeJSON(w, resp.StatusCode, responsesResp)
}

func handleResponsesToAnthropic(w http.ResponseWriter, r *http.Request, p *Provider, ir *IRRequest) {
	ir.Model = p.Name
	anthReq := irToAnthropicRequest(ir)
	anthBody, err := json.Marshal(anthReq)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
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

		if err := translateAnthropicToResponsesStream(ctx, resp.Body, w, flusher, ir.Model); err != nil {
			return
		}
		return
	}

	// Non-streaming: Anthropic → IR → Responses
	resBody, _ := io.ReadAll(resp.Body)
	irResp := anthropicToIR(resBody, ir.Model)
	responsesResp := irToResponses(irResp)
	writeJSON(w, resp.StatusCode, responsesResp)
}

// ============================================================
// 流式转换：Chat Completions SSE → Responses SSE
// ============================================================

// buildOutput constructs the output array for response.completed events.
type completedToolCall struct {
	callID, name, args string
}

func buildOutput(reasoningText, textContent string, toolCalls []completedToolCall, reasoningID, msgID string) []interface{} {
	var output []interface{}
	if reasoningText != "" {
		output = append(output, map[string]interface{}{
			"type": "reasoning", "id": reasoningID,
			"content": []interface{}{map[string]interface{}{"type": "reasoning_text", "text": reasoningText}},
			"summary": []interface{}{map[string]interface{}{"type": "summary_text", "text": reasoningText}},
			"status":  "completed",
		})
	}
	if textContent != "" {
		output = append(output, map[string]interface{}{
			"type": "message", "id": msgID, "role": "assistant",
			"content": []interface{}{map[string]interface{}{"type": "output_text", "text": textContent, "annotations": []interface{}{}}},
			"status":  "completed",
		})
	}
	for _, tc := range toolCalls {
		output = append(output, map[string]interface{}{
			"type": "function_call", "id": newFunctionCallID(),
			"call_id": tc.callID, "name": tc.name, "arguments": tc.args,
			"status": "completed",
		})
	}
	return output
}

func translateChatCompletionsToResponsesStream(ctx context.Context, upstream io.Reader, w io.Writer, flusher http.Flusher, model string) error {
	respID := newResponseID()
	msgID := newMessageID()
	reasoningID := newReasoningID()
	var started, hasReasoning, hasMessage, hasContentPart bool
	var outputIdx int
	var inputTokens, outputTokens int
	var reasoningContent strings.Builder
	var textContent strings.Builder
	var pendingToolID, pendingToolName, pendingToolCallID, pendingToolArgs string // deferred close for function_call items
	var completedToolCalls []completedToolCall

	emit := func(event string, data interface{}) error {
		return emitSSE(w, flusher, event, data)
	}

	scanner := newSSEScanner(upstream)

	ensureStarted := func() error {
		if started {
			return nil
		}
		started = true
		return emit("response.created", map[string]interface{}{
			"type": "response.created",
			"response": map[string]interface{}{
				"id": respID, "object": "response", "created_at": time.Now().Unix(),
				"model": model, "output": []interface{}{}, "status": "in_progress",
			},
		})
	}

	flushReasoning := func() error {
		if !hasReasoning {
			return nil
		}
		reasoningText := reasoningContent.String()
		if err := emit("response.output_item.done", map[string]interface{}{
			"type": "response.output_item.done", "output_index": outputIdx,
			"item": map[string]interface{}{
				"type": "reasoning", "id": reasoningID,
				"content": []interface{}{map[string]interface{}{"type": "reasoning_text", "text": reasoningText}},
				"summary": []interface{}{map[string]interface{}{"type": "summary_text", "text": reasoningText}},
				"status": "completed",
			},
		}); err != nil {
			return err
		}
		outputIdx++
		hasReasoning = false
		reasoningID = newReasoningID()
		reasoningContent.Reset()
		return nil
	}

	flushMessage := func() error {
		if hasContentPart {
			if err := emit("response.content_part.done", map[string]interface{}{
				"type": "response.content_part.done", "output_index": outputIdx, "content_index": 0,
				"part": map[string]interface{}{"type": "output_text", "text": "", "annotations": []interface{}{}},
			}); err != nil {
				return err
			}
			hasContentPart = false
		}
		if hasMessage {
			if err := emit("response.output_item.done", map[string]interface{}{
				"type": "response.output_item.done", "output_index": outputIdx,
				"item": map[string]interface{}{
					"type": "message", "id": msgID, "role": "assistant",
					"content": []interface{}{map[string]interface{}{"type": "output_text", "text": textContent.String(), "annotations": []interface{}{}}},
					"status": "completed",
				},
			}); err != nil {
				return err
			}
			outputIdx++
			hasMessage = false
			msgID = newMessageID()
		}
		return nil
	}

	closePendingTool := func() error {
		if pendingToolID != "" {
			if err := emit("response.output_item.done", map[string]interface{}{
				"type": "response.output_item.done", "output_index": outputIdx,
				"item": map[string]interface{}{
					"type": "function_call", "id": pendingToolID,
					"call_id": pendingToolCallID, "name": pendingToolName, "arguments": pendingToolArgs,
					"status": "completed",
				},
			}); err != nil {
				return err
			}
			completedToolCalls = append(completedToolCalls, completedToolCall{
				callID: pendingToolCallID, name: pendingToolName, args: pendingToolArgs,
			})
			outputIdx++
			pendingToolID = ""
			pendingToolName = ""
			pendingToolCallID = ""
			pendingToolArgs = ""
		}
		return nil
	}

	closeAll := func() error {
		if err := closePendingTool(); err != nil {
			return err
		}
		if err := flushMessage(); err != nil {
			return err
		}
		return flushReasoning()
	}

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
		if data == "[DONE]" {
			if err := ensureStarted(); err != nil {
				return err
			}
			if err := closeAll(); err != nil {
				return err
			}
			return emit("response.completed", map[string]interface{}{
				"type": "response.completed",
				"response": map[string]interface{}{
					"id": respID, "object": "response", "created_at": time.Now().Unix(),
					"model": model, "status": "completed",
					"output":   buildOutput(reasoningContent.String(), textContent.String(), completedToolCalls, reasoningID, msgID),
					"usage": map[string]interface{}{
						"input_tokens": inputTokens, "output_tokens": outputTokens,
						"total_tokens": inputTokens + outputTokens,
					},
				},
			})
		}

		var chunk struct {
			Choices []struct {
				Delta struct {
					Role             string `json:"role"`
					Content          string `json:"content"`
					ReasoningContent string `json:"reasoning_content"`
					ToolCalls        []struct {
						Index    int    `json:"index"`
						ID       string `json:"id"`
						Type     string `json:"type"`
						Function struct {
							Name      string `json:"name"`
							Arguments string `json:"arguments"`
						} `json:"function"`
					} `json:"tool_calls"`
				} `json:"delta"`
				FinishReason *string `json:"finish_reason"`
			} `json:"choices"`
			Usage *struct {
				PromptTokens     int `json:"prompt_tokens"`
				CompletionTokens int `json:"completion_tokens"`
			} `json:"usage"`
		}
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}
		if len(chunk.Choices) == 0 {
			if chunk.Usage != nil {
				inputTokens = chunk.Usage.PromptTokens
				outputTokens = chunk.Usage.CompletionTokens
			}
			continue
		}
		choice := chunk.Choices[0]

		// Reasoning content → reasoning output item
		if choice.Delta.ReasoningContent != "" {
			if err := ensureStarted(); err != nil {
				return err
			}
			if !hasReasoning {
				hasReasoning = true
				if err := emit("response.output_item.added", map[string]interface{}{
					"type": "response.output_item.added", "output_index": outputIdx,
					"item": map[string]interface{}{
						"type": "reasoning", "id": reasoningID,
						"content": []interface{}{}, "summary": []interface{}{},
						"status": "in_progress",
					},
				}); err != nil {
					return err
				}
			}
			reasoningContent.WriteString(choice.Delta.ReasoningContent)
		}

		// Text content → output_text delta
		if choice.Delta.Content != "" {
			if err := flushReasoning(); err != nil {
				return err
			}
			if !hasMessage {
				hasMessage = true
				if err := emit("response.output_item.added", map[string]interface{}{
					"type": "response.output_item.added", "output_index": outputIdx,
					"item": map[string]interface{}{
						"type": "message", "id": msgID, "role": "assistant",
						"content": []interface{}{}, "status": "in_progress",
					},
				}); err != nil {
					return err
				}
			}
			if !hasContentPart {
				hasContentPart = true
				if err := emit("response.content_part.added", map[string]interface{}{
					"type": "response.content_part.added", "output_index": outputIdx, "content_index": 0,
					"part": map[string]interface{}{"type": "output_text", "text": "", "annotations": []interface{}{}},
				}); err != nil {
					return err
				}
			}
			textContent.WriteString(choice.Delta.Content)
			if err := emit("response.output_text.delta", map[string]interface{}{
				"type": "response.output_text.delta", "output_index": outputIdx, "content_index": 0,
				"delta": choice.Delta.Content,
			}); err != nil {
				return err
			}
		}

		// Tool calls → function_call items
		for _, tc := range choice.Delta.ToolCalls {
			if tc.Function.Name != "" {
				// Close any previous pending tool call
				if err := closePendingTool(); err != nil {
					return err
				}
				// Close current message/reasoning
				if err := flushMessage(); err != nil {
					return err
				}
				if err := flushReasoning(); err != nil {
					return err
				}
				pendingToolID = newFunctionCallID()
				pendingToolName = tc.Function.Name
				pendingToolCallID = tc.ID
				pendingToolArgs = ""
				if err := emit("response.output_item.added", map[string]interface{}{
					"type": "response.output_item.added", "output_index": outputIdx,
					"item": map[string]interface{}{
						"type": "function_call", "id": pendingToolID,
						"call_id": tc.ID, "name": pendingToolName, "arguments": "",
					},
				}); err != nil {
					return err
				}
			}
			if tc.Function.Arguments != "" {
				pendingToolArgs += tc.Function.Arguments
				if err := emit("response.function_call_arguments.delta", map[string]interface{}{
					"type": "response.function_call_arguments.delta", "output_index": outputIdx,
					"delta": tc.Function.Arguments,
				}); err != nil {
					return err
				}
			}
		}

		if choice.FinishReason != nil {
			if chunk.Usage != nil {
				inputTokens = chunk.Usage.PromptTokens
				outputTokens = chunk.Usage.CompletionTokens
			}
			if err := closeAll(); err != nil {
				return err
			}
			status := "completed"
			if choice.FinishReason != nil && *choice.FinishReason == "length" {
				status = "incomplete"
			}
			return emit("response.completed", map[string]interface{}{
				"type": "response.completed",
				"response": map[string]interface{}{
					"id": respID, "object": "response", "created_at": time.Now().Unix(),
					"model": model, "status": status,
					"output":   buildOutput(reasoningContent.String(), textContent.String(), completedToolCalls, reasoningID, msgID),
					"usage": map[string]interface{}{
						"input_tokens": inputTokens, "output_tokens": outputTokens,
						"total_tokens": inputTokens + outputTokens,
					},
				},
			})
		}
	}
	return scanner.Err()
}

// ============================================================
// 流式转换：Anthropic SSE → Responses SSE
// ============================================================

func translateAnthropicToResponsesStream(ctx context.Context, upstream io.Reader, w io.Writer, flusher http.Flusher, model string) error {
	respID := newResponseID()
	msgID := newMessageID()
	reasoningID := newReasoningID()
	var started, hasReasoning, hasMessage, hasContentPart, hasFunctionCall bool
	var inputTokens, outputTokens int
	var outputIdx int
	var reasoningContent strings.Builder
	var textContent strings.Builder
	var completedToolCalls []completedToolCall
	var functionCallID, functionCallName, functionCallArgs, functionCallOriginalID string
	var anthStopReason string

	emit := func(event string, data interface{}) error {
		return emitSSE(w, flusher, event, data)
	}

	ensureStarted := func() error {
		if started {
			return nil
		}
		started = true
		return emit("response.created", map[string]interface{}{
			"type": "response.created",
			"response": map[string]interface{}{
				"id": respID, "object": "response", "created_at": time.Now().Unix(),
				"model": model, "output": []interface{}{}, "status": "in_progress",
			},
		})
	}

	flushFunctionCall := func() error {
		if !hasFunctionCall {
			return nil
		}
		if err := emit("response.output_item.done", map[string]interface{}{
			"type": "response.output_item.done", "output_index": outputIdx,
			"item": map[string]interface{}{
				"type": "function_call", "id": functionCallID,
				"call_id": functionCallOriginalID, "name": functionCallName, "arguments": functionCallArgs,
				"status": "completed",
			},
		}); err != nil {
			return err
		}
		completedToolCalls = append(completedToolCalls, completedToolCall{
			callID: functionCallOriginalID, name: functionCallName, args: functionCallArgs,
		})
		outputIdx++
		hasFunctionCall = false
		functionCallArgs = ""
		return nil
	}

	flushReasoning := func() error {
		if !hasReasoning {
			return nil
		}
		if err := emit("response.output_item.done", map[string]interface{}{
			"type": "response.output_item.done", "output_index": outputIdx,
			"item": map[string]interface{}{
				"type": "reasoning", "id": reasoningID,
				"content": []interface{}{map[string]interface{}{"type": "reasoning_text", "text": reasoningContent.String()}},
				"summary": []interface{}{map[string]interface{}{"type": "summary_text", "text": reasoningContent.String()}},
				"status": "completed",
			},
		}); err != nil {
			return err
		}
		outputIdx++
		hasReasoning = false
		reasoningID = newReasoningID()
		reasoningContent.Reset()
		return nil
	}

	flushMessage := func() error {
		if hasContentPart {
			if err := emit("response.content_part.done", map[string]interface{}{
				"type": "response.content_part.done", "output_index": outputIdx, "content_index": 0,
				"part": map[string]interface{}{"type": "output_text", "text": "", "annotations": []interface{}{}},
			}); err != nil {
				return err
			}
			hasContentPart = false
		}
		if hasMessage {
			if err := emit("response.output_item.done", map[string]interface{}{
				"type": "response.output_item.done", "output_index": outputIdx,
				"item": map[string]interface{}{
					"type": "message", "id": msgID, "role": "assistant",
					"content": []interface{}{map[string]interface{}{"type": "output_text", "text": textContent.String(), "annotations": []interface{}{}}},
					"status": "completed",
				},
			}); err != nil {
				return err
			}
			outputIdx++
			hasMessage = false
			msgID = newMessageID()
		}
		return nil
	}

	scanner := newSSEScanner(upstream)

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
		dataStr := strings.TrimPrefix(line, "data: ")
		if dataStr == "" {
			continue
		}

		var event map[string]interface{}
		if err := json.Unmarshal([]byte(dataStr), &event); err != nil {
			continue
		}

		etype, _ := event["type"].(string)

		switch etype {
		case "message_start":
			if err := ensureStarted(); err != nil {
				return err
			}
			if msg, ok := event["message"].(map[string]interface{}); ok {
				if usage, ok := msg["usage"].(map[string]interface{}); ok {
					if v, ok := usage["input_tokens"].(float64); ok {
						inputTokens = int(v)
					}
				}
			}

		case "content_block_start":
			if err := ensureStarted(); err != nil {
				return err
			}
			cb, _ := event["content_block"].(map[string]interface{})
			cbType, _ := cb["type"].(string)

			switch cbType {
			case "thinking":
				hasReasoning = true
				if err := emit("response.output_item.added", map[string]interface{}{
					"type": "response.output_item.added", "output_index": outputIdx,
					"item": map[string]interface{}{
						"type": "reasoning", "id": reasoningID,
						"content": []interface{}{}, "summary": []interface{}{},
						"status": "in_progress",
					},
				}); err != nil {
					return err
				}
			case "text":
				if err := flushReasoning(); err != nil {
					return err
				}
				hasMessage = true
				if err := emit("response.output_item.added", map[string]interface{}{
					"type": "response.output_item.added", "output_index": outputIdx,
					"item": map[string]interface{}{
						"type": "message", "id": msgID, "role": "assistant",
						"content": []interface{}{}, "status": "in_progress",
					},
				}); err != nil {
					return err
				}
				hasContentPart = true
				if err := emit("response.content_part.added", map[string]interface{}{
					"type": "response.content_part.added", "output_index": outputIdx, "content_index": 0,
					"part": map[string]interface{}{"type": "output_text", "text": "", "annotations": []interface{}{}},
				}); err != nil {
					return err
				}
			case "tool_use":
				if err := flushFunctionCall(); err != nil {
					return err
				}
				if err := flushMessage(); err != nil {
					return err
				}
				if err := flushReasoning(); err != nil {
					return err
				}
				toolID, _ := cb["id"].(string)
				toolName, _ := cb["name"].(string)
				functionCallID = newFunctionCallID()
				functionCallName = toolName
				functionCallArgs = ""
				functionCallOriginalID = toolID
				hasFunctionCall = true
				if err := emit("response.output_item.added", map[string]interface{}{
					"type": "response.output_item.added", "output_index": outputIdx,
					"item": map[string]interface{}{
						"type": "function_call", "id": functionCallID,
						"call_id": toolID, "name": toolName, "arguments": "",
					},
				}); err != nil {
					return err
				}
			}

		case "content_block_delta":
			if err := ensureStarted(); err != nil {
				return err
			}
			delta, _ := event["delta"].(map[string]interface{})
			deltaType, _ := delta["type"].(string)

			switch deltaType {
			case "thinking_delta":
				thinking, _ := delta["thinking"].(string)
				if thinking != "" && hasReasoning {
					reasoningContent.WriteString(thinking)
					if err := emit("response.output_text.delta", map[string]interface{}{
						"type": "response.output_text.delta", "output_index": outputIdx, "content_index": 0,
						"delta": thinking,
					}); err != nil {
						return err
					}
				}
			case "text_delta":
				text, _ := delta["text"].(string)
				if text != "" {
					textContent.WriteString(text)
				}
				if text != "" && hasMessage {
					if err := emit("response.output_text.delta", map[string]interface{}{
						"type": "response.output_text.delta", "output_index": outputIdx, "content_index": 0,
						"delta": text,
					}); err != nil {
						return err
					}
				}
			case "input_json_delta":
				partialJSON, _ := delta["partial_json"].(string)
				if partialJSON != "" {
					functionCallArgs += partialJSON
					if err := emit("response.function_call_arguments.delta", map[string]interface{}{
						"type": "response.function_call_arguments.delta", "output_index": outputIdx,
						"delta": partialJSON,
					}); err != nil {
						return err
					}
				}
			}

		case "content_block_stop":
			cbType, _ := event["content_block_type"].(string)
			_ = cbType // type info not always present; rely on state flags

			if hasContentPart {
				if err := flushMessage(); err != nil {
					return err
				}
			} else if hasReasoning {
				if err := flushReasoning(); err != nil {
					return err
				}
			} else if hasFunctionCall {
				if err := flushFunctionCall(); err != nil {
					return err
				}
			}

		case "message_delta":
			if err := ensureStarted(); err != nil {
				return err
			}
			if usage, ok := event["usage"].(map[string]interface{}); ok {
				if v, ok := usage["output_tokens"].(float64); ok {
					outputTokens = int(v)
				}
			}
			if delta, ok := event["delta"].(map[string]interface{}); ok {
				if sr, ok := delta["stop_reason"].(string); ok {
					anthStopReason = sr
				}
			}

		case "message_stop":
			if err := ensureStarted(); err != nil {
				return err
			}
			// Flush any remaining items
			if err := flushMessage(); err != nil {
				return err
			}
			if err := flushReasoning(); err != nil {
				return err
			}
			if err := flushFunctionCall(); err != nil {
				return err
			}
			status := "completed"
			if anthStopReason == "max_tokens" {
				status = "incomplete"
			}
			return emit("response.completed", map[string]interface{}{
				"type": "response.completed",
				"response": map[string]interface{}{
					"id": respID, "object": "response", "created_at": time.Now().Unix(),
					"model": model, "status": status,
					"output":   buildOutput(reasoningContent.String(), textContent.String(), completedToolCalls, reasoningID, msgID),
					"usage": map[string]interface{}{
						"input_tokens": inputTokens, "output_tokens": outputTokens,
						"total_tokens": inputTokens + outputTokens,
					},
				},
			})
		}
	}
	return scanner.Err()
}


// ============================================================
// Responses 响应 → IR 转换
// ============================================================

func responsesToIRResponse(body []byte, model string) *IRResponse {
	var resp responsesResp
	if err := json.Unmarshal(body, &resp); err != nil {
		return &IRResponse{
			ID:         newResponseID(),
			Model:      model,
			Role:       "assistant",
			Content:    []IRContentBlock{{Type: "text", Text: "[upstream response parse error]"}},
			StopReason: "end_turn",
		}
	}

	ir := &IRResponse{
		ID:         resp.ID,
		Model:      model,
		Role:       "assistant",
		StopReason: "end_turn",
	}

	if resp.Error != nil {
		ir.Content = []IRContentBlock{{Type: "text", Text: resp.Error.Message}}
		return ir
	}

	for _, item := range resp.Output {
		switch item.Type {
		case "message":
			for _, c := range item.Content {
				switch c.Type {
				case "output_text":
					ir.Content = append(ir.Content, IRContentBlock{Type: "text", Text: c.Text})
				}
			}
		case "reasoning":
			for _, c := range item.Content {
				if c.Type == "reasoning_text" {
					ir.Content = append(ir.Content, IRContentBlock{Type: "thinking", Thinking: c.Text})
				}
			}
		case "function_call":
			var input map[string]interface{}
			json.Unmarshal([]byte(item.Arguments), &input)
			ir.Content = append(ir.Content, IRContentBlock{
				Type:      "tool_use",
				ToolUseID: item.CallID,
				ToolName:  item.Name,
				ToolInput: input,
			})
		}
	}

	// Map upstream status to stop_reason
	switch resp.Status {
	case "incomplete":
		ir.StopReason = "max_tokens"
	case "failed":
		ir.StopReason = "end_turn"
	}

	// If output contains function_call items, set stop reason to tool_use
	for _, b := range ir.Content {
		if b.Type == "tool_use" {
			ir.StopReason = "tool_use"
			break
		}
	}

	if resp.Usage != nil {
		ir.Usage = IRUsage{
			InputTokens:  resp.Usage.InputTokens,
			OutputTokens: resp.Usage.OutputTokens,
		}
	}
	return ir
}

// ============================================================
// Responses SSE → Anthropic SSE 流式转换
// ============================================================

func translateResponsesToAnthropicStream(ctx context.Context, upstream io.Reader, w io.Writer, flusher http.Flusher, model string) error {
	msgID := newMessageID()
	var started, isReasoning, hasToolUse bool
	var inputTokens, outputTokens int
	var blockIndex int
	var hasTextBlock, hasThinkingBlock, hasToolBlock bool

	emit := func(event string, data interface{}) error {
		return emitSSE(w, flusher, event, data)
	}

	emitMessageStart := func() error {
		return emit("message_start", map[string]interface{}{
			"type": "message_start",
			"message": map[string]interface{}{
				"id": msgID, "type": "message", "role": "assistant", "content": []interface{}{},
				"model": model, "stop_reason": nil, "stop_sequence": nil,
				"usage": map[string]interface{}{"input_tokens": inputTokens, "output_tokens": 0},
			},
		})
	}

	closeCurrentBlock := func() error {
		if hasTextBlock || hasThinkingBlock || hasToolBlock {
			if err := emit("content_block_stop", map[string]interface{}{
				"type": "content_block_stop", "index": blockIndex,
			}); err != nil {
				return err
			}
			blockIndex++
			hasTextBlock = false
			hasThinkingBlock = false
			hasToolBlock = false
		}
		return nil
	}

	scanner := newSSEScanner(upstream)

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
		case "response.created", "response.in_progress":
			if started {
				continue
			}
			started = true
			if err := emitMessageStart(); err != nil {
				return err
			}

		case "response.output_item.added":
			item, _ := event["item"].(map[string]interface{})
			itemType, _ := item["type"].(string)
			switch itemType {
			case "reasoning":
				isReasoning = true
				if err := closeCurrentBlock(); err != nil {
					return err
				}
				hasThinkingBlock = true
				if err := emit("content_block_start", map[string]interface{}{
					"type": "content_block_start", "index": blockIndex,
					"content_block": map[string]interface{}{"type": "thinking", "thinking": ""},
				}); err != nil {
					return err
				}
			case "message":
				isReasoning = false
			case "function_call":
				hasToolUse = true
				if err := closeCurrentBlock(); err != nil {
					return err
				}
				hasToolBlock = true
				if err := emit("content_block_start", map[string]interface{}{
					"type": "content_block_start", "index": blockIndex,
					"content_block": map[string]interface{}{
						"type": "tool_use", "id": item["call_id"], "name": item["name"], "input": map[string]interface{}{},
					},
				}); err != nil {
					return err
				}
			}

		case "response.content_part.added":
			if !started {
				started = true
				if err := emitMessageStart(); err != nil {
					return err
				}
			}
			if err := closeCurrentBlock(); err != nil {
				return err
			}
			hasTextBlock = true
			if err := emit("content_block_start", map[string]interface{}{
				"type": "content_block_start", "index": blockIndex,
				"content_block": map[string]interface{}{"type": "text", "text": ""},
			}); err != nil {
				return err
			}

		case "response.output_text.delta":
			delta, _ := event["delta"].(string)
			if delta != "" {
				if isReasoning {
					if err := emit("content_block_delta", map[string]interface{}{
						"type": "content_block_delta", "index": blockIndex,
						"delta": map[string]interface{}{"type": "thinking_delta", "thinking": delta},
					}); err != nil {
						return err
					}
				} else {
					if err := emit("content_block_delta", map[string]interface{}{
						"type": "content_block_delta", "index": blockIndex,
						"delta": map[string]interface{}{"type": "text_delta", "text": delta},
					}); err != nil {
						return err
					}
				}
			}

		case "response.function_call_arguments.delta":
			delta, _ := event["delta"].(string)
			if delta != "" {
				if err := emit("content_block_delta", map[string]interface{}{
					"type": "content_block_delta", "index": blockIndex,
					"delta": map[string]interface{}{"type": "input_json_delta", "partial_json": delta},
				}); err != nil {
					return err
				}
			}

		case "response.content_part.done":
			if err := closeCurrentBlock(); err != nil {
				return err
			}

		case "response.output_item.done":
			item, _ := event["item"].(map[string]interface{})
			itemType, _ := item["type"].(string)
			switch itemType {
			case "reasoning":
				if hasThinkingBlock {
					if err := closeCurrentBlock(); err != nil {
						return err
					}
				}
				isReasoning = false
			case "function_call":
				// Don't close block here; let response.completed handle it
				// to avoid resetting hasToolBlock before the final close
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
			if !started {
				started = true
				if err := emitMessageStart(); err != nil {
					return err
				}
			}
			// Close any remaining open block
			if err := closeCurrentBlock(); err != nil {
				return err
			}
			stopReason := "end_turn"
			if hasToolUse {
				stopReason = "tool_use"
			}
			// Check upstream status for incomplete responses
			if respStatus, ok := respData["status"].(string); ok && respStatus == "incomplete" {
				stopReason = "max_tokens"
			}
			if err := emit("message_delta", map[string]interface{}{
				"type": "message_delta",
				"delta": map[string]interface{}{"stop_reason": stopReason, "stop_sequence": nil},
				"usage": map[string]interface{}{"output_tokens": outputTokens},
			}); err != nil {
				return err
			}
			return emit("message_stop", map[string]interface{}{"type": "message_stop"})
		}
	}
	return scanner.Err()
}
