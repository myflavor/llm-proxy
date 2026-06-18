package main

import (
	"bufio"
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
}

type responsesInputItem struct {
	Type    string      `json:"type"` // "message", "function_call_output"
	Role    string      `json:"role,omitempty"`
	Content interface{} `json:"content,omitempty"` // string or []responsesContentPart
	CallID  string      `json:"call_id,omitempty"` // for function_call_output
	Output  string      `json:"output,omitempty"`  // for function_call_output
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

// applyExtraParams 将 Provider 的 ExtraParams 合并到请求体 map 中。
func applyExtraParams(req map[string]interface{}, extra map[string]interface{}) {
	for k, v := range extra {
		req[k] = v
	}
}

// ============================================================
// Responses → IR 转换
// ============================================================

func responsesToIR(body []byte) (*IRRequest, error) {
	var req responsesReq
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, err
	}

	ir := &IRRequest{
		Model:       req.Model,
		Stream:      req.Stream,
		MaxTokens:   req.MaxOutputTokens,
		Temperature: req.Temperature,
		TopP:        req.TopP,
	}

	// instructions → system prompt
	ir.SystemPrompt = req.Instructions

	// input → messages
	switch v := req.Input.(type) {
	case string:
		if v != "" {
			ir.Messages = append(ir.Messages, IRMessage{
				Role:    "user",
				Content: []IRContentBlock{{Type: "text", Text: v}},
			})
		}
	case []interface{}:
		for _, raw := range v {
			itemBytes, _ := json.Marshal(raw)
			var item responsesInputItem
			if err := json.Unmarshal(itemBytes, &item); err != nil {
				continue
			}
			switch item.Type {
			case "message":
				msg := IRMessage{Role: item.Role}
				switch c := item.Content.(type) {
				case string:
					if c != "" {
						msg.Content = append(msg.Content, IRContentBlock{Type: "text", Text: c})
					}
				case []interface{}:
					for _, partRaw := range c {
						partBytes, _ := json.Marshal(partRaw)
						var part responsesContentPart
						if err := json.Unmarshal(partBytes, &part); err != nil {
							continue
						}
						switch part.Type {
						case "input_text":
							msg.Content = append(msg.Content, IRContentBlock{Type: "text", Text: part.Text})
						case "input_image":
							msg.Content = append(msg.Content, IRContentBlock{Type: "image", ImageURL: part.ImageURL})
						}
					}
				}
				if len(msg.Content) > 0 {
					ir.Messages = append(ir.Messages, msg)
				}
			case "function_call_output":
				ir.Messages = append(ir.Messages, IRMessage{
					Role:       "tool",
					ToolCallID: item.CallID,
					Content:    []IRContentBlock{{Type: "text", Text: item.Output}},
				})
			}
		}
	}

	// tools
	for _, t := range req.Tools {
		if t.Type == "function" {
			ir.Tools = append(ir.Tools, IRTool{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  t.Parameters,
			})
		}
	}

	// reasoning
	if req.Reasoning != nil {
		ir.Thinking = &IRThinking{Enabled: true, Effort: req.Reasoning.Effort}
	}

	// text.format → extensions
	if req.Text != nil && req.Text.Format != nil {
		ir.Extensions = map[string]interface{}{
			"response_format": req.Text.Format,
		}
	}

	return ir, nil
}

// ============================================================
// IR → Chat Completions 请求转换
// ============================================================

func irToChatCompletions(ir *IRRequest) map[string]interface{} {
	oa := map[string]interface{}{
		"model":  ir.Model,
		"stream": ir.Stream,
	}
	if ir.Stream {
		oa["stream_options"] = map[string]interface{}{"include_usage": true}
	}
	if ir.MaxTokens > 0 {
		oa["max_tokens"] = ir.MaxTokens
	}
	if ir.Temperature != nil {
		oa["temperature"] = *ir.Temperature
	}
	if ir.TopP != nil {
		oa["top_p"] = *ir.TopP
	}
	if len(ir.StopSequences) > 0 {
		oa["stop"] = ir.StopSequences
	}
	if ir.Metadata != nil && ir.Metadata.UserID != "" {
		oa["user"] = ir.Metadata.UserID
	}
	// structured output
	if ir.Extensions != nil {
		if rf, ok := ir.Extensions["response_format"]; ok {
			oa["response_format"] = rf
		}
	}

	// messages
	var messages []interface{}
	if ir.SystemPrompt != "" {
		messages = append(messages, map[string]interface{}{"role": "system", "content": ir.SystemPrompt})
	}
	for _, m := range ir.Messages {
		if m.Role == "assistant" {
			msg := map[string]interface{}{"role": "assistant"}
			var textParts []string
			var reasoningParts []string
			var toolCalls []interface{}
			for _, b := range m.Content {
				switch b.Type {
				case "text":
					textParts = append(textParts, b.Text)
				case "thinking":
					reasoningParts = append(reasoningParts, b.Thinking)
				case "tool_use":
					toolCalls = append(toolCalls, map[string]interface{}{
						"id":   b.ToolUseID,
						"type": "function",
						"function": map[string]interface{}{
							"name":      b.ToolName,
							"arguments": mustJSON(b.ToolInput),
						},
					})
				}
			}
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
		} else if m.Role == "tool" {
			messages = append(messages, map[string]interface{}{
				"role":         "tool",
				"tool_call_id": m.ToolCallID,
				"content":      extractText(m.Content),
			})
		} else {
			// user message — check for image blocks
			var contentParts []interface{}
			for _, b := range m.Content {
				switch b.Type {
				case "text":
					contentParts = append(contentParts, map[string]interface{}{"type": "text", "text": b.Text})
				case "image":
					contentParts = append(contentParts, map[string]interface{}{
						"type": "image_url",
						"image_url": map[string]interface{}{"url": b.ImageURL},
					})
				}
			}
			if len(contentParts) == 0 {
				continue
			}
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
	}
	oa["messages"] = messages

	// tools
	if len(ir.Tools) > 0 {
		var tools []interface{}
		for _, t := range ir.Tools {
			tools = append(tools, map[string]interface{}{
				"type": "function",
				"function": map[string]interface{}{
					"name":        t.Name,
					"description": t.Description,
					"parameters":  t.Parameters,
				},
			})
		}
		oa["tools"] = tools
	}

	// tool_choice
	if ir.ToolChoice != nil {
		switch ir.ToolChoice.Type {
		case "auto":
			oa["tool_choice"] = "auto"
		case "required", "any":
			oa["tool_choice"] = "required"
		case "none":
			oa["tool_choice"] = "none"
		case "specific":
			if ir.ToolChoice.Name != "" {
				oa["tool_choice"] = map[string]interface{}{
					"type": "function",
					"function": map[string]interface{}{"name": ir.ToolChoice.Name},
				}
			}
		}
	}

	// reasoning_effort
	if ir.Thinking != nil && ir.Thinking.Effort != "" {
		oa["reasoning_effort"] = ir.Thinking.Effort
	}

	return oa
}

// ============================================================
// Anthropic Messages 请求 → IR 转换
// ============================================================

func anthropicToIRRequest(req anthropicMsgReq) *IRRequest {
	ir := &IRRequest{
		Model:         req.Model,
		MaxTokens:     req.MaxTokens,
		Temperature:   req.Temperature,
		TopP:          req.TopP,
		StopSequences: req.StopSequences,
		Stream:        req.Stream,
	}

	// system → system prompt
	ir.SystemPrompt = extractText(req.System)

	// thinking
	if req.Thinking != nil && req.Thinking.Type == "enabled" {
		ir.Thinking = &IRThinking{Enabled: true, BudgetTokens: req.Thinking.BudgetTokens}
		if req.Thinking.BudgetTokens > 0 && ir.Thinking.Effort == "" {
			ir.Thinking.Effort = budgetToEffort(req.Thinking.BudgetTokens)
		}
	}
	if req.Effort != "" {
		if ir.Thinking == nil {
			ir.Thinking = &IRThinking{Enabled: true}
		}
		ir.Thinking.Effort = req.Effort
	}

	// metadata
	if req.Metadata != nil && req.Metadata.UserID != "" {
		ir.Metadata = &IRMetadata{UserID: req.Metadata.UserID}
	}

	// messages
	for _, m := range req.Messages {
		msg := IRMessage{Role: m.Role}
		if blocks, ok := m.Content.([]interface{}); ok {
			// Separate tool_results from other content.
			// Anthropic: tool_result is a block inside user message.
			// IR/OpenAI: tool result must be a separate role:"tool" message.
			var toolResults []IRMessage
			for _, block := range blocks {
				b, ok := block.(map[string]interface{})
				if !ok {
					continue
				}
				switch b["type"] {
				case "text":
					if txt, _ := b["text"].(string); txt != "" {
						msg.Content = append(msg.Content, IRContentBlock{Type: "text", Text: txt})
					}
				case "thinking":
					if txt, _ := b["thinking"].(string); txt != "" {
						msg.Content = append(msg.Content, IRContentBlock{Type: "thinking", Thinking: txt})
					}
				case "tool_use":
					input, _ := b["input"].(map[string]interface{})
					msg.Content = append(msg.Content, IRContentBlock{
						Type:      "tool_use",
						ToolUseID: b["id"].(string),
						ToolName:  b["name"].(string),
						ToolInput: input,
					})
				case "tool_result":
					toolCallID, _ := b["tool_use_id"].(string)
					resultContent := ""
					if c, ok := b["content"].(string); ok {
						resultContent = c
					} else if blocks2, ok := b["content"].([]interface{}); ok {
						resultContent = extractText(blocks2)
					}
					toolResults = append(toolResults, IRMessage{
						Role:       "tool",
						ToolCallID: toolCallID,
						Content:    []IRContentBlock{{Type: "text", Text: resultContent}},
					})
				case "image":
					if url := extractImageURL(b); url != "" {
						msg.Content = append(msg.Content, IRContentBlock{Type: "image", ImageURL: url})
					}
				}
			}
			// Append user message first (if it has non-tool content).
			if len(msg.Content) > 0 {
				ir.Messages = append(ir.Messages, msg)
			}
			// Then append tool result messages (must follow immediately).
			ir.Messages = append(ir.Messages, toolResults...)
		} else if contentStr, ok := m.Content.(string); ok {
			msg.Content = []IRContentBlock{{Type: "text", Text: contentStr}}
			if len(msg.Content) > 0 {
				ir.Messages = append(ir.Messages, msg)
			}
		}
	}

	// tools
	for _, t := range req.Tools {
		ir.Tools = append(ir.Tools, IRTool{
			Name:        t.Name,
			Description: t.Description,
			Parameters:  t.InputSchema,
		})
	}

	// tool_choice
	if req.ToolChoice != nil {
		tc := &IRToolChoice{}
		if m, ok := req.ToolChoice.(map[string]interface{}); ok {
			switch m["type"] {
			case "auto":
				tc.Type = "auto"
			case "any":
				tc.Type = "required"
			case "none":
				tc.Type = "none"
			case "tool":
				tc.Type = "specific"
				tc.Name, _ = m["name"].(string)
			}
		}
		ir.ToolChoice = tc
	}

	return ir
}

// ============================================================
// IR → Anthropic Messages 请求转换
// ============================================================

func irToAnthropicRequest(ir *IRRequest) *anthropicMsgReq {
	req := &anthropicMsgReq{
		Model:         ir.Model,
		MaxTokens:     ir.MaxTokens,
		Temperature:   ir.Temperature,
		TopP:          ir.TopP,
		StopSequences: ir.StopSequences,
		Stream:        ir.Stream,
	}

	if ir.SystemPrompt != "" {
		req.System = ir.SystemPrompt
	}

	for _, m := range ir.Messages {
		if m.Role == "assistant" {
			msg := anthropicMsg{Role: "assistant"}
			var blocks []interface{}
			for _, b := range m.Content {
				switch b.Type {
				case "text":
					blocks = append(blocks, map[string]interface{}{"type": "text", "text": b.Text})
				case "thinking":
					blocks = append(blocks, map[string]interface{}{"type": "thinking", "thinking": b.Thinking})
				case "tool_use":
					blocks = append(blocks, map[string]interface{}{
						"type": "tool_use", "id": b.ToolUseID, "name": b.ToolName, "input": b.ToolInput,
					})
				}
			}
			if len(blocks) > 0 {
				msg.Content = blocks
				req.Messages = append(req.Messages, msg)
			}
		} else if m.Role == "tool" {
			resultContent := extractText(m.Content)
			toolResult := map[string]interface{}{
				"type":        "tool_result",
				"tool_use_id": m.ToolCallID,
				"content":     resultContent,
			}
			req.Messages = append(req.Messages, anthropicMsg{
				Role:    "user",
				Content: []interface{}{toolResult},
			})
		} else {
			// user message — check for image blocks
			msg := anthropicMsg{Role: "user"}
			var blocks []interface{}
			for _, b := range m.Content {
				switch b.Type {
				case "text":
					blocks = append(blocks, map[string]interface{}{"type": "text", "text": b.Text})
				case "image":
					if strings.HasPrefix(b.ImageURL, "data:") {
						// data URL → base64 source
						parts := strings.SplitN(b.ImageURL, ",", 2)
						if len(parts) == 2 {
							headerParts := strings.SplitN(parts[0], ":", 2)
							mediaType := ""
							if len(headerParts) == 2 {
								mediaType = strings.TrimSuffix(strings.TrimPrefix(headerParts[1], "application/"), ";base64")
							}
							blocks = append(blocks, map[string]interface{}{
								"type": "image",
								"source": map[string]interface{}{
									"type":       "base64",
									"media_type": mediaType,
									"data":       parts[1],
								},
							})
						}
					} else {
						blocks = append(blocks, map[string]interface{}{
							"type": "image",
							"source": map[string]interface{}{
								"type": "url",
								"url":  b.ImageURL,
							},
						})
					}
				}
			}
			if len(blocks) > 0 {
				if len(blocks) == 1 {
					if b, ok := blocks[0].(map[string]interface{}); ok && b["type"] == "text" {
						msg.Content = b["text"]
					} else {
						msg.Content = blocks
					}
				} else {
					msg.Content = blocks
				}
				req.Messages = append(req.Messages, msg)
			}
		}
	}

	// tools
	for _, t := range ir.Tools {
		req.Tools = append(req.Tools, anthropicTool{
			Name:        t.Name,
			Description: t.Description,
			InputSchema: t.Parameters,
		})
	}

	// tool_choice
	if ir.ToolChoice != nil {
		switch ir.ToolChoice.Type {
		case "auto":
			req.ToolChoice = map[string]interface{}{"type": "auto"}
		case "required", "any":
			req.ToolChoice = map[string]interface{}{"type": "any"}
		case "none":
			req.ToolChoice = map[string]interface{}{"type": "none"}
		case "specific":
			if ir.ToolChoice.Name != "" {
				req.ToolChoice = map[string]interface{}{"type": "tool", "name": ir.ToolChoice.Name}
			}
		}
	}

	// thinking
	if ir.Thinking != nil && ir.Thinking.Enabled {
		if ir.Thinking.Effort != "" {
			// 新格式：adaptive thinking + effort
			req.Thinking = &anthropicThinking{Type: "adaptive"}
			req.Effort = clampEffortForAnthropic(ir.Thinking.Effort)
		} else if ir.Thinking.BudgetTokens > 0 {
			// 旧格式：enabled + budget_tokens
			req.Thinking = &anthropicThinking{Type: "enabled", BudgetTokens: ir.Thinking.BudgetTokens}
		} else {
			req.Thinking = &anthropicThinking{Type: "adaptive"}
		}
	}

	// metadata
	if ir.Metadata != nil && ir.Metadata.UserID != "" {
		req.Metadata = &anthropicMetadata{UserID: ir.Metadata.UserID}
	}

	return req
}

// ============================================================
// Chat Completions 请求 → IR 转换
// ============================================================

func chatCompletionsToIRRequest(body []byte) (*IRRequest, error) {
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
		ToolChoice      interface{} `json:"tool_choice,omitempty"`
		ResponseFormat  interface{} `json:"response_format,omitempty"`
		ReasoningEffort string      `json:"reasoning_effort,omitempty"`
	}
	if err := json.Unmarshal(body, &oa); err != nil {
		return nil, err
	}

	ir := &IRRequest{
		Model:         oa.Model,
		MaxTokens:     oa.MaxTokens,
		Temperature:   oa.Temperature,
		TopP:          oa.TopP,
		StopSequences: oa.Stop,
		Stream:        oa.Stream,
	}

	if oa.ResponseFormat != nil {
		ir.Extensions = map[string]interface{}{"response_format": oa.ResponseFormat}
	}

	for _, m := range oa.Messages {
		role, _ := m["role"].(string)
		if role == "system" {
			ir.SystemPrompt = extractText(m["content"])
			continue
		}
		if role == "tool" {
			toolCallID, _ := m["tool_call_id"].(string)
			content, _ := m["content"].(string)
			ir.Messages = append(ir.Messages, IRMessage{
				Role:       "tool",
				ToolCallID: toolCallID,
				Content:    []IRContentBlock{{Type: "text", Text: content}},
			})
			continue
		}
		msg := IRMessage{Role: role}
		if contentStr, ok := m["content"].(string); ok {
			msg.Content = []IRContentBlock{{Type: "text", Text: contentStr}}
		} else if contentArr, ok := m["content"].([]interface{}); ok {
			for _, part := range contentArr {
				pm, ok := part.(map[string]interface{})
				if !ok {
					continue
				}
				switch pm["type"] {
				case "text":
					if txt, _ := pm["text"].(string); txt != "" {
						msg.Content = append(msg.Content, IRContentBlock{Type: "text", Text: txt})
					}
				case "image_url":
					if imgURL, ok := pm["image_url"].(map[string]interface{}); ok {
						if url, _ := imgURL["url"].(string); url != "" {
							msg.Content = append(msg.Content, IRContentBlock{Type: "image", ImageURL: url})
						}
					}
				}
			}
		}
		// Assistant messages with tool_calls
		if role == "assistant" {
			if tcRaw, ok := m["tool_calls"].([]interface{}); ok {
				for _, t := range tcRaw {
					tc, ok := t.(map[string]interface{})
					if !ok {
						continue
					}
					fn, _ := tc["function"].(map[string]interface{})
					name, _ := fn["name"].(string)
					args, _ := fn["arguments"].(string)
					var input map[string]interface{}
					json.Unmarshal([]byte(args), &input)
					msg.Content = append(msg.Content, IRContentBlock{
						Type:      "tool_use",
						ToolUseID: tc["id"].(string),
						ToolName:  name,
						ToolInput: input,
					})
				}
				// reasoning_content
				if rc, ok := m["reasoning_content"].(string); ok && rc != "" {
					msg.Content = append([]IRContentBlock{{Type: "thinking", Thinking: rc}}, msg.Content...)
				}
			}
		}
		if len(msg.Content) > 0 {
			ir.Messages = append(ir.Messages, msg)
		}
	}

	for _, t := range oa.Tools {
		ir.Tools = append(ir.Tools, IRTool{
			Name:        t.Function.Name,
			Description: t.Function.Description,
			Parameters:  t.Function.Parameters,
		})
	}

	if oa.ToolChoice != nil {
		tc := &IRToolChoice{}
		switch v := oa.ToolChoice.(type) {
		case string:
			tc.Type = v
		case map[string]interface{}:
			if fn, ok := v["function"].(map[string]interface{}); ok {
				tc.Type = "specific"
				tc.Name, _ = fn["name"].(string)
			}
		}
		ir.ToolChoice = tc
	}

	if oa.ReasoningEffort != "" {
		ir.Thinking = &IRThinking{Enabled: true, Effort: oa.ReasoningEffort}
	}

	return ir, nil
}

// ============================================================
// IR → Chat Completions 响应转换
// ============================================================

func irToChatCompletionsResponse(ir *IRResponse) map[string]interface{} {
	msg := map[string]interface{}{"role": "assistant"}
	var content string
	var toolCalls []interface{}

	for _, b := range ir.Content {
		switch b.Type {
		case "text":
			content += b.Text
		case "thinking":
			msg["reasoning_content"] = b.Thinking
		case "tool_use":
			toolCalls = append(toolCalls, map[string]interface{}{
				"id":   b.ToolUseID,
				"type": "function",
				"function": map[string]interface{}{
					"name":      b.ToolName,
					"arguments": mustJSON(b.ToolInput),
				},
			})
		}
	}
	if content != "" {
		msg["content"] = content
	}
	if len(toolCalls) > 0 {
		msg["tool_calls"] = toolCalls
	}

	resp := map[string]interface{}{
		"id": "chatcmpl-" + randomHex(12), "object": "chat.completion", "created": time.Now().Unix(),
		"model": ir.Model,
		"choices": []interface{}{map[string]interface{}{
			"index": 0, "message": msg, "finish_reason": mapFinishReasonReverse(ir.StopReason),
		}},
	}
	if ir.Usage.InputTokens > 0 || ir.Usage.OutputTokens > 0 {
		resp["usage"] = map[string]interface{}{
			"prompt_tokens":     ir.Usage.InputTokens,
			"completion_tokens": ir.Usage.OutputTokens,
			"total_tokens":      ir.Usage.InputTokens + ir.Usage.OutputTokens,
		}
	}
	return resp
}

// ============================================================
// Chat Completions 响应 → IR 转换
// ============================================================

func chatCompletionsToIR(body []byte, model string) *IRResponse {
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

	if err := json.Unmarshal(body, &oa); err != nil {
		var errResp struct {
			Error struct {
				Message string `json:"message"`
			} `json:"error"`
		}
		msg := "[upstream response parse error]"
		if json.Unmarshal(body, &errResp) == nil && errResp.Error.Message != "" {
			msg = errResp.Error.Message
		}
		return &IRResponse{
			ID:         "resp_" + randomHex(12),
			Model:      model,
			Role:       "assistant",
			Content:    []IRContentBlock{{Type: "text", Text: msg}},
			StopReason: "end_turn",
		}
	}

	ir := &IRResponse{
		ID:    "resp_" + randomHex(12),
		Model: model,
		Role:  "assistant",
	}

	if len(oa.Choices) > 0 {
		msg := oa.Choices[0].Message
		if msg.Role != "" {
			ir.Role = msg.Role
		}
		if msg.ReasoningContent != "" {
			ir.Content = append(ir.Content, IRContentBlock{Type: "thinking", Thinking: msg.ReasoningContent})
		}
		if msg.Content != "" {
			ir.Content = append(ir.Content, IRContentBlock{Type: "text", Text: msg.Content})
		}
		for _, tc := range msg.ToolCalls {
			var input map[string]interface{}
			json.Unmarshal([]byte(tc.Function.Arguments), &input)
			ir.Content = append(ir.Content, IRContentBlock{
				Type:      "tool_use",
				ToolUseID: tc.ID,
				ToolName:  tc.Function.Name,
				ToolInput: input,
			})
		}
		if oa.Choices[0].FinishReason != nil {
			ir.StopReason = mapFinishReason(*oa.Choices[0].FinishReason)
		}
	}

	if oa.Usage != nil {
		ir.Usage = IRUsage{
			InputTokens:  oa.Usage.PromptTokens,
			OutputTokens: oa.Usage.CompletionTokens,
		}
	}
	return ir
}

// ============================================================
// Anthropic Messages 响应 → IR 转换
// ============================================================

func anthropicToIR(body []byte, model string) *IRResponse {
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
		return &IRResponse{
			ID:         "resp_" + randomHex(12),
			Model:      model,
			Role:       "assistant",
			Content:    []IRContentBlock{{Type: "text", Text: msg}},
			StopReason: "end_turn",
		}
	}

	ir := &IRResponse{
		ID:         "resp_" + randomHex(12),
		Model:      model,
		Role:       "assistant",
		StopReason: "end_turn",
	}

	if anth.Role != "" {
		ir.Role = anth.Role
	}
	if anth.StopReason != "" {
		ir.StopReason = anth.StopReason
	}

	for _, c := range anth.Content {
		switch c.Type {
		case "text":
			ir.Content = append(ir.Content, IRContentBlock{Type: "text", Text: c.Text})
		case "thinking":
			ir.Content = append(ir.Content, IRContentBlock{Type: "thinking", Thinking: c.Thinking})
		case "tool_use":
			input, _ := c.Input.(map[string]interface{})
			ir.Content = append(ir.Content, IRContentBlock{
				Type:      "tool_use",
				ToolUseID: c.ID,
				ToolName:  c.Name,
				ToolInput: input,
			})
		}
	}

	if anth.Usage != nil {
		ir.Usage = IRUsage{
			InputTokens:  anth.Usage.InputTokens,
			OutputTokens: anth.Usage.OutputTokens,
		}
	}
	return ir
}

// ============================================================
// IR → Anthropic Messages 响应转换
// ============================================================

func irToAnthropicResponse(ir *IRResponse) map[string]interface{} {
	var contentBlocks []interface{}
	for _, b := range ir.Content {
		switch b.Type {
		case "text":
			contentBlocks = append(contentBlocks, map[string]interface{}{"type": "text", "text": b.Text})
		case "thinking":
			contentBlocks = append(contentBlocks, map[string]interface{}{"type": "thinking", "thinking": b.Thinking})
		case "tool_use":
			contentBlocks = append(contentBlocks, map[string]interface{}{
				"type": "tool_use", "id": b.ToolUseID, "name": b.ToolName, "input": b.ToolInput,
			})
		}
	}
	if len(contentBlocks) == 0 {
		contentBlocks = []interface{}{map[string]interface{}{"type": "text", "text": ""}}
	}

	stopReason := ir.StopReason
	if stopReason == "" {
		stopReason = "end_turn"
	}

	// Anthropic 用 msg_ 前缀
	id := ir.ID
	if !strings.HasPrefix(id, "msg_") {
		id = "msg_" + randomHex(16)
	}

	resp := map[string]interface{}{
		"id":          id,
		"type":        "message",
		"role":        ir.Role,
		"content":     contentBlocks,
		"model":       ir.Model,
		"stop_reason": stopReason,
	}
	if ir.Usage.InputTokens > 0 || ir.Usage.OutputTokens > 0 {
		resp["usage"] = map[string]interface{}{
			"input_tokens":  ir.Usage.InputTokens,
			"output_tokens": ir.Usage.OutputTokens,
		}
	}
	return resp
}

// ============================================================
// IR → Responses 响应转换
// ============================================================

// irToResponsesRequest converts IR request to OpenAI Responses API request format.
func irToResponsesRequest(ir *IRRequest) map[string]interface{} {
	req := map[string]interface{}{
		"model":  ir.Model,
		"stream": ir.Stream,
	}

	// instructions
	if ir.SystemPrompt != "" {
		req["instructions"] = ir.SystemPrompt
	}

	// input: build InputItem array
	var input []interface{}
	for _, m := range ir.Messages {
		if m.Role == "tool" {
			// tool result → function_call_output
			input = append(input, map[string]interface{}{
				"type":   "function_call_output",
				"call_id": m.ToolCallID,
				"output":  extractText(m.Content),
			})
			continue
		}
		// regular message
		item := map[string]interface{}{
			"type": "message",
			"role": m.Role,
		}
		var parts []interface{}
		for _, b := range m.Content {
			switch b.Type {
			case "text":
				parts = append(parts, map[string]interface{}{"type": "input_text", "text": b.Text})
			case "image":
				parts = append(parts, map[string]interface{}{
					"type":      "input_image",
					"image_url": b.ImageURL,
				})
			}
		}
		if len(parts) > 0 {
			item["content"] = parts
		}
		input = append(input, item)
	}
	req["input"] = input

	// max_output_tokens
	if ir.MaxTokens > 0 {
		req["max_output_tokens"] = ir.MaxTokens
	}
	if ir.Temperature != nil {
		req["temperature"] = *ir.Temperature
	}
	if ir.TopP != nil {
		req["top_p"] = *ir.TopP
	}

	// tools
	if len(ir.Tools) > 0 {
		var tools []interface{}
		for _, t := range ir.Tools {
			tools = append(tools, map[string]interface{}{
				"type":        "function",
				"name":        t.Name,
				"description": t.Description,
				"parameters":  t.Parameters,
			})
		}
		req["tools"] = tools
	}

	// tool_choice
	if ir.ToolChoice != nil {
		switch ir.ToolChoice.Type {
		case "auto":
			req["tool_choice"] = "auto"
		case "none":
			req["tool_choice"] = "none"
		}
	}

	// reasoning effort
	if ir.Thinking != nil && ir.Thinking.Effort != "" {
		req["reasoning"] = map[string]interface{}{
			"effort": clampEffortForResponses(ir.Thinking.Effort),
		}
	}

	return req
}

func irToResponses(ir *IRResponse) map[string]interface{} {
	resp := map[string]interface{}{
		"id":          ir.ID,
		"object":      "response",
		"created_at":  time.Now().Unix(),
		"model":       ir.Model,
		"status":      "completed",
	}

	var output []interface{}
	var messageContent []interface{}

	for _, b := range ir.Content {
		switch b.Type {
		case "thinking":
			output = append(output, map[string]interface{}{
				"type":    "reasoning",
				"id":      "rs_" + randomHex(12),
				"content": []interface{}{},
				"summary": []interface{}{},
			})
		case "text":
			messageContent = append(messageContent, map[string]interface{}{
				"type": "output_text",
				"text": b.Text,
				"annotations": []interface{}{},
			})
		case "tool_use":
			// Flush any pending message content first
			if len(messageContent) > 0 {
				output = append(output, map[string]interface{}{
					"type":    "message",
					"id":      "msg_" + randomHex(12),
					"role":    "assistant",
					"content": messageContent,
					"status":  "completed",
				})
				messageContent = nil
			}
			output = append(output, map[string]interface{}{
				"type":      "function_call",
				"id":        "fc_" + randomHex(12),
				"call_id":   b.ToolUseID,
				"name":      b.ToolName,
				"arguments": mustJSON(b.ToolInput),
			})
		}
	}

	// Flush remaining message content
	if len(messageContent) > 0 {
		output = append(output, map[string]interface{}{
			"type":    "message",
			"id":      "msg_" + randomHex(12),
			"role":    "assistant",
			"content": messageContent,
			"status":  "completed",
		})
	}

	resp["output"] = output

	if ir.Usage.InputTokens > 0 || ir.Usage.OutputTokens > 0 {
		resp["usage"] = map[string]interface{}{
			"input_tokens":  ir.Usage.InputTokens,
			"output_tokens": ir.Usage.OutputTokens,
			"total_tokens":  ir.Usage.InputTokens + ir.Usage.OutputTokens,
		}
	}
	return resp
}

// ============================================================
// handleResponses — POST /v1/responses
// ============================================================

func handleResponses(w http.ResponseWriter, r *http.Request) {
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
		forwardResponses(w, r, p, rewritten)
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
		writeJSON(w, http.StatusInternalServerError, map[string]interface{}{
			"error": map[string]interface{}{"message": err.Error(), "type": "server_error"},
		})
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

func translateChatCompletionsToResponsesStream(ctx context.Context, upstream io.Reader, w io.Writer, flusher http.Flusher, model string) error {
	respID := "resp_" + randomHex(12)
	msgID := "msg_" + randomHex(12)
	var started, itemAdded, partAdded, reasoningDone bool
	var outputTokens int
	var inputTokens int

	emit := func(event string, data interface{}) error {
		return emitSSE(w, flusher, event, data)
	}

	scanner := bufio.NewScanner(upstream)
	scanner.Buffer(make([]byte, 64*1024), 1024*1024)

	ensureStarted := func() error {
		if started {
			return nil
		}
		started = true
		// response.created
		if err := emit("response.created", map[string]interface{}{
			"type":   "response.created",
			"response": map[string]interface{}{
				"id": respID, "object": "response", "created_at": time.Now().Unix(),
				"model": model, "output": []interface{}{}, "status": "in_progress",
			},
		}); err != nil {
			return err
		}
		return nil
	}

	ensureItemAdded := func() error {
		if err := ensureStarted(); err != nil {
			return err
		}
		if itemAdded {
			return nil
		}
		itemAdded = true
		return emit("response.output_item.added", map[string]interface{}{
			"type": "response.output_item.added", "output_index": 0,
			"item": map[string]interface{}{
				"type": "message", "id": msgID, "role": "assistant",
				"content": []interface{}{}, "status": "in_progress",
			},
		})
	}

	ensurePartAdded := func(contentType string) error {
		if err := ensureItemAdded(); err != nil {
			return err
		}
		if partAdded {
			return nil
		}
		partAdded = true
		return emit("response.content_part.added", map[string]interface{}{
			"type": "response.content_part.added", "output_index": 0, "content_index": 0,
			"part": map[string]interface{}{"type": contentType, "text": "", "annotations": []interface{}{}},
		})
	}

	closeSequence := func() error {
		// Close content_part if open
		if partAdded {
			if err := emit("response.content_part.done", map[string]interface{}{
				"type": "response.content_part.done", "output_index": 0, "content_index": 0,
				"part": map[string]interface{}{"type": "output_text", "text": "", "annotations": []interface{}{}},
			}); err != nil {
				return err
			}
		}
		// Close output_item if open
		if itemAdded {
			if err := emit("response.output_item.done", map[string]interface{}{
				"type": "response.output_item.done", "output_index": 0,
				"item": map[string]interface{}{
					"type": "message", "id": msgID, "role": "assistant",
					"content": []interface{}{map[string]interface{}{"type": "output_text", "text": "", "annotations": []interface{}{}}},
					"status": "completed",
				},
			}); err != nil {
				return err
			}
		}
		return nil
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
			if err := closeSequence(); err != nil {
				return err
			}
			return emit("response.completed", map[string]interface{}{
				"type": "response.completed",
				"response": map[string]interface{}{
					"id": respID, "object": "response", "created_at": time.Now().Unix(),
					"model": model, "status": "completed",
					"output": []interface{}{map[string]interface{}{
						"type": "message", "id": msgID, "role": "assistant",
						"content": []interface{}{map[string]interface{}{"type": "output_text", "text": "", "annotations": []interface{}{}}},
						"status": "completed",
					}},
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
			if !reasoningDone {
				if err := ensureStarted(); err != nil {
					return err
				}
				if err := emit("response.output_item.added", map[string]interface{}{
					"type": "response.output_item.added", "output_index": 0,
					"item": map[string]interface{}{
						"type": "reasoning", "id": "rs_" + randomHex(12),
						"content": []interface{}{}, "summary": []interface{}{},
					},
				}); err != nil {
					return err
				}
				reasoningDone = true
			}
		}

		// Text content → output_text delta
		if choice.Delta.Content != "" {
			if reasoningDone && !itemAdded {
				// Close reasoning item first
				if err := emit("response.output_item.done", map[string]interface{}{
					"type": "response.output_item.done", "output_index": 0,
					"item": map[string]interface{}{
						"type": "reasoning", "id": "rs_" + randomHex(12),
						"content": []interface{}{}, "summary": []interface{}{},
					},
				}); err != nil {
					return err
				}
			}
			if err := ensurePartAdded("output_text"); err != nil {
				return err
			}
			if err := emit("response.output_text.delta", map[string]interface{}{
				"type": "response.output_text.delta", "output_index": 0, "content_index": 0,
				"delta": choice.Delta.Content,
			}); err != nil {
				return err
			}
		}

		// Tool calls → function_call items
		for _, tc := range choice.Delta.ToolCalls {
			if tc.Function.Name != "" {
				if err := closeSequence(); err != nil {
					return err
				}
				itemAdded = false
				partAdded = false
				if err := emit("response.output_item.added", map[string]interface{}{
					"type": "response.output_item.added", "output_index": 0,
					"item": map[string]interface{}{
						"type": "function_call", "id": "fc_" + randomHex(12),
						"call_id": tc.ID, "name": tc.Function.Name, "arguments": "",
					},
				}); err != nil {
					return err
				}
			}
			if tc.Function.Arguments != "" {
				if err := emit("response.function_call_arguments.delta", map[string]interface{}{
					"type": "response.function_call_arguments.delta", "output_index": 0,
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
			if err := closeSequence(); err != nil {
				return err
			}
			return emit("response.completed", map[string]interface{}{
				"type": "response.completed",
				"response": map[string]interface{}{
					"id": respID, "object": "response", "created_at": time.Now().Unix(),
					"model": model, "status": "completed",
					"output": []interface{}{},
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
	respID := "resp_" + randomHex(12)
	msgID := "msg_" + randomHex(12)
	var started bool
	var inputTokens, outputTokens int
	var blockIndex int
	var hasTextBlock bool

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

	scanner := bufio.NewScanner(upstream)
	scanner.Buffer(make([]byte, 64*1024), 1024*1024)

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
			idx, _ := event["index"].(float64)
			blockIndex = int(idx)

			switch cbType {
			case "thinking":
				if err := emit("response.output_item.added", map[string]interface{}{
					"type": "response.output_item.added", "output_index": 0,
					"item": map[string]interface{}{
						"type": "reasoning", "id": "rs_" + randomHex(12),
						"content": []interface{}{}, "summary": []interface{}{},
					},
				}); err != nil {
					return err
				}
			case "text":
				hasTextBlock = true
				if err := emit("response.output_item.added", map[string]interface{}{
					"type": "response.output_item.added", "output_index": 0,
					"item": map[string]interface{}{
						"type": "message", "id": msgID, "role": "assistant",
						"content": []interface{}{}, "status": "in_progress",
					},
				}); err != nil {
					return err
				}
				if err := emit("response.content_part.added", map[string]interface{}{
					"type": "response.content_part.added", "output_index": 0, "content_index": blockIndex,
					"part": map[string]interface{}{"type": "output_text", "text": "", "annotations": []interface{}{}},
				}); err != nil {
					return err
				}
			case "tool_use":
				toolID, _ := cb["id"].(string)
				toolName, _ := cb["name"].(string)
				if err := emit("response.output_item.added", map[string]interface{}{
					"type": "response.output_item.added", "output_index": 0,
					"item": map[string]interface{}{
						"type": "function_call", "id": "fc_" + randomHex(12),
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
			idx, _ := event["index"].(float64)
			blockIndex = int(idx)

			switch deltaType {
			case "thinking_delta":
				thinking, _ := delta["thinking"].(string)
				if thinking != "" {
					if err := emit("response.output_text.delta", map[string]interface{}{
						"type": "response.output_text.delta", "output_index": 0, "content_index": blockIndex,
						"delta": thinking,
					}); err != nil {
						return err
					}
				}
			case "text_delta":
				text, _ := delta["text"].(string)
				if text != "" {
					if err := emit("response.output_text.delta", map[string]interface{}{
						"type": "response.output_text.delta", "output_index": 0, "content_index": blockIndex,
						"delta": text,
					}); err != nil {
						return err
					}
				}
			case "input_json_delta":
				partialJSON, _ := delta["partial_json"].(string)
				if partialJSON != "" {
					if err := emit("response.function_call_arguments.delta", map[string]interface{}{
						"type": "response.function_call_arguments.delta", "output_index": 0,
						"delta": partialJSON,
					}); err != nil {
						return err
					}
				}
			}

		case "content_block_stop":
			idx, _ := event["index"].(float64)
			blockIndex = int(idx)
			if hasTextBlock {
				if err := emit("response.content_part.done", map[string]interface{}{
					"type": "response.content_part.done", "output_index": 0, "content_index": blockIndex,
					"part": map[string]interface{}{"type": "output_text", "text": "", "annotations": []interface{}{}},
				}); err != nil {
					return err
				}
				if err := emit("response.output_item.done", map[string]interface{}{
					"type": "response.output_item.done", "output_index": 0,
					"item": map[string]interface{}{
						"type": "message", "id": msgID, "role": "assistant",
						"content": []interface{}{map[string]interface{}{"type": "output_text", "text": "", "annotations": []interface{}{}}},
						"status": "completed",
					},
				}); err != nil {
					return err
				}
				hasTextBlock = false
			}

		case "message_delta":
			if err := ensureStarted(); err != nil {
				return err
			}
			delta, _ := event["delta"].(map[string]interface{})
			if usage, ok := event["usage"].(map[string]interface{}); ok {
				if v, ok := usage["output_tokens"].(float64); ok {
					outputTokens = int(v)
				}
			}
			_ = delta

		case "message_stop":
			if err := ensureStarted(); err != nil {
				return err
			}
			return emit("response.completed", map[string]interface{}{
				"type": "response.completed",
				"response": map[string]interface{}{
					"id": respID, "object": "response", "created_at": time.Now().Unix(),
					"model": model, "status": "completed",
					"output": []interface{}{},
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

// forwardResponses 透传 Responses 格式请求到上游 Responses-native 端点。
func forwardResponses(w http.ResponseWriter, r *http.Request, p *Provider, body []byte) {
	req, err := http.NewRequestWithContext(r.Context(), "POST", p.ResponsesURL, bytes.NewReader(body))
	if err != nil {
		writeProxyError(w, r, err)
		return
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+p.APIKey)

	resp, err := httpClient.Do(req)
	if err != nil {
		writeProxyError(w, r, err)
		return
	}
	defer resp.Body.Close()

	streamPassthrough(w, resp)
}

// ============================================================
// Responses 响应 → IR 转换
// ============================================================

func responsesToIRResponse(body []byte, model string) *IRResponse {
	var resp responsesResp
	if err := json.Unmarshal(body, &resp); err != nil {
		return &IRResponse{
			ID:         "resp_" + randomHex(12),
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
	msgID := "msg_" + randomHex(16)
	var started bool
	var inputTokens, outputTokens int
	var blockIndex int
	var hasTextBlock bool

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

	scanner := bufio.NewScanner(upstream)
	scanner.Buffer(make([]byte, 64*1024), 1024*1024)

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

		// Read next line for data
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
			case "message":
				// Will emit content_block_start when content_part arrives
			case "function_call":
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
				if err := emit("content_block_delta", map[string]interface{}{
					"type": "content_block_delta", "index": blockIndex,
					"delta": map[string]interface{}{"type": "text_delta", "text": delta},
				}); err != nil {
					return err
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
			if hasTextBlock {
				if err := emit("content_block_stop", map[string]interface{}{
					"type": "content_block_stop", "index": blockIndex,
				}); err != nil {
					return err
				}
				blockIndex++
				hasTextBlock = false
			}

		case "response.output_item.done":
			item, _ := event["item"].(map[string]interface{})
			itemType, _ := item["type"].(string)
			if itemType == "function_call" {
				if err := emit("content_block_stop", map[string]interface{}{
					"type": "content_block_stop", "index": blockIndex,
				}); err != nil {
					return err
				}
				blockIndex++
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
			if err := emit("message_delta", map[string]interface{}{
				"type": "message_delta",
				"delta": map[string]interface{}{"stop_reason": "end_turn", "stop_sequence": nil},
				"usage": map[string]interface{}{"output_tokens": outputTokens},
			}); err != nil {
				return err
			}
			return emit("message_stop", map[string]interface{}{"type": "message_stop"})
		}
	}
	return scanner.Err()
}
