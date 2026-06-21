package main

import (
	"encoding/json"
	"log"
	"strings"
	"time"
)

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
			case "message", "":
				// Default to message when type is missing (codex doesn't always set it)
				if item.Role == "" {
					item.Role = "user"
				}
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
						case "input_text", "output_text":
							msg.Content = append(msg.Content, IRContentBlock{Type: "text", Text: part.Text})
						case "input_image":
							msg.Content = append(msg.Content, IRContentBlock{Type: "image", ImageURL: part.ImageURL})
						}
					}
				}
				if len(msg.Content) > 0 {
					ir.Messages = append(ir.Messages, msg)
				}
			case "function_call":
				// Codex sends function_call items from previous response in the input.
				// Convert to assistant message with tool_use content.
				var input map[string]interface{}
				if item.Arguments != "" {
					json.Unmarshal([]byte(item.Arguments), &input)
				}
				ir.Messages = append(ir.Messages, IRMessage{
					Role: "assistant",
					Content: []IRContentBlock{{
						Type:      "tool_use",
						ToolUseID: item.CallID,
						ToolName:  item.Name,
						ToolInput: input,
					}},
				})
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
		ir.Thinking = &IRThinking{Enabled: true, Effort: req.Reasoning.Effort, Summary: req.Reasoning.Summary}
	}

	// tool_choice
	if req.ToolChoice != nil {
		tc := &IRToolChoice{}
		switch v := req.ToolChoice.(type) {
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
		oa["max_completion_tokens"] = ir.MaxTokens
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
			msg["content"] = nil
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

	// reasoning_effort (OpenAI standard)
	if ir.Thinking != nil && ir.Thinking.Effort != "" && ir.Thinking.Effort != "none" {
		oa["reasoning_effort"] = ir.Thinking.Effort
		log.Printf("[effort→] reasoning_effort=%s", ir.Thinking.Effort)
	}

	return oa
}

// ============================================================
// Anthropic Messages 请求 → IR 转换
// ============================================================

func anthropicToIRRequest(req anthropicMsgReq) *IRRequest {
	maxTokens := req.MaxTokens
	if maxTokens == 0 {
		maxTokens = 4096 // Anthropic requires max_tokens > 0 for actual generation
	}
	ir := &IRRequest{
		Model:         req.Model,
		MaxTokens:     maxTokens,
		Temperature:   req.Temperature,
		TopP:          req.TopP,
		TopK:          req.TopK,
		StopSequences: req.StopSequences,
		Stream:        req.Stream,
	}

	// system → system prompt
	ir.SystemPrompt = extractText(req.System)

	// thinking
	if req.Thinking != nil && req.Thinking.Type == "enabled" {
		ir.Thinking = &IRThinking{Enabled: true, BudgetTokens: req.Thinking.BudgetTokens, Display: req.Thinking.Display}
		if req.Thinking.BudgetTokens > 0 && ir.Thinking.Effort == "" {
			ir.Thinking.Effort = budgetToEffort(req.Thinking.BudgetTokens)
		}
	} else if req.Thinking != nil && req.Thinking.Type == "adaptive" {
		ir.Thinking = &IRThinking{Enabled: true, Display: req.Thinking.Display}
	}
	if req.OutputConfig != nil && req.OutputConfig.Effort != "" {
		if ir.Thinking == nil {
			ir.Thinking = &IRThinking{Enabled: true}
		}
		ir.Thinking.Effort = req.OutputConfig.Effort
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
						sig, _ := b["signature"].(string)
						msg.Content = append(msg.Content, IRContentBlock{Type: "thinking", Thinking: txt, Signature: sig})
					}
				case "redacted_thinking":
					if data, _ := b["data"].(string); data != "" {
						msg.Content = append(msg.Content, IRContentBlock{Type: "redacted_thinking", Data: data})
					}
				case "tool_use":
					input, _ := b["input"].(map[string]interface{})
					id, _ := b["id"].(string)
					name, _ := b["name"].(string)
					msg.Content = append(msg.Content, IRContentBlock{
						Type:      "tool_use",
						ToolUseID: id,
						ToolName:  name,
						ToolInput: input,
					})
				case "tool_result":
					toolCallID, _ := b["tool_use_id"].(string)
					isErr, _ := b["is_error"].(bool)
					resultContent := ""
					if c, ok := b["content"].(string); ok {
						resultContent = c
					} else if blocks2, ok := b["content"].([]interface{}); ok {
						resultContent = extractText(blocks2)
					}
					toolResults = append(toolResults, IRMessage{
						Role:       "tool",
						ToolCallID: toolCallID,
						Content:    []IRContentBlock{{Type: "text", Text: resultContent, IsError: isErr}},
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
			if v, ok := m["disable_parallel_tool_use"].(bool); ok {
				tc.DisableParallelToolUse = v
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
		TopK:          ir.TopK,
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
					block := map[string]interface{}{"type": "thinking", "thinking": b.Thinking}
					if b.Signature != "" {
						block["signature"] = b.Signature
					}
					blocks = append(blocks, block)
				case "redacted_thinking":
					blocks = append(blocks, map[string]interface{}{"type": "redacted_thinking", "data": b.Data})
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
			// Preserve is_error flag if present
			for _, b := range m.Content {
				if b.IsError {
					toolResult["is_error"] = true
					break
				}
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
								mediaType = strings.TrimSuffix(headerParts[1], ";base64")
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
		tc := map[string]interface{}{}
		switch ir.ToolChoice.Type {
		case "auto":
			tc["type"] = "auto"
		case "required", "any":
			tc["type"] = "any"
		case "none":
			tc["type"] = "none"
		case "specific":
			if ir.ToolChoice.Name != "" {
				tc["type"] = "tool"
				tc["name"] = ir.ToolChoice.Name
			}
		}
		if ir.ToolChoice.DisableParallelToolUse {
			tc["disable_parallel_tool_use"] = true
		}
		if len(tc) > 0 {
			req.ToolChoice = tc
		}
	}

	// thinking
	if ir.Thinking != nil && ir.Thinking.Enabled {
		display := ir.Thinking.Display
		if display == "" {
			display = "summarized"
		}
		if ir.Thinking.Effort != "" {
			// 新格式：adaptive thinking + output_config.effort
			clamped := clampEffortForAnthropic(ir.Thinking.Effort)
			req.Thinking = &anthropicThinking{Type: "adaptive", Display: display}
			req.OutputConfig = &anthropicOutputCfg{Effort: clamped}
			if clamped != ir.Thinking.Effort {
				log.Printf("[effort→] output_config.effort=%s (clamped from %s)", clamped, ir.Thinking.Effort)
			} else {
				log.Printf("[effort→] output_config.effort=%s", clamped)
			}
		} else if ir.Thinking.BudgetTokens > 0 {
			// 旧格式：enabled + budget_tokens
			req.Thinking = &anthropicThinking{Type: "enabled", BudgetTokens: ir.Thinking.BudgetTokens, Display: display}
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
		MaxTokens           int     `json:"max_tokens"`
		MaxCompletionTokens int     `json:"max_completion_tokens"`
		Temperature         *float64 `json:"temperature,omitempty"`
		TopP                *float64 `json:"top_p,omitempty"`
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

	maxTokens := oa.MaxTokens
	if maxTokens == 0 {
		maxTokens = oa.MaxCompletionTokens // max_completion_tokens is the newer field
	}
	ir := &IRRequest{
		Model:         oa.Model,
		MaxTokens:     maxTokens,
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
		if role == "system" || role == "developer" {
			text := extractText(m["content"])
			if ir.SystemPrompt != "" && text != "" {
				ir.SystemPrompt += "\n" + text
			} else if text != "" {
				ir.SystemPrompt = text
			}
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
			// reasoning_content (can exist with or without tool_calls)
			if rc, ok := m["reasoning_content"].(string); ok && rc != "" {
				msg.Content = append(msg.Content, IRContentBlock{Type: "thinking", Thinking: rc})
			}
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
					id, _ := tc["id"].(string)
					msg.Content = append(msg.Content, IRContentBlock{
						Type:      "tool_use",
						ToolUseID: id,
						ToolName:  name,
						ToolInput: input,
					})
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
	var reasoningParts []string
	var toolCalls []interface{}

	for _, b := range ir.Content {
		switch b.Type {
		case "text":
			content += b.Text
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
	if content != "" {
		msg["content"] = content
	} else if len(toolCalls) > 0 {
		msg["content"] = nil // spec requires null for tool-call-only responses
	}
	if len(reasoningParts) > 0 {
		msg["reasoning_content"] = strings.Join(reasoningParts, "")
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
		} else {
			ir.StopReason = "end_turn"
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
			Type      string      `json:"type"`
			Text      string      `json:"text,omitempty"`
			Thinking  string      `json:"thinking,omitempty"`
			Signature string      `json:"signature,omitempty"`
			Data      string      `json:"data,omitempty"`
			ID        string      `json:"id,omitempty"`
			Name      string      `json:"name,omitempty"`
			Input     interface{} `json:"input,omitempty"`
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
			ir.Content = append(ir.Content, IRContentBlock{Type: "thinking", Thinking: c.Thinking, Signature: c.Signature})
		case "redacted_thinking":
			ir.Content = append(ir.Content, IRContentBlock{Type: "redacted_thinking", Data: c.Data})
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
			block := map[string]interface{}{"type": "thinking", "thinking": b.Thinking}
			if b.Signature != "" {
				block["signature"] = b.Signature
			}
			contentBlocks = append(contentBlocks, block)
		case "redacted_thinking":
			contentBlocks = append(contentBlocks, map[string]interface{}{"type": "redacted_thinking", "data": b.Data})
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
	// Map non-Anthropic stop reasons to valid ones
	switch stopReason {
	case "refusal":
		stopReason = "end_turn"
	case "incomplete":
		stopReason = "max_tokens"
	}

	// Anthropic 用 msg_ 前缀
	id := ir.ID
	if !strings.HasPrefix(id, "msg_") {
		id = "msg_" + randomHex(16)
	}

	resp := map[string]interface{}{
		"id":            id,
		"type":          "message",
		"role":          ir.Role,
		"content":       contentBlocks,
		"model":         ir.Model,
		"stop_reason":   stopReason,
		"stop_sequence": nil,
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
		// Check for tool_use blocks in assistant messages → function_call items
		if m.Role == "assistant" {
			hasToolUse := false
			for _, b := range m.Content {
				if b.Type == "tool_use" {
					hasToolUse = true
					input = append(input, map[string]interface{}{
						"type":      "function_call",
						"id":        b.ToolUseID,
						"call_id":   b.ToolUseID,
						"name":      b.ToolName,
						"arguments": mustJSON(b.ToolInput),
					})
				}
			}
			if hasToolUse {
				// Also emit text content as a message if present
				var textParts []string
				for _, b := range m.Content {
					if b.Type == "text" {
						textParts = append(textParts, b.Text)
					}
				}
				if len(textParts) > 0 {
					input = append(input, map[string]interface{}{
						"type": "message",
						"role": "assistant",
						"content": []interface{}{
							map[string]interface{}{"type": "output_text", "text": strings.Join(textParts, "")},
						},
					})
				}
				continue
			}
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
		case "required", "any":
			req["tool_choice"] = "required"
		case "specific":
			if ir.ToolChoice.Name != "" {
				req["tool_choice"] = map[string]interface{}{
					"type": "function",
					"function": map[string]interface{}{"name": ir.ToolChoice.Name},
				}
			}
		}
	}

	// reasoning effort + summary
	if ir.Thinking != nil && ir.Thinking.Effort != "" {
		clamped := clampEffortForResponses(ir.Thinking.Effort)
		reasoning := map[string]interface{}{"effort": clamped}
		if ir.Thinking.Summary != "" {
			reasoning["summary"] = ir.Thinking.Summary
		}
		req["reasoning"] = reasoning
		if clamped != ir.Thinking.Effort {
			log.Printf("[effort→] reasoning.effort=%s (clamped from %s)", clamped, ir.Thinking.Effort)
		} else {
			log.Printf("[effort→] reasoning.effort=%s", clamped)
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
			reasoningContent := []interface{}{}
			reasoningSummary := []interface{}{}
			if b.Thinking != "" {
				reasoningContent = []interface{}{map[string]interface{}{
					"type": "reasoning_text", "text": b.Thinking,
				}}
				reasoningSummary = []interface{}{map[string]interface{}{
					"type": "summary_text", "text": b.Thinking,
				}}
			}
			output = append(output, map[string]interface{}{
				"type":    "reasoning",
				"id":      "rs_" + randomHex(12),
				"content": reasoningContent,
				"summary": reasoningSummary,
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
