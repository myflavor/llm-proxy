package main

import (
	"encoding/json"
	"fmt"
	"strings"
	"sync/atomic"
	"time"
)

// --- Anthropic request types ---

// anthropicMsgReq is the Anthropic Messages API request.
type anthropicMsgReq struct {
	Model         string             `json:"model"`
	Messages      []anthropicMsg     `json:"messages"`
	System        interface{}        `json:"system"`
	MaxTokens     int                `json:"max_tokens"`
	Temperature   *float64           `json:"temperature,omitempty"`
	TopP          *float64           `json:"top_p,omitempty"`
	StopSequences []string           `json:"stop_sequences,omitempty"`
	Stream        bool               `json:"stream,omitempty"`
	Tools         []anthropicTool    `json:"tools,omitempty"`
	ToolChoice    interface{}        `json:"tool_choice,omitempty"`
	Thinking      *anthropicThinking `json:"thinking,omitempty"`
	Metadata      *anthropicMetadata `json:"metadata,omitempty"`
}

type anthropicMsg struct {
	Role    string      `json:"role"`
	Content interface{} `json:"content"`
}

type anthropicTool struct {
	Name        string      `json:"name"`
	Description string      `json:"description,omitempty"`
	InputSchema interface{} `json:"input_schema"`
}

type anthropicThinking struct {
	Type         string `json:"type"`
	BudgetTokens int    `json:"budget_tokens,omitempty"`
}

type anthropicMetadata struct {
	UserID string `json:"user_id,omitempty"`
}

// --- Content extraction helpers ---

// extractText handles Anthropic's polymorphic content field (string or []contentBlock).
func extractText(v interface{}) string {
	switch c := v.(type) {
	case string:
		return c
	case []interface{}:
		var sb strings.Builder
		for _, block := range c {
			b, ok := block.(map[string]interface{})
			if !ok {
				continue
			}
			if t, _ := b["type"].(string); t == "text" {
				if txt, ok := b["text"].(string); ok {
					sb.WriteString(txt)
				}
			}
		}
		return sb.String()
	}
	return ""
}

// extractImageURL converts an Anthropic image content block to a data URL.
func extractImageURL(b map[string]interface{}) string {
	source, ok := b["source"].(map[string]interface{})
	if !ok {
		return ""
	}
	switch source["type"] {
	case "base64":
		mediaType, _ := source["media_type"].(string)
		data, _ := source["data"].(string)
		if mediaType != "" && data != "" {
			return "data:" + mediaType + ";base64," + data
		}
	case "url":
		if url, _ := source["url"].(string); url != "" {
			return url
		}
	}
	return ""
}

func mustJSON(v interface{}) string {
	b, _ := json.Marshal(v)
	return string(b)
}

// --- Mapping helpers ---

func mapFinishReason(r string) string {
	switch r {
	case "stop":
		return "end_turn"
	case "length":
		return "max_tokens"
	case "tool_calls":
		return "tool_use"
	}
	return r
}

func mapFinishReasonReverse(r string) string {
	switch r {
	case "end_turn":
		return "stop"
	case "max_tokens":
		return "length"
	case "tool_use":
		return "tool_calls"
	}
	return r
}

// --- ID generation ---

var hexCounter uint64

func randomHex(n int) string {
	counter := atomic.AddUint64(&hexCounter, 1)
	result := fmt.Sprintf("%x%x", time.Now().UnixNano(), counter)
	if len(result) < n {
		return result
	}
	return result[:n]
}

// --- Token estimation ---

func countTokens(req map[string]interface{}) int {
	body, _ := json.Marshal(req)
	var ascii, nonASCII int
	for i := 0; i < len(body); i++ {
		if body[i] < 128 {
			ascii++
		} else {
			nonASCII++
		}
	}
	return ascii/4 + nonASCII*2/3
}
