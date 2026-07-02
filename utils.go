package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"sync/atomic"
	"time"
)

// HTTP constants
const (
	authBearerPrefix    = "Bearer "
	headerContentType   = "Content-Type"
	headerAuthorization = "Authorization"
	contentTypeJSON     = "application/json"
	contentTypeSSE      = "text/event-stream"
)

// --- Anthropic request types ---

// anthropicMsgReq is the Anthropic Messages API request.
type anthropicMsgReq struct {
	Model         string              `json:"model"`
	Messages      []anthropicMsg      `json:"messages"`
	System        interface{}         `json:"system"`
	MaxTokens     int                 `json:"max_tokens"`
	Temperature   *float64            `json:"temperature,omitempty"`
	TopP          *float64            `json:"top_p,omitempty"`
	TopK          *int                `json:"top_k,omitempty"`
	StopSequences []string            `json:"stop_sequences,omitempty"`
	Stream        bool                `json:"stream,omitempty"`
	Tools         []anthropicTool     `json:"tools,omitempty"`
	ToolChoice    interface{}         `json:"tool_choice,omitempty"`
	Thinking      *anthropicThinking  `json:"thinking,omitempty"`
	OutputConfig  *anthropicOutputCfg `json:"output_config,omitempty"`
	Metadata      *anthropicMetadata  `json:"metadata,omitempty"`
}

type anthropicOutputCfg struct {
	Effort string `json:"effort,omitempty"`
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
	Display      string `json:"display,omitempty"` // summarized / omitted
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
	case []IRContentBlock:
		var sb strings.Builder
		for _, b := range c {
			if b.Type == "text" {
				sb.WriteString(b.Text)
			}
		}
		return sb.String()
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
	if v == nil {
		return "{}"
	}
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
	case "content_filter":
		return "refusal"
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
	case "stop_sequence":
		return "stop"
	case "pause_turn":
		return "stop"
	case "refusal":
		return "content_filter"
	case "incomplete":
		return "length"
	}
	return r
}

// --- Effort 转换 ---

// budgetToEffort 将 Anthropic 的 budget_tokens 转为 effort 字符串。
func budgetToEffort(budget int) string {
	switch {
	case budget <= 0:
		return "none"
	case budget <= 2048:
		return "low"
	case budget <= 8192:
		return "medium"
	default:
		return "high"
	}
}

// clampEffortForChatCompletions 将 effort 限制在 Chat Completions API 支持范围内。
func clampEffortForChatCompletions(effort string) string {
	switch effort {
	case "low", "medium", "high", "xhigh":
		return effort
	case "minimal":
		return "low"
	case "max", "ultracode":
		return "xhigh"
	default:
		return effort
	}
}

// clampEffortForResponses 将 effort 限制在 Responses API 支持范围内。
func clampEffortForResponses(effort string) string {
	switch effort {
	case "none", "minimal", "low", "medium", "high", "xhigh":
		return effort
	case "max", "ultracode":
		return "xhigh"
	default:
		return effort
	}
}

// clampEffortForAnthropic 将 effort 限制在 Anthropic 支持范围内。
func clampEffortForAnthropic(effort string) string {
	switch effort {
	case "none", "low", "medium", "high":
		return effort
	case "minimal":
		return "low"
	case "xhigh", "max", "ultracode":
		return "high"
	default:
		return effort
	}
}

// writeError writes a standardized error response
func writeError(w http.ResponseWriter, statusCode int, message string) {
	writeJSON(w, statusCode, map[string]interface{}{
		"error": map[string]interface{}{
			"message": message,
			"type":    "server_error",
		},
	})
}

// readResponseBody reads response body and logs if error occurs
func readResponseBody(body io.Reader, context string) []byte {
	data, err := io.ReadAll(body)
	if err != nil {
		log.Printf("ERROR: failed to read %s body: %v", context, err)
		return nil
	}
	return data
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

// ID generation helpers - standardized ID formats and lengths
func newMessageID() string        { return "msg_" + randomHex(12) }      // 28 chars (Anthropic standard)
func newResponseID() string       { return "resp_" + randomHex(12) }     // 29 chars
func newFunctionCallID() string   { return "fc_" + randomHex(12) }       // 27 chars
func newReasoningID() string      { return "rs_" + randomHex(12) }       // 27 chars
func newChatCompletionID() string { return "chatcmpl-" + randomHex(12) } // 33 chars

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
