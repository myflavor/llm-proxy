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

// streamPassthrough copies an upstream response to the client, supporting SSE.
func streamPassthrough(w http.ResponseWriter, resp *http.Response) {
	contentType := resp.Header.Get("Content-Type")
	if contentType == "" {
		contentType = "application/json"
	}
	w.Header().Set("Content-Type", contentType)
	w.Header().Set("Cache-Control", "no-cache")

	if flusher, ok := w.(http.Flusher); ok {
		http.NewResponseController(w).SetWriteDeadline(time.Time{})
		w.WriteHeader(resp.StatusCode)
		buf := make([]byte, 4096)
		for {
			n, readErr := resp.Body.Read(buf)
			if n > 0 {
				if _, writeErr := w.Write(buf[:n]); writeErr != nil {
					return
				}
				flusher.Flush()
			}
			if readErr != nil {
				break
			}
		}
	} else {
		w.WriteHeader(resp.StatusCode)
		io.Copy(w, resp.Body)
	}
}

// startSSEStream sets response headers for an SSE stream.
func startSSEStream(w http.ResponseWriter, statusCode int) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	http.NewResponseController(w).SetWriteDeadline(time.Time{})
	w.WriteHeader(statusCode)
}

// newSSEScanner creates a scanner tuned for SSE line parsing.
func newSSEScanner(r io.Reader) *bufio.Scanner {
	s := bufio.NewScanner(r)
	s.Buffer(make([]byte, 64*1024), 1024*1024)
	return s
}

// forwardUpstream forwards a request body to an upstream endpoint and streams the response back.
func forwardUpstream(w http.ResponseWriter, r *http.Request, url string, apiKey string, body []byte, extraHeaders map[string]string) {
	req, err := http.NewRequestWithContext(r.Context(), "POST", url, bytes.NewReader(body))
	if err != nil {
		writeProxyError(w, r, err)
		return
	}
	req.Header.Set("Content-Type", "application/json")
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}
	for k, v := range extraHeaders {
		req.Header.Set(k, v)
	}

	resp, err := httpClient.Do(req)
	if err != nil {
		writeProxyError(w, r, err)
		return
	}
	defer resp.Body.Close()

	streamPassthrough(w, resp)
}

// emitSSE writes an SSE event.
func emitSSE(w io.Writer, flusher http.Flusher, event string, data interface{}) error {
	jsonBytes, err := json.Marshal(data)
	if err != nil {
		return err
	}
	if _, err := fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, jsonBytes); err != nil {
		return err
	}
	flusher.Flush()
	return nil
}

// translateStream translates OpenAI SSE → Anthropic SSE.
func translateStream(ctx context.Context, upstream io.Reader, w io.Writer, flusher http.Flusher, model string, inputTokens int) error {
	msgID := "msg_" + randomHex(16)
	var started, stopped bool
	var outputTokens int

	type blockState struct {
		index     int
		blockType string
		closed    bool
		name      string
	}
	var blocks []*blockState
	toolBlocks := map[int]*blockState{}

	allocBlock := func(bt string) *blockState {
		bs := &blockState{index: len(blocks), blockType: bt}
		blocks = append(blocks, bs)
		return bs
	}

	emitMessageStart := func() error {
		return emitSSE(w, flusher, "message_start", map[string]interface{}{
			"type": "message_start",
			"message": map[string]interface{}{
				"id": msgID, "type": "message", "role": "assistant", "content": []interface{}{},
				"model": model, "stop_reason": nil, "stop_sequence": nil,
				"usage": map[string]int{"input_tokens": inputTokens, "output_tokens": 0},
			},
		})
	}
	closeBlock := func(bs *blockState) error {
		if !bs.closed {
			bs.closed = true
			return emitSSE(w, flusher, "content_block_stop", map[string]interface{}{"type": "content_block_stop", "index": bs.index})
		}
		return nil
	}
	closeLastOpenBlock := func() error {
		for i := len(blocks) - 1; i >= 0; i-- {
			if !blocks[i].closed {
				return closeBlock(blocks[i])
			}
		}
		return nil
	}
	emitStopEvents := func(stopReason string) error {
		if err := closeLastOpenBlock(); err != nil {
			return err
		}
		if err := emitSSE(w, flusher, "message_delta", map[string]interface{}{
			"type": "message_delta",
			"delta": map[string]interface{}{"stop_reason": stopReason, "stop_sequence": nil},
			"usage": map[string]int{"output_tokens": outputTokens},
		}); err != nil {
			return err
		}
		return emitSSE(w, flusher, "message_stop", map[string]interface{}{"type": "message_stop"})
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
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			if !started {
				if err := emitMessageStart(); err != nil {
					return err
				}
			}
			if len(blocks) == 0 {
				bs := allocBlock("text")
				if err := emitSSE(w, flusher, "content_block_start", map[string]interface{}{
					"type": "content_block_start", "index": bs.index,
					"content_block": map[string]interface{}{"type": "text", "text": ""},
				}); err != nil {
					return err
				}
			}
			if !stopped {
				if err := emitStopEvents("end_turn"); err != nil {
					return err
				}
			}
			break
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
			continue
		}
		choice := chunk.Choices[0]

		if !started {
			started = true
			if err := emitMessageStart(); err != nil {
				return err
			}
		}

		// Reasoning/thinking.
		if choice.Delta.ReasoningContent != "" {
			var thinkingBlock *blockState
			for _, bs := range blocks {
				if bs.blockType == "thinking" && !bs.closed {
					thinkingBlock = bs
					break
				}
			}
			if thinkingBlock == nil {
				thinkingBlock = allocBlock("thinking")
				if err := emitSSE(w, flusher, "content_block_start", map[string]interface{}{
					"type": "content_block_start", "index": thinkingBlock.index,
					"content_block": map[string]interface{}{"type": "thinking", "thinking": ""},
				}); err != nil {
					return err
				}
			}
			if err := emitSSE(w, flusher, "content_block_delta", map[string]interface{}{
				"type": "content_block_delta", "index": thinkingBlock.index,
				"delta": map[string]interface{}{"type": "thinking_delta", "thinking": choice.Delta.ReasoningContent},
			}); err != nil {
				return err
			}
		}

		// Text content.
		if choice.Delta.Content != "" {
			for _, bs := range blocks {
				if bs.blockType == "thinking" && !bs.closed {
					if err := closeBlock(bs); err != nil {
						return err
					}
				}
			}
			var textBlock *blockState
			for _, bs := range blocks {
				if bs.blockType == "text" && !bs.closed {
					textBlock = bs
					break
				}
			}
			if textBlock == nil {
				textBlock = allocBlock("text")
				if err := emitSSE(w, flusher, "content_block_start", map[string]interface{}{
					"type": "content_block_start", "index": textBlock.index,
					"content_block": map[string]interface{}{"type": "text", "text": ""},
				}); err != nil {
					return err
				}
			}
			if err := emitSSE(w, flusher, "content_block_delta", map[string]interface{}{
				"type": "content_block_delta", "index": textBlock.index,
				"delta": map[string]interface{}{"type": "text_delta", "text": choice.Delta.Content},
			}); err != nil {
				return err
			}
		}

		// Tool calls.
		for _, tc := range choice.Delta.ToolCalls {
			tb, exists := toolBlocks[tc.Index]
			if !exists {
				for _, bs := range blocks {
					if (bs.blockType == "thinking" || bs.blockType == "text") && !bs.closed {
						if err := closeBlock(bs); err != nil {
							return err
						}
					}
				}
				tb = allocBlock("tool_use")
				tb.name = tc.Function.Name
				toolBlocks[tc.Index] = tb
				if err := emitSSE(w, flusher, "content_block_start", map[string]interface{}{
					"type": "content_block_start", "index": tb.index,
					"content_block": map[string]interface{}{"type": "tool_use", "id": tc.ID, "name": tb.name, "input": map[string]interface{}{}},
				}); err != nil {
					return err
				}
			}
			if tc.Function.Arguments != "" {
				if err := emitSSE(w, flusher, "content_block_delta", map[string]interface{}{
					"type": "content_block_delta", "index": tb.index,
					"delta": map[string]interface{}{"type": "input_json_delta", "partial_json": tc.Function.Arguments},
				}); err != nil {
					return err
				}
			}
		}

		if choice.FinishReason != nil {
			stopped = true
			if chunk.Usage != nil {
				inputTokens = chunk.Usage.PromptTokens
				outputTokens = chunk.Usage.CompletionTokens
			}
			if err := emitStopEvents(mapFinishReason(*choice.FinishReason)); err != nil {
				return err
			}
		}
	}
	return scanner.Err()
}
