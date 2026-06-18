package main

import (
	"bytes"
	"encoding/json"
	"io"
	"log"
	"net/http"
)

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

	if req.OutputConfig != nil && req.OutputConfig.Effort != "" {
		log.Printf("[effort] %s: %s → %s/%s", req.Model, req.OutputConfig.Effort, p.Type, p.Name)
	}

	switch p.Type {
	case ProviderAnthropic:
		// Rewrite model name and forward.
		req.Model = p.Name
		rewritten, _ := json.Marshal(req)
		forwardUpstream(w, r, p.MessagesURL, p.APIKey, rewritten, map[string]string{
			"x-api-key":         p.APIKey,
			"anthropic-version": "2023-06-01",
		})

	case ProviderResponses:
		// Convert Anthropic request → IR → Responses format.
		ir := anthropicToIRRequest(req)
		ir.Model = p.Name
		responsesReq := irToResponsesRequest(ir)
		applyExtraParams(responsesReq, p.ExtraParams)
		responsesBody, err := json.Marshal(responsesReq)
		if err != nil {
			writeJSON(w, http.StatusInternalServerError, map[string]interface{}{
				"error": map[string]interface{}{"message": err.Error(), "type": "server_error"},
			})
			return
		}

		ctx := r.Context()
		req2, err := http.NewRequestWithContext(ctx, "POST", p.ResponsesURL, bytes.NewReader(responsesBody))
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
			startSSEStream(w, resp.StatusCode)

			if err := translateResponsesToAnthropicStream(ctx, resp.Body, w, flusher, req.Model); err != nil {
				return
			}
			return
		}

		// Non-streaming: Responses response → IR → Anthropic format.
		resBody, _ := io.ReadAll(resp.Body)
		irResp := responsesToIRResponse(resBody, req.Model)
		anthResp := irToAnthropicResponse(irResp)
		writeJSON(w, resp.StatusCode, anthResp)

	case ProviderOpenAI:
		// Convert Anthropic request → IR → OpenAI format.
		ir := anthropicToIRRequest(req)
		if p.DropParams {
			ir.Thinking = nil
			ir.ToolChoice = nil
		}
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
			startSSEStream(w, resp.StatusCode)

			inputTokens := countTokens(oaReq)
			if err := translateStream(ctx, resp.Body, w, flusher, req.Model, inputTokens); err != nil {
				return
			}
			return
		}

		// Non-streaming: OpenAI response → IR → Anthropic format.
		resBody, _ := io.ReadAll(resp.Body)
		irResp := chatCompletionsToIR(resBody, req.Model)
		anthResp := irToAnthropicResponse(irResp)
		writeJSON(w, resp.StatusCode, anthResp)
	}
}

