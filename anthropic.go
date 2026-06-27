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

	body, err := readRequestBody(r)
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
		if len(p.ExtraParams) > 0 {
			var m map[string]interface{}
			if err := json.Unmarshal(rewritten, &m); err == nil && m != nil {
				if len(p.ExtraParams) > 0 {
					applyExtraParams(m, p.ExtraParams)
				}
				rewritten, _ = json.Marshal(m)
			}
		}
		forwardUpstream(w, r, p.MessagesURL, "", rewritten, map[string]string{
			"x-api-key":         p.APIKey,
			"anthropic-version": anthropicAPIVersion,
		})

	case ProviderResponses:
		// Convert Anthropic request → IR → Responses format.
		ir := anthropicToIRRequest(req)
		if p.DropParams {
			ir.Thinking = nil
			ir.ToolChoice = nil
		}
		ir.Model = p.Name
		responsesReq := irToResponsesRequest(ir)
		applyExtraParams(responsesReq, p.ExtraParams)
		responsesBody, err := json.Marshal(responsesReq)
		if err != nil {
			writeError(w, http.StatusInternalServerError, err.Error())
			return
		}

		ctx := r.Context()
		req2, err := http.NewRequestWithContext(ctx, "POST", p.ResponsesURL, bytes.NewReader(responsesBody))
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

	default:
		writeError(w, http.StatusInternalServerError, "unsupported provider type")
	}
}

