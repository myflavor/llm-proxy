package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sync/atomic"
	"time"
)

// bugReportEnabled writes bugreports/ on upstream errors. Toggle via config
// `server.bug_report: true` or env var LLM_PROXY_BUG_REPORT=1.
var bugReportEnabled bool

// requestIDSeq is a per-process counter for unique, sortable request IDs.
var requestIDSeq uint64

type requestCtxKey struct{}

// requestContext carries per-request metadata for bug reporting.
type requestContext struct {
	ID           string
	StartTime    time.Time
	Method       string
	Path         string
	ClientBody   []byte // original inbound request body (client → proxy)
	OutboundURL  string // upstream URL the proxy is calling
	OutboundBody []byte // converted request body sent upstream
	Model        string
}

func newRequestID() string {
	seq := atomic.AddUint64(&requestIDSeq, 1)
	return fmt.Sprintf("%x-%x", time.Now().UnixNano()/int64(time.Millisecond), seq)
}

func withRequestContext(ctx context.Context, rc *requestContext) context.Context {
	return context.WithValue(ctx, requestCtxKey{}, rc)
}

func requestContextFrom(ctx context.Context) *requestContext {
	if rc, ok := ctx.Value(requestCtxKey{}).(*requestContext); ok {
		return rc
	}
	return nil
}

// setOutbound stashes the about-to-be-sent upstream URL/body/model on the
// requestContext so writeProxyError can record a bug report if the request
// fails at the transport level.
func setOutbound(ctx context.Context, model, url string, body []byte) {
	if rc := requestContextFrom(ctx); rc != nil {
		rc.Model = model
		rc.OutboundURL = url
		rc.OutboundBody = body
	}
}

// BugReport is the on-disk JSON structure.
type BugReport struct {
	RequestID      string          `json:"request_id"`
	Timestamp      string          `json:"timestamp"`
	Method         string          `json:"method"`
	Path           string          `json:"path"`
	Model          string          `json:"model"`
	UpstreamURL    string          `json:"upstream_url"`
	UpstreamStatus int             `json:"upstream_status"`
	Note           string          `json:"note,omitempty"`
	ClientRequest  json.RawMessage `json:"client_request,omitempty"`
	OutboundRequest json.RawMessage `json:"outbound_request,omitempty"`
	UpstreamError  json.RawMessage `json:"upstream_error,omitempty"`
}

// writeBugReport is the single entry point. It pulls all context from the
// requestContext in ctx, filters by status, then builds and saves the report.
// Only statuses that indicate real proxy/upstream bugs are recorded:
//   - 400, 422  — bad/invalid request (conversion bug or model fault)
//   - >= 500    — upstream/internal server error
//   - 0         — transport failure (httpClient.Do returned err)
//
// Client/environment errors (401/403/404/429 etc.) are excluded — the caller
// already knows about them from the response.
func writeBugReport(ctx context.Context, status int, upstreamErr []byte, note string) {
	if !bugReportEnabled {
		return
	}
	rc := requestContextFrom(ctx)
	if rc == nil {
		return
	}
	if !(status == 0 || status == 400 || status == 422 || status >= 500) {
		return
	}

	report := buildBugReport(rc, status, upstreamErr, note)
	saveBugReport(report)
}

// buildBugReport assembles a BugReport from the request context.
func buildBugReport(rc *requestContext, status int, upstreamErr []byte, note string) BugReport {
	return BugReport{
		RequestID:       rc.ID,
		Timestamp:       time.Now().Format(time.RFC3339Nano),
		Method:          rc.Method,
		Path:            rc.Path,
		Model:           rc.Model,
		UpstreamURL:     rc.OutboundURL,
		UpstreamStatus:  status,
		Note:            note,
		ClientRequest:   asJSONRaw(rc.ClientBody),
		OutboundRequest: asJSONRaw(rc.OutboundBody),
		UpstreamError:   asJSONRaw(upstreamErr),
	}
}

// saveBugReport writes a BugReport to bugreports/ with a unique filename
// (requestID + nanosecond timestamp) so repeated failures never overwrite.
func saveBugReport(report BugReport) {
	dir := "bugreports"
	if err := os.MkdirAll(dir, 0755); err != nil {
		log.Printf("[bugreport] mkdir %s failed: %v", dir, err)
		return
	}

	fname := fmt.Sprintf("bug-%s.json", report.RequestID)
	fpath := filepath.Join(dir, fname)

	data, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		log.Printf("[bugreport] marshal failed: %v", err)
		return
	}
	if err := os.WriteFile(fpath, data, 0644); err != nil {
		log.Printf("[bugreport] write %s failed: %v", fpath, err)
		return
	}
	log.Printf("[bugreport] wrote %s (req=%s model=%s upstream=%d)",
		fpath, report.RequestID, report.Model, report.UpstreamStatus)
}

// asJSONRaw returns b as a json.RawMessage if valid JSON, else wraps as JSON
// string. nil in → nil out.
func asJSONRaw(b []byte) json.RawMessage {
	if len(b) == 0 {
		return nil
	}
	var v interface{}
	if json.Unmarshal(b, &v) == nil {
		return b
	}
	enc, _ := json.Marshal(string(b))
	return enc
}
