package main

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"time"
)

// bugReportEnabled writes bugreports/ on upstream errors. Toggle via config
// `server.bug_report: true` or env var LLM_PROXY_BUG_REPORT=1.
var bugReportEnabled bool

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
	Report       *BugReport // built by writeBugReport, saved by logMiddleware after request
}

func newRequestID() string {
	var uuid [16]byte
	rand.Read(uuid[:])
	uuid[6] = (uuid[6] & 0x0f) | 0x40 // version 4
	uuid[8] = (uuid[8] & 0x3f) | 0x80 // variant 10
	return fmt.Sprintf("%s-%s-%s-%s-%s",
		hex.EncodeToString(uuid[:4]),
		hex.EncodeToString(uuid[4:6]),
		hex.EncodeToString(uuid[6:8]),
		hex.EncodeToString(uuid[8:10]),
		hex.EncodeToString(uuid[10:]),
	)
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
// requestContext so writeBugReport can record a bug report if the request
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

// writeBugReport builds a BugReport from the request context and stores it on
// rc.Report. The actual file I/O happens later in logMiddleware's defer (after
// the request completes), so the handler never blocks on disk writes.
//
// Only statuses that indicate real proxy/upstream bugs are recorded:
//   - 400, 422  — bad/invalid request (conversion bug or model fault)
//   - >= 500    — upstream/internal server error
//   - 0         — transport failure (httpClient.Do returned err)
//
// Client/environment errors (401/403/404/429 etc.) are excluded.
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

	report := BugReport{
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
	rc.Report = &report
}

// saveBugReport writes a BugReport to bugreports/<uuid>.json.
// Returns the filename, or empty on error.
func saveBugReport(report *BugReport) string {
	dir := "bugreports"
	if err := os.MkdirAll(dir, 0755); err != nil {
		log.Printf("[bugreport] mkdir %s failed: %v", dir, err)
		return ""
	}

	fname := report.RequestID + ".json"
	fpath := filepath.Join(dir, fname)
	data, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		log.Printf("[bugreport] marshal failed: %v", err)
		return ""
	}
	if err := os.WriteFile(fpath, data, 0644); err != nil {
		log.Printf("[bugreport] write %s failed: %v", fpath, err)
		return ""
	}
	return fname
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
