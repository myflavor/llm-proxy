package main

import (
	"bytes"
	"crypto/subtle"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"
)

// Public endpoints that bypass authentication
var publicPaths = []string{
	"/health",
	"/v1/models",
	"/models",
}

func isPublicPath(path string) bool {
	for _, p := range publicPaths {
		if path == p {
			return true
		}
	}
	return false
}

// responseWriter wraps http.ResponseWriter to capture status code.
type responseWriter struct {
	http.ResponseWriter
	status        int
	headerWritten bool
}

func (rw *responseWriter) WriteHeader(code int) {
	if !rw.headerWritten {
		rw.status = code
		rw.headerWritten = true
		rw.ResponseWriter.WriteHeader(code)
	}
}

func (rw *responseWriter) Write(b []byte) (int, error) {
	if !rw.headerWritten {
		rw.WriteHeader(http.StatusOK)
	}
	return rw.ResponseWriter.Write(b)
}

// Flush implements http.Flusher by delegating to the underlying writer.
func (rw *responseWriter) Flush() {
	if f, ok := rw.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

// authMiddleware checks Authorization: Bearer <key> if serverAPIKey is set.
func authMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if serverAPIKey == "" {
			next.ServeHTTP(w, r)
			return
		}
		// Health, models, and CORS bypass auth.
		path := urlPath(r.URL.Path)
		if isPublicPath(path) || r.Method == http.MethodOptions {
			next.ServeHTTP(w, r)
			return
		}

		// Check Authorization: Bearer <key> (OpenAI style)
		key := ""
		if auth := r.Header.Get(headerAuthorization); strings.HasPrefix(auth, authBearerPrefix) {
			key = strings.TrimPrefix(auth, authBearerPrefix)
		}
		// Also check x-api-key header (Anthropic style)
		if key == "" {
			key = r.Header.Get("x-api-key")
		}
		if key == "" {
			writeJSON(w, http.StatusUnauthorized, map[string]interface{}{
				"error": map[string]interface{}{"message": "missing API key (use Authorization: Bearer or x-api-key)", "type": "auth_error"},
			})
			return
		}
		if subtle.ConstantTimeCompare([]byte(key), []byte(serverAPIKey)) != 1 {
			writeJSON(w, http.StatusUnauthorized, map[string]interface{}{
				"error": map[string]interface{}{"message": "invalid API key", "type": "auth_error"},
			})
			return
		}
		next.ServeHTTP(w, r)
	})
}

// logMiddleware logs each request and recovers from panics.
func logMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		rw := &responseWriter{ResponseWriter: w, status: 200, headerWritten: false}

		log.Printf("%s %s", r.Method, urlPath(r.URL.Path))

		// Build a request context for bug reporting: a unique request ID plus the
		// original inbound body. Tee the body so handlers still read it normally.
		rc := &requestContext{
			ID:        newRequestID(),
			StartTime: start,
			Method:    r.Method,
			Path:      urlPath(r.URL.Path),
		}
		if r.Body != nil {
			buf, err := io.ReadAll(r.Body)
			r.Body.Close()
			if err != nil {
				log.Printf("[bugreport] read inbound body err: %v", err)
			} else {
				rc.ClientBody = buf
				r.Body = io.NopCloser(bytes.NewReader(buf))
			}
		}
		r = r.WithContext(withRequestContext(r.Context(), rc))

		defer func() {
			if rec := recover(); rec != nil {
				log.Printf("PANIC %s %s: %v", r.Method, r.URL.Path, rec)
				// Only write error response if headers haven't been sent yet
				if !rw.headerWritten {
					writeError(rw, http.StatusInternalServerError, fmt.Sprintf("internal error: %v", rec))
					rw.status = 500
				}
			}
			log.Printf("  [response] %d %s", rw.status, time.Since(start).Round(time.Millisecond))
			if rc.Report != nil {
				if fname := saveBugReport(rc.Report); fname != "" {
					log.Printf("  [bugreport] %s", fname)
				}
			}
		}()

		next.ServeHTTP(rw, r)
	})
}

// --- HTTP helpers ---

func urlPath(u string) string {
	if i := strings.Index(u, "?"); i != -1 {
		return u[:i]
	}
	return u
}

func setCORS(w http.ResponseWriter) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Headers", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
}

func writeJSON(w http.ResponseWriter, status int, v interface{}) {
	w.Header().Set(headerContentType, contentTypeJSON)
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

func writeProxyError(w http.ResponseWriter, r *http.Request, err error) {
	// Transport / request-construction failure: record a bug report (status 0)
	// since this is an upstream-unreachable or proxy-internal failure worth
	// diagnosing. The outbound URL/body/model are stashed on the requestContext
	// by the handler before building req2.
	if rc := requestContextFrom(r.Context()); rc != nil {
		writeBugReport(r.Context(), 0, nil, "proxy error: "+err.Error())
	}
	status := http.StatusBadGateway
	if r.Context().Err() != nil {
		status = 499
	}
	writeJSON(w, status, map[string]interface{}{
		"error": map[string]interface{}{"message": err.Error(), "type": "proxy_error"},
	})
}

func extractUpstreamError(body []byte) string {
	var errResp struct {
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if json.Unmarshal(body, &errResp) == nil && errResp.Error.Message != "" {
		return errResp.Error.Message
	}
	return string(body)
}

func readRequestBody(r *http.Request) ([]byte, error) {
	const maxBody = 32 * 1024 * 1024

	// If the middleware already read the body for bugreport, reuse it and
	// avoid a second read of r.Body.
	if rc := requestContextFrom(r.Context()); rc != nil && len(rc.ClientBody) > 0 {
		if len(rc.ClientBody) > maxBody {
			return nil, fmt.Errorf("request body exceeds %d bytes", maxBody)
		}
		return rc.ClientBody, nil
	}

	bodyBytes, err := io.ReadAll(io.LimitReader(r.Body, maxBody+1))
	if err != nil {
		return nil, err
	}
	if len(bodyBytes) > maxBody {
		return nil, fmt.Errorf("request body exceeds %d bytes", maxBody)
	}
	return bodyBytes, nil
}
