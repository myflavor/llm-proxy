package main

import (
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
		if auth := r.Header.Get("Authorization"); strings.HasPrefix(auth, "Bearer ") {
			key = strings.TrimPrefix(auth, "Bearer ")
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

		defer func() {
			if rec := recover(); rec != nil {
				log.Printf("PANIC %s %s: %v", r.Method, r.URL.Path, rec)
				// Only write error response if headers haven't been sent yet
				if !rw.headerWritten {
					writeJSON(rw, http.StatusInternalServerError, map[string]interface{}{
						"error": map[string]interface{}{"message": fmt.Sprintf("internal error: %v", rec), "type": "server_error"},
					})
					rw.status = 500
				}
			}
			log.Printf("%s %s %d %s", r.Method, r.URL.Path, rw.status, time.Since(start).Round(time.Millisecond))
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
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

func writeProxyError(w http.ResponseWriter, r *http.Request, err error) {
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

func readBody(r *http.Request) ([]byte, error) {
	const maxBody = 32 * 1024 * 1024
	bodyBytes, err := io.ReadAll(io.LimitReader(r.Body, maxBody+1))
	if err != nil {
		return nil, err
	}
	if len(bodyBytes) > maxBody {
		return nil, fmt.Errorf("request body exceeds %d bytes", maxBody)
	}
	return bodyBytes, nil
}
