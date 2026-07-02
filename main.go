package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"
)

func main() {
	configPath := "config.yaml"
	if len(os.Args) > 1 {
		configPath = os.Args[1]
	}

	if err := loadConfig(configPath); err != nil {
		log.Fatalf("load config: %v", err)
	}

	log.Printf("loaded %d models", len(providerList))
	for _, p := range providerList {
		log.Printf("  %s → %s/%s (%s)", p.ModelName, p.Type, p.Name, p.BaseURL)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		setCORS(w)
		writeJSON(w, http.StatusOK, map[string]interface{}{"ok": true, "models": len(providerList)})
	})
	mux.HandleFunc("/v1/models", handleModels)
	mux.HandleFunc("/models", handleModels)
	mux.HandleFunc("/v1/chat/completions", handleOpenAI)
	mux.HandleFunc("/chat/completions", handleOpenAI)
	mux.HandleFunc("/v1/messages", handleAnthropic)
	mux.HandleFunc("/messages", handleAnthropic)
	mux.HandleFunc("/v1/responses", handleResponses)
	mux.HandleFunc("/responses", handleResponses)
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodOptions {
			setCORS(w)
			w.WriteHeader(http.StatusNoContent)
			return
		}
		setCORS(w)
		writeJSON(w, http.StatusNotFound, map[string]interface{}{
			"error": map[string]interface{}{"message": "Not found", "type": "invalid_request_error"},
		})
	})

	port := serverPort
	if envPort := os.Getenv("PORT"); envPort != "" {
		port = envPort
	}

	srv := &http.Server{
		Addr:         "0.0.0.0:" + port,
		Handler:      authMiddleware(logMiddleware(mux)),
		ReadTimeout:  10 * time.Minute,
		WriteTimeout: 10 * time.Minute,
		IdleTimeout:  20 * time.Minute,
	}

	go func() {
		log.Printf("llm-proxy listening on http://0.0.0.0:%s", port)
		if serverAPIKey != "" {
			log.Printf("auth: enabled (server.api_key is set)")
		} else {
			log.Printf("auth: disabled (server.api_key is empty)")
		}
		if err := srv.ListenAndServe(); err != http.ErrServerClosed {
			log.Fatalf("server error: %v", err)
		}
	}()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGTERM, syscall.SIGINT)
	sig := <-sigCh
	log.Printf("\n%s received, shutting down...", sig)

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	srv.Shutdown(ctx)
	log.Println("server stopped")
}
