package main

import (
	"fmt"
	"net/http"
	"time"
)

var (
	providersByModel = map[string]*Provider{} // model_name → Provider
	providerList     []*Provider
	serverAPIKey     string // empty = no auth required
	serverPort       string // listen port, default "5000"
	startedAt        = time.Now().Unix()     // fixed timestamp for model listing
)

var httpClient = &http.Client{Timeout: 10 * time.Minute}

// getProvider looks up a provider by model name.
func getProvider(modelName string) (*Provider, error) {
	p, ok := providersByModel[modelName]
	if !ok {
		return nil, fmt.Errorf("model %q not found", modelName)
	}
	return p, nil
}

// handleModels handles GET /v1/models.
func handleModels(w http.ResponseWriter, r *http.Request) {
	setCORS(w)
	models := make([]map[string]interface{}, 0, len(providerList))
	for _, p := range providerList {
		models = append(models, map[string]interface{}{
			"id":       p.ModelName,
			"object":   "model",
			"created":  startedAt,
			"owned_by": string(p.Type),
		})
	}
	writeJSON(w, http.StatusOK, map[string]interface{}{"object": "list", "data": models})
}
