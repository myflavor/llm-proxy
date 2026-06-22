package main

import (
	"fmt"
	"log"
	"net/url"
	"os"
	"strings"

	"gopkg.in/yaml.v3"
)

// Anthropic API version for all requests
const anthropicAPIVersion = "2023-06-01"

// Config represents the YAML configuration file.
type Config struct {
	Server ServerConfig `yaml:"server"`
	Models []ModelEntry `yaml:"models"`
}

// ServerConfig holds server-level settings.
type ServerConfig struct {
	Port   string `yaml:"port"`    // listen port, default "5000"
	APIKey string `yaml:"api_key"`
}

// ModelEntry is a single model definition in the config.
type ModelEntry struct {
	Name        string                 `yaml:"name"`         // 客户端使用的模型名
	Provider    string                 `yaml:"provider"`     // openai / anthropic / responses
	Model       string                 `yaml:"model"`        // 上游模型名
	APIKey      string                 `yaml:"api_key"`
	BaseURL     string                 `yaml:"base_url"`
	DropParams  bool                   `yaml:"drop_params"`  // 丢弃上游不支持的参数
	ExtraParams map[string]interface{} `yaml:"extra_params"` // 注入到上游请求体的额外参数
}

// ProviderType represents the upstream API format.
type ProviderType string

const (
	ProviderOpenAI    ProviderType = "openai"
	ProviderAnthropic ProviderType = "anthropic"
	ProviderResponses ProviderType = "responses" // 上游原生支持 Responses API
)

// Provider is a resolved upstream endpoint.
type Provider struct {
	Name         string                 // 上游模型名
	Type         ProviderType           // openai / anthropic / responses
	APIKey       string
	BaseURL      string                 // base URL
	ChatURL      string                 // full chat completions URL
	MessagesURL  string                 // full messages URL (for anthropic)
	ResponsesURL string                 // full responses URL
	DropParams   bool
	ExtraParams  map[string]interface{} // 注入到上游请求体的额外参数
	ModelName    string                 // 客户端使用的模型名
}

// validateModelEntry validates a model configuration entry
func validateModelEntry(entry ModelEntry) error {
	if entry.Name == "" {
		return fmt.Errorf("model name cannot be empty")
	}
	if entry.Model == "" {
		return fmt.Errorf("model %q: upstream model name cannot be empty", entry.Name)
	}
	if entry.BaseURL == "" {
		return fmt.Errorf("model %q: base_url cannot be empty", entry.Name)
	}
	if _, err := url.Parse(entry.BaseURL); err != nil {
		return fmt.Errorf("model %q: invalid base_url: %w", entry.Name, err)
	}
	// If api_key is provided as empty string, reject it
	if key, ok := entry.ExtraParams["api_key"].(string); ok && key == "" {
		return fmt.Errorf("model %q: api_key cannot be empty string", entry.Name)
	}
	return nil
}

// loadConfig reads and parses the YAML config file.
// Supports ${ENV_VAR} substitution in api_key fields.
func loadConfig(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("read config: %w", err)
	}

	content := os.ExpandEnv(string(data))

	var cfg Config
	if err := yaml.Unmarshal([]byte(content), &cfg); err != nil {
		return fmt.Errorf("parse config: %w", err)
	}

	serverAPIKey = cfg.Server.APIKey
	serverPort = cfg.Server.Port
	if serverPort == "" {
		serverPort = "5000"
	}

	for _, entry := range cfg.Models {
		if err := validateModelEntry(entry); err != nil {
			return err
		}

		providerType := ProviderType(entry.Provider)
		baseURL := strings.TrimRight(entry.BaseURL, "/")

		p := &Provider{
			Name:        entry.Model,
			Type:        providerType,
			APIKey:      entry.APIKey,
			BaseURL:     baseURL,
			DropParams:  entry.DropParams,
			ExtraParams: entry.ExtraParams,
			ModelName:   entry.Name,
		}

		switch providerType {
		case ProviderOpenAI:
			p.ChatURL = baseURL + "/chat/completions"
			p.ResponsesURL = baseURL + "/responses"
		case ProviderAnthropic:
			p.MessagesURL = baseURL + "/v1/messages"
		case ProviderResponses:
			p.ResponsesURL = baseURL + "/responses"
		default:
			return fmt.Errorf("unknown provider type %q for model %s (expected openai/anthropic/responses)", entry.Provider, entry.Name)
		}

		if _, exists := providersByModel[entry.Name]; exists {
			log.Printf("WARNING: duplicate model name %q, overwriting previous entry", entry.Name)
		}
		providersByModel[entry.Name] = p
		providerList = append(providerList, p)
	}

	return nil
}
