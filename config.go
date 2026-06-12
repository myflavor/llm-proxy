package main

import (
	"fmt"
	"os"
	"strings"

	"gopkg.in/yaml.v3"
)

// Config represents the YAML configuration file.
type Config struct {
	Server    ServerConfig `yaml:"server"`
	ModelList []ModelEntry `yaml:"model_list"`
}

// ServerConfig holds server-level settings.
type ServerConfig struct {
	APIKey string `yaml:"api_key"`
}

// ModelEntry is a single model definition in the config.
type ModelEntry struct {
	ModelName     string        `yaml:"model_name"`
	LitellmParams LitellmParams `yaml:"litellm_params"`
}

// LitellmParams holds the upstream provider parameters.
type LitellmParams struct {
	Model      string `yaml:"model"`       // "provider/upstream_model"
	APIKey     string `yaml:"api_key"`
	APIBase    string `yaml:"api_base"`
	DropParams bool   `yaml:"drop_params"` // drop unsupported params
}

// ProviderType represents the upstream API format.
type ProviderType string

const (
	ProviderOpenAI    ProviderType = "openai"
	ProviderAnthropic ProviderType = "anthropic"
)

// Provider is a resolved upstream endpoint.
type Provider struct {
	Name        string       // upstream model name (without prefix)
	Type        ProviderType // openai or anthropic
	APIKey      string
	APIBase     string // base URL
	ChatURL     string // full chat completions URL
	MessagesURL string // full messages URL (for anthropic)
	DropParams  bool
	ModelName   string // the model_name clients use to select this provider
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

	for _, entry := range cfg.ModelList {
		parts := strings.SplitN(entry.LitellmParams.Model, "/", 2)
		if len(parts) != 2 {
			return fmt.Errorf("invalid model format %q (expected provider/name)", entry.LitellmParams.Model)
		}
		providerType := ProviderType(parts[0])
		upstreamModel := parts[1]

		baseURL := strings.TrimRight(entry.LitellmParams.APIBase, "/")

		p := &Provider{
			Name:       upstreamModel,
			Type:       providerType,
			APIKey:     entry.LitellmParams.APIKey,
			APIBase:    baseURL,
			DropParams: entry.LitellmParams.DropParams,
			ModelName:  entry.ModelName,
		}

		switch providerType {
		case ProviderOpenAI:
			p.ChatURL = baseURL + "/chat/completions"
		case ProviderAnthropic:
			p.MessagesURL = baseURL + "/v1/messages"
		default:
			return fmt.Errorf("unknown provider type %q for model %s", providerType, entry.ModelName)
		}

		providersByModel[entry.ModelName] = p
		providerList = append(providerList, p)
	}

	return nil
}
