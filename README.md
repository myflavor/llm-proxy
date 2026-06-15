# LLM Proxy

一个轻量级的 LLM API 代理服务，支持 OpenAI 和 Anthropic 双协议入口，自动处理上游协议转换。

## 功能特性

- 🔄 **双协议支持**：同时提供 OpenAI (`/v1/chat/completions`) 和 Anthropic (`/v1/messages`) 接口
- 🔀 **自动协议转换**：客户端和上游协议可任意组合，代理自动处理转换
- 🚀 高性能并发处理
- 📝 流式响应支持
- 🔧 灵活的模型路由配置
- 🔑 统一的 API Key 认证
- 🎯 支持多个上游提供商（OpenAI、Anthropic、OpenCode Zen、MiniMax、Gemini 等）

## 快速开始

### 下载

从 [Releases](../../releases) 页面下载对应平台的预编译版本：

- **Linux AMD64**: `llm-proxy-linux-amd64.zip`
- **Linux ARM64**: `llm-proxy-linux-arm64.zip`
- **macOS Intel**: `llm-proxy-darwin-amd64.zip`
- **macOS Apple Silicon**: `llm-proxy-darwin-arm64.zip`
- **Windows AMD64**: `llm-proxy-windows-amd64.zip`
- **Windows ARM64**: `llm-proxy-windows-arm64.zip`

### 安装

解压下载的 zip 文件：

```bash
unzip llm-proxy-linux-amd64.zip
cd llm-proxy
```

### 配置

编辑 `config.yaml` 文件，配置你的 API 密钥和模型映射：

```yaml
server:
  port: "5000"  # 监听端口，可通过环境变量 PORT 覆盖
  api_key: "sk-your-secret-key"  # 客户端访问代理服务时使用的密钥，留空则不需要认证

model_list:
  # OpenAI 兼容的上游
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4
      api_key: sk-your-openai-key
      api_base: https://api.openai.com/v1

  - model_name: deepseek-chat
    litellm_params:
      model: openai/deepseek-chat
      api_key: sk-your-deepseek-key
      api_base: https://api.deepseek.com/v1

  # Anthropic 兼容的上游（通过 MiniMax）
  - model_name: MiniMax-M3
    litellm_params:
      model: anthropic/MiniMax-M3
      api_key: your-minimax-key
      api_base: https://api.minimaxi.com/anthropic

  # 免费模型示例（OpenCode Zen）
  - model_name: mimo-v2.5-free
    litellm_params:
      model: openai/mimo-v2.5-free
      api_key: public
      api_base: https://opencode.ai/zen/v1
```

### 运行

Linux/macOS:
```bash
chmod +x llm-proxy
./llm-proxy
```

Windows:
```cmd
llm-proxy.exe
```

服务默认在 `http://localhost:5000` 启动（可通过环境变量 `PORT` 修改）。

## 使用示例

### 方式一：使用 OpenAI 协议调用

```bash
# 调用 OpenAI 上游模型
curl http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-your-secret-key" \
  -d '{
    "model": "gpt-4",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'

# 通过 OpenAI 协议调用 Anthropic 上游（自动转换）
curl http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-your-secret-key" \
  -d '{
    "model": "MiniMax-M3",
    "messages": [
      {"role": "user", "content": "你好！"}
    ]
  }'
```

### 方式二：使用 Anthropic 协议调用

```bash
# 调用 Anthropic 上游模型
curl http://localhost:5000/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk-your-secret-key" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "MiniMax-M3",
    "max_tokens": 1024,
    "messages": [
      {"role": "user", "content": "你好！"}
    ]
  }'

# 通过 Anthropic 协议调用 OpenAI 上游（自动转换）
curl http://localhost:5000/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk-your-secret-key" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "gpt-4",
    "max_tokens": 1024,
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

### 流式响应

```bash
# OpenAI 格式流式
curl http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-your-secret-key" \
  -d '{
    "model": "mimo-v2.5-free",
    "messages": [{"role": "user", "content": "写一首诗"}],
    "stream": true
  }'

# Anthropic 格式流式
curl http://localhost:5000/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk-your-secret-key" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "gpt-4",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "写一首诗"}],
    "stream": true
  }'
```

### 查询可用模型

```bash
curl http://localhost:5000/v1/models \
  -H "Authorization: Bearer sk-your-secret-key"
```

## 配置说明

### Server 配置

- `server.port`: 监听端口，默认 `"5000"`，可通过环境变量 `PORT` 覆盖
- `server.api_key`: 客户端访问代理服务时需要的认证密钥，留空则不需要认证

### Model 配置

每个模型条目包含：

- `model_name`: 客户端请求时使用的模型名称
- `litellm_params.model`: 上游模型格式，格式为 `provider/model`
  - `provider` 可以是 `openai` 或 `anthropic`
  - `model` 是上游实际的模型名称
- `litellm_params.api_key`: 上游服务的 API 密钥，支持环境变量 `${ENV_VAR}`
- `litellm_params.api_base`: 上游服务的 API 基础 URL
- `litellm_params.drop_params`: (可选) 是否丢弃上游不支持的参数

### 环境变量支持

配置文件中的 `api_key` 字段支持环境变量替换：

```yaml
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4
      api_key: ${OPENAI_API_KEY}
      api_base: https://api.openai.com/v1
```

## 从源码编译

需要 Go 1.26 或更高版本：

```bash
git clone <repository-url>
cd llm-proxy
go build -o llm-proxy
```

## 支持的上游协议

代理服务支持两种客户端协议和两种上游协议的任意组合：

### 客户端协议（入口）

- **OpenAI 协议**: `/v1/chat/completions`
  - 使用 `Authorization: Bearer <token>` 认证
  - 标准 OpenAI Chat Completions 格式

- **Anthropic 协议**: `/v1/messages`
  - 使用 `x-api-key: <token>` 认证
  - 需要 `anthropic-version` 头（如 `2023-06-01`）
  - Anthropic Messages API 格式

### 上游协议（配置）

- **OpenAI 上游** (`litellm_params.model: "openai/..."`)
  - 标准 OpenAI API 格式
  - 调用 `{api_base}/chat/completions`

- **Anthropic 上游** (`litellm_params.model: "anthropic/..."`)
  - Anthropic Messages API 格式
  - 调用 `{api_base}/v1/messages`

### 协议转换

代理自动处理以下四种组合：

| 客户端协议 | 上游协议 | 处理方式 |
|----------|---------|---------|
| OpenAI   | OpenAI  | 直接转发（仅替换模型名） |
| OpenAI   | Anthropic | 自动转换：OpenAI → Anthropic |
| Anthropic | Anthropic | 直接转发（仅替换模型名） |
| Anthropic | OpenAI | 自动转换：Anthropic → OpenAI |

## 许可证

MIT

## 贡献

欢迎提交 Issue 和 Pull Request！
