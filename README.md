# LLM Proxy

一个轻量级的 LLM API 代理服务，支持 OpenAI Chat Completions、Anthropic Messages 和 OpenAI Responses 三协议入口，通过 IR（中间表示）自动处理上游协议转换。

## 功能特性

- 🔄 **三协议支持**：同时提供 OpenAI (`/v1/chat/completions`)、Anthropic (`/v1/messages`) 和 Responses (`/v1/responses`) 接口
- 🔀 **自动协议转换**：客户端和上游协议可任意组合，代理自动处理转换
- 🏗️ **IR 架构**：通过中间表示解耦各格式，加新格式只需 2 个转换函数
- 🚀 高性能并发处理
- 📝 流式响应支持（SSE）
- 🔧 灵活的模型路由配置
- 🔑 统一的 API Key 认证
- 🎯 支持多个上游提供商（OpenAI、Anthropic、MiniMax 等）

## 快速开始

### 配置

编辑 `config.yaml`：

```yaml
server:
  port: "5000"
  api_key: "sk-your-secret-key"  # 留空则不需要认证

models:
  # OpenAI 兼容上游
  - name: gpt-4
    provider: openai
    model: gpt-4
    api_key: sk-your-openai-key
    base_url: https://api.openai.com/v1

  # Anthropic 兼容上游
  - name: claude-sonnet
    provider: anthropic
    model: claude-sonnet-4-20250514
    api_key: sk-your-anthropic-key
    base_url: https://api.anthropic.com

  # Responses API 原生上游
  - name: gpt-5-responses
    provider: responses
    model: gpt-5
    api_key: sk-your-openai-key
    base_url: https://api.openai.com/v1

  # 带额外参数的模型
  - name: some-model
    provider: openai
    model: some-model
    api_key: xxx
    base_url: https://api.example.com/v1
    drop_params: true          # 丢弃上游不支持的参数
    extra_params:              # 注入到上游请求体的额外参数
      custom_param: value
```

### 运行

```bash
# Linux/macOS
./llm-proxy config.yaml

# Windows
llm-proxy.exe config.yaml
```

服务默认在 `http://localhost:5000` 启动（可通过环境变量 `PORT` 覆盖）。

## 三种协议

### 1. OpenAI Chat Completions

```bash
curl http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-your-secret-key" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### 2. Anthropic Messages

```bash
curl http://localhost:5000/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk-your-secret-key" \
  -d '{
    "model": "claude-sonnet",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### 3. OpenAI Responses

```bash
curl http://localhost:5000/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-your-secret-key" \
  -d '{
    "model": "gpt-5-responses",
    "input": "Hello!"
  }'
```

### 流式响应

三种协议都支持流式，加 `"stream": true` 即可。

### 查询可用模型

```bash
curl http://localhost:5000/v1/models \
  -H "Authorization: Bearer sk-your-secret-key"
```

## 配置说明

### Server

| 字段 | 说明 | 默认值 |
|------|------|--------|
| `port` | 监听端口（环境变量 `PORT` 可覆盖） | `"5000"` |
| `api_key` | 客户端认证密钥，留空则不需要认证 | - |

### Models

| 字段 | 必填 | 说明 |
|------|------|------|
| `name` | ✅ | 客户端请求时使用的模型名 |
| `provider` | ✅ | 上游类型：`openai` / `anthropic` / `responses` |
| `model` | ✅ | 上游实际的模型名 |
| `api_key` | ✅ | 上游 API 密钥，支持 `${ENV_VAR}` |
| `base_url` | ✅ | 上游 API 基础 URL |
| `drop_params` | ❌ | 丢弃上游不支持的参数（如 thinking、tool_choice） |
| `extra_params` | ❌ | 注入到上游请求体的额外参数（key-value） |

### Provider 类型

| provider | 客户端请求 | 上游端点 |
|----------|----------|---------|
| `openai` | Chat Completions | `{base_url}/chat/completions` |
| `anthropic` | Messages | `{base_url}/v1/messages` |
| `responses` | Responses | `{base_url}/responses` |

### 协议转换矩阵

客户端请求 ↓ / 上游 → | `openai` | `anthropic` | `responses` |
|---|---|---|---|
| **Chat Completions** | 直接转发 | 自动转换 | - |
| **Messages** | 自动转换 | 直接转发 | 自动转换 |
| **Responses** | 自动转换 | - | 直接转发 |

## Thinking / Effort 支持

代理支持跨协议的 thinking effort 参数转换：

| 客户端格式 | effort 字段 | 值 |
|-----------|------------|---|
| Anthropic Messages | `output_config.effort` | `low` / `medium` / `high` / `xhigh` / `max` |
| OpenAI Responses | `reasoning.effort` | `none` / `minimal` / `low` / `medium` / `high` / `xhigh` |
| OpenAI Chat Completions | `reasoning_effort` | `low` / `medium` / `high` / `xhigh` |

effort 值在转换时自动映射，超出目标协议范围的值会降级到最高支持等级。

代理日志会显示 effort 转换过程：
```
[effort] MiniMax-M3-responses: max → responses/MiniMax-M3
[effort→] reasoning.effort=xhigh (clamped from max)
```

## 架构

```
客户端请求 → Handler → IR(中间表示) → 上游格式 → 上游 API
上游响应   → IR(中间表示) → 客户端格式 → 客户端
```

所有格式通过 IR 互转，每加一种新格式只需实现：
1. `formatToIR()` — 请求转换
2. `irToFormat()` — 响应转换

流式传输通过 IR 流式事件桥接，不经过完整 IR 缓冲。

### 文件结构

| 文件 | 职责 |
|------|------|
| `ir.go` | IR 中间表示类型定义 |
| `config.go` | 配置解析 |
| `convert.go` | 公共工具函数、Anthropic 类型定义 |
| `convert_ir.go` | 所有 IR 转换函数（各格式 ↔ IR） |
| `middleware.go` | 认证、日志、CORS |
| `provider.go` | 模型查找 |
| `main.go` | 入口、路由注册 |
| `anthropic.go` | Anthropic handler |
| `openai.go` | OpenAI handler |
| `responses.go` | Responses handler + 流式翻译 |
| `stream.go` | 流式透传 |

## 从源码编译

需要 Go 1.21 或更高版本：

```bash
git clone <repository-url>
cd llm-proxy
go build -o llm-proxy

# 交叉编译 Windows
GOOS=windows GOARCH=amd64 go build -o llm-proxy.exe
```

## 许可证

MIT
