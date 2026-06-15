# LLM Proxy

一个轻量级的 LLM API 代理服务，支持多种 AI 模型提供商之间的协议转换。

## 功能特性

- 🔄 支持 OpenAI 和 Anthropic API 协议互转
- 🚀 高性能并发处理
- 📝 流式响应支持
- 🔧 灵活的配置管理
- 🎯 多模型路由支持

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
listen: ":8080"

providers:
  - name: "openai"
    type: "openai"
    base_url: "https://api.openai.com/v1"
    api_key: "sk-your-openai-key"
    models:
      - "gpt-4"
      - "gpt-3.5-turbo"

  - name: "anthropic"
    type: "anthropic"
    base_url: "https://api.anthropic.com"
    api_key: "sk-ant-your-anthropic-key"
    models:
      - "claude-3-opus-20240229"
      - "claude-3-sonnet-20240229"
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

服务将在配置的端口（默认 8080）上启动。

## 使用示例

### 通过 OpenAI 协议访问 Anthropic 模型

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-ant-your-anthropic-key" \
  -d '{
    "model": "claude-3-opus-20240229",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

### 流式响应

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

## 配置说明

### 基本配置

- `listen`: 监听地址和端口，格式为 `":8080"` 或 `"0.0.0.0:8080"`

### Provider 配置

每个 provider 包含以下字段：

- `name`: Provider 名称（唯一标识）
- `type`: Provider 类型（`openai` 或 `anthropic`）
- `base_url`: API 基础 URL
- `api_key`: API 密钥
- `models`: 支持的模型列表

## 从源码编译

需要 Go 1.26 或更高版本：

```bash
git clone <repository-url>
cd llm-proxy
go build -o llm-proxy
```

## 许可证

[添加你的许可证信息]

## 贡献

欢迎提交 Issue 和 Pull Request！
