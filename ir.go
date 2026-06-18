package main

// IR (Intermediate Representation) — 统一的内部数据格式。
// 所有 API 格式（Chat Completions、Anthropic Messages、Responses）都转换为 IR，
// 再从 IR 转换为目标格式。这样每加一种新格式只需写 2 个函数（进 IR + 出 IR）。

// --- IR Request ---

type IRRequest struct {
	Model         string
	SystemPrompt  string
	Messages      []IRMessage
	MaxTokens     int
	Temperature   *float64
	TopP          *float64
	StopSequences []string
	Stream        bool
	Tools         []IRTool
	ToolChoice    *IRToolChoice
	Thinking      *IRThinking
	Metadata      *IRMetadata
	Extensions    map[string]interface{} // 格式特有参数
}

type IRMessage struct {
	Role       string // user / assistant / tool
	Content    []IRContentBlock
	ToolCallID string // tool result 关联的 call id
}

type IRContentBlock struct {
	Type      string                 // text / image / thinking / redacted_thinking / tool_use
	Text      string                 // text 内容
	ImageURL  string                 // 图片 URL (data: 或 https:)
	Thinking  string                 // thinking/reasoning 文本
	Signature string                 // thinking block 签名（Anthropic 多轮对话需要）
	Data      string                 // redacted_thinking 的加密数据
	ToolUseID string                 // tool_use 的 ID
	ToolName  string                 // tool_use 的函数名
	ToolInput map[string]interface{} // tool_use 的参数
}

type IRTool struct {
	Name        string
	Description string
	Parameters  interface{} // JSON Schema
}

type IRToolChoice struct {
	Type string // auto / required / none / any / specific
	Name string // 当 type=specific 时指定函数名
}

type IRThinking struct {
	Enabled      bool
	BudgetTokens int    // Anthropic 旧格式 budget_tokens
	Effort       string // 统一 effort 等级：none/minimal/low/medium/high/xhigh/max/ultracode
}

type IRMetadata struct {
	UserID string
}

// --- IR Response ---

type IRResponse struct {
	ID         string
	Model      string
	Role       string
	Content    []IRContentBlock
	StopReason string // end_turn / max_tokens / tool_use
	Usage      IRUsage
}

type IRUsage struct {
	InputTokens  int
	OutputTokens int
}

// --- IR Stream Event ---

type IRStreamEvent struct {
	Type       string // message_start / content_delta / message_stop
	Index      int    // content block index (for content_delta)
	Delta      *IRContentBlock
	StopReason string
	Usage      *IRUsage
}
