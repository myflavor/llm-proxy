#!/usr/bin/env bash
#
# test_proxy.sh — llm-proxy 集成测试脚本
#
# 用法:  bash test_proxy.sh
#
# 分两部分:
#   A. 确定性测试   —— 构建/格式/路由/认证/参数校验,不依赖上游,必须全绿。
#   B. 上游冒烟测试 —— 真实调用 opencode zen 免费模型,受上游可用性/限流影响,
#                     仅打印状态,不计入失败(上游 4xx/5xx 不算代理 bug)。
#
# 退出码: 确定性测试有任何一条失败 → 非 0;否则 0。

set -u

# ---- 配置 ----
PORT=5999
BASE="http://127.0.0.1:${PORT}"
KEY="sk-xin"                 # 与 config.yaml 的 server.api_key 一致
MODEL="mimo-v2.5-free"       # config.yaml 中存在的模型
CFG="config.yaml"
BIN="./llm-proxy.test.bin"
LOG="/tmp/llm-proxy-test.log"

PASS=0
FAIL=0
SRV_PID=""

# ---- 工具函数 ----
green() { printf '\033[32m%s\033[0m\n' "$1"; }
red()   { printf '\033[31m%s\033[0m\n' "$1"; }
info()  { printf '\033[36m%s\033[0m\n' "$1"; }

# ok <描述> <条件字符串: "实际" == "期望">
ok() { # $1 desc  $2 actual  $3 expected
  if [ "$2" = "$3" ]; then
    green "  PASS  $1  (got $2)"; PASS=$((PASS+1))
  else
    red   "  FAIL  $1  (got $2, want $3)"; FAIL=$((FAIL+1))
  fi
}

# 发请求,回显 HTTP 状态码
code() { # $@ curl args
  curl -s -o /dev/null -w '%{http_code}' --max-time 60 "$@"
}

cleanup() {
  [ -n "$SRV_PID" ] && kill "$SRV_PID" 2>/dev/null
  rm -f "$BIN"
}
trap cleanup EXIT

# ============================================================
# A. 确定性测试
# ============================================================
info "== A. 确定性测试(必须全绿) =="

# A1. 构建
if go build -o "$BIN" . 2>build.err; then
  ok "go build" "ok" "ok"
else
  ok "go build" "fail" "ok"; cat build.err; rm -f build.err
  echo "构建失败,终止。"; exit 1
fi
rm -f build.err

# A2. gofmt 干净
UNFMT=$(gofmt -l *.go | wc -l | tr -d ' ')
ok "gofmt 无未格式化文件" "$UNFMT" "0"

# A3. go vet 干净
if go vet ./... 2>vet.err; then
  ok "go vet" "ok" "ok"
else
  ok "go vet" "fail" "ok"; cat vet.err
fi
rm -f vet.err

# 启动服务
info "启动服务 (PORT=$PORT) ..."
PORT=$PORT "$BIN" "$CFG" >"$LOG" 2>&1 &
SRV_PID=$!

# 等待就绪(最多 10s)
READY=""
for i in $(seq 1 50); do
  if curl -s -o /dev/null "${BASE}/health" 2>/dev/null; then READY="yes"; break; fi
  sleep 0.2
done
ok "服务启动并响应 /health" "${READY:-no}" "yes"
if [ "${READY:-no}" != "yes" ]; then
  echo "服务未就绪,日志:"; cat "$LOG"; exit 1
fi

# A4. /health -> 200
ok "GET /health" "$(code "${BASE}/health")" "200"

# A5. /v1/models 无需认证 -> 200
ok "GET /v1/models (公开)" "$(code "${BASE}/v1/models")" "200"

# A6. /v1/models 返回含配置的模型名
if curl -s "${BASE}/v1/models" | grep -q "$MODEL"; then
  ok "/v1/models 含 $MODEL" "yes" "yes"
else
  ok "/v1/models 含 $MODEL" "no" "yes"
fi

# A7. 缺少 API key -> 401
ok "POST /v1/chat/completions 无 key -> 401" \
  "$(code -X POST "${BASE}/v1/chat/completions" -H 'Content-Type: application/json' -d '{"model":"'"$MODEL"'","messages":[]}')" \
  "401"

# A8. 错误 API key -> 401
ok "POST 错误 key -> 401" \
  "$(code -X POST "${BASE}/v1/chat/completions" -H 'Content-Type: application/json' -H 'Authorization: Bearer wrong-key' -d '{"model":"'"$MODEL"'","messages":[]}')" \
  "401"

# A9. 未知模型 -> 404
ok "POST 未知模型 -> 404" \
  "$(code -X POST "${BASE}/v1/chat/completions" -H 'Content-Type: application/json' -H "Authorization: Bearer $KEY" -d '{"model":"no-such-model","messages":[]}')" \
  "404"

# A10. 非法 JSON -> 400
ok "POST 非法 JSON -> 400" \
  "$(code -X POST "${BASE}/v1/chat/completions" -H 'Content-Type: application/json' -H "Authorization: Bearer $KEY" -d 'not-json')" \
  "400"

# A11. x-api-key 头(Anthropic 风格)也能通过认证
#      用未知模型触发 404(而非 401),证明认证已通过
ok "x-api-key 认证通过(-> 404 而非 401)" \
  "$(code -X POST "${BASE}/v1/messages" -H 'Content-Type: application/json' -H "x-api-key: $KEY" -d '{"model":"no-such-model","messages":[]}')" \
  "404"

# ============================================================
# B. 上游冒烟测试(仅报告,不计失败)
# ============================================================
info ""
info "== B. 上游冒烟测试(真实调用 opencode zen,仅报告状态) =="
info "   200=成功  4xx/5xx=上游问题(限流/不可用),非代理 bug"

smoke() { # $1 desc  $2... curl args
  local desc="$1"; shift
  local c; c=$(code "$@")
  if [ "$c" = "200" ]; then green "  OK    $desc  (HTTP $c)"
  else                       info  "  ~     $desc  (HTTP $c, 上游相关)"; fi
}

# B1. OpenAI 格式 (passthrough: openai -> openai)
smoke "POST /v1/chat/completions 非流式" \
  -X POST "${BASE}/v1/chat/completions" -H 'Content-Type: application/json' -H "Authorization: Bearer $KEY" \
  -d '{"model":"'"$MODEL"'","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}'

# B2. OpenAI 格式 流式
smoke "POST /v1/chat/completions 流式" \
  -X POST "${BASE}/v1/chat/completions" -H 'Content-Type: application/json' -H "Authorization: Bearer $KEY" \
  -d '{"model":"'"$MODEL"'","max_tokens":16,"stream":true,"messages":[{"role":"user","content":"hi"}]}'

# B3. Anthropic 格式 (转换: anthropic -> openai)
smoke "POST /v1/messages 非流式" \
  -X POST "${BASE}/v1/messages" -H 'Content-Type: application/json' -H "x-api-key: $KEY" \
  -d '{"model":"'"$MODEL"'","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}'

# B4. Anthropic 格式 流式
smoke "POST /v1/messages 流式" \
  -X POST "${BASE}/v1/messages" -H 'Content-Type: application/json' -H "x-api-key: $KEY" \
  -d '{"model":"'"$MODEL"'","max_tokens":16,"stream":true,"messages":[{"role":"user","content":"hi"}]}'

# B5. Responses 格式 (转换: responses -> openai)
smoke "POST /v1/responses 非流式" \
  -X POST "${BASE}/v1/responses" -H 'Content-Type: application/json' -H "Authorization: Bearer $KEY" \
  -d '{"model":"'"$MODEL"'","max_output_tokens":16,"input":"hi"}'

# ============================================================
# 汇总
# ============================================================
info ""
info "== 汇总 =="
echo "  确定性测试: PASS=$PASS  FAIL=$FAIL"
echo "  服务日志:   $LOG"
if [ "$FAIL" -eq 0 ]; then
  green "确定性测试全部通过 ✅"
  exit 0
else
  red "有 $FAIL 条确定性测试失败 ❌"
  exit 1
fi
