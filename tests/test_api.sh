#!/usr/bin/env bash
set -euo pipefail

SERVICE_URL="${EMBEDDINGS_SERVICE_URL:-http://localhost:${EMBEDDINGS_PORT:-11445}}"
MODEL_ID="${EMBEDDINGS_MODEL_ID:-ibm-granite/granite-embedding-30m-english}"
API_KEY="${EMBEDDINGS_API_KEY:-test-key}"

require() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

require curl
require python3

expect_http() {
  local expected="$1"
  local actual="$2"
  if [[ "$actual" != "$expected" ]]; then
    echo "Expected HTTP $expected, got $actual" >&2
    exit 1
  fi
}

request() {
  local payload="$1"
  curl -sS -w "\n%{http_code}" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${API_KEY}" \
    -d "$payload" \
    "$SERVICE_URL/v1/embeddings"
}

request_no_auth() {
  local payload="$1"
  curl -sS -w "\n%{http_code}" \
    -H "Content-Type: application/json" \
    -d "$payload" \
    "$SERVICE_URL/v1/embeddings"
}

# 1) auth required
body_and_status=$(request_no_auth "{\"model\":\"$MODEL_ID\",\"input\":\"test\"}")
status=$(echo "$body_and_status" | tail -n1)
expect_http 401 "$status"

# 2) basic request
body_and_status=$(request "{\"model\":\"$MODEL_ID\",\"input\":\"hello\"}")
status=$(echo "$body_and_status" | tail -n1)
expect_http 200 "$status"
body=$(echo "$body_and_status" | sed '$d')
python3 - <<'PY' "$body"
import json, sys
payload=json.loads(sys.argv[1])
assert payload["object"] == "list"
assert payload["model"]
assert payload["data"] and isinstance(payload["data"], list)
item=payload["data"][0]
assert item["object"] == "embedding"
assert isinstance(item["embedding"], list)
assert len(item["embedding"]) > 0
usage=payload["usage"]
assert "prompt_tokens" in usage and "total_tokens" in usage
PY

# 3) batch request
body_and_status=$(request "{\"model\":\"$MODEL_ID\",\"input\":[\"a\",\"b\"]}")
status=$(echo "$body_and_status" | tail -n1)
expect_http 200 "$status"
body=$(echo "$body_and_status" | sed '$d')
python3 - <<'PY' "$body"
import json, sys
payload=json.loads(sys.argv[1])
assert len(payload["data"]) == 2
assert payload["data"][0]["index"] == 0
assert payload["data"][1]["index"] == 1
PY

# 4) token array input
body_and_status=$(request "{\"model\":\"$MODEL_ID\",\"input\":[1,2,3,4]}")
status=$(echo "$body_and_status" | tail -n1)
expect_http 200 "$status"

# 5) dimensions truncation
body_and_status=$(request "{\"model\":\"$MODEL_ID\",\"input\":\"hello\",\"dimensions\":8}")
status=$(echo "$body_and_status" | tail -n1)
expect_http 200 "$status"
body=$(echo "$body_and_status" | sed '$d')
python3 - <<'PY' "$body"
import json, sys
payload=json.loads(sys.argv[1])
vec=payload["data"][0]["embedding"]
assert len(vec) == 8
PY

# 6) base64 output
body_and_status=$(request "{\"model\":\"$MODEL_ID\",\"input\":\"hello\",\"encoding_format\":\"base64\"}")
status=$(echo "$body_and_status" | tail -n1)
expect_http 200 "$status"
body=$(echo "$body_and_status" | sed '$d')
python3 - <<'PY' "$body"
import base64, json, sys
payload=json.loads(sys.argv[1])
emb=payload["data"][0]["embedding"]
assert isinstance(emb, str)
raw=base64.b64decode(emb)
assert len(raw) % 4 == 0
assert len(raw) >= 4
PY

# 7) max batch size enforcement
body_and_status=$(request "{\"model\":\"$MODEL_ID\",\"input\":[\"a\",\"b\",\"c\"]}")
status=$(echo "$body_and_status" | tail -n1)
expect_http 400 "$status"

# 8) max total tokens enforcement (token array length > limit)
payload=$(python3 - <<PY
import json
print(json.dumps({"model": "${MODEL_ID}", "input": list(range(0, 101))}))
PY
)
body_and_status=$(request "$payload")
status=$(echo "$body_and_status" | tail -n1)
expect_http 400 "$status"

# 9) request size limit enforcement
payload=$(python3 - <<PY
import json
big = "a" * 60000
print(json.dumps({"model": "${MODEL_ID}", "input": big}))
PY
)
body_and_status=$(request "$payload")
status=$(echo "$body_and_status" | tail -n1)
expect_http 413 "$status"

# 10) schema is published
schema_status=$(curl -sS -o /dev/null -w "%{http_code}" "$SERVICE_URL/schema.json")
expect_http 200 "$schema_status"

echo "embeddings API tests passed"
