#!/usr/bin/env bash
# scripts/pre_validate.sh — Pre-submission validation script for SocialGuard-RL.
#
# Checks:
#   1. Docker image builds
#   2. Container starts and /healthz returns 200
#   3. /reset endpoint returns 200
#   4. /grade/task_spam, /grade/task_misinfo, /grade/task_cib return scores in [0.0, 1.0]
#   5. inference.py runs and emits [START]/[STEP]/[END] format
#   6. Stops and removes the test container
#
# Requirements: docker, curl, python3, jq
# Usage: bash scripts/pre_validate.sh

set -euo pipefail

IMAGE_NAME="socialguard-rl-test"
CONTAINER_NAME="socialguard-rl-validate"
PORT=7860
BASE_URL="http://localhost:${PORT}"

PASS=0
FAIL=0
ERRORS=()

_pass() { echo "  ✅  $1"; ((PASS++)) || true; }
_fail() { echo "  ❌  $1"; ((FAIL++)) || true; ERRORS+=("$1"); }
_info() { echo "  ℹ️   $1"; }

# ---------------------------------------------------------------------------
# Cleanup on exit
# ---------------------------------------------------------------------------
cleanup() {
  _info "Stopping container..."
  docker stop "${CONTAINER_NAME}" 2>/dev/null || true
  docker rm   "${CONTAINER_NAME}" 2>/dev/null || true
}
trap cleanup EXIT

echo ""
echo "=== SocialGuard-RL Pre-Submission Validator ==="
echo ""

# ---------------------------------------------------------------------------
# 1. Docker build
# ---------------------------------------------------------------------------
echo "[1/6] Building Docker image..."
if docker build -t "${IMAGE_NAME}" . --quiet; then
  _pass "Docker build succeeded"
else
  _fail "Docker build FAILED — fix Dockerfile before submitting"
  exit 1
fi

# ---------------------------------------------------------------------------
# 2. Start container
# ---------------------------------------------------------------------------
echo "[2/6] Starting container on port ${PORT}..."
docker run -d \
  --name "${CONTAINER_NAME}" \
  -p "${PORT}:${PORT}" \
  "${IMAGE_NAME}" > /dev/null

_info "Waiting 15 seconds for server to boot..."
sleep 15

# ---------------------------------------------------------------------------
# 3. Health check
# ---------------------------------------------------------------------------
echo "[3/6] Checking /healthz..."
STATUS=$(curl -s -o /dev/null -w "%{http_code}" "${BASE_URL}/healthz" || echo "000")
if [ "${STATUS}" = "200" ]; then
  _pass "/healthz returned 200"
else
  _fail "/healthz returned ${STATUS} (expected 200)"
fi

# ---------------------------------------------------------------------------
# 4. /reset smoke test
# ---------------------------------------------------------------------------
echo "[4/6] Testing /reset..."
RESET_STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
  -X POST "${BASE_URL}/reset" \
  -H "Content-Type: application/json" \
  -d '{"task":"task_spam","seed":42}' || echo "000")
if [ "${RESET_STATUS}" = "200" ]; then
  _pass "/reset returned 200"
else
  _fail "/reset returned ${RESET_STATUS} (expected 200)"
fi

# ---------------------------------------------------------------------------
# 5. /grade endpoints — scores must be in [0.0, 1.0]
# ---------------------------------------------------------------------------
echo "[5/6] Testing /grade endpoints..."
for TASK in task_spam task_misinfo task_cib; do
  RESPONSE=$(curl -s "${BASE_URL}/grade/${TASK}" || echo "{}")
  SCORE=$(echo "${RESPONSE}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('score','ERROR'))" 2>/dev/null || echo "ERROR")

  if python3 -c "s=float('${SCORE}'); assert 0.0 <= s <= 1.0" 2>/dev/null; then
    _pass "/grade/${TASK} score=${SCORE} (in [0.0, 1.0])"
  else
    _fail "/grade/${TASK} score=${SCORE} (expected float in [0.0, 1.0])"
  fi
done

# ---------------------------------------------------------------------------
# 6. inference.py stdout format
# ---------------------------------------------------------------------------
echo "[6/6] Validating inference.py stdout format..."
# Run inference locally (not in container) to avoid Docker overhead.
# Use safe placeholder env defaults so format validation works even without real credentials.
INF_API_BASE_URL="${API_BASE_URL:-http://127.0.0.1:9/v1}"
INF_MODEL_NAME="${MODEL_NAME:-baseline}"
INF_HF_TOKEN="${HF_TOKEN:-hf_dummy_for_format_check}"
INFERENCE_OUT=$(API_BASE_URL="${INF_API_BASE_URL}" MODEL_NAME="${INF_MODEL_NAME}" HF_TOKEN="${INF_HF_TOKEN}" \
  timeout 300 python3 inference.py 2>/dev/null | head -20 || echo "ERROR")

if echo "${INFERENCE_OUT}" | grep -q "^\[START\]"; then
  _pass "inference.py emits [START] lines"
else
  _fail "inference.py did not emit [START] — check output format"
fi

if echo "${INFERENCE_OUT}" | grep -q "^\[STEP\]"; then
  _pass "inference.py emits [STEP] lines"
else
  _fail "inference.py did not emit [STEP] lines"
fi

if API_BASE_URL="${INF_API_BASE_URL}" MODEL_NAME="${INF_MODEL_NAME}" HF_TOKEN="${INF_HF_TOKEN}" \
  timeout 300 python3 inference.py 2>/dev/null | grep -q "^\[END\]"; then
  _pass "inference.py emits [END] lines"
else
  _fail "inference.py did not emit [END] lines"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "=== Validation Summary ==="
echo "  Passed: ${PASS}"
echo "  Failed: ${FAIL}"
if [ ${FAIL} -gt 0 ]; then
  echo ""
  echo "  Failures:"
  for ERR in "${ERRORS[@]}"; do
    echo "    - ${ERR}"
  done
  echo ""
  echo "❌ Pre-validation FAILED — fix the issues above before submitting."
  exit 1
else
  echo ""
  echo "✅ All checks passed — ready for submission!"
  exit 0
fi
