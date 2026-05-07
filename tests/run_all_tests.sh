#!/usr/bin/env bash
# tests/run_all_tests.sh - run all AST smoke tests sequentially.
set -u
cd "$(dirname "$0")/.."
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  if command -v python >/dev/null 2>&1; then PY=python
  elif command -v python3 >/dev/null 2>&1; then PY=python3
  else echo "No python interpreter found"; exit 2
  fi
fi
fail=0
for f in tests/test_*.py; do
  echo "== $f =="
  if ! "$PY" "$f"; then
    echo "FAIL: $f"
    fail=$((fail+1))
  fi
done
if [ "$fail" -ne 0 ]; then
  echo "TOTAL FAIL: $fail"
  exit 1
fi
echo "ALL TESTS PASS"
