#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

SELF=$$
PATTERNS=("python .*src/pipeline.py" "python .*run_model_grid.py")
PIDS=""

for pat in "${PATTERNS[@]}"; do
  PIDS+=" $(pgrep -u "$USER" -f -e "$pat" || true)"
done
PIDS=$(echo "$PIDS" | xargs -n1 | sort -u | grep -v -e "^$SELF$" \
                             -e "jupyter-lab" -e "ipykernel" || true)

if [[ -z "$PIDS" ]]; then
  echo "No grid workers found."
  exit 0
fi

echo "Killing PIDs: $PIDS"
kill $PIDS
sleep 2
kill -9 $PIDS 2>/dev/null || true
echo "Done."
