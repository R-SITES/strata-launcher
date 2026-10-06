#!/bin/bash
# Start the Strata Launcher and open it. Probes the port rather than trusting a PID file.
PORT="${STRATA_LAUNCHER_PORT:-9877}"
DIR="$(cd "$(dirname "$0")" && pwd)"

if curl -s -o /dev/null --max-time 2 "http://127.0.0.1:$PORT/api/models"; then
    xdg-open "http://127.0.0.1:$PORT" &>/dev/null &
    exit 0
fi

cd "$DIR" || exit 1
nohup python3 server.py >> "$DIR/launcher.out" 2>&1 &
for _ in $(seq 1 20); do
    if curl -s -o /dev/null --max-time 2 "http://127.0.0.1:$PORT/api/models"; then
        xdg-open "http://127.0.0.1:$PORT" &>/dev/null &
        exit 0
    fi
    sleep 0.5
done
echo "launcher did not come up; see $DIR/launcher.out" >&2
exit 1
