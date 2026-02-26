#!/bin/bash

set -e

CONTAINER_NAME="ast_rag_qdrant"
VOLUMES=("qdrant_storage")

create_volumes() {
    for vol in "${VOLUMES[@]}"; do
        podman volume create "${vol}" 2>/dev/null || true
    done
}

start() {
    create_volumes

    podman run -d \
        --name "$CONTAINER_NAME" \
        --restart "unless-stopped" \
        -p 6333:6333 \
        -p 6334:6334 \
        -v qdrant_storage:/qdrant/storage:z \
        docker.io/qdrant/qdrant:latest
}

stop() {
    podman stop "$CONTAINER_NAME" 2>/dev/null || true
    podman rm "$CONTAINER_NAME" 2>/dev/null || true
}

down() {
    stop
    for vol in "${VOLUMES[@]}"; do
        podman volume rm "$vol" 2>/dev/null || true
    done
}

case "${1:-start}" in
    up|start)
        start
        ;;
    down)
        down
        ;;
    stop)
        stop
        ;;
    restart)
        stop
        start
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|down}"
        exit 1
        ;;
esac
