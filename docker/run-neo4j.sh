#!/bin/bash

set -e

CONTAINER_NAME="ast_rag_neo4j"
VOLUMES=("neo4j_data" "neo4j_logs" "neo4j_import" "neo4j_plugins")

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
        -p 7474:7474 \
        -p 7687:7687 \
        -e "NEO4J_AUTH=neo4j/password" \
        -e 'NEO4J_PLUGINS=["apoc"]' \
        -e "NEO4J_dbms_memory_heap_initial__size=512m" \
        -e "NEO4J_dbms_memory_heap_max__size=2g" \
        -e "NEO4J_dbms_memory_pagecache_size=512m" \
        -v neo4j_data:/data \
        -v neo4j_logs:/logs \
        -v neo4j_import:/import \
        -v neo4j_plugins:/plugins \
        --healthcheck-interval 10s \
        --healthcheck-timeout 10s \
        --healthcheck-start-period 30s \
        --healthcheck-retries 10 \
        docker.io/neo4j:5.18-community
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
