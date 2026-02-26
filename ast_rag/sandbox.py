"""
sandbox.py - Docker-based sandbox for running language-specific test commands.

Provides run_in_sandbox(command, workdir, lang) which:
- Mounts the workdir into a Docker container
- Runs the given command with no network access
- Enforces memory and CPU limits
- Returns (stdout, stderr, exit_code)

NOTE: This is a PoC implementation. The Docker images used are standard ones
(maven, openjdk, gcc, rust, python, node). For production use you should build
custom images with only the tools you need, add seccomp/AppArmor profiles, etc.

The caller must have Docker running and the working directory accessible by Docker.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default test commands per language
# ---------------------------------------------------------------------------

DEFAULT_COMMANDS: dict[str, str] = {
    "java":       "mvn test -B --no-transfer-progress",
    "cpp":        "cmake --build /workspace/build && cd /workspace/build && ctest --output-on-failure",
    "rust":       "cargo test",
    "python":     "pytest --tb=short -q",
    "typescript": "npm test",
}

# Docker images to use per language
# NOTE: these are public Docker Hub images; pin versions in production.
_DOCKER_IMAGES: dict[str, str] = {
    "java":       "maven:3.9-eclipse-temurin-21",
    "cpp":        "gcc:13",
    "rust":       "rust:1.76",
    "python":     "python:3.12-slim",
    "typescript": "node:20-slim",
}

# Resource constraints
_MEM_LIMIT = "512m"    # maximum RAM per container
_CPU_PERIOD = 100_000  # microseconds
_CPU_QUOTA  = 50_000   # 0.5 CPU cores
_TIMEOUT    = 300      # seconds before we kill the container


def run_in_sandbox(
    command: str,
    workdir: str = ".",
    lang: str = "python",
    image: Optional[str] = None,
    timeout: int = _TIMEOUT,
    mem_limit: str = _MEM_LIMIT,
    cpu_period: int = _CPU_PERIOD,
    cpu_quota: int = _CPU_QUOTA,
) -> tuple[str, str, int]:
    """Run a shell command inside a Docker container and return its output.

    Args:
        command:    Shell command to run inside the container.
        workdir:    Host directory to mount as /workspace inside the container.
        lang:       Language key; used to select the Docker image if `image` is None.
        image:      Override Docker image name.
        timeout:    Seconds before the container is killed.
        mem_limit:  Docker memory limit string (e.g. "512m").
        cpu_period: Docker CPU period in microseconds.
        cpu_quota:  Docker CPU quota in microseconds (0.5 * cpu_period = 50% of 1 core).

    Returns:
        (stdout, stderr, exit_code) tuple.
    """
    import docker  # imported here so missing docker SDK doesn't break other imports
    import docker.errors

    image = image or _DOCKER_IMAGES.get(lang)
    if not image:
        raise ValueError(f"No Docker image configured for language: {lang!r}")

    abs_workdir = os.path.abspath(workdir)
    if not os.path.isdir(abs_workdir):
        raise FileNotFoundError(f"workdir does not exist: {abs_workdir}")

    client = docker.from_env()

    logger.info("Sandbox: lang=%s image=%s command=%r workdir=%s", lang, image, command, abs_workdir)

    try:
        container = client.containers.run(
            image=image,
            command=["/bin/sh", "-c", command],
            volumes={abs_workdir: {"bind": "/workspace", "mode": "rw"}},
            working_dir="/workspace",
            # Security: disable network
            network_disabled=True,
            network_mode="none",
            # Resource limits
            mem_limit=mem_limit,
            cpu_period=cpu_period,
            cpu_quota=cpu_quota,
            # Don't keep the container after it exits
            remove=True,
            # Capture all output
            stdout=True,
            stderr=True,
            # Run synchronously
            detach=False,
            # Drop all capabilities, run as non-root if possible
            security_opt=["no-new-privileges"],
            # Prevent fork bombs
            pids_limit=256,
        )
        # When detach=False and stdout/stderr=True, .run() returns the combined log bytes
        stdout_bytes = container if isinstance(container, bytes) else b""
        return stdout_bytes.decode("utf-8", errors="replace"), "", 0

    except docker.errors.ContainerError as exc:
        # Non-zero exit code
        stdout_text = exc.container.logs(stdout=True, stderr=False).decode("utf-8", errors="replace") \
            if exc.container else ""
        stderr_text = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else str(exc)
        return stdout_text, stderr_text, exc.exit_status

    except docker.errors.ImageNotFound:
        error = f"Docker image not found: {image}. Run `docker pull {image}` first."
        logger.error(error)
        return "", error, 127

    except docker.errors.APIError as exc:
        error = f"Docker API error: {exc}"
        logger.error(error)
        return "", error, 1

    except Exception as exc:
        error = f"Unexpected sandbox error: {exc}"
        logger.error(error)
        return "", error, 1


def run_tests(lang: str, workdir: str, command: Optional[str] = None) -> tuple[str, str, int]:
    """Convenience wrapper: run the default test suite for the given language.

    Args:
        lang:     Language identifier.
        workdir:  Directory containing the project to test.
        command:  Override default test command.

    Returns:
        (stdout, stderr, exit_code)
    """
    cmd = command or DEFAULT_COMMANDS.get(lang)
    if not cmd:
        raise ValueError(f"Unknown language for sandbox: {lang!r}")
    return run_in_sandbox(cmd, workdir=workdir, lang=lang)
