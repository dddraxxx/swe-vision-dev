"""
Agentic VLM Tool Call Framework (Docker Runtime)
==================================================
This framework provides a VLM (Vision Language Model) agent that can:
1. Accept user queries with optional images
2. Use a stateful Jupyter notebook tool running inside a Docker container
3. Iteratively reason and call tools until it arrives at a final answer
4. Return structured results via a `finish` tool call

Architecture:
- JupyterNotebookKernel: manages a Jupyter kernel inside a Docker container
- OpenAI function calling: defines `execute_code` and `finish` tools
- Agent loop: sends messages to GPT, dispatches tool calls, feeds results back

The Docker container is built from env/Dockerfile and provides an isolated
Python environment with pre-installed image processing packages.

Usage:
    python sample_vlm_toolcall_docker.py --image path/to/image.png "Describe what's in this image"

    # Or without an image:
    python sample_vlm_toolcall_docker.py "What is 2^100? Use python to compute it."

    # Interactive mode:
    python sample_vlm_toolcall_docker.py --interactive

Environment Variables:
    OPENAI_API_KEY      - Your OpenAI API key
    OPENAI_BASE_URL     - (Optional) Custom API base URL
    OPENAI_MODEL        - (Optional) Model name, default: gpt-4o

Dependencies (host-side):
    pip install openai jupyter_client docker
"""

import argparse
import asyncio
import base64
import copy
import datetime
import io
import json
import logging
import mimetypes
import os
import queue
import re
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from openai import OpenAI

# ─────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("vlm_agent")

# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
MAX_ITERATIONS = 100  # safety limit for the agentic loop
CELL_TIMEOUT = 120.0  # max seconds per code cell execution
MAX_OUTPUT_CHARS = 50000  # truncate very long outputs

# Container-side working directory (visible to the kernel)
CONTAINER_WORK_DIR = "/mnt/data"

# Host-side directory that is volume-mounted into the container.
# Each runtime gets its own subdirectory named with a timestamp to avoid conflicts.
_HOST_WORK_BASE = os.environ.get(
    "VLM_HOST_WORK_DIR",
    os.path.join(os.path.expanduser("~"), "tmp", "vlm_docker_workdir"),
)
HOST_WORK_DIR = os.path.join(
    _HOST_WORK_BASE,
    datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
)

# Docker image / build settings
DOCKER_IMAGE_NAME = os.environ.get("VLM_DOCKER_IMAGE", "vlm-jupyter-kernel:latest")
DOCKERFILE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "env")

# Pre-assigned ZMQ ports for the Jupyter kernel inside the container
_KERNEL_BASE_PORT = 55555
KERNEL_PORTS = {
    "shell_port":   _KERNEL_BASE_PORT,
    "iopub_port":   _KERNEL_BASE_PORT + 1,
    "stdin_port":   _KERNEL_BASE_PORT + 2,
    "control_port": _KERNEL_BASE_PORT + 3,
    "hb_port":      _KERNEL_BASE_PORT + 4,
}

# ─────────────────────────────────────────────────────────────────────
# Tool Definitions (OpenAI function calling format)
# ─────────────────────────────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": (
                "Execute Python code in a **stateful** Jupyter notebook environment. "
                "The kernel persists across calls, so variables, imports, and state are retained. "
                "Use this to process images, perform calculations, create visualizations, "
                "or run any Python code. "
                "Any images generated (e.g. via matplotlib plt.show() or PIL Image.save()) "
                "will be captured and returned as base64-encoded images."
                "Print statements and expression results are captured as text output. "
                "All uploaded files are available under /mnt/data/. "
                "The kernel's working directory is /mnt/data/."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "The Python code to execute. The code runs in a Jupyter kernel "
                            "so you can use magics, display(), etc. "
                            "Use print() for text output. "
                            "Images from matplotlib will be auto-captured."
                        ),
                    },
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": (
                "Call this tool when you have determined the final answer. "
                "This ends the agentic workflow and returns the answer to the user."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The final answer to the user's question.",
                    },
                },
                "required": ["answer"],
            },
        },
    },
]

# ─────────────────────────────────────────────────────────────────────
# System Prompt
# ─────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an expert AI assistant with access to a **stateful Jupyter notebook** environment. \
You can execute Python code to help answer the user's questions.

## Available Tools

1. **execute_code**: Run Python code in a persistent Jupyter notebook. The kernel state \
(variables, imports, loaded data) is preserved between calls. Use this for:
   - Image processing and analysis (PIL/Pillow, OpenCV, skimage, etc.)
   - Data analysis and computation (numpy, pandas, scipy, etc.)
   - Visualization (matplotlib, seaborn, plotly, etc.)
   - Any Python computation

2. **finish**: Call this when you have the final answer. This ends the workflow.

## File System

- All uploaded files (images, data files, etc.) are placed in `/mnt/data/`.
- The Jupyter kernel's working directory is `/mnt/data/`, so you can reference files \
by their filename directly (e.g. `open('image.png')`) or by absolute path \
(e.g. `open('/mnt/data/image.png')`).
- Any files you create or save will also go into `/mnt/data/`.

## Guidelines

- When given an image, you can load it in the notebook using PIL or OpenCV. \
The image file will be available at `/mnt/data/<filename>`.
- You can call execute_code **multiple times** to iteratively explore and process data.
- Always use print() to output results you want to see.
- When you generate plots with matplotlib, use plt.show() — the plot image will be \
captured and returned to you.
- Think step by step. Examine intermediate results before giving a final answer.
- When you're confident in your answer, call the **finish** tool with your final response.
- If code produces an error, analyze the error and try a different approach.
"""


# ═════════════════════════════════════════════════════════════════════
# Docker Jupyter Notebook Kernel Manager
# ═════════════════════════════════════════════════════════════════════
class JupyterNotebookKernel:
    """
    Manages a persistent Jupyter kernel running inside a Docker container.

    The container is built from env/Dockerfile and runs an IPython kernel
    with pre-assigned ZMQ ports.  The host connects to it via
    jupyter_client using the same ports (forwarded from the container).

    A host directory is volume-mounted to /mnt/data inside the container
    so files can be shared between host and kernel.
    """

    def __init__(
        self,
        timeout: float = CELL_TIMEOUT,
        host_work_dir: str = HOST_WORK_DIR,
        container_work_dir: str = CONTAINER_WORK_DIR,
        docker_image: str = DOCKER_IMAGE_NAME,
        dockerfile_dir: str = DOCKERFILE_DIR,
    ):
        self._timeout = timeout
        self._host_work_dir = host_work_dir
        self._container_work_dir = container_work_dir
        self._docker_image = docker_image
        self._dockerfile_dir = dockerfile_dir

        self._container = None  # docker.models.containers.Container
        self._docker_client = None  # docker.DockerClient
        self._kc = None  # jupyter_client.BlockingKernelClient
        self._started = False

        # Unique key for ZMQ HMAC signing
        self._kernel_key = uuid.uuid4().hex

    @property
    def host_work_dir(self) -> str:
        """Return the host-side mount directory (files go here)."""
        return self._host_work_dir

    @property
    def container_work_dir(self) -> str:
        """Return the container-side working directory (/mnt/data)."""
        return self._container_work_dir

    # ── Docker helpers ─────────────────────────────────────────────

    def _build_image(self):
        """Build the Docker image from env/Dockerfile if it doesn't exist."""
        import docker

        if self._docker_client is None:
            self._docker_client = docker.from_env()

        # Check if image already exists
        try:
            self._docker_client.images.get(self._docker_image)
            logger.info("Docker image '%s' already exists, skipping build.", self._docker_image)
            return
        except docker.errors.ImageNotFound:
            pass

        logger.info(
            "Building Docker image '%s' from %s ...",
            self._docker_image, self._dockerfile_dir,
        )
        image, build_logs = self._docker_client.images.build(
            path=self._dockerfile_dir,
            tag=self._docker_image,
            rm=True,
        )
        for chunk in build_logs:
            if "stream" in chunk:
                line = chunk["stream"].rstrip()
                if line:
                    logger.debug("  [docker build] %s", line)
        logger.info("Docker image '%s' built successfully.", self._docker_image)

    def _write_connection_file(self) -> str:
        """
        Write a Jupyter kernel connection JSON into the mounted volume.
        Returns the container-side path of the connection file.
        """
        conn = {
            "shell_port":   KERNEL_PORTS["shell_port"],
            "iopub_port":   KERNEL_PORTS["iopub_port"],
            "stdin_port":   KERNEL_PORTS["stdin_port"],
            "control_port": KERNEL_PORTS["control_port"],
            "hb_port":      KERNEL_PORTS["hb_port"],
            "ip": "0.0.0.0",
            "key": self._kernel_key,
            "transport": "tcp",
            "signature_scheme": "hmac-sha256",
            "kernel_name": "python3",
        }
        host_path = os.path.join(self._host_work_dir, ".kernel_connection.json")
        with open(host_path, "w") as f:
            json.dump(conn, f)
        logger.info("Wrote kernel connection file to %s", host_path)
        return os.path.join(self._container_work_dir, ".kernel_connection.json")

    def _start_container(self):
        """Start a Docker container with port mappings and volume mount."""
        import docker

        if self._docker_client is None:
            self._docker_client = docker.from_env()

        # Port mappings: host_port -> container_port  (all TCP)
        port_bindings = {
            f"{p}/tcp": ("127.0.0.1", p) for p in KERNEL_PORTS.values()
        }
        ports_exposure = {f"{p}/tcp": None for p in KERNEL_PORTS.values()}

        container_name = f"vlm-jupyter-{uuid.uuid4().hex[:8]}"

        logger.info(
            "Starting Docker container '%s' (image=%s, mount=%s -> %s) ...",
            container_name, self._docker_image,
            self._host_work_dir, self._container_work_dir,
        )

        self._container = self._docker_client.containers.run(
            image=self._docker_image,
            name=container_name,
            command="sleep infinity",  # keep container alive
            ports=port_bindings,
            volumes={
                self._host_work_dir: {
                    "bind": self._container_work_dir,
                    "mode": "rw",
                },
            },
            working_dir=self._container_work_dir,
            detach=True,
            remove=False,
        )
        logger.info("Container '%s' started (id=%s).", container_name, self._container.short_id)

    def _start_kernel_in_container(self, connection_file: str):
        """
        Start ipykernel inside the container (detached) using the
        pre-written connection file.
        """
        cmd = (
            f"python -m ipykernel_launcher -f {connection_file} "
            f"--IPKernelApp.matplotlib='inline'"
        )
        logger.info("Starting kernel inside container: %s", cmd)
        self._container.exec_run(
            cmd=["bash", "-c", cmd],
            detach=True,
            workdir=self._container_work_dir,
        )
        # Give the kernel a moment to start
        time.sleep(2)

    def _connect_client(self):
        """
        Create a jupyter_client.BlockingKernelClient that talks to
        the kernel inside the container via the forwarded ports.
        """
        from jupyter_client import BlockingKernelClient

        self._kc = BlockingKernelClient()
        self._kc.ip = "127.0.0.1"
        self._kc.shell_port = KERNEL_PORTS["shell_port"]
        self._kc.iopub_port = KERNEL_PORTS["iopub_port"]
        self._kc.stdin_port = KERNEL_PORTS["stdin_port"]
        self._kc.control_port = KERNEL_PORTS["control_port"]
        self._kc.hb_port = KERNEL_PORTS["hb_port"]
        self._kc.session.key = self._kernel_key.encode("utf-8")
        self._kc.start_channels()
        logger.info("Kernel client connected to 127.0.0.1 ports %s", list(KERNEL_PORTS.values()))

    # ── Public API ─────────────────────────────────────────────────

    async def start(self):
        """Build image, start container, launch kernel, connect client."""
        if self._started:
            return

        # Ensure the host-side work directory exists
        os.makedirs(self._host_work_dir, exist_ok=True)

        # 1) Build the Docker image (no-op if already exists)
        self._build_image()

        # 2) Write the kernel connection file into the mounted volume
        conn_file = self._write_connection_file()

        # 3) Start the container
        self._start_container()

        # 4) Start the kernel inside the container
        self._start_kernel_in_container(conn_file)

        # 5) Connect from the host
        self._connect_client()

        # 6) Wait for kernel to be ready
        try:
            self._kc.wait_for_ready(timeout=self._timeout)
        except RuntimeError:
            logger.warning("Kernel wait_for_ready timed out, retrying after 3s...")
            time.sleep(3)
            self._kc.wait_for_ready(timeout=self._timeout)

        # 7) Verify kernel works
        test_result = await self._execute_raw("print('kernel_ready')")
        if "kernel_ready" not in test_result.get("stdout", ""):
            raise RuntimeError("Docker Jupyter kernel failed health check")

        # 8) Configure inline backend to use PNG format
        await self._execute_raw("%config InlineBackend.figure_format = 'png'")

        self._started = True
        logger.info(
            "Docker Jupyter kernel started successfully "
            "(container=%s, work_dir=%s).",
            self._container.short_id, self._container_work_dir,
        )

    async def _execute_raw(self, code: str) -> Dict[str, Any]:
        """
        Execute code and collect raw results from the kernel.
        Returns a dict with stdout, stderr, display (images), error info.

        Uses BlockingKernelClient (synchronous) wrapped in asyncio
        so the rest of the agent can remain async.
        """
        cell_result: Dict[str, Any] = {
            "stdout": "",
            "stderr": "",
            "display": [],   # list of dicts like {"image/png": "<base64>", ...}
            "error": [],
            "status": "ok",
        }

        def _sync_execute():
            msg_id = self._kc.execute(code)
            # Collect iopub messages until we see an "execute_reply" on shell
            deadline = time.time() + self._timeout
            reply_received = False
            while time.time() < deadline:
                try:
                    msg = self._kc.get_iopub_msg(timeout=1)
                except queue.Empty:
                    if reply_received:
                        break
                    continue

                msg_type = msg["header"]["msg_type"]
                parent_id = msg.get("parent_header", {}).get("msg_id")
                if parent_id != msg_id:
                    continue  # not our message

                content = msg["content"]

                if msg_type == "stream":
                    cell_result[content["name"]] += content["text"]
                elif msg_type in ("display_data", "execute_result"):
                    cell_result["display"].append(content["data"])
                elif msg_type == "error":
                    cell_result["error"].append(content)
                elif msg_type == "status" and content.get("execution_state") == "idle":
                    reply_received = True

            # Get the execute_reply from the shell channel
            try:
                reply = self._kc.get_shell_msg(timeout=self._timeout)
                cell_result["status"] = reply["content"]["status"]
            except queue.Empty:
                cell_result["status"] = "error"

        await asyncio.get_event_loop().run_in_executor(None, _sync_execute)
        return cell_result

    async def execute(self, code: str) -> Dict[str, Any]:
        """
        Execute code in the kernel and return structured results.

        Returns:
            {
                "text_output": str,      # combined stdout + error text
                "images": [str, ...],    # list of base64-encoded PNG images
                "status": "ok" | "error",
                "error_traceback": str | None,
            }
        """
        if not self._started:
            await self.start()

        raw = await self._execute_raw(code)

        # Collect text output
        text_parts = []
        if raw["stdout"]:
            text_parts.append(raw["stdout"])
        if raw["stderr"]:
            text_parts.append(f"[STDERR] {raw['stderr']}")
        if raw["error"]:
            for err in raw["error"]:
                tb_text = "\n".join(err.get("traceback", []))
                # Strip ANSI escape codes
                tb_text = re.sub(r"\x1b\[[0-9;]*m", "", tb_text)
                text_parts.append(f"[ERROR] {tb_text}")

        # Collect display data (text representations)
        for display_item in raw["display"]:
            if "text/plain" in display_item and "image/png" not in display_item:
                text_parts.append(display_item["text/plain"])

        # Collect images
        images = []
        for display_item in raw["display"]:
            if "image/png" in display_item:
                images.append(display_item["image/png"])
            elif "image/jpeg" in display_item:
                images.append(display_item["image/jpeg"])

        text_output = "\n".join(text_parts).strip()
        if not text_output and not images:
            text_output = "[No output produced. Use print() to see results.]"

        # Truncate very long output
        if len(text_output) > MAX_OUTPUT_CHARS:
            text_output = text_output[:MAX_OUTPUT_CHARS] + "\n... [output truncated]"

        error_tb = None
        if raw["error"]:
            error_tb = "\n".join(
                "\n".join(e.get("traceback", [])) for e in raw["error"]
            )
            error_tb = re.sub(r"\x1b\[[0-9;]*m", "", error_tb)

        return {
            "text_output": text_output,
            "images": images,
            "status": raw["status"],
            "error_traceback": error_tb,
        }

    async def shutdown(self, cleanup_work_dir: bool = False):
        """Stop and remove the Docker container; optionally clean up host mount."""
        # Close kernel client channels
        if self._kc is not None:
            try:
                self._kc.stop_channels()
            except Exception as e:
                logger.warning("Failed to stop kernel client channels: %s", e)
            self._kc = None

        # Stop and remove the container
        if self._container is not None:
            try:
                logger.info(
                    "Stopping Docker container '%s' ...", self._container.short_id
                )
                self._container.stop(timeout=5)
                self._container.remove(force=True)
                logger.info("Container removed.")
            except Exception as e:
                logger.warning("Failed to stop/remove container: %s", e)
            self._container = None

        self._started = False

        if cleanup_work_dir and os.path.isdir(self._host_work_dir):
            try:
                shutil.rmtree(self._host_work_dir)
                logger.info("Cleaned up host work directory: %s", self._host_work_dir)
            except Exception as e:
                logger.warning("Failed to clean up work directory %s: %s", self._host_work_dir, e)


# ═════════════════════════════════════════════════════════════════════
# Image Helper Utilities
# ═════════════════════════════════════════════════════════════════════

from utils import image_file_to_base64

def guess_mime_type(file_path: str) -> str:
    """Guess MIME type from file extension."""
    mime, _ = mimetypes.guess_type(file_path)
    return mime or "image/png"


def make_image_content_part(file_path: str) -> Dict[str, Any]:
    """Create an OpenAI image_url content part from a local file."""
    b64 = image_file_to_base64(file_path)
    return {
        "type": "image_url",
        "image_url": {
            "url": b64,
        },
    }


def make_base64_image_content_part(b64_data: str, mime: str = "image/png") -> Dict[str, Any]:
    """Create an OpenAI image_url content part from base64 data."""
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{mime};base64,{b64_data}",
        },
    }


# ═════════════════════════════════════════════════════════════════════
# File Management for Notebook
# ═════════════════════════════════════════════════════════════════════
@dataclass
class NotebookFileManager:
    """
    Manages files that should be accessible in the Jupyter kernel's
    Docker container.

    Files are copied to the **host-side** mount directory, which is
    volume-mounted into the container at /mnt/data.  The model is told
    to reference files using the container-side path (/mnt/data/<name>).
    """
    host_work_dir: str = HOST_WORK_DIR
    container_work_dir: str = CONTAINER_WORK_DIR

    def setup_work_dir(
        self,
        host_work_dir: Optional[str] = None,
        container_work_dir: Optional[str] = None,
    ):
        """
        Set up the working directories.
        Creates the host directory if it doesn't exist.
        """
        if host_work_dir:
            self.host_work_dir = host_work_dir
        if container_work_dir:
            self.container_work_dir = container_work_dir
        os.makedirs(self.host_work_dir, exist_ok=True)
        logger.info(
            "NotebookFileManager: host_work_dir=%s, container_work_dir=%s",
            self.host_work_dir, self.container_work_dir,
        )

    def copy_file_to_workdir(self, src_path: str, dest_name: Optional[str] = None) -> str:
        """
        Copy a file into the host mount directory so the container kernel
        can access it at /mnt/data/<filename>.
        Returns the **container-side** path for use in prompts / hints.
        """
        if dest_name is None:
            dest_name = os.path.basename(src_path)
        os.makedirs(self.host_work_dir, exist_ok=True)
        host_dest = os.path.join(self.host_work_dir, dest_name)
        if os.path.abspath(src_path) != os.path.abspath(host_dest):
            shutil.copy2(src_path, host_dest)
            logger.info("Copied %s -> %s (container: %s)", src_path, host_dest,
                        os.path.join(self.container_work_dir, dest_name))
        # Return the container-side path so prompts reference the right location
        return os.path.join(self.container_work_dir, dest_name)

    def get_kernel_path(self, filename: str) -> str:
        """Return the full path a file would have inside the container."""
        return os.path.join(self.container_work_dir, filename)


# ═════════════════════════════════════════════════════════════════════
# Trajectory Recorder — saves the full agent trace to disk
# ═════════════════════════════════════════════════════════════════════
class TrajectoryRecorder:
    """
    Records every step of the agentic loop and persists it to a local
    directory.  The output structure looks like:

        <save_dir>/
        ├── trajectory.json          # full structured trajectory
        ├── messages_raw.json        # raw OpenAI messages list (with base64 replaced by file refs)
        └── images/
            ├── user_input_0.png     # images the user uploaded
            ├── step_2_tool_0.png    # images generated by notebook at step 2
            └── ...

    `trajectory.json` contains:
    {
        "metadata": { model, start_time, end_time, total_iterations, query, image_paths, ... },
        "steps": [
            {
                "step":        int,
                "role":        "user" | "assistant" | "tool",
                "timestamp":   ISO-8601 string,
                "content_text": str | null,           # text portion
                "tool_calls":  [ { name, arguments } ] | null,
                "tool_call_id": str | null,
                "code":        str | null,             # code that was executed (for tool steps)
                "images":      [ "images/xxx.png" ],   # paths relative to save_dir
            },
            ...
        ],
        "final_answer": str | null,
    }
    """

    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        self.image_dir = os.path.join(save_dir, "images")
        os.makedirs(self.image_dir, exist_ok=True)

        self.steps: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}
        self.final_answer: Optional[str] = None
        self._image_counter = 0

    # ── helpers ─────────────────────────────────────────────────────

    def _next_image_name(self, prefix: str, ext: str = "png") -> str:
        self._image_counter += 1
        return f"{prefix}_{self._image_counter}.{ext}"

    def _save_base64_image(self, b64_data: str, prefix: str) -> str:
        """Decode base64 image, save to disk, return path relative to save_dir."""
        fname = self._next_image_name(prefix)
        fpath = os.path.join(self.image_dir, fname)
        with open(fpath, "wb") as f:
            f.write(base64.b64decode(b64_data))
        return os.path.join("images", fname)

    def _save_image_file(self, src_path: str, prefix: str) -> str:
        """Copy an image file into the trajectory images dir."""
        ext = Path(src_path).suffix.lstrip(".") or "png"
        fname = self._next_image_name(prefix, ext)
        dst = os.path.join(self.image_dir, fname)
        shutil.copy2(src_path, dst)
        return os.path.join("images", fname)

    @staticmethod
    def _now_iso() -> str:
        return datetime.datetime.now().isoformat(timespec="milliseconds")

    # ── public recording API ───────────────────────────────────────

    def set_metadata(self, **kwargs):
        self.metadata.update(kwargs)

    def record_step(
        self,
        *,
        role: str,
        content_text: Optional[str] = None,
        reasoning_details: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        tool_call_id: Optional[str] = None,
        code: Optional[str] = None,
        images: Optional[List[str]] = None,
    ):
        """Append one step to the trajectory.  `images` are *relative* paths."""
        step = {
            "step": len(self.steps),
            "role": role,
            "timestamp": self._now_iso(),
            "content_text": content_text,
            "tool_calls": tool_calls,
            "tool_call_id": tool_call_id,
            "code": code,
            "images": images or [],
        }
        if reasoning_details:
            step["reasoning_details"] = reasoning_details
        self.steps.append(step)

    def record_user_step(
        self,
        query: str,
        image_paths: Optional[List[str]] = None,
    ):
        """Record the initial user message (text + optional images)."""
        saved_images = []
        for p in (image_paths or []):
            if os.path.exists(p):
                saved_images.append(self._save_image_file(p, "user_input"))
        self.record_step(role="user", content_text=query, images=saved_images)

    def record_assistant_step(
        self,
        content_text: Optional[str],
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        reasoning_details: Optional[str] = None,
    ):
        """Record an assistant message (text + optional tool calls + optional reasoning)."""
        # Simplify tool_calls for readability
        simplified = None
        if tool_calls:
            simplified = []
            for tc in tool_calls:
                fn = tc.get("function", tc)
                simplified.append({
                    "id": tc.get("id"),
                    "name": fn.get("name"),
                    "arguments": fn.get("arguments"),
                })
        self.record_step(
            role="assistant",
            content_text=content_text,
            reasoning_details=reasoning_details,
            tool_calls=simplified,
        )

    def record_tool_step(
        self,
        tool_call_id: str,
        tool_name: str,
        code: Optional[str],
        text_output: str,
        base64_images: Optional[List[str]] = None,
    ):
        """Record a tool execution result (code + output + images)."""
        saved_images = []
        step_idx = len(self.steps)
        for img_b64 in (base64_images or []):
            rel = self._save_base64_image(img_b64, f"step_{step_idx}_tool")
            saved_images.append(rel)
        self.record_step(
            role="tool",
            content_text=text_output,
            tool_call_id=tool_call_id,
            code=code,
            images=saved_images,
        )

    def record_finish(self, answer: str):
        self.final_answer = answer

    # ── persistence ────────────────────────────────────────────────

    def save(self):
        """Write trajectory.json and messages_raw.json to save_dir."""
        self.metadata.setdefault("end_time", self._now_iso())
        self.metadata.setdefault("total_steps", len(self.steps))

        traj = {
            "metadata": self.metadata,
            "steps": self.steps,
            "final_answer": self.final_answer,
        }

        traj_path = os.path.join(self.save_dir, "trajectory.json")
        with open(traj_path, "w", encoding="utf-8") as f:
            json.dump(traj, f, ensure_ascii=False, indent=2)

        logger.info("Trajectory saved to %s (%d steps, %d images)",
                     self.save_dir, len(self.steps), self._image_counter)

    def save_messages_raw(self, messages: List[Dict[str, Any]]):
        """
        Save the raw OpenAI messages list, but replace inline base64 image
        data with file references so the JSON stays human-readable.
        """
        sanitized = _sanitize_messages_for_save(messages, self.image_dir, self.save_dir)
        raw_path = os.path.join(self.save_dir, "messages_raw.json")
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(sanitized, f, ensure_ascii=False, indent=2)


def _sanitize_messages_for_save(
    messages: List[Dict[str, Any]],
    image_dir: str,
    save_dir: str,
) -> List[Dict[str, Any]]:
    """
    Deep-copy messages and replace base64 data URIs with saved file paths.
    This keeps the JSON readable.
    """
    counter = [0]

    def _replace_b64(obj):
        if isinstance(obj, dict):
            # Handle image_url with data URI
            if obj.get("type") == "image_url":
                url = obj.get("image_url", {}).get("url", "")
                if url.startswith("data:"):
                    # Extract base64 payload
                    match = re.match(r"data:([^;]+);base64,(.*)", url, re.DOTALL)
                    if match:
                        mime, b64 = match.group(1), match.group(2)
                        ext = mime.split("/")[-1]
                        if ext == "jpeg":
                            ext = "jpg"
                        counter[0] += 1
                        fname = f"msg_image_{counter[0]}.{ext}"
                        fpath = os.path.join(image_dir, fname)
                        if not os.path.exists(fpath):
                            with open(fpath, "wb") as fp:
                                fp.write(base64.b64decode(b64))
                        return {
                            "type": "image_url",
                            "image_url": {"url": os.path.join("images", fname)},
                        }
            return {k: _replace_b64(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_replace_b64(item) for item in obj]
        return obj

    return _replace_b64(copy.deepcopy(messages))


# ═════════════════════════════════════════════════════════════════════
# VLM Agent
# ═════════════════════════════════════════════════════════════════════
class VLMToolCallAgent:
    """
    An agentic VLM framework that uses OpenAI's function calling to
    give a vision-language model access to a stateful Jupyter notebook
    running inside a Docker container.

    The agent loop:
    1. Send user message (with optional images) to the VLM
    2. If the model calls `execute_code`, run the code in the Docker kernel
    3. Feed results (text + images) back to the model
    4. Repeat until the model calls `finish` or max iterations reached
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        system_prompt: str = SYSTEM_PROMPT,
        max_iterations: int = MAX_ITERATIONS,
        verbose: bool = True,
        save_trajectory: Optional[str] = None,
        reasoning: bool = True,
    ):
        self.model = model
        self.max_iterations = max_iterations
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.reasoning = reasoning

        # Trajectory save directory (None = auto-generate under ./trajectories/)
        self._save_trajectory_dir = save_trajectory

        # Initialize OpenAI client
        client_kwargs = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url
        elif os.environ.get("OPENAI_BASE_URL"):
            client_kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]

        self.client = OpenAI(**client_kwargs)

        print(f"Using model: {self.model}")
        print(f"Using API key: {api_key}")
        print(f"Using base URL: {base_url}")

        # Jupyter kernel (lazy init — runs inside Docker)
        self.kernel: Optional[JupyterNotebookKernel] = None
        self.file_manager = NotebookFileManager()

        # Conversation history
        self.messages: List[Dict[str, Any]] = []

        # Trajectory recorder (created per run)
        self.trajectory: Optional[TrajectoryRecorder] = None

    async def _ensure_kernel(self):
        """Ensure the Docker Jupyter kernel is running."""
        if self.kernel is None:
            self.kernel = JupyterNotebookKernel()
        if not self.kernel._started:
            await self.kernel.start()
            # Point the file manager at the host mount directory
            self.file_manager.setup_work_dir(
                host_work_dir=self.kernel.host_work_dir,
                container_work_dir=self.kernel.container_work_dir,
            )

    def _log(self, msg: str, *args, level: str = "info"):
        """Log with optional verbose printing."""
        getattr(logger, level)(msg, *args)
        if self.verbose:
            formatted = msg % args if args else msg
            print(f"  [{level.upper()}] {formatted}", flush=True)

    def _build_user_message(
        self,
        query: str,
        image_paths: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Build a user message with text and optional images."""
        content = []

        # Build text — append file-path hints so the model knows where to find them
        file_hints = []
        if image_paths:
            # Detect basename collisions so we can prefix an index when needed
            basenames = [os.path.basename(os.path.abspath(p)) for p in image_paths]
            has_collision = len(basenames) != len(set(basenames))

            for idx, img_path in enumerate(image_paths):
                img_path = os.path.abspath(img_path)
                if not os.path.exists(img_path):
                    self._log("Warning: image not found: %s", img_path, level="warning")
                    continue
                content.append(make_image_content_part(img_path))
                # Use an indexed dest name when basenames collide or
                # there are multiple images (safe default)
                dest_name = None
                if has_collision or len(image_paths) > 1:
                    base = os.path.basename(img_path)
                    name, ext = os.path.splitext(base)
                    dest_name = f"{idx}_{name}{ext}"
                # Copy to the host mount dir (appears at /mnt/data/ inside container)
                container_path = self.file_manager.copy_file_to_workdir(
                    img_path, dest_name=dest_name,
                )
                file_hints.append(container_path)

        # Main text
        text = query
        if file_hints:
            paths_str = ", ".join(f"`{p}`" for p in file_hints)
            text += f"\n\n[Uploaded file(s) available at: {paths_str}]"
        content.insert(0, {"type": "text", "text": text})

        return {"role": "user", "content": content}

    def _call_llm(self) -> Any:
        """Call the OpenAI API with current messages and tools."""
        kwargs = dict(
            model=self.model,
            messages=self.messages,
            tools=TOOLS,
            tool_choice="auto",
            
        )
        if self.reasoning:
            kwargs["extra_body"] = {"reasoning": {"enabled": True, 'effort': 'xhigh'}}
            kwargs["reasoning_effort"] = 'xhigh'
        else:
            kwargs["extra_body"] = {"reasoning": {"enabled": False, 'effort': 'minimal'}}
        
        
        response = self.client.chat.completions.create(**kwargs)

        return response

    async def _handle_execute_code(self, code: str) -> Dict[str, Any]:
        """
        Execute code in the Docker Jupyter kernel and return structured results.

        Returns a dict with:
            - "text_output": str        — text for the tool message (string content)
            - "image_parts": list       — image_url content parts for a follow-up user message
            - "base64_images": list     — raw base64 strings (for trajectory recording)
        """
        await self._ensure_kernel()

        self._log("Executing code in Docker Jupyter notebook:\n%s", code[:200] + ("..." if len(code) > 200 else ""))

        result = await self.kernel.execute(code)

        # Text output (goes into the tool message as a string)
        text = result["text_output"]
        if result["status"] == "error":
            text = f"[Execution Error]\n{text}"

        # Image parts (will be injected via a follow-up user message)
        image_parts = []
        for img_b64 in result["images"]:
            image_parts.append(make_base64_image_content_part(img_b64))

        return {
            "text_output": text,
            "image_parts": image_parts,
            "base64_images": result["images"],
        }

    def _init_trajectory(self, query: str, image_paths: Optional[List[str]]) -> TrajectoryRecorder:
        """Create a TrajectoryRecorder for this run."""
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if self._save_trajectory_dir:
            save_dir = f"{self._save_trajectory_dir}_{ts}"
        else:
            save_dir = os.path.join("trajectories", f"run_{ts}")
        recorder = TrajectoryRecorder(save_dir)
        recorder.set_metadata(
            model=self.model,
            start_time=TrajectoryRecorder._now_iso(),
            query=query,
            image_paths=image_paths or [],
            max_iterations=self.max_iterations,
            system_prompt=self.system_prompt,
        )
        return recorder

    async def run(
        self,
        query: str,
        image_paths: Optional[List[str]] = None,
    ) -> str:
        """
        Run the agentic loop for a single user query.

        Args:
            query: The user's question/instruction
            image_paths: Optional list of image file paths to include

        Returns:
            The final answer string
        """
        # ── Initialize trajectory recorder ──
        self.trajectory = self._init_trajectory(query, image_paths)

        # Initialize conversation
        self.messages = [
            {"role": "system", "content": self.system_prompt},
        ]

        # Build and add user message
        user_msg = self._build_user_message(query, image_paths)
        self.messages.append(user_msg)

        # Record user step
        self.trajectory.record_user_step(query, image_paths)

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"User Query: {query}")
            if image_paths:
                print(f"Images: {image_paths}")
            print(f"{'='*60}\n")

        final_answer = None
        try:
            final_answer = await self._run_loop()
        finally:
            # ── Always save trajectory, even on error ──
            if final_answer is not None:
                self.trajectory.record_finish(final_answer)
            self.trajectory.save()
            self.trajectory.save_messages_raw(self.messages)

        return final_answer

    async def _run_loop(self) -> str:
        """Core agentic loop — separated so trajectory save happens in `run`."""
        for iteration in range(1, self.max_iterations + 1):
            if self.verbose:
                print(f"\n--- Iteration {iteration}/{self.max_iterations} ---")

            
            MAX_RETRIES = 10
            for retry in range(MAX_RETRIES):
                try:
                    response = self._call_llm()
                    break
                except Exception as e:
                    self._log("OpenAI API error: %s, retry %d/%d", str(e), retry, MAX_RETRIES, level="error")
                    
            if retry == MAX_RETRIES - 1:
                return f"[Error] Failed to call LLM: {e}"

            choice = response.choices[0]
            message = choice.message

            # Serialize the full message to a dict, preserving every field
            # exactly as the API returned it (content, reasoning_details,
            # tool_calls, etc.) so nothing is lost or renamed.
            if hasattr(message, "to_dict"):
                assistant_msg = message.to_dict()
            elif hasattr(message, "model_dump"):
                assistant_msg = message.model_dump()
            else:
                assistant_msg = {"role": "assistant", "content": message.content}
            assistant_msg.setdefault("role", "assistant")
            self.messages.append(assistant_msg)

            # Extract fields for trajectory recording / display
            tool_call_dicts = assistant_msg.get("tool_calls")
            reasoning_details = assistant_msg.get("reasoning_details")

            # Record assistant step (including reasoning for trajectory)
            self.trajectory.record_assistant_step(
                message.content, tool_call_dicts, reasoning_details=reasoning_details,
            )

            # Show reasoning content if available

            try:
                if message.reasoning and self.verbose:
                    summary = message.reasoning if isinstance(message.reasoning, str) else ""
                    preview = summary[:300] + ("..." if len(summary) > 300 else "")
                    print(f"\n[Reasoning] {preview}")
            except:
                # no reasoning use content
                summary = message.reasoning_content[:300]
                #print(f"No reasoning, use content: {summary}")

            # If the model has text content (final text / partial answer), show it
            if message.content:
                if self.verbose:
                    print(f"\n[Assistant] {message.content[:500]}")

            # If no tool calls, the model finished without calling finish tool
            if not message.tool_calls:
                if choice.finish_reason == "stop":
                    self._log("Model stopped without calling finish tool.")
                    return message.content or "[No response]"
                continue

            # Process each tool call
            for tool_call in message.tool_calls:
                fn_name = tool_call.function.name
                try:
                    fn_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError as e:
                    self._log("Failed to parse tool arguments: %s", e, level="error")
                    err_text = f"[Error] Invalid JSON arguments: {e}"
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": err_text,
                    })
                    self.trajectory.record_tool_step(
                        tool_call_id=tool_call.id,
                        tool_name=fn_name,
                        code=None,
                        text_output=err_text,
                    )
                    continue

                if fn_name == "finish":
                    # ─── Workflow complete ───
                    answer = fn_args.get("answer", "")
                    if self.verbose:
                        print(f"\n{'='*60}")
                        print(f"[FINISH] Final Answer:")
                        print(answer)
                        print(f"{'='*60}\n")
                    return answer

                elif fn_name == "execute_code":
                    # ─── Execute code in Docker Jupyter ───
                    code = fn_args.get("code", "")
                    text_output = ""
                    image_parts: List[Dict[str, Any]] = []
                    base64_images: List[str] = []
                    try:
                        exec_result = await self._handle_execute_code(code)
                        text_output = exec_result["text_output"]
                        image_parts = exec_result["image_parts"]
                        base64_images = exec_result["base64_images"]
                    except Exception as e:
                        tb = traceback.format_exc()
                        self._log("Code execution failed: %s", e, level="error")
                        text_output = f"[Execution Error] {e}\n{tb}"

                    # Build tool message content — OpenAI API supports
                    # multimodal content (text + images) in tool messages.
                    if image_parts:
                        tool_content: List[Dict[str, Any]] = [
                            {"type": "text", "text": text_output},
                        ] + image_parts
                    else:
                        tool_content = text_output

                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_content,
                    })

                    # Record in trajectory
                    self.trajectory.record_tool_step(
                        tool_call_id=tool_call.id,
                        tool_name=fn_name,
                        code=code,
                        text_output=text_output,
                        base64_images=base64_images,
                    )

                    if self.verbose:
                        print(f"\n[Code Output] {text_output[:500]}")
                        if image_parts:
                            print(f"  [{len(image_parts)} image(s) returned to model in tool message]")

                else:
                    self._log("Unknown tool: %s", fn_name, level="warning")
                    err_text = f"[Error] Unknown tool: {fn_name}"
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": err_text,
                    })
                    self.trajectory.record_tool_step(
                        tool_call_id=tool_call.id,
                        tool_name=fn_name,
                        code=None,
                        text_output=err_text,
                    )

        self._log("Max iterations reached (%d)", self.max_iterations, level="warning")
        return "[Error] Max iterations reached without a final answer."

    async def run_interactive(self, image_paths: Optional[List[str]] = None):
        """
        Run in interactive mode — the user can keep asking questions
        and the kernel state is preserved.
        """
        print("\n" + "="*60)
        print("VLM Tool Call Agent - Interactive Mode (Docker Runtime)")
        print("Type 'quit' or 'exit' to stop.")
        print("Type 'image:<path>' to add an image to the next query.")
        print("="*60 + "\n")

        session_images = list(image_paths or [])

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit"):
                print("Goodbye!")
                break

            # Check for image attachment
            if user_input.lower().startswith("image:"):
                img_path = user_input[6:].strip()
                if os.path.exists(img_path):
                    session_images.append(img_path)
                    print(f"  Added image: {img_path}")
                else:
                    print(f"  Image not found: {img_path}")
                continue

            # Run the query
            answer = await self.run(user_input, session_images if session_images else None)
            print(f"\nAnswer: {answer}\n")

            # Reset images after each query (keep kernel state)
            session_images = []

    async def cleanup(self):
        """Shut down the Docker kernel and clean up resources."""
        if self.kernel:
            await self.kernel.shutdown()


# ═════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═════════════════════════════════════════════════════════════════════
async def async_main():
    parser = argparse.ArgumentParser(
        description="VLM Tool Call Agent - Agentic VLM with Docker Jupyter Notebook tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ask a question with an image
  python sample_vlm_toolcall_docker.py --image image.png "What objects are in this image?"

  # Computation question (no image)
  python sample_vlm_toolcall_docker.py "Compute the first 20 Fibonacci numbers"

  # Multiple images
  python sample_vlm_toolcall_docker.py --image img1.png --image img2.png "Compare these two images"

  # Interactive mode
  python sample_vlm_toolcall_docker.py --interactive

  # Custom model and API
  python sample_vlm_toolcall_docker.py --model gpt-4o --base-url https://api.openai.com/v1 "Hello"
        """,
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="The question or instruction for the agent",
    )
    parser.add_argument(
        "--image", "-i",
        action="append",
        default=[],
        help="Path to an image file (can be specified multiple times)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )
    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_MODEL,
        help=f"Model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenAI API key (or set OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Custom API base URL (or set OPENAI_BASE_URL env var)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=MAX_ITERATIONS,
        help=f"Max agentic loop iterations (default: {MAX_ITERATIONS})",
    )
    parser.add_argument(
        "--save-trajectory",
        default=None,
        help="Directory to save trajectory (default: auto-generated under ./trajectories/)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=True,
        help="Verbose output (default: True)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Quiet mode (minimal output)",
    )
    parser.add_argument(
        "--reasoning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable reasoning mode (default: True). Use --no-reasoning to disable.",
    )

    args = parser.parse_args()

    if args.quiet:
        args.verbose = False

    # Create agent
    agent = VLMToolCallAgent(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        max_iterations=args.max_iterations,
        verbose=args.verbose,
        save_trajectory=args.save_trajectory,
        reasoning=args.reasoning,
    )

    try:
        if args.interactive:
            await agent.run_interactive(args.image)
        elif args.query:
            answer = await agent.run(
                args.query,
                args.image if args.image else None,
            )
            if not args.verbose:
                print(answer)
        else:
            parser.print_help()
            print("\nError: Please provide a query or use --interactive mode.")
            sys.exit(1)
    finally:
        await agent.cleanup()


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()