"""
Notebook kernel manager.

    Supports:
    - local managed Jupyter kernel
    - local_sandbox managed Jupyter kernel under unshare + setpriv
    - docker-backed Jupyter runtime
    - podman-backed Jupyter runtime
"""

import asyncio
import base64
import json
import os
import queue
import re
import shutil
import socket
import subprocess
import sys
import time
import uuid
from io import BytesIO
from threading import Lock
from typing import Any, Dict, List

from swe_vision.config import (
    CELL_TIMEOUT,
    CONTAINER_WORK_DIR,
    DOCKER_IMAGE_NAME,
    DOCKERFILE_DIR,
    HOST_WORK_BASE,
    HOST_WORK_DIR,
    KERNEL_PORTS,
    MAX_OUTPUT_CHARS,
    PODMAN_ROOT,
    PODMAN_RUNROOT,
    RUNTIME,
    logger,
)


class JupyterNotebookKernel:
    """
    Manages a persistent notebook runtime for tool execution.

    Local mode runs a managed Jupyter kernel process in the current Python
    environment. local_sandbox adds user, pid, and mount namespace
    isolation with no-new-privs. Docker mode preserves the original
    container-backed runtime. Podman mode mirrors Docker but uses a
    rootful Podman runtime configured for this host.
    """

    _reserved_kernel_ports_lock = Lock()
    _reserved_kernel_ports: set[int] = set()

    def __init__(
        self,
        timeout: float = CELL_TIMEOUT,
        host_work_dir: str = HOST_WORK_DIR,
        container_work_dir: str = CONTAINER_WORK_DIR,
        docker_image: str = DOCKER_IMAGE_NAME,
        dockerfile_dir: str = DOCKERFILE_DIR,
        runtime: str = RUNTIME,
    ):
        self._timeout = timeout
        self._host_work_dir = self._resolve_host_work_dir(host_work_dir)
        self._runtime = runtime
        self._container_work_dir = self._host_work_dir if runtime in {"local", "local_sandbox"} else container_work_dir
        self._docker_image = docker_image
        self._dockerfile_dir = dockerfile_dir

        self._container = None
        self._docker_client = None
        self._podman_container_name = None
        self._km = None
        self._kc = None
        self._started = False

        self._kernel_key = uuid.uuid4().hex
        self._kernel_ports = None

    @staticmethod
    def _resolve_host_work_dir(host_work_dir: str) -> str:
        """
        Give each kernel instance its own work directory so failed or slow
        startups do not reuse stale connection files or notebook artifacts.
        """
        base_dir = os.path.abspath(host_work_dir or HOST_WORK_BASE)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        return os.path.join(base_dir, f"{timestamp}_{uuid.uuid4().hex[:8]}")

    @property
    def host_work_dir(self) -> str:
        return self._host_work_dir

    @property
    def container_work_dir(self) -> str:
        return self._container_work_dir

    @property
    def supports_remote_files(self) -> bool:
        return False

    # Docker helpers

    def _build_image(self) -> None:
        import docker

        if self._docker_client is None:
            self._docker_client = docker.from_env()

        try:
            self._docker_client.images.get(self._docker_image)
            logger.info("Docker image '%s' already exists, skipping build.", self._docker_image)
            return
        except docker.errors.ImageNotFound:
            pass

        logger.info("Building Docker image '%s' from %s ...", self._docker_image, self._dockerfile_dir)
        _, build_logs = self._docker_client.images.build(
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

    @classmethod
    def _allocate_kernel_ports(cls) -> Dict[str, int]:
        ports: Dict[str, int] = {}
        with cls._reserved_kernel_ports_lock:
            for name in (
                "shell_port",
                "iopub_port",
                "stdin_port",
                "control_port",
                "hb_port",
            ):
                while True:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    try:
                        sock.bind(("127.0.0.1", 0))
                        port = sock.getsockname()[1]
                    finally:
                        sock.close()
                    if port in cls._reserved_kernel_ports:
                        continue
                    cls._reserved_kernel_ports.add(port)
                    ports[name] = port
                    break
        return ports

    @classmethod
    def _release_kernel_ports(cls, ports: List[int]) -> None:
        with cls._reserved_kernel_ports_lock:
            for port in ports:
                cls._reserved_kernel_ports.discard(port)

    def _write_connection_file(self) -> str:
        conn = {
            "shell_port": self._kernel_ports["shell_port"],
            "iopub_port": self._kernel_ports["iopub_port"],
            "stdin_port": self._kernel_ports["stdin_port"],
            "control_port": self._kernel_ports["control_port"],
            "hb_port": self._kernel_ports["hb_port"],
            "ip": "0.0.0.0",
            "key": self._kernel_key,
            "transport": "tcp",
            "signature_scheme": "hmac-sha256",
            "kernel_name": "python3",
        }
        host_path = os.path.join(self._host_work_dir, ".kernel_connection.json")
        with open(host_path, "w") as handle:
            json.dump(conn, handle)
        logger.info("Wrote kernel connection file to %s", host_path)
        return os.path.join(self._container_work_dir, ".kernel_connection.json")

    def _start_container(self) -> None:
        import docker

        if self._docker_client is None:
            self._docker_client = docker.from_env()

        port_bindings = {f"{port}/tcp": ("127.0.0.1", port) for port in self._kernel_ports.values()}
        container_name = f"vlm-jupyter-{uuid.uuid4().hex[:8]}"

        logger.info(
            "Starting Docker container '%s' (image=%s, mount=%s -> %s) ...",
            container_name,
            self._docker_image,
            self._host_work_dir,
            self._container_work_dir,
        )

        self._container = self._docker_client.containers.run(
            image=self._docker_image,
            name=container_name,
            command="sleep infinity",
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

    def _start_kernel_in_container(self, connection_file: str) -> None:
        cmd = f"python -m ipykernel_launcher -f {connection_file} --IPKernelApp.matplotlib='inline'"
        logger.info("Starting kernel inside container: %s", cmd)
        self._container.exec_run(
            cmd=["sh", "-lc", cmd],
            detach=True,
            workdir=self._container_work_dir,
        )
        time.sleep(2)

    def _connect_docker_client(self) -> None:
        from jupyter_client import BlockingKernelClient

        self._kc = BlockingKernelClient()
        self._kc.ip = "127.0.0.1"
        self._kc.shell_port = self._kernel_ports["shell_port"]
        self._kc.iopub_port = self._kernel_ports["iopub_port"]
        self._kc.stdin_port = self._kernel_ports["stdin_port"]
        self._kc.control_port = self._kernel_ports["control_port"]
        self._kc.hb_port = self._kernel_ports["hb_port"]
        self._kc.session.key = self._kernel_key.encode("utf-8")
        self._kc.start_channels()
        logger.info("Kernel client connected to 127.0.0.1 ports %s", list(self._kernel_ports.values()))

    def _stop_kernel_client_channels(self) -> None:
        if self._kc is None:
            return
        try:
            self._kc.stop_channels()
        except Exception as exc:
            logger.warning("Failed to stop kernel client channels: %s", exc)
        self._kc = None

    # Podman helpers

    def _podman_cmd(self) -> List[str]:
        cmd = ["sudo", "podman"]
        if PODMAN_ROOT:
            cmd += ["--root", PODMAN_ROOT]
        if PODMAN_RUNROOT:
            cmd += ["--runroot", PODMAN_RUNROOT]
        return cmd

    def _podman_image_exists(self) -> bool:
        result = subprocess.run(
            self._podman_cmd() + ["image", "exists", self._docker_image],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return result.returncode == 0

    def _build_podman_image(self) -> None:
        if self._podman_image_exists():
            logger.info("Podman image '%s' already exists, skipping build.", self._docker_image)
            return

        logger.info("Building Podman image '%s' from %s ...", self._docker_image, self._dockerfile_dir)
        subprocess.run(
            self._podman_cmd() + ["build", "--isolation=chroot", "-t", self._docker_image, self._dockerfile_dir],
            check=True,
        )
        logger.info("Podman image '%s' built successfully.", self._docker_image)

    def _start_podman_container(self) -> None:
        container_name = f"vlm-jupyter-{uuid.uuid4().hex[:8]}"

        logger.info(
            "Starting Podman container '%s' (image=%s, mount=%s -> %s) ...",
            container_name,
            self._docker_image,
            self._host_work_dir,
            self._container_work_dir,
        )

        cmd = self._podman_cmd() + [
            "run",
            "-d",
            "--rm",
            "--name",
            container_name,
            "--network=host",
            "--runtime=crun",
            "--cgroups=disabled",
            "-v",
            f"{self._host_work_dir}:{self._container_work_dir}:rw",
            "-w",
            self._container_work_dir,
            self._docker_image,
            "sleep",
            "infinity",
        ]
        subprocess.run(cmd, check=True)
        self._podman_container_name = container_name
        logger.info("Podman container '%s' started.", container_name)

    def _start_kernel_in_podman(self, connection_file: str) -> None:
        cmd = f"python -m ipykernel_launcher -f {connection_file} --IPKernelApp.matplotlib='inline'"
        logger.info("Starting kernel inside Podman container: %s", cmd)
        subprocess.run(
            self._podman_cmd()
            + ["exec", "-d", self._podman_container_name, "sh", "-lc", cmd],
            check=True,
        )
        time.sleep(2)

    # Local helpers

    def _start_local_kernel(self) -> None:
        from jupyter_client import KernelManager

        logger.info("Starting local Jupyter kernel in %s ...", self._host_work_dir)
        self._km = KernelManager(
            kernel_cmd=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        )
        self._km.start_kernel(
            cwd=self._host_work_dir,
            extra_arguments=["--IPKernelApp.matplotlib=inline"],
            env=os.environ.copy(),
        )
        self._kc = self._km.blocking_client()
        self._kc.start_channels()
        self._kc.wait_for_ready(timeout=self._timeout)
        logger.info("Local Jupyter kernel started.")

    def _start_local_sandbox_kernel(self) -> None:
        from jupyter_client import KernelManager

        sandbox_cmd = [
            "setpriv",
            "--no-new-privs",
            "unshare",
            "--user",
            "--map-root-user",
            "--mount",
            "--pid",
            "--fork",
            "--mount-proc",
            sys.executable,
            "-m",
            "ipykernel_launcher",
            "-f",
            "{connection_file}",
        ]
        logger.info("Starting local sandboxed Jupyter kernel in %s ...", self._host_work_dir)
        self._km = KernelManager(kernel_cmd=sandbox_cmd)
        self._km.start_kernel(
            cwd=self._host_work_dir,
            extra_arguments=["--IPKernelApp.matplotlib=inline"],
            env=os.environ.copy(),
        )
        self._kc = self._km.blocking_client()
        self._kc.start_channels()
        self._kc.wait_for_ready(timeout=self._timeout)
        logger.info("Local sandboxed Jupyter kernel started.")

    # Public API

    async def start(self) -> None:
        if self._started:
            return

        if self._kernel_ports is None:
            self._kernel_ports = self._allocate_kernel_ports()

        os.makedirs(self._host_work_dir, exist_ok=True)

        try:
            if self._runtime == "docker":
                self._build_image()
                conn_file = self._write_connection_file()
                self._start_container()
                self._start_kernel_in_container(conn_file)
                self._connect_docker_client()
                try:
                    self._kc.wait_for_ready(timeout=self._timeout)
                except RuntimeError:
                    logger.warning("Kernel wait_for_ready timed out, retrying after 3s...")
                    self._stop_kernel_client_channels()
                    time.sleep(3)
                    self._connect_docker_client()
                    self._kc.wait_for_ready(timeout=self._timeout)
            elif self._runtime == "podman":
                if PODMAN_ROOT:
                    os.makedirs(PODMAN_ROOT, exist_ok=True)
                if PODMAN_RUNROOT:
                    os.makedirs(PODMAN_RUNROOT, exist_ok=True)
                self._build_podman_image()
                conn_file = self._write_connection_file()
                self._start_podman_container()
                self._start_kernel_in_podman(conn_file)
                self._connect_docker_client()
                try:
                    self._kc.wait_for_ready(timeout=self._timeout)
                except RuntimeError:
                    logger.warning("Kernel wait_for_ready timed out, retrying after 3s...")
                    self._stop_kernel_client_channels()
                    time.sleep(3)
                    self._connect_docker_client()
                    self._kc.wait_for_ready(timeout=self._timeout)
            elif self._runtime == "local":
                self._start_local_kernel()
            elif self._runtime == "local_sandbox":
                self._start_local_sandbox_kernel()
            else:
                raise ValueError(f"Unsupported runtime: {self._runtime}")

            test_result = await self._execute_jupyter("print('kernel_ready')")
            if "kernel_ready" not in test_result.get("stdout", ""):
                raise RuntimeError(f"{self._runtime} notebook runtime failed health check")

            await self._execute_jupyter("%config InlineBackend.figure_format = 'png'")

            self._started = True
            logger.info("Notebook runtime started successfully (runtime=%s).", self._runtime)
        except Exception:
            await self.shutdown(cleanup_work_dir=True)
            raise

    async def _execute_jupyter(self, code: str) -> Dict[str, Any]:
        cell_result: Dict[str, Any] = {
            "stdout": "",
            "stderr": "",
            "display": [],
            "error": [],
            "status": "ok",
        }

        def _sync_execute() -> None:
            msg_id = self._kc.execute(code)
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
                    continue

                content = msg["content"]
                if msg_type == "stream":
                    cell_result[content["name"]] += content["text"]
                elif msg_type in ("display_data", "execute_result"):
                    cell_result["display"].append(content["data"])
                elif msg_type == "error":
                    cell_result["error"].append(content)
                elif msg_type == "status" and content.get("execution_state") == "idle":
                    reply_received = True

            try:
                reply = self._kc.get_shell_msg(timeout=self._timeout)
                cell_result["status"] = reply["content"]["status"]
            except queue.Empty:
                cell_result["status"] = "error"

        await asyncio.get_event_loop().run_in_executor(None, _sync_execute)
        return cell_result

    def _resolve_display_image_src(self, src: str) -> str | None:
        src = (src or "").strip()
        if not src or src.startswith(("http://", "https://", "data:")):
            return None
        if src.startswith("file://"):
            src = src[len("file://") :]
        if not os.path.isabs(src):
            src = os.path.join(self._host_work_dir, src)
        return src

    @staticmethod
    def _encode_image_file_as_png_base64(image_path: str) -> str:
        from PIL import Image

        with Image.open(image_path) as image:
            if image.mode not in {"RGB", "RGBA", "L"}:
                image = image.convert("RGBA")
            buf = BytesIO()
            image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _extract_display_images(self, display_item: Dict[str, Any]) -> List[str]:
        images: List[str] = []
        if "image/png" in display_item:
            images.append(display_item["image/png"])
            return images
        if "image/jpeg" in display_item:
            # Normalize notebook JPEG output to PNG so downstream handling can
            # treat all notebook images as a single format.
            try:
                from PIL import Image

                jpeg_bytes = base64.b64decode(display_item["image/jpeg"])
                with Image.open(BytesIO(jpeg_bytes)) as image:
                    if image.mode not in {"RGB", "RGBA", "L"}:
                        image = image.convert("RGBA")
                    buf = BytesIO()
                    image.save(buf, format="PNG")
                images.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
            except Exception as exc:
                logger.warning("Failed to normalize JPEG display output: %s", exc)
            return images

        html = display_item.get("text/html")
        if not isinstance(html, str):
            return images

        srcs = re.findall(r"""<img[^>]+src=["']([^"']+)["']""", html, flags=re.IGNORECASE)
        for src in srcs:
            resolved = self._resolve_display_image_src(src)
            if not resolved or not os.path.exists(resolved):
                continue
            try:
                images.append(self._encode_image_file_as_png_base64(resolved))
            except Exception as exc:
                logger.warning("Failed to capture displayed image '%s': %s", resolved, exc)
        return images

    async def execute(self, code: str) -> Dict[str, Any]:
        if not self._started:
            await self.start()

        raw = await self._execute_jupyter(code)

        text_parts = []
        if raw["stdout"]:
            text_parts.append(raw["stdout"])
        if raw["stderr"]:
            text_parts.append(f"[STDERR] {raw['stderr']}")
        if raw["error"]:
            for err in raw["error"]:
                tb_text = "\n".join(err.get("traceback", []))
                tb_text = re.sub(r"\x1b\[[0-9;]*m", "", tb_text)
                text_parts.append(f"[ERROR] {tb_text}")

        for display_item in raw["display"]:
            if "text/plain" in display_item and "image/png" not in display_item:
                text_parts.append(display_item["text/plain"])

        images = []
        for display_item in raw["display"]:
            images.extend(self._extract_display_images(display_item))

        text_output = "\n".join(text_parts).strip()
        if not text_output and not images:
            text_output = "[No output produced. Use print() to see results.]"

        if len(text_output) > MAX_OUTPUT_CHARS:
            text_output = text_output[:MAX_OUTPUT_CHARS] + "\n... [output truncated]"

        error_tb = None
        if raw["error"]:
            error_tb = "\n".join("\n".join(e.get("traceback", [])) for e in raw["error"])
            error_tb = re.sub(r"\x1b\[[0-9;]*m", "", error_tb)

        return {
            "text_output": text_output,
            "images": images,
            "status": raw["status"],
            "error_traceback": error_tb,
        }

    async def shutdown(self, cleanup_work_dir: bool = False) -> None:
        self._stop_kernel_client_channels()

        if self._km is not None:
            try:
                logger.info("Stopping local Jupyter kernel ...")
                self._km.shutdown_kernel(now=True)
                logger.info("Local kernel stopped.")
            except Exception as exc:
                logger.warning("Failed to stop local kernel: %s", exc)
            self._km = None

        if self._container is not None:
            try:
                logger.info("Stopping Docker container '%s' ...", self._container.short_id)
                self._container.stop(timeout=5)
                self._container.remove(force=True)
                logger.info("Container removed.")
            except Exception as exc:
                logger.warning("Failed to stop/remove container: %s", exc)
            self._container = None

        if self._podman_container_name is not None:
            try:
                logger.info("Stopping Podman container '%s' ...", self._podman_container_name)
                subprocess.run(
                    self._podman_cmd() + ["rm", "-f", self._podman_container_name],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                logger.info("Podman container removed.")
            except Exception as exc:
                logger.warning("Failed to stop/remove Podman container: %s", exc)
            self._podman_container_name = None

        self._started = False

        if self._kernel_ports is not None:
            self._release_kernel_ports(list(self._kernel_ports.values()))
            self._kernel_ports = None

        if cleanup_work_dir and os.path.isdir(self._host_work_dir):
            try:
                shutil.rmtree(self._host_work_dir)
                logger.info("Cleaned up host work directory: %s", self._host_work_dir)
            except Exception as exc:
                logger.warning("Failed to clean up work directory %s: %s", self._host_work_dir, exc)
