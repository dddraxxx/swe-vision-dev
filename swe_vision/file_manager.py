"""
File management for the notebook runtime work directory.
"""

import os
import shutil
from dataclasses import dataclass
from typing import Optional

from swe_vision.config import CONTAINER_WORK_DIR, HOST_WORK_DIR, logger


@dataclass
class NotebookFileManager:
    """
    Manages files that should be accessible in the notebook runtime.

    Files are copied into the host work directory. In docker mode that
    directory is mounted into the container. In local mode it is the
    kernel's working directory directly.
    """
    host_work_dir: str = HOST_WORK_DIR
    container_work_dir: str = CONTAINER_WORK_DIR
    kernel: object | None = None

    def setup_work_dir(
        self,
        host_work_dir: Optional[str] = None,
        container_work_dir: Optional[str] = None,
        kernel=None,
    ):
        if host_work_dir:
            self.host_work_dir = host_work_dir
        if container_work_dir:
            self.container_work_dir = container_work_dir
        self.kernel = kernel
        os.makedirs(self.host_work_dir, exist_ok=True)
        logger.info(
            "NotebookFileManager: host_work_dir=%s, container_work_dir=%s",
            self.host_work_dir, self.container_work_dir,
        )

    def copy_file_to_workdir(self, src_path: str, dest_name: Optional[str] = None) -> str:
        """
        Copy a file into the runtime work directory.
        Returns the runtime-visible path for use in prompts / hints.
        """
        if dest_name is None:
            dest_name = os.path.basename(src_path)
        os.makedirs(self.host_work_dir, exist_ok=True)
        host_dest = os.path.join(self.host_work_dir, dest_name)
        if os.path.abspath(src_path) != os.path.abspath(host_dest):
            shutil.copy2(src_path, host_dest)
            logger.info(
                "Copied %s -> %s (kernel path: %s)",
                src_path,
                host_dest,
                os.path.join(self.container_work_dir, dest_name),
            )
        return os.path.join(self.container_work_dir, dest_name)

    def get_kernel_path(self, filename: str) -> str:
        """Return the full path a file would have inside the runtime."""
        return os.path.join(self.container_work_dir, filename)
