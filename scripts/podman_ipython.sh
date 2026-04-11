#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_DIR"

WORKDIR_INPUT="${1:-${REPO_DIR}/workspaces/charxiv_reasoning}"
shift || true

WORKDIR="$(cd "$(dirname "$WORKDIR_INPUT")" && pwd)/$(basename "$WORKDIR_INPUT")"
IMAGE_NAME="${VLM_DOCKER_IMAGE:-swe-vision}"

if ! sudo podman image exists "$IMAGE_NAME"; then
  sudo podman build --isolation=chroot -t "$IMAGE_NAME" ./env
fi

mkdir -p "$WORKDIR"

exec sudo podman run --rm -it \
  --runtime=crun \
  --cgroups=disabled \
  --network=host \
  -v "${WORKDIR}:/mnt/data:rw" \
  -w /mnt/data \
  "$IMAGE_NAME" \
  ipython "$@"
