FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

RUN pip install \
    ipython==8.18.1 \
    ipykernel \
    matplotlib \
    numpy \
    pandas \
    Pillow \
    scipy \
    opencv-python-headless \
    pytesseract \
    scikit-image

WORKDIR /mnt/data
