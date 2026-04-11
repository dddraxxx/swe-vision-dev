FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN pip install \
    ipython==8.18.1 \
    ipykernel \
    matplotlib \
    numpy \
    pandas \
    Pillow \
    scipy

WORKDIR /mnt/data
