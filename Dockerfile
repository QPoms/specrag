# syntax=docker/dockerfile:1

FROM node:22-alpine AS web-build

WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build


FROM nginx:1.27-alpine AS web

COPY docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=web-build /build/frontend/dist /usr/share/nginx/html

EXPOSE 80
HEALTHCHECK --interval=15s --timeout=3s --retries=5 \
    CMD wget -q -O /dev/null http://127.0.0.1/ || exit 1


FROM python:3.12-slim-bookworm AS backend

ARG UV_VERSION=0.12.8

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:${PATH}" \
    HF_HOME=/home/specrag/.cache/huggingface \
    TORCH_HOME=/home/specrag/.cache/torch

RUN groupadd --gid 10001 specrag \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin specrag \
    && python -m pip install --no-cache-dir "uv==${UV_VERSION}"

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ ./src/
COPY config.yaml ./
COPY data/QA/ ./data/QA/
COPY data/memory/README.md ./data/memory/README.md

RUN uv sync --frozen --no-dev \
    && mkdir -p /app/.local /app/data/3gpp /home/specrag/.cache \
    && chown -R specrag:specrag /app /home/specrag/.cache

USER specrag

EXPOSE 8000 8766

CMD ["specrag-api", "--host", "0.0.0.0", "--port", "8000", "--worker-mode", "conversation"]
