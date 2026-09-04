FROM ghcr.io/astral-sh/uv:0.12.5 AS uv

FROM python:3.12-slim
COPY --from=uv /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/home/app/.cache/uv \
    HOME=/home/app

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --home-dir /home/app --create-home \
        --shell /usr/sbin/nologin app \
    && mkdir -p /app/artifacts /app/charts /home/app/.cache/uv \
    && chown -R app:app /app /home/app
WORKDIR /app
COPY --chown=app:app pyproject.toml uv.lock README.md ./
USER app
RUN uv sync --locked --no-install-project

COPY --chown=app:app src ./src
RUN uv sync --locked --no-editable

COPY --chown=app:app . .

CMD ["uv", "run", "--frozen", "lead-scoring", "--help"]
