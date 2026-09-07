# Flexa: forecasting + battery steering in one reproducible image.
#
# Build:  docker build -t flexa:latest .
# Run:    docker run --rm -v "$(pwd)/data:/app/data" flexa:latest forecast
#         docker run --rm -v "$(pwd)/data:/app/data" flexa:latest battery
#
# Dependencies are installed from uv.lock, so the image resolves to exactly the
# same package versions on any machine. The `chronos` extra (torch, transformers,
# accelerate) is deliberately NOT installed: the Chronos-2 foundation model is a
# benchmark comparator, and neither entry point uses it. That keeps the image in
# the hundreds of MB instead of several GB. To include it, add --extra chronos to
# both uv sync calls below.

FROM python:3.11-slim-bookworm

# uv is copied from its official distroless image: no curl/install script needed.
COPY --from=ghcr.io/astral-sh/uv:0.11.4 /uv /uvx /usr/local/bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Layer 1: dependencies only. Rebuilt solely when pyproject.toml or uv.lock change,
# so day-to-day source edits reuse the (slow) dependency layer from cache.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Layer 2: project source.
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

COPY docker/entrypoint.sh /usr/local/bin/flexa-entrypoint
RUN chmod +x /usr/local/bin/flexa-entrypoint

# data/ is a mount point. Bind the host directory over it at run time
# (-v "$(pwd)/data:/app/data") so inputs are read and outputs land on the host.
VOLUME ["/app/data"]

ENTRYPOINT ["flexa-entrypoint"]
CMD ["help"]
