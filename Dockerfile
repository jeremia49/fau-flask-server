FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      curl \
      ca-certificates \
      gcc libjpeg-dev zlib1g-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN curl -LsSf https://astral.sh/uv/install.sh | sh 

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH:/root/.local/bin/"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


RUN uv sync
RUN python install_dep.py

EXPOSE 8000

CMD ["gunicorn", "--workers", "3", "--bind", "0.0.0.0:8000", "app:app"]