FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y curl gcc bash git

ADD https://astral.sh/uv/install.sh /uv-installer.sh

RUN sh /uv-installer.sh && rm /uv-installer.sh

ENV PATH="/root/.local/bin/:$PATH"

COPY . /app/

RUN uv sync --frozen

CMD ["bash", "run_app.sh"]