FROM python:3.9.12-slim-bullseye as python
#  prevents python from buffering logs before printout to stderr stdout
ENV PYTHONUNBUFFERED=true
# prevents python from creating pyc files
ENV PYTHONDONTWRITEBYTECODE=true
WORKDIR /app
RUN apt-get update && apt-get install -y  curl nano && rm -fr /var/lib/apt/lists/*

FROM python as poetry
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VIRTUALENVS_IN_PROJECT=true
ENV PATH="$POETRY_HOME/bin:$PATH"
COPY poetry.lock poetry.lock
COPY pyproject.toml pyproject.toml
RUN python -c 'from urllib.request import urlopen; print(urlopen("https://install.python-poetry.org").read().decode())' | python - && \
    poetry install --no-interaction --no-ansi -vvv && yes | poetry cache clear . --all && rm -fr poetry.lock  pyproject.toml



FROM python as runtime
ENV PATH="/app/.venv/bin:$PATH"
COPY --from=poetry /app /app
EXPOSE 8000
COPY api api
WORKDIR /app/api
RUN curl -o /app/torch-2.2.0%252Bcpu-cp39-cp39-linux_x86_64.whl https://download.pytorch.org/whl/cpu/torch-2.2.0%2Bcpu-cp39-cp39-linux_x86_64.whl && \
    pip3 install --no-cache --no-cache-dir /app/torch-2.2.0%252Bcpu-cp39-cp39-linux_x86_64.whl && rm /app/torch-2.2.0%252Bcpu-cp39-cp39-linux_x86_64.whl
CMD gunicorn app:app --workers ${WORKERS:-1} --threads=${THREADS:-2} --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --reload --log-level=${LOGLEVEL:-debug} --timeout ${TIMEOUT:-1000}

#HEALTHCHECK CMD http://localhost:8000/app_health