# docker/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# системные зависимости для сборки некоторых пакетов
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ставим poetry
RUN pip install --no-cache-dir poetry==1.8.3

# копируем только файлы зависимостей для кэширования слоёв
COPY pyproject.toml poetry.lock* /app/

# ставим зависимости (без dev)
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --without dev

# копируем код
COPY src /app/src
COPY main.py /app/
COPY generate_info.py /app/
COPY batch_answers.py /app/

# информация о сборке
RUN python generate_info.py --output /app/info.json || true

EXPOSE 8000

CMD ["python", "main.py"]