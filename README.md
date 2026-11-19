# Чат по документам

Загружаешь ZIP с документами в формате DOCX, TXT или Markdown, ждёшь окончания импорта и задаёшь вопросы в чате. В ответе можно посмотреть, на какие документы опиралась модель.

Под капотом OpenSearch ищет по словам (BM25) и по смыслу (BGE-M3 и multilingual E5). Результаты объединяются через RRF, а ответ по найденным фрагментам формирует `openai/gpt-oss-120b`.

## Запуск

Нужны Python 3.11+, Poetry и Docker. В `.env` укажи параметры подключения к OpenSearch, PostgreSQL и API языковой модели. Для BGE и E5 можно указать один адрес сервера эмбеддингов. В проекте для него есть `emb_server.py`, который запускается отдельно и требует `sentence-transformers` и `torch`.

```powershell
docker compose -f docker/docker-compose.yml up -d
poetry install
poetry run pip install "langfuse==4.6.1" "python-docx==1.2.0"
poetry run python -m src.main
```

Открой `http://localhost:8000` и нажми «Загрузить документы». Повторная загрузка ZIP заменяет текущий индекс.
