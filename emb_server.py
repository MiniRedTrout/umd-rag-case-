from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import uvicorn

app = FastAPI()
print("Загрузка модели deepvk/USER2-small...")
model = SentenceTransformer("deepvk/USER2-small")
print("Модель загружена.")


class EmbedRequest(BaseModel):
    input: list[str] | str
    model: str | None = None


import asyncio

@app.post("/v1/embeddings")
@app.post("/embeddings")
async def embed(request: EmbedRequest):
    inputs = request.input
    if isinstance(inputs, str):
        inputs = [inputs]
    vectors = await asyncio.to_thread(
        model.encode, inputs, normalize_embeddings=True
    )
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": i, "embedding": v.tolist()}
            for i, v in enumerate(vectors)
        ],
        "model": request.model or "deepvk/USER2-small",
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="debug", access_log=True)