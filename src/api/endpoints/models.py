from fastapi import APIRouter, HTTPException
import httpx
from loguru import logger

from src.core import settings
from src.models.api_credentials import get_api_key, get_default_headers

router = APIRouter()


@router.get("/models")
async def list_models():
    """Proxy the OpenAI-compatible /models endpoint of the configured LLM."""
    url = f"{settings.VLLM_LLM_URL.rstrip('/')}/models"
    api_key = get_api_key(settings.VLLM_LLM_URL)
    headers = get_default_headers(settings.VLLM_LLM_URL) or {}
    
    if api_key and api_key != "not-needed":
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        async with httpx.AsyncClient(timeout=10.0, proxy=None) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            models = []
            for model in data.get("data", []):
                out_mods = model.get("output_modalities", ["text"])
                if "text" in out_mods:
                    models.append({
                        "id": model.get("id"),
                        "name": model.get("name") or model.get("id"),
                        "context_length": model.get("context_length") or model.get("context_window"),
                        "active": model.get("active", True)
                    })
            
            return {"object": "list", "data": models}
            
    except httpx.HTTPStatusError as exc:
        logger.error(f"[models] Groq returned {exc.response.status_code}: {exc.response.text}")
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=f"LLM provider error: {exc.response.text}"
        )
    except Exception as exc:
        logger.exception("[models] failed to fetch models")
        raise HTTPException(
            status_code=502,
            detail=f"Could not reach LLM provider: {type(exc).__name__}: {exc}"
        )