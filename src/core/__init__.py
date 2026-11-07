__all__ = ["settings", "llm_settings"]


def __getattr__(name: str):
    if name == "settings":
        from src.core.config.config import settings

        return settings
    if name == "llm_settings":
        from src.core.config.llm_config import llm_settings

        return llm_settings
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
