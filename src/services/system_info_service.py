import os
import json

from loguru import logger

from src.schemas.info import InfoResponse


class InfoService:
    def __init__(self):
        try:
            info_file_path = os.getenv("SERVICE_INFO_PATH", "info.json")
            with open(info_file_path, "r", encoding="utf-8") as f:
                self.info = json.load(f)
        except Exception as e:
            logger.error(f"An error occurred while trying to generate info: {e}")

    async def get_info(self) -> InfoResponse:
        try:
            return self.info
        except Exception as e:
            logger.error(f"An error occurred while trying to get info: {e}")
            return InfoResponse(
                name="spok-case-nmd",
                version="0.1.0",
                git_current_branch="unknown",
                git_head_commit="unknown",
                lang="python",
                builder="poetry",
                lang_version="unknown",
                builder_version="unknown",
                build_time="unknown",
            )
