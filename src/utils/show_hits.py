from loguru import logger


def show_hits(hits):
    for i, hit in enumerate(hits):
        logger.debug(f"Hit #{i}")
        logger.debug(hit.get("score", -1))
        logger.debug(f"Filename: {hit.get('_source').get('filename', 'unknown')}")
        logger.debug(f"Course: {hit.get('_source').get('course', 'unknown')}")
        logger.debug(f"Source: {hit.get('_source').get('source', 'unknown')}")
        logger.debug(f"Chunk index: {hit.get('_source').get('chunk_index', -1)}")
        logger.debug(f"Text: {hit.get('_source').get('text', 'unknown')}")
