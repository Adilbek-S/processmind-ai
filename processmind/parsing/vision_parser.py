"""Vision Parser — сценарий Б: извлечение процесса из изображения через GPT-4o-mini Vision.

Реальный вызов OpenAI Chat Completions с image-контентом. Никаких заготовленных ответов:
если ключа нет, изображение битое или модель ничего не распознала — наружу уходит
`ExtractionError`, а не «правдоподобный» результат.
"""

from __future__ import annotations

import base64
import io

from openai import OpenAI
from PIL import Image, UnidentifiedImageError

from processmind.parsing.llm import request_raw_process
from processmind.parsing.prompts import VISION_SYSTEM_PROMPT, VISION_USER_PROMPT
from processmind.parsing.schemas import ExtractionError, RawProcess

MAX_IMAGE_BYTES = 20 * 1024 * 1024  # лимит OpenAI на изображение в запросе
_MIME_BY_FORMAT = {"PNG": "image/png", "JPEG": "image/jpeg"}


def encode_image_data_url(image_bytes: bytes) -> str:
    """Проверяет, что это настоящий PNG/JPEG (по содержимому, не по расширению), и кодирует в data URL."""
    if not image_bytes:
        raise ExtractionError("Файл изображения пуст.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ExtractionError("Изображение больше 20 МБ — уменьшите его и загрузите снова.")
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img.verify()
            fmt = img.format
    except (UnidentifiedImageError, OSError) as exc:
        raise ExtractionError("Файл не является корректным изображением PNG/JPEG.") from exc
    if fmt not in _MIME_BY_FORMAT:
        raise ExtractionError(f"Формат изображения {fmt} не поддерживается (нужен PNG или JPEG).")
    return f"data:{_MIME_BY_FORMAT[fmt]};base64,{base64.b64encode(image_bytes).decode('ascii')}"


class VisionParser:
    """Распознаёт операции процесса и связи между ними на изображении схемы."""

    def __init__(self, client: OpenAI, model: str) -> None:
        self._client = client
        self._model = model

    def extract_process_from_image(self, image_bytes: bytes) -> RawProcess:
        data_url = encode_image_data_url(image_bytes)
        messages = [
            {"role": "system", "content": VISION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VISION_USER_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
                ],
            },
        ]
        return request_raw_process(self._client, self._model, messages)
