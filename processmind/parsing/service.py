"""Точка входа модуля извлечения: файл (PDF/PNG/JPG/JPEG) -> ProcessSpec."""

from __future__ import annotations

from pathlib import PurePath

from openai import OpenAI

from processmind.config import Settings, get_settings
from processmind.parsing.document_parser import DocumentParser
from processmind.parsing.llm import get_client
from processmind.parsing.process_builder import build_process_spec
from processmind.parsing.schemas import ExtractionError, ExtractionResult
from processmind.parsing.text_extractor import TextProcessExtractor
from processmind.parsing.vision_parser import VisionParser, encode_image_data_url

PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
SUPPORTED_EXTENSIONS = PDF_EXTENSIONS | IMAGE_EXTENSIONS

# Ниже этого порога текстового слоя PDF считаем скан без текста: извлекать нечего.
MIN_PDF_TEXT_CHARS = 20


def extract_process(
    filename: str,
    file_bytes: bytes,
    *,
    client: OpenAI | None = None,
    settings: Settings | None = None,
) -> ExtractionResult:
    """Извлекает ProcessSpec из файла. При любой проблеме бросает ExtractionError.

    Локальные проверки (формат, чтение PDF, валидность картинки) выполняются до создания
    клиента OpenAI — поэтому плохой файл получает точное сообщение даже без API-ключа.
    """
    settings = settings or get_settings()
    ext = PurePath(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ExtractionError(
            f"Формат «{ext or filename}» не поддерживается. Допустимо: PDF, PNG, JPG, JPEG."
        )

    if ext in PDF_EXTENSIONS:
        text = DocumentParser().parse_pdf(file_bytes)
        if len(text) < MIN_PDF_TEXT_CHARS:
            raise ExtractionError(
                "В PDF нет текстового слоя (похоже на скан). Загрузите схему как PNG/JPG — "
                "тогда она будет распознана Vision-моделью."
            )
        client = client or get_client(settings)
        raw = TextProcessExtractor(client, settings.openai_model).extract(text)
        return build_process_spec(
            raw,
            source_name=filename,
            source_type="pdf",
            source_bytes=file_bytes,
            model=settings.openai_model,
            source_text=text,
        )

    encode_image_data_url(file_bytes)  # быстрая локальная проверка до обращения к API
    client = client or get_client(settings)
    raw = VisionParser(client, settings.openai_model).extract_process_from_image(file_bytes)
    return build_process_spec(
        raw,
        source_name=filename,
        source_type="image",
        source_bytes=file_bytes,
        model=settings.openai_model,
    )
