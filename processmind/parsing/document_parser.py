"""Document Parser: извлечение текста из текстовых файлов и PDF (PyMuPDF)."""

from __future__ import annotations

import pymupdf

from processmind.parsing.schemas import ExtractionError


class DocumentParser:
    """Извлекает сырой текст из загруженного документа."""

    def parse_pdf(self, file_bytes: bytes) -> str:
        text_parts: list[str] = []
        try:
            with pymupdf.open(stream=file_bytes, filetype="pdf") as doc:
                if doc.needs_pass:
                    raise ExtractionError("PDF защищён паролем — распознавание невозможно.")
                for page in doc:
                    text_parts.append(page.get_text())
        except ExtractionError:
            raise
        except Exception as exc:  # pymupdf бросает разные типы на повреждённых файлах
            raise ExtractionError(f"Не удалось прочитать PDF: {exc}") from exc
        text = "\n".join(text_parts)
        # PyMuPDF отдаёт неразрывные пробелы и мягкие переносы — для LLM это шум.
        return text.replace(" ", " ").replace("­", "").strip()

    def parse_text_file(self, file_bytes: bytes, encoding: str = "utf-8") -> str:
        return file_bytes.decode(encoding, errors="ignore").strip()

    def parse(self, filename: str, file_bytes: bytes) -> str:
        if filename.lower().endswith(".pdf"):
            return self.parse_pdf(file_bytes)
        return self.parse_text_file(file_bytes)
