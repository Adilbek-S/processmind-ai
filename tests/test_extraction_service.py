"""Тесты сервиса извлечения на синтетических PDF/PNG.

Клиент OpenAI здесь подменён: тесты проверяют нашу обработку (PyMuPDF на настоящем PDF, формирование
запроса с изображением, валидацию, обработку ошибок), а НЕ качество распознавания моделью.
Качество проверяется живым прогоном `python -m scripts.verify_live` (нужен OPENAI_API_KEY).
"""

import base64
from types import SimpleNamespace

import httpx
import openai
import pytest

from processmind.config import Settings
from processmind.parsing.schemas import ExtractionError, RawProcess, RawStep
from processmind.parsing.service import extract_process
from scripts import demo_data

UNIT = {"seconds": "seconds", "minutes": "minutes", "hours": "hours", "days": "days"}


def perfect_raw_process() -> RawProcess:
    """Ответ «идеальной модели» по эталону демо-процесса."""
    steps = []
    for i, s in enumerate(demo_data.TRUTH_STEPS, start=1):
        minutes = s["duration_minutes"]
        steps.append(
            RawStep(
                order=i,
                name=s["name"],
                description=None,
                actor=s["actor"],
                duration_value=minutes,
                duration_unit="minutes" if minutes is not None else None,
                duration_quote=s["duration_text"] if minutes is not None else None,
                execution_type={True: "manual", False: "automated", None: "unknown"}[s["is_manual"]],
                next_orders=[i + 1] if i < len(demo_data.TRUTH_STEPS) else [],
            )
        )
    return RawProcess(
        name=demo_data.PROCESS_NAME,
        goal=demo_data.PROCESS_GOAL,
        participants=demo_data.PARTICIPANTS,
        steps=steps,
        uncertainties=[],
    )


class FakeClient:
    """Минимальная замена openai.OpenAI: запоминает запрос, отдаёт заданный ответ или ошибку."""

    def __init__(self, parsed=None, refusal=None, error=None):
        self.calls: list[dict] = []
        self._parsed, self._refusal, self._error = parsed, refusal, error
        self.chat = SimpleNamespace(completions=SimpleNamespace(parse=self._parse))

    def _parse(self, **kwargs):
        self.calls.append(kwargs)
        if self._error:
            raise self._error
        message = SimpleNamespace(parsed=self._parsed, refusal=self._refusal)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


@pytest.fixture(scope="module")
def demo_pdf() -> bytes:
    try:
        return demo_data.build_pdf()
    except FileNotFoundError:
        pytest.skip("нет шрифта с кириллицей для генерации демо-файлов")


@pytest.fixture(scope="module")
def demo_png() -> bytes:
    try:
        return demo_data.build_png()
    except FileNotFoundError:
        pytest.skip("нет шрифта с кириллицей для генерации демо-файлов")


SETTINGS = Settings(_env_file=None, openai_api_key="test-key", openai_model="gpt-4o-mini")
NO_KEY = Settings(_env_file=None, openai_api_key=None)


# ---------- сценарий А: PDF ----------


def test_pdf_scenario_extracts_all_operations(demo_pdf):
    client = FakeClient(parsed=perfect_raw_process())
    result = extract_process("demo.pdf", demo_pdf, client=client, settings=SETTINGS)

    spec = result.spec
    assert len(spec.steps) == len(demo_data.TRUTH_STEPS) == 8
    assert [s.name for s in spec.steps] == [s["name"] for s in demo_data.TRUTH_STEPS]
    assert spec.name == demo_data.PROCESS_NAME
    assert spec.goal == demo_data.PROCESS_GOAL
    assert spec.metadata["source_type"] == "pdf"


def test_pdf_scenario_sends_real_pymupdf_text_to_llm(demo_pdf):
    client = FakeClient(parsed=perfect_raw_process())
    extract_process("demo.pdf", demo_pdf, client=client, settings=SETTINGS)

    (call,) = client.calls
    assert call["model"] == "gpt-4o-mini"
    assert call["response_format"] is RawProcess
    user_text = call["messages"][1]["content"]
    assert "Менеджер по закупкам вручную проверяет комплектность заявки (30 минут)" in user_text


def test_pdf_missing_values_are_absent_not_invented(demo_pdf):
    spec = extract_process("demo.pdf", demo_pdf, client=FakeClient(parsed=perfect_raw_process()), settings=SETTINGS).spec
    by_name = {s.name: s for s in spec.steps}
    assert by_name["Подать заявку на закупку"].duration_minutes is None
    assert by_name["Подать заявку на закупку"].is_manual is None
    assert by_name["Заключить договор с поставщиком"].actor is None
    assert by_name["Согласовать бюджет"].duration_minutes == 120
    assert by_name["Заключить договор с поставщиком"].duration_minutes == 4320


def test_pdf_hallucinated_duration_and_actor_from_model_are_removed(demo_pdf):
    raw = perfect_raw_process()
    # «модель» приписала длительность и исполнителя там, где в документе их нет
    raw.steps[0] = raw.steps[0].model_copy(
        update={"duration_value": 20, "duration_unit": "minutes", "duration_quote": "20 минут"}
    )
    raw.steps[4] = raw.steps[4].model_copy(update={"actor": "Юрист"})

    result = extract_process("demo.pdf", demo_pdf, client=FakeClient(parsed=raw), settings=SETTINGS)
    assert result.spec.steps[0].duration_minutes is None
    assert result.spec.steps[4].actor is None
    assert "Юрист" not in result.spec.actors
    assert len(result.warnings) >= 2


def test_pdf_without_text_layer_is_reported(tmp_path):
    import pymupdf

    doc = pymupdf.open()
    doc.new_page()
    blank = doc.tobytes()
    client = FakeClient(parsed=perfect_raw_process())
    with pytest.raises(ExtractionError, match="текстового слоя"):
        extract_process("scan.pdf", blank, client=client, settings=SETTINGS)
    assert client.calls == []  # без текста LLM не вызывается


def test_broken_pdf_is_reported():
    with pytest.raises(ExtractionError, match="PDF"):
        extract_process("broken.pdf", b"%PDF-1.4 not really a pdf", client=FakeClient(), settings=SETTINGS)


# ---------- сценарий Б: изображение ----------


def test_image_scenario_sends_image_to_vision_model(demo_png):
    client = FakeClient(parsed=perfect_raw_process())
    result = extract_process("scheme.png", demo_png, client=client, settings=SETTINGS)

    (call,) = client.calls
    assert call["model"] == "gpt-4o-mini"
    user_content = call["messages"][1]["content"]
    image_part = next(p for p in user_content if p["type"] == "image_url")
    url = image_part["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == demo_png  # передана именно загруженная картинка
    assert result.spec.metadata["source_type"] == "image"
    assert len(result.spec.steps) == 8


def test_jpeg_uses_jpeg_mime_by_content_not_extension(demo_png, tmp_path):
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.open(io.BytesIO(demo_png)).convert("RGB").save(buf, format="JPEG")
    client = FakeClient(parsed=perfect_raw_process())
    extract_process("scheme.jpg", buf.getvalue(), client=client, settings=SETTINGS)
    url = next(p for p in client.calls[0]["messages"][1]["content"] if p["type"] == "image_url")["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")


def test_image_recognition_failure_is_not_assumed_success(demo_png):
    """Модель ничего не нашла на картинке -> ошибка, а не пустой/выдуманный процесс."""
    empty = RawProcess(name=None, goal=None, participants=[], steps=[], uncertainties=["это фотография кота"])
    with pytest.raises(ExtractionError, match="ни одной операции"):
        extract_process("cat.png", demo_png, client=FakeClient(parsed=empty), settings=SETTINGS)


def test_corrupted_image_rejected_before_api_call():
    client = FakeClient(parsed=perfect_raw_process())
    with pytest.raises(ExtractionError, match="корректным изображением"):
        extract_process("bad.png", b"definitely not a png", client=client, settings=SETTINGS)
    assert client.calls == []


# ---------- общие ошибки ----------


@pytest.mark.parametrize("name", ["notes.txt", "scheme.gif", "noext"])
def test_unsupported_format_rejected(name):
    with pytest.raises(ExtractionError, match="не поддерживается"):
        extract_process(name, b"data", client=FakeClient(), settings=SETTINGS)


def test_missing_api_key_gives_clear_error_not_stub(demo_png, demo_pdf):
    for name, data in (("s.png", demo_png), ("d.pdf", demo_pdf)):
        with pytest.raises(ExtractionError, match="OPENAI_API_KEY"):
            extract_process(name, data, settings=NO_KEY)


def test_model_refusal_is_reported(demo_png):
    with pytest.raises(ExtractionError, match="отказалась"):
        extract_process("s.png", demo_png, client=FakeClient(refusal="нельзя"), settings=SETTINGS)


def test_unparseable_model_output_is_reported(demo_png):
    with pytest.raises(ExtractionError, match="не соответствующий схеме"):
        extract_process("s.png", demo_png, client=FakeClient(parsed=None), settings=SETTINGS)


def test_openai_errors_are_wrapped(demo_png):
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    for error, fragment in (
        (openai.APIConnectionError(request=request), "подключиться"),
        (openai.RateLimitError("limit", response=httpx.Response(429, request=request), body=None), "лимит"),
        (openai.AuthenticationError("bad", response=httpx.Response(401, request=request), body=None), "ключ"),
    ):
        with pytest.raises(ExtractionError, match=fragment):
            extract_process("s.png", demo_png, client=FakeClient(error=error), settings=SETTINGS)
