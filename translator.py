from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from google import genai
from google.genai import types
from google.genai.errors import ClientError

from utils.log import LOG
from utils.textblock import TextBlock

DEFAULT_MODEL = "gemini-2.5-flash"
FALLBACK_MODELS = (
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
)
BATCH_SIZE = 5
BATCH_DELAY_SEC = 4.5
MAX_RETRIES = 4


def _load_env() -> None:
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value:
            os.environ.setdefault(key, value)


def _load_api_key() -> str:
    _load_env()
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "Gemini API key not found. Set GEMINI_API_KEY in your environment or in a .env file."
        )
    return api_key


def _load_model(explicit: str | None) -> str:
    _load_env()
    return explicit or os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL


def _extract_json_array(text: str) -> list[dict]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", text)
        if not match:
            raise ValueError(f"Could not parse JSON from Gemini response: {text[:200]}")
        payload = json.loads(match.group())

    if isinstance(payload, dict) and "translations" in payload:
        payload = payload["translations"]
    if not isinstance(payload, list):
        raise ValueError("Gemini response must be a JSON array of translations.")
    return payload


def _parse_retry_seconds(error: ClientError) -> float:
    message = str(error)
    match = re.search(r"retry in ([0-9]+(?:\.[0-9]+)?)s", message, re.IGNORECASE)
    if match:
        return float(match.group(1)) + 1.0

    details = getattr(error, "details", None)
    if isinstance(details, dict):
        for item in details.get("details", []):
            if item.get("@type", "").endswith("RetryInfo"):
                delay = item.get("retryDelay", "")
                if isinstance(delay, str) and delay.endswith("s"):
                    return float(delay[:-1]) + 1.0
    return 60.0


class GeminiTranslator:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        target_language: str = "English",
        batch_size: int = BATCH_SIZE,
    ) -> None:
        self.api_key = api_key or _load_api_key()
        self.model = _load_model(model)
        self.target_language = target_language
        self.batch_size = batch_size
        self.client = genai.Client(api_key=self.api_key)

    def _build_prompt(self, entries: list[dict]) -> str:
        compact = json.dumps(entries, ensure_ascii=False, separators=(",", ":"))
        return (
            f"Translate manga/comic dialogue to {self.target_language}. "
            "Keep tone natural for speech bubbles. Keep names consistent. "
            'Return JSON array only: [{"index":0,"translation":"..."}]. '
            f"Entries: {compact}"
        )

    def _generate_with_retry(self, prompt: str, model: str) -> str:
        last_error: ClientError | None = None
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.3,
                        response_mime_type="application/json",
                    ),
                )
                return response.text or ""
            except ClientError as error:
                last_error = error
                if error.status_code != 429 or attempt == MAX_RETRIES - 1:
                    raise
                wait = _parse_retry_seconds(error)
                LOG.info("[translate] Rate limited on %s, retrying in %.0fs...", model, wait)
                time.sleep(wait)
        if last_error:
            raise last_error
        return ""

    def _translate_batch(self, entries: list[dict], model: str) -> dict[int, str]:
        text = self._generate_with_retry(self._build_prompt(entries), model)
        translations = _extract_json_array(text)
        result: dict[int, str] = {}
        for item in translations:
            if not isinstance(item, dict):
                continue
            index = item.get("index")
            translation = item.get("translation", "")
            if index is None:
                continue
            result[int(index)] = str(translation).strip()
        return result

    def translate_texts(self, entries: list[dict]) -> dict[int, str]:
        if not entries:
            return {}

        models_to_try = [self.model] + [m for m in FALLBACK_MODELS if m != self.model]
        all_results: dict[int, str] = {}
        batches = [
            entries[i : i + self.batch_size]
            for i in range(0, len(entries), self.batch_size)
        ]

        for batch_idx, batch in enumerate(batches):
            if batch_idx > 0:
                time.sleep(BATCH_DELAY_SEC)

            LOG.info(
                "[translate] Batch %d/%d (%d blocks, model=%s)",
                batch_idx + 1,
                len(batches),
                len(batch),
                self.model,
            )

            translated = False
            for model in models_to_try:
                try:
                    all_results.update(self._translate_batch(batch, model))
                    self.model = model
                    translated = True
                    break
                except ClientError as error:
                    if error.status_code == 429:
                        LOG.info("[translate] %s quota/rate limit hit, trying next model...", model)
                        continue
                    raise

            if not translated:
                raise RuntimeError(
                    "All Gemini models hit rate/quota limits. "
                    "Wait ~1 minute and retry, or set GEMINI_MODEL in .env "
                    f"(tried: {', '.join(models_to_try)}). "
                    "See https://ai.google.dev/gemini-api/docs/rate-limits"
                )

        return all_results

    def translate_blocks(self, text_blocks: list[TextBlock]) -> list[TextBlock]:
        entries = []
        for i, block in enumerate(text_blocks):
            source = block.get_text()
            if not source:
                block.translation = ""
                continue
            entries.append({"index": i, "text": source})

        if not entries:
            return text_blocks

        LOG.info("[translate] Translating %d blocks to %s...", len(entries), self.target_language)
        translations = self.translate_texts(entries)
        for i, block in enumerate(text_blocks):
            if block.get_text():
                block.translation = translations.get(i, "")
                block.target_lang = self.target_language
        return text_blocks
