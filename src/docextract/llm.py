"""LLM extraction with a small local model (Ollama), constrained to a JSON schema.

Numbers are requested as STRINGS exactly as printed, so the model cannot silently reformat or "correct" them; the code parses
them afterwards. The prompt forbids computing or guessing. Even so, a 1.7B model can still invent digits, which is why its
output goes through the same validator as the rule-based extraction instead of being trusted.
"""

from __future__ import annotations

import json

import httpx
from pydantic import BaseModel, Field


class LLMRecord(BaseModel):
    currency: str | None = Field(None, description="3-letter currency code in the title")
    report_date: str | None = Field(None, description="date in the subtitle, YYYY-MM-DD")
    bid: str | None = None
    ask: str | None = None
    open: str | None = None
    previous_close: str | None = None
    high: str | None = None
    low: str | None = None
    change: str | None = None
    change_percent: str | None = Field(None, description="the value printed next to 'Change %'")
    high_52week: str | None = None
    low_52week: str | None = None
    news_count: int | None = Field(None, description="the number in parentheses after 'Relevant news'")
    headlines: list[str] = Field(default_factory=list, description="the headline text of each news item, without the source or time")


SYSTEM = ("You extract fields from the OCR text of a scanned market-snapshot report. Copy each number exactly as printed, as a string, digit for digit. "
          "Do not calculate, correct or guess. If a field is not present or is unreadable, use null. The OCR text may contain recognition errors; "
          "copy them as they are.")


class OllamaExtractor:
    def __init__(self, model: str = "qwen3:1.7b", host: str = "http://127.0.0.1:11434", num_thread: int = 3, timeout: float = 240.0):
        self.model, self.host, self.num_thread, self.timeout = model, host, num_thread, timeout

    def extract(self, ocr_text: str) -> LLMRecord:
        body = {
            "model": self.model, "stream": False, "think": False, "format": LLMRecord.model_json_schema(),
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": ocr_text}],
            "options": {"temperature": 0, "seed": 7, "num_ctx": 4096, "num_predict": 600, "num_thread": self.num_thread},
        }
        r = httpx.post(f"{self.host}/api/chat", json=body, timeout=self.timeout)
        r.raise_for_status()
        return LLMRecord.model_validate(json.loads(r.json()["message"]["content"]))
