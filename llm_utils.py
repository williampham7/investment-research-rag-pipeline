"""
Shared Gemini calling infrastructure used by structured_extract.py and qa.py:
client creation, schema-validated structured-output calls, and retry/backoff
that treats a per-day quota violation (won't clear within this run) as
distinct from a transient error (will clear on retry).
"""

import logging
import os
import random
import time
from typing import TypeVar

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from pydantic import BaseModel

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.5-flash-lite"  # generous free-tier daily quota; gemini-3.5-flash caps at 20 req/day
MAX_RETRIES = 5
BASE_BACKOFF_SECONDS = 20.0
MAX_BACKOFF_SECONDS = 180.0

ModelT = TypeVar("ModelT", bound=BaseModel)


class DailyQuotaExhausted(Exception):
    """Raised when the API reports a per-day (not per-minute) quota violation --
    retrying within the same run cannot possibly help, so callers should stop
    rather than burn minutes retrying against the same wall."""


def get_client(api_key_env: tuple[str, ...] = ("GEMINI_API_KEY", "GOOGLE_API_KEY")) -> genai.Client:
    api_key = next((os.environ[k] for k in api_key_env if os.environ.get(k)), None)
    if not api_key:
        raise RuntimeError(
            f"No API key found in {api_key_env}. Get a free key at https://aistudio.google.com/apikey"
        )
    return genai.Client(api_key=api_key)


def _is_daily_quota_error(e: "genai_errors.APIError") -> bool:
    if e.code != 429:
        return False
    try:
        violations = next(
            d["violations"]
            for d in e.details.get("error", {}).get("details", [])
            if d.get("@type", "").endswith("QuotaFailure")
        )
        return any("perday" in v.get("quotaId", "").lower() for v in violations)
    except (AttributeError, KeyError, StopIteration, TypeError):
        return False


def _call_once(
    client: genai.Client,
    model: str,
    contents: str,
    response_schema: type[ModelT],
    system_instruction: str | None,
    temperature: float,
) -> ModelT:
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=genai_types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=response_schema,
            temperature=temperature,
        ),
    )
    if response.parsed is not None:
        return response.parsed
    # Fall back to manual validation if the SDK couldn't auto-parse (e.g. a
    # schema-adjacent but not identical response).
    if not response.text:
        raise ValueError(
            f"Empty response (finish reason on candidates: "
            f"{[c.finish_reason for c in (response.candidates or [])]})"
        )
    return response_schema.model_validate_json(response.text)


def generate_structured(
    client: genai.Client,
    model: str,
    contents: str,
    response_schema: type[ModelT],
    system_instruction: str | None = None,
    temperature: float = 0.1,
    max_retries: int = MAX_RETRIES,
) -> ModelT:
    """Structured-output call with retry/backoff. Raises DailyQuotaExhausted
    immediately (no retry) on a per-day quota violation; retries everything
    else (transient 5xx/429/network/parse hiccups) with exponential backoff."""
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return _call_once(client, model, contents, response_schema, system_instruction, temperature)
        except genai_errors.APIError as e:
            last_error = e
            if e.code in (400, 403, 404):
                logger.error("Non-retryable API error (%s): %s", e.code, e)
                raise
            if _is_daily_quota_error(e):
                logger.error("Daily quota exhausted for model %s: %s", model, e)
                raise DailyQuotaExhausted(str(e)) from e
            backoff = min(BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)), MAX_BACKOFF_SECONDS)
            backoff += random.uniform(0, backoff * 0.25)
            logger.warning(
                "API error (%s) on attempt %d/%d, retrying in %.0fs: %s",
                e.code, attempt, max_retries, backoff, e,
            )
            time.sleep(backoff)
        except Exception as e:  # noqa: BLE001 - network/JSON/validation hiccups all retry the same way
            last_error = e
            backoff = min(BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)), MAX_BACKOFF_SECONDS) / 2
            logger.warning(
                "Error on attempt %d/%d, retrying in %.0fs: %s", attempt, max_retries, backoff, e
            )
            time.sleep(backoff)
    raise RuntimeError(f"Exhausted {max_retries} retries") from last_error
