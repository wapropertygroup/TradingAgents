import logging
import os
import re
import threading
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI

from .base_client import BaseLLMClient, normalize_content
from .validators import validate_model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Daily-quota fallback
#
# Gemini enforces a per-model, per-project *daily* request cap (250/day for
# gemini-3.1-pro on the current tier). A deep run makes roughly 22 model calls,
# so a day of Pro-on-everything exhausts it after about eleven runs and every
# later run then dies mid-debate with
#     429 RESOURCE_EXHAUSTED ... generate_requests_per_model_per_day
# having already spent whatever it spent. Because the limit is daily, retrying
# the same model is pointless -- the API's own retryDelay comes back in hours.
#
# Each model id has its own bucket, so the recovery is to continue on a
# different model rather than to wait. The run degrades in quality instead of
# failing outright, which for a report that took ten minutes is plainly the
# better trade.
# ---------------------------------------------------------------------------

# Order is deliberate and configurable: strongest first, and each entry is a
# separate daily bucket. gemini-3.7-flash is newer than 3.5, so a caller who
# prefers it can simply reorder this list without a deploy.
DEFAULT_FALLBACK_MODELS = "gemini-3.5-flash,gemini-3.7-flash,gemini-2.5-flash"

# Models exhausted in this process. Per-process is the right scope: a run is one
# subprocess, so this stops the remaining ~20 calls of a run from each
# rediscovering the same dead quota, and it resets naturally on the next run
# (which costs one probe call to learn the quota is back).
_exhausted: set[str] = set()
_exhausted_lock = threading.Lock()


def _fallback_models() -> list[str]:
    raw = os.environ.get("TRADINGAGENTS_GOOGLE_FALLBACK_MODELS")
    if raw is None:
        raw = DEFAULT_FALLBACK_MODELS
    return [m.strip() for m in raw.split(",") if m.strip()]


def _is_daily_quota_error(exc: BaseException) -> bool:
    """Whether this failure means "this model is out of quota", not "retry me".

    Matched on the message because langchain_google_genai wraps the provider
    error in its own ChatGoogleGenerativeAIError rather than surfacing
    google.api_core.exceptions.ResourceExhausted, so there is no exception type
    to catch. Both markers are required for a *daily* cap; a per-minute 429 also
    says RESOURCE_EXHAUSTED but is worth retrying on the same model, and that
    case is already handled by max_retries.
    """
    text = str(exc)
    if "RESOURCE_EXHAUSTED" not in text and "429" not in text:
        return False
    return ("per_day" in text or "PerDay" in text
            or "generate_requests_per_model_per_day" in text)


def _supports_thinking_level(model: str) -> bool:
    """``thinking_level`` is a Gemini 3.x parameter.

    The 2.5 line takes the integer ``thinking_budget`` and rejects the string
    outright with 400 INVALID_ARGUMENT, so carrying the caller's thinking_level
    into a 2.5 fallback would make that rung of the chain fail on every call --
    turning a graceful degradation into a second, more confusing outage.
    """
    return bool(re.match(r"^gemini-3", model.strip(), re.IGNORECASE))


def _mark_exhausted(model: str) -> None:
    with _exhausted_lock:
        first = model not in _exhausted
        _exhausted.add(model)
    if first:
        logger.warning(
            "Model %s has hit its daily request quota; falling back for the "
            "rest of this run", model)


def _is_exhausted(model: str) -> bool:
    with _exhausted_lock:
        return model in _exhausted


class NormalizedChatGoogleGenerativeAI(ChatGoogleGenerativeAI):
    """ChatGoogleGenerativeAI with normalized content output.

    Gemini 3 models return content as list of typed blocks.
    This normalizes to string for consistent downstream handling.

    Also falls back to another model when this one's daily quota is exhausted;
    see the module comment. The override is on ``_generate`` rather than
    ``invoke`` on purpose: the analysts call ``llm.bind_tools(tools)``, and a
    bound runnable routes through ``_generate`` as well, so one override covers
    both the plain and the tool-calling paths.
    """

    def invoke(self, input, config=None, **kwargs):
        return normalize_content(super().invoke(input, config, **kwargs))

    def _sibling(self, model: str) -> ChatGoogleGenerativeAI:
        """A plain client for ``model``, carrying this one's configuration.

        Built fresh rather than with ``model_copy(update=...)``, which does not
        clear ``thinking_level`` -- it is not a declared field on the langchain
        model, so an update is silently ignored and a 2.5 fallback would still
        be sent the parameter it rejects.

        Deliberately a plain ChatGoogleGenerativeAI, not this subclass, so a
        sibling cannot start its own fallback chain and recurse; the loop in
        :meth:`_generate` owns the sequencing. Normalisation still happens,
        because it is applied by the primary's ``invoke``.
        """
        kwargs: dict[str, Any] = {"model": model}
        if self.google_api_key is not None:
            kwargs["google_api_key"] = self.google_api_key
        for name in ("temperature", "timeout", "max_retries", "base_url",
                     "callbacks", "http_client", "http_async_client"):
            value = getattr(self, name, None)
            if value is not None:
                kwargs[name] = value
        level = getattr(self, "thinking_level", None)
        if level and _supports_thinking_level(model):
            kwargs["thinking_level"] = level
        return ChatGoogleGenerativeAI(**kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        primary = self.model
        candidates = [primary] + [m for m in _fallback_models() if m != primary]
        last_error: BaseException | None = None
        tried: list[str] = []

        for model in candidates:
            if _is_exhausted(model):
                continue
            tried.append(model)
            try:
                if model == primary:
                    return super()._generate(messages, stop, run_manager, **kwargs)
                logger.info("Retrying on %s after %s ran out of daily quota",
                            model, primary)
                return self._sibling(model)._generate(
                    messages, stop, run_manager, **kwargs)
            except Exception as exc:  # noqa: BLE001 - re-raised unless it is a cap
                if not _is_daily_quota_error(exc):
                    raise
                _mark_exhausted(model)
                last_error = exc

        if last_error is not None:
            logger.error("Every candidate model is out of daily quota (%s)",
                         ", ".join(tried) or "none available")
            raise last_error
        # Everything was already known-exhausted, so nothing was attempted.
        raise RuntimeError(
            "All configured Gemini models have exhausted their daily quota: "
            + ", ".join(candidates)
        )


class GoogleClient(BaseLLMClient):
    """Client for Google Gemini models."""

    def __init__(self, model: str, base_url: str | None = None, **kwargs):
        super().__init__(model, base_url, **kwargs)

    def get_llm(self) -> Any:
        """Return configured ChatGoogleGenerativeAI instance."""
        self.warn_if_unknown_model()
        llm_kwargs = {"model": self.model}

        if self.base_url:
            llm_kwargs["base_url"] = self.base_url

        for key in ("timeout", "max_retries", "temperature", "callbacks", "http_client", "http_async_client"):
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        # Unified api_key maps to provider-specific google_api_key
        google_api_key = self.kwargs.get("api_key") or self.kwargs.get("google_api_key")
        if google_api_key:
            llm_kwargs["google_api_key"] = google_api_key

        # Gemini 3.x takes the string ``thinking_level`` (the integer
        # ``thinking_budget`` was for the now-retired 2.5 line). Pro accepts
        # low/high; Flash also accepts minimal/medium — so map an unsupported
        # "minimal" on Pro to the nearest level it does accept.
        thinking_level = self.kwargs.get("thinking_level")
        if thinking_level:
            if "pro" in self.model.lower() and thinking_level == "minimal":
                thinking_level = "low"
            llm_kwargs["thinking_level"] = thinking_level

        return NormalizedChatGoogleGenerativeAI(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model for Google."""
        return validate_model("google", self.model)
