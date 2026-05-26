"""Cloud LLM client supporting Ollama, OpenAI, and Anthropic providers.

Provides async generate() and generate_stream() methods with automatic
retry logic, structured logging, and provider-specific API formatting.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncIterator

import httpx

from app.infrastructure.observability import get_logger, metrics

logger = get_logger("llm_client")


class LLMError(Exception):
    """Raised when an LLM API call fails after retries."""

    def __init__(self, message: str, provider: str = "", status_code: int = 0) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code


class LLMClient:
    """Async LLM client supporting multiple cloud providers.

    Supports:
        - Ollama (local): POST {base_url}/api/chat
        - OpenAI-compatible: POST {base_url}/v1/chat/completions
        - Anthropic: POST {base_url}/v1/messages

    Includes 1 retry with 5-second backoff on timeout/5xx errors.
    """

    SUPPORTED_PROVIDERS = ("ollama", "openai", "anthropic")

    def __init__(
        self,
        provider: str,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout: float = 60.0,
    ) -> None:
        """Initialize the LLM client.

        Args:
            provider: One of 'ollama', 'openai', 'anthropic'.
            base_url: API base URL (e.g., http://localhost:11434).
            model: Model name/ID.
            api_key: API key for authenticated providers.
            timeout: Request timeout in seconds.

        Raises:
            ValueError: If provider is not supported.
        """
        if provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported provider '{provider}'. "
                f"Supported: {self.SUPPORTED_PROVIDERS}"
            )
        self._provider = provider
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._timeout = timeout
        self._max_retries = 1
        self._retry_backoff = 5.0

    async def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        """Generate a complete response from the LLM.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            The generated text response.

        Raises:
            LLMError: If all retries fail.
        """
        start = time.perf_counter()
        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                result = await self._call_provider(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False,
                )
                elapsed_ms = (time.perf_counter() - start) * 1000
                metrics.record_latency("llm_generate", elapsed_ms)
                metrics.increment("llm_calls_total")
                logger.info(
                    "llm_generate_complete",
                    provider=self._provider,
                    model=self._model,
                    latency_ms=round(elapsed_ms, 1),
                    response_length=len(result),
                    attempt=attempt + 1,
                )
                return result

            except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
                last_error = e
                status = getattr(e, "response", None)
                status_code = status.status_code if status else 0

                if attempt < self._max_retries and (
                    isinstance(e, httpx.TimeoutException) or status_code >= 500
                ):
                    logger.warning(
                        "llm_retry",
                        provider=self._provider,
                        attempt=attempt + 1,
                        error=str(e),
                        backoff_seconds=self._retry_backoff,
                    )
                    await asyncio.sleep(self._retry_backoff)
                else:
                    break

        metrics.increment("llm_errors_total")
        raise LLMError(
            message=f"LLM call failed after {self._max_retries + 1} attempts: {last_error}",
            provider=self._provider,
            status_code=getattr(getattr(last_error, "response", None), "status_code", 0),
        )

    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> AsyncIterator[str]:
        """Generate a streaming response from the LLM.

        Args:
            messages: List of message dicts.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Yields:
            Text chunks as they arrive from the LLM.

        Raises:
            LLMError: If the streaming call fails.
        """
        start = time.perf_counter()

        try:
            url, headers, body = self._build_request(
                messages, temperature, max_tokens, stream=True
            )

            async with httpx.AsyncClient(timeout=self._timeout) as client:
                async with client.stream("POST", url, headers=headers, json=body) as resp:
                    resp.raise_for_status()
                    async for chunk_text in self._parse_stream(resp):
                        yield chunk_text

            elapsed_ms = (time.perf_counter() - start) * 1000
            metrics.record_latency("llm_stream", elapsed_ms)
            metrics.increment("llm_stream_calls_total")

        except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
            metrics.increment("llm_errors_total")
            raise LLMError(
                message=f"LLM stream failed: {e}",
                provider=self._provider,
            ) from e

    # ------------------------------------------------------------------
    # Provider-specific formatting
    # ------------------------------------------------------------------

    def _build_request(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        stream: bool,
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        """Build provider-specific URL, headers, and request body."""
        if self._provider == "ollama":
            return self._build_ollama(messages, temperature, max_tokens, stream)
        elif self._provider == "openai":
            return self._build_openai(messages, temperature, max_tokens, stream)
        elif self._provider == "anthropic":
            return self._build_anthropic(messages, temperature, max_tokens, stream)
        else:
            raise ValueError(f"Unsupported provider: {self._provider}")

    def _build_ollama(
        self, messages: list[dict], temperature: float, max_tokens: int, stream: bool
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        url = f"{self._base_url}/api/chat"
        headers = {"Content-Type": "application/json"}
        body = {
            "model": self._model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        return url, headers, body

    def _build_openai(
        self, messages: list[dict], temperature: float, max_tokens: int, stream: bool
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        base = self._base_url.rstrip("/")
        if base.endswith("/v1"):
            url = f"{base}/chat/completions"
        else:
            url = f"{base}/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        body = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        return url, headers, body

    def _build_anthropic(
        self, messages: list[dict], temperature: float, max_tokens: int, stream: bool
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        base = self._base_url.rstrip("/")
        if base.endswith("/v1"):
            url = f"{base}/messages"
        else:
            url = f"{base}/v1/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
        }
        # Anthropic uses system message separately
        system_msg = ""
        user_messages: list[dict[str, str]] = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg += msg["content"] + "\n"
            else:
                user_messages.append(msg)

        body: dict[str, Any] = {
            "model": self._model,
            "messages": user_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        if system_msg.strip():
            body["system"] = system_msg.strip()
        return url, headers, body

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    async def _call_provider(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        stream: bool,
    ) -> str:
        """Make a non-streaming API call and extract the response text."""
        url, headers, body = self._build_request(
            messages, temperature, max_tokens, stream=False
        )

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(url, headers=headers, json=body)
            response.raise_for_status()
            data = response.json()

        return self._extract_text(data)

    def _extract_text(self, data: dict[str, Any]) -> str:
        """Extract response text from provider-specific response format."""
        if self._provider == "ollama":
            return data.get("message", {}).get("content", "")
        elif self._provider == "openai":
            choices = data.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
            return ""
        elif self._provider == "anthropic":
            content_blocks = data.get("content", [])
            texts = [b.get("text", "") for b in content_blocks if b.get("type") == "text"]
            return "".join(texts)
        return ""

    async def _parse_stream(self, response: httpx.Response) -> AsyncIterator[str]:
        """Parse SSE stream from any provider."""
        if self._provider == "ollama":
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    content = data.get("message", {}).get("content", "")
                    if content:
                        yield content
                    if data.get("done", False):
                        break
                except json.JSONDecodeError:
                    continue

        elif self._provider == "openai":
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    break
                try:
                    data = json.loads(payload)
                    delta = data.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield content
                except (json.JSONDecodeError, IndexError):
                    continue

        elif self._provider == "anthropic":
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                try:
                    data = json.loads(payload)
                    event_type = data.get("type", "")
                    if event_type == "content_block_delta":
                        delta = data.get("delta", {})
                        text = delta.get("text", "")
                        if text:
                            yield text
                    elif event_type == "message_stop":
                        break
                except json.JSONDecodeError:
                    continue

    @property
    def provider(self) -> str:
        """Return the provider name."""
        return self._provider

    @property
    def model(self) -> str:
        """Return the model name."""
        return self._model
