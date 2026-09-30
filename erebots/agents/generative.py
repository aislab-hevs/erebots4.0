"""Generative layer for interacting with LLM models via OpenAI-compatible APIs."""

from __future__ import annotations

import contextvars
import logging
from collections.abc import AsyncGenerator

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from erebots.config import settings

logger = logging.getLogger(__name__)

# (completion_tokens, prompt_tokens) of the most recent LLM call, stored per
# asyncio task: agents are process-wide singletons serving concurrent requests,
# so an instance attribute would let one turn's trace read another turn's
# numbers.
_last_usage: contextvars.ContextVar[tuple[int, int]] = contextvars.ContextVar(
    "erebots_llm_usage", default=(0, 0)
)

# Stands in for content a text-only deployment had to drop, so the model still
# receives a coherent turn and can tell the user why it cannot look.
_IMAGE_DROPPED = (
    "[An image was attached. This assistant is running on a text-only model and "
    "cannot see it. Say so plainly and ask the user to describe it in words.]"
)


def _strip_image_parts(messages: list[dict]) -> list[dict]:
    """Remove image parts from every message, for a text-only deployment.

    Multimodal content arrives as a list of typed parts. Text parts are kept and
    flattened back to a plain string, which every OpenAI-compatible backend
    accepts; a turn left with no text at all becomes the placeholder, since an
    empty user message is what makes a model answer nonsense.
    """
    cleaned: list[dict] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            cleaned.append(message)
            continue
        texts = [
            part["text"]
            for part in content
            if isinstance(part, dict) and part.get("type") == "text" and part.get("text")
        ]
        dropped = len(content) - len(texts)
        text = "\n".join(texts).strip()
        if dropped:
            text = f"{text}\n\n{_IMAGE_DROPPED}".strip() if text else _IMAGE_DROPPED
        cleaned.append({**message, "content": text})
    return cleaned


class GenerativeLayer:
    """LLM wrapper with streaming and single-response modes.

    Handles communication with the OpenAI-compatible API, system prompt
    management, token usage tracking, and — when ``settings.enable_parser_retry``
    is on — an opt-in retry that flattens the prompt on backend chat-template
    parser errors (vLLM/Qwen).
    """

    def __init__(
        self,
        api_key: str,
        api_base_url: str,
        model: str,
        system_prompt: str,
        thinking: bool = False,
    ):
        """Initialize the generative layer with an OpenAI client and model configuration.

        Args:
            api_key: API key for LLM service authentication.
            api_base_url: Base URL of the API endpoint.
            model: LLM model identifier to use.
            system_prompt: Default system prompt to prepend to messages.
            thinking: If True, enables the model's explicit reasoning mode.
        """
        self.client = AsyncOpenAI(api_key=api_key, base_url=api_base_url)
        self.model = model
        self.system_prompt = system_prompt
        self.thinking = thinking
        self.enable_parser_retry = settings.enable_parser_retry
        # Bumped by stop(); streams compare it against the value they captured
        # at start, so stopping never leaks into streams started afterwards.
        self._stop_epoch = 0

    @property
    def last_completion_tokens(self) -> int:
        """Completion tokens of the current task's most recent call."""
        return _last_usage.get()[0]

    @property
    def last_prompt_tokens(self) -> int:
        """Prompt tokens of the current task's most recent call."""
        return _last_usage.get()[1]

    def _stringify_content(self, content) -> str:
        """Convert message content to a string, handling multiple formats.

        Supports plain text content, multi-modal block lists (text and images),
        and None values.

        Args:
            content: Message content as a string, list of dictionaries, or None.

        Returns:
            Text representation of the content with extra whitespace removed.
        """
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text" and item.get("text"):
                    parts.append(str(item["text"]).strip())
                elif item.get("type") == "image_url":
                    parts.append("[Image provided]")
            return "\n".join(part for part in parts if part).strip()
        if content is None:
            return ""
        return str(content).strip()

    def _prepend_system(
        self,
        messages: list[dict],
        system_prompt: str | None = None,
    ) -> list[dict]:
        """Prepend the system prompt to the message list.

        If the first message already has role 'system', the prompt is merged
        with the existing content. If the prompt exactly matches the first
        system message's content, the list is returned unchanged.

        Args:
            messages: List of conversation messages.
            system_prompt: Custom system prompt. If None, uses the default.

        Returns:
            New message list with the system prompt correctly inserted.
        """
        prompt = self.system_prompt if system_prompt is None else system_prompt
        if not prompt:
            return messages
        if not messages:
            return [{"role": "system", "content": prompt}]

        first_message = messages[0]
        if first_message.get("role") != "system":
            return [{"role": "system", "content": prompt}] + messages

        if first_message.get("content") == prompt:
            return messages

        # Merge the system prompt with the existing first message content
        merged_first = dict(first_message)
        existing_content = first_message.get("content", "")
        existing_text = existing_content if isinstance(existing_content, str) else str(existing_content)
        if existing_text.strip():
            merged_first["content"] = f"{prompt}\n\nAdditional context:\n{existing_text}"
        else:
            merged_first["content"] = prompt
        return [merged_first] + messages[1:]

    def _uses_parser_retry(self, exc: Exception) -> bool:
        """Check if the exception is caused by a chat template parser error.

        Checks whether the error message indicates automatic parser generation
        failure, in which case a retry with a flattened prompt is possible.

        Args:
            exc: Exception raised during the API call.

        Returns:
            True if the error is recoverable via retry with a flattened prompt.
        """
        text = str(exc)
        return (
            "Unable to generate parser for this template" in text
            or "Automatic parser generation failed" in text
            or "No user query found in messages" in text
        )

    def _flatten_messages_for_retry(self, messages: list[dict]) -> list[dict]:
        """Flatten the conversation into a single user message for retry.

        Combines all system and conversation messages into a single message
        with role 'user', useful as a fallback when the backend cannot handle
        the multi-message format.

        Args:
            messages: List of original conversation messages.

        Returns:
            List containing a single dictionary with role 'user' and flattened content.
        """
        transcript: list[str] = []
        system_sections: list[str] = []

        for message in messages:
            role = str(message.get("role", "user")).strip().lower() or "user"
            content = self._stringify_content(message.get("content", ""))
            if not content:
                continue
            if role == "system":
                system_sections.append(content)
                continue
            transcript.append(f"{role.upper()}:\n{content}")

        flattened_parts: list[str] = []
        if system_sections:
            flattened_parts.append("Instructions:\n" + "\n\n".join(system_sections))
        if transcript:
            flattened_parts.append("Conversation:\n" + "\n\n".join(transcript))
        else:
            flattened_parts.append("Conversation:\nUSER:\nPlease answer the request.")
        flattened_parts.append(
            "Respond as the assistant to the latest user request in the conversation above."
        )

        return [{"role": "user", "content": "\n\n".join(flattened_parts)}]

    def _build_kwargs(
        self,
        messages: list[dict],
        stream: bool = False,
        include_thinking_flag: bool = True,
        tools: list[dict] | None = None,
    ) -> dict:
        """Build the keyword arguments dictionary for the completion API call.

        Args:
            messages: List of messages to send to the model.
            stream: If True, enables response streaming.
            include_thinking_flag: If True and thinking is disabled, adds the
                ``enable_thinking: False`` flag to the request body.
            tools: Optional list of tools in OpenAI function-calling format.

        Returns:
            Dictionary with arguments ready for ``client.chat.completions.create()``.
        """
        if not settings.multimodal_enabled:
            messages = _strip_image_parts(messages)
        kwargs: dict = {"model": self.model, "messages": messages}
        if stream:
            kwargs["stream"] = True
            # Without this the usage block never arrives on a streamed response,
            # and prompt size — the only stable measure of what a turn costs —
            # stays invisible. Supported by vLLM and llama.cpp.
            kwargs["stream_options"] = {"include_usage": True}
        if tools:
            kwargs["tools"] = tools
        if not self.thinking and include_thinking_flag:
            # Two spellings of the same request, because the backends disagree
            # on where it goes. vLLM reads a top-level ``enable_thinking``;
            # llama.cpp reads ``chat_template_kwargs``, which is what actually
            # reaches the model's chat template. Sending only the first left
            # ``thinking=False`` with no effect on llama.cpp — measured on one
            # classification prompt, 10.4s with reasoning against 0.4s without.
            # Each backend ignores the spelling it does not know.
            kwargs["extra_body"] = {
                "enable_thinking": False,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        return kwargs

    def _record_usage(self, obj) -> None:
        """Store the usage block of a response or of a streamed usage chunk.

        Streamed usage arrives in a final chunk that carries no choices, so it
        is read where those chunks are skipped.
        """
        usage = getattr(obj, "usage", None)
        if usage:
            _last_usage.set(
                (
                    getattr(usage, "completion_tokens", 0) or 0,
                    getattr(usage, "prompt_tokens", 0) or 0,
                )
            )

    async def chat_stream(
        self,
        messages: list[dict],
        system_prompt: str | None = None,
    ) -> AsyncGenerator[str]:
        """Generate the model's response in streaming mode (async generator).

        Some backends (e.g. Qwen via vLLM) split the thinking output into a
        separate ``delta.reasoning_content`` field. To keep downstream
        parsing uniform, those chunks are wrapped here in inline
        ``<think>...</think>`` tags so a single :class:`ThinkSplitter`
        can consume the stream.

        Args:
            messages: The conversation so far.
            system_prompt: Overrides the layer's own system prompt for this call.

        Yields:
            Text chunks, with reasoning delimited inline by ``<think>`` tags.
        """
        epoch = self._stop_epoch
        full_messages = self._prepend_system(messages, system_prompt=system_prompt)
        kwargs = self._build_kwargs(full_messages, stream=True)
        try:
            stream = await self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            if not self.enable_parser_retry or not self._uses_parser_retry(exc):
                raise
            logger.warning("Retrying streaming generation with flattened prompt due to backend chat-template parser error")
            fallback_text = await self.chat(messages, system_prompt=system_prompt)
            if fallback_text:
                yield fallback_text
            return

        in_reasoning = False
        try:
            async for chunk in stream:
                if self._stop_epoch != epoch:
                    await stream.close()
                    break
                if not chunk.choices:
                    self._record_usage(chunk)
                    continue
                delta = chunk.choices[0].delta
                if delta is None:
                    continue
                reasoning = getattr(delta, "reasoning_content", None)
                content = getattr(delta, "content", None)
                if reasoning:
                    if not in_reasoning:
                        yield "<think>"
                        in_reasoning = True
                    yield reasoning
                if content:
                    if in_reasoning:
                        yield "</think>"
                        in_reasoning = False
                    yield content
        finally:
            # Whatever ends the loop — exhaustion, a stop, the consumer being
            # cancelled mid-turn — the HTTP connection is released here.
            await stream.close()
        if in_reasoning:
            yield "</think>"

    async def stream_with_tools(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system_prompt: str | None = None,
    ):
        """Stream one turn, tool-aware.

        Yields ``("text", chunk)`` as content/reasoning arrives (reasoning wrapped
        in ``<think>`` inline), then ``("tools", calls)`` if the model requested
        tool calls instead. This keeps token streaming for tool-using agents: a
        tool-call turn produces no text, the answer turn streams.

        Args:
            messages: The conversation so far.
            tools: Tool specifications in OpenAI format, or ``None`` to call
                the model without a ``tools`` field at all.
            system_prompt: Overrides the layer's own system prompt for this call.

        Yields:
            ``("text", chunk)`` pairs, then a single ``("tools", calls)`` if
            the model requested tool calls.
        """
        epoch = self._stop_epoch
        full = self._prepend_system(messages, system_prompt=system_prompt)
        stream = await self.client.chat.completions.create(
            **self._build_kwargs(full, stream=True, tools=tools)
        )
        frags: dict[int, dict] = {}
        in_reasoning = False
        try:
            async for chunk in stream:
                if self._stop_epoch != epoch:
                    await stream.close()
                    break
                if not chunk.choices:
                    self._record_usage(chunk)
                    continue
                delta = chunk.choices[0].delta
                if delta is None:
                    continue
                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning:
                    if not in_reasoning:
                        yield ("text", "<think>")
                        in_reasoning = True
                    yield ("text", reasoning)
                if delta.content:
                    if in_reasoning:
                        yield ("text", "</think>")
                        in_reasoning = False
                    yield ("text", delta.content)
                for tc in delta.tool_calls or []:
                    f = frags.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                    if tc.id:
                        f["id"] = tc.id
                    if tc.function and tc.function.name:
                        f["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        f["args"] += tc.function.arguments
        finally:
            await stream.close()
        if in_reasoning:
            yield ("text", "</think>")
        if frags:
            yield ("tools", [frags[i] for i in sorted(frags)])

    async def chat(
        self,
        messages: list[dict],
        system_prompt: str | None = None,
        tools: list[dict] | None = None,
    ) -> ChatCompletion | str:
        """Generate a single (non-streaming) response from the model (async)."""
        full_messages = self._prepend_system(messages, system_prompt=system_prompt)
        kwargs = self._build_kwargs(full_messages, tools=tools)
        try:
            response = await self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            if not self.enable_parser_retry or not self._uses_parser_retry(exc):
                raise
            logger.warning(
                "Retrying generation with flattened prompt due to backend chat-template parser error"
            )
            flattened_messages = self._flatten_messages_for_retry(full_messages)
            response = await self.client.chat.completions.create(
                **self._build_kwargs(
                    flattened_messages,
                    include_thinking_flag=False,
                )
            )
        self._record_usage(response)
        if tools:
            return response
        content = response.choices[0].message.content or ""
        return content.strip()

    def stop(self):
        """Stop the streams currently in flight.

        Their loops observe the epoch change and close the HTTP stream; streams
        started after this call are unaffected. This is instance-wide: every
        stream this layer is serving stops, whoever it is for. A frontend that
        serves several users must cancel the turn's task instead (the stream
        loops release their connection on cancellation too).
        """
        self._stop_epoch += 1
