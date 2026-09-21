import json
import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from app.serving.local_model import AdapterLoadError, LocalModelCache
from app.shared.model_config import (
    RuntimeModelSettings,
    configured_runtime_settings,
)

if TYPE_CHECKING:
    from app.serving.model_runtime import ModelRuntimeSession


class ProviderUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProviderToolCall:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class ProviderResponse:
    content: str
    tool_calls: tuple[ProviderToolCall, ...] = ()
    usage_metadata: dict[str, Any] | None = None


class ModelProvider(ABC):
    name: str
    runtime_type: str
    default_model: str

    def initialize(self, base_model: str) -> None:
        return None

    @abstractmethod
    def model_candidates(self) -> tuple[str, ...]:
        raise NotImplementedError

    @abstractmethod
    def capabilities(self) -> dict[str, bool]:
        raise NotImplementedError

    @abstractmethod
    def is_available(self, session: "ModelRuntimeSession") -> bool:
        raise NotImplementedError

    @abstractmethod
    def complete(
        self,
        session: "ModelRuntimeSession",
        *,
        model: str,
        messages: Any,
        temperature: float,
    ) -> ProviderResponse:
        raise NotImplementedError

    @abstractmethod
    def complete_with_tools(
        self,
        session: "ModelRuntimeSession",
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: str | None,
        temperature: float,
        max_tokens: int,
    ) -> ProviderResponse:
        raise NotImplementedError


class LocalModelProvider(ModelProvider):
    """Run all inference through the shared local model and project adapter."""

    name = "local"
    runtime_type = "local_peft"
    _tool_call_pattern = re.compile(
        r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
        re.DOTALL,
    )

    def __init__(
        self,
        settings: RuntimeModelSettings | None = None,
        cache: LocalModelCache | None = None,
    ):
        self.settings = settings or configured_runtime_settings()
        self.default_model = self.settings.base_model
        self.cache = cache or LocalModelCache(self.settings)

    def initialize(self, base_model: str) -> None:
        if base_model != self.settings.base_model:
            raise ProviderUnavailableError(
                f"Runtime base model {base_model!r} does not match configured local "
                f"model {self.settings.base_model!r}"
            )
        self.cache.initialize()

    def model_candidates(self) -> tuple[str, ...]:
        return (self.default_model,)

    def capabilities(self) -> dict[str, bool]:
        return {
            "text_generation": True,
            "structured_output": True,
            "tool_calling": True,
            "model_fallbacks": False,
            "local_inference": True,
        }

    def is_available(self, session: "ModelRuntimeSession") -> bool:
        return bool(
            self.cache.initialized
            and session.active_artifact
            and session.adapter_path
            and not session.validation_warnings
        )

    def complete(
        self,
        session: "ModelRuntimeSession",
        *,
        model: str,
        messages: Any,
        temperature: float,
    ) -> ProviderResponse:
        return self._generate(
            session,
            model=model,
            messages=self._normalize_messages(messages),
            tools=None,
            temperature=temperature,
            max_tokens=self.settings.max_new_tokens,
        )

    def complete_with_tools(
        self,
        session: "ModelRuntimeSession",
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: str | None,
        temperature: float,
        max_tokens: int,
    ) -> ProviderResponse:
        response = self._generate(
            session,
            model=model,
            messages=self._normalize_messages(messages),
            tools=tools if tool_choice != "none" else None,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content, tool_calls = self._parse_tool_calls(response.content)
        return ProviderResponse(
            content=content,
            tool_calls=tool_calls,
            usage_metadata=response.usage_metadata,
        )

    def _generate(
        self,
        session: "ModelRuntimeSession",
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        temperature: float,
        max_tokens: int,
    ) -> ProviderResponse:
        if model != self.default_model:
            raise ProviderUnavailableError(
                f"Local runtime only serves configured model {self.default_model!r}"
            )
        artifact = self._require_artifact(session)
        generated = self.cache.generate(
            project_id=session.project_id,
            artifact_id=artifact.id,
            adapter_path=artifact.adapter_path,
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_new_tokens=max_tokens,
        )
        return ProviderResponse(
            content=generated.content,
            usage_metadata={
                "input_tokens": generated.input_tokens,
                "output_tokens": generated.output_tokens,
                "total_tokens": generated.input_tokens + generated.output_tokens,
            },
        )

    @staticmethod
    def _require_artifact(session: "ModelRuntimeSession"):
        if session.validation_warnings:
            raise AdapterLoadError("; ".join(session.validation_warnings))
        artifact = session.active_artifact
        if artifact is None:
            raise AdapterLoadError(
                f"Project {session.project_id!r} has no READY active model artifact"
            )
        if not artifact.adapter_path:
            raise AdapterLoadError(
                f"Active artifact {artifact.id!r} has no adapter path"
            )
        if not artifact.metadata:
            raise AdapterLoadError(
                f"Active artifact {artifact.id!r} has no valid metadata"
            )
        if artifact.metadata.get("contains_adapter_weights") is not True:
            raise AdapterLoadError(
                f"Active artifact {artifact.id!r} does not declare adapter weights"
            )
        return artifact

    @classmethod
    def _parse_tool_calls(
        cls, content: str
    ) -> tuple[str, tuple[ProviderToolCall, ...]]:
        calls = []

        def replace(match):
            try:
                payload = json.loads(match.group(1))
                name = payload["name"]
                arguments = payload.get("arguments", {})
                if not isinstance(name, str) or not isinstance(arguments, dict):
                    return match.group(0)
            except (KeyError, TypeError, json.JSONDecodeError):
                return match.group(0)
            calls.append(
                ProviderToolCall(
                    id=f"local-{uuid.uuid4()}",
                    name=name,
                    arguments=json.dumps(
                        arguments,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                )
            )
            return ""

        remaining = cls._tool_call_pattern.sub(replace, content).strip()
        return remaining, tuple(calls)

    @staticmethod
    def _normalize_messages(messages: Any) -> list[dict[str, Any]]:
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]

        normalized = []
        for message in messages:
            if isinstance(message, dict):
                item = dict(message)
            else:
                role = {
                    "human": "user",
                    "ai": "assistant",
                }.get(getattr(message, "type", ""), getattr(message, "type", "user"))
                item = {
                    "role": role,
                    "content": getattr(message, "content", str(message)),
                }

            item.setdefault("content", "")
            if item.get("role") == "assistant" and item.get("tool_calls"):
                tool_calls = []
                for tool_call in item["tool_calls"]:
                    tool_call = dict(tool_call)
                    function = dict(tool_call.get("function", tool_call))
                    arguments = function.get("arguments", {})
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            arguments = {}
                    function["arguments"] = arguments
                    tool_calls.append({"type": "function", "function": function})
                item["tool_calls"] = tool_calls
            if item.get("role") == "tool":
                item.pop("tool_call_id", None)
            normalized.append(item)
        return normalized
