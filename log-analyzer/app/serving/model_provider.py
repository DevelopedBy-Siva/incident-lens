import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.serving.model_runtime import ModelRuntimeSession

DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"


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


class GroqProvider(ModelProvider):
    """Current Groq behavior behind the provider-neutral runtime contract."""

    name = "groq"
    runtime_type = "remote_provider"
    default_model = DEFAULT_GROQ_MODEL

    def capabilities(self) -> dict[str, bool]:
        return {
            "text_generation": True,
            "structured_output": True,
            "tool_calling": True,
            "model_fallbacks": True,
        }

    def model_candidates(self) -> tuple[str, ...]:
        primary = os.getenv("GROQ_MODEL", "").strip()
        fallbacks = [
            model.strip()
            for model in os.getenv("GROQ_MODEL_FALLBACKS", "").split(",")
            if model.strip()
        ]
        models = [primary] if primary else []
        models.append(self.default_model)
        models.extend(fallbacks)
        return tuple(dict.fromkeys(models))

    def is_available(self, session: "ModelRuntimeSession") -> bool:
        return bool(self._api_keys(session))

    def complete(
        self,
        session: "ModelRuntimeSession",
        *,
        model: str,
        messages: Any,
        temperature: float,
    ) -> ProviderResponse:
        from langchain_groq import ChatGroq

        api_key = self._select_api_key(session)
        client = ChatGroq(model=model, temperature=temperature, api_key=api_key)
        response = client.invoke(messages)
        return ProviderResponse(
            content=str(response.content),
            usage_metadata=getattr(response, "usage_metadata", None),
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
        from groq import Groq

        client = Groq(api_key=self._select_api_key(session))
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        message = response.choices[0].message
        tool_calls = tuple(
            ProviderToolCall(
                id=tool_call.id,
                name=tool_call.function.name,
                arguments=tool_call.function.arguments,
            )
            for tool_call in (message.tool_calls or [])
        )
        return ProviderResponse(
            content=message.content or "",
            tool_calls=tool_calls,
            usage_metadata=self._raw_usage(response),
        )

    @staticmethod
    def _api_keys(session: "ModelRuntimeSession") -> list[str]:
        if session._provider_api_key:
            return [session._provider_api_key]
        return [
            os.getenv(variable, "").strip()
            for variable in ("GROQ_API_KEY", "GROQ_API_KEY_2", "GROQ_API_KEY_3")
            if os.getenv(variable, "").strip()
        ]

    def _select_api_key(self, session: "ModelRuntimeSession") -> str:
        keys = self._api_keys(session)
        if not keys:
            raise ProviderUnavailableError("No Groq API key configured")
        return random.choice(keys)

    @staticmethod
    def _raw_usage(response) -> dict[str, int] | None:
        usage = getattr(response, "usage", None)
        if not usage:
            return None
        return {
            "input_tokens": getattr(usage, "prompt_tokens", 0),
            "output_tokens": getattr(usage, "completion_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0),
        }
