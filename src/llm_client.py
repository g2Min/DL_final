import json
import os
from typing import TypeVar

import requests as _requests
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError


T = TypeVar("T", bound=BaseModel)


class StructuredLLMClient:
    """
    Pydantic 스키마에 맞는 JSON을 생성하는 LLM 클라이언트.

    두 가지 백엔드 지원:
    - HuggingFace Inference API (기본): base_url 미설정 시
    - OpenAI 호환 API (vLLM, Ollama 등): base_url 설정 시
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-32B",
        provider: str = "auto",
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        load_dotenv()
        self.model_id = model_id

        if base_url:
            # 로컬 vLLM / Ollama 등 OpenAI 호환 서버
            self._base_url = base_url.rstrip("/")
            self._api_key = api_key or "dummy"
            self._backend = "openai_compat"
        else:
            # HuggingFace Inference Providers
            from huggingface_hub import InferenceClient

            token = os.getenv("HF_TOKEN")
            if not token:
                raise RuntimeError(
                    "HF_TOKEN이 없습니다. "
                    "프로젝트 루트의 .env 파일을 확인하세요."
                )
            self._hf_client = InferenceClient(
                provider=provider,
                api_key=token,
            )
            self._backend = "hf"

    def _build_system_prompt(
        self,
        system_prompt: str,
        schema_text: str,
    ) -> str:
        return f"""
{system_prompt}

반드시 유효한 JSON 객체 하나만 반환하세요.
마크다운 코드 블록, 설명, 부가 문장은 출력하지 마세요.

아래 JSON Schema를 정확하게 따르세요.
- schema에 정의되지 않은 필드는 추가하지 마세요.
- required 필드는 절대 생략하지 마세요.
- 각 필드의 데이터 타입을 정확히 지키세요.
- 모든 필드 이름을 schema와 동일하게 작성하세요.

JSON Schema:
{schema_text}
""".strip()

    def _call_openai_compat(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> str:
        url = f"{self._base_url}/chat/completions"
        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        resp = _requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _call_hf(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> str:
        response = self._hf_client.chat_completion(
            model=self.model_id,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> T:
        schema_text = json.dumps(
            response_model.model_json_schema(),
            ensure_ascii=False,
            indent=2,
        )

        messages = [
            {
                "role": "system",
                "content": self._build_system_prompt(
                    system_prompt, schema_text
                ),
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

        if self._backend == "openai_compat":
            content = self._call_openai_compat(
                messages, temperature, max_tokens
            )
        else:
            content = self._call_hf(
                messages, temperature, max_tokens
            )

        if not content:
            raise RuntimeError("LLM 응답이 비어 있습니다.")

        try:
            return response_model.model_validate_json(content)
        except ValidationError as error:
            raise ValueError(
                "LLM 응답이 스키마와 일치하지 않습니다.\n"
                f"원본 응답:\n{content}\n\n"
                f"검증 오류:\n{error}"
            ) from error
