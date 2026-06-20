import json
import os
from typing import TypeVar

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from pydantic import BaseModel, ValidationError


T = TypeVar("T", bound=BaseModel)


class StructuredLLMClient:
    """
    Hugging Face Inference Providers를 사용해
    Pydantic 스키마에 맞는 JSON을 생성한다.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-32B",
        provider: str = "auto",
    ) -> None:
        load_dotenv()

        token = os.getenv("HF_TOKEN")

        if not token:
            raise RuntimeError(
                "HF_TOKEN이 없습니다. "
                "프로젝트 루트의 .env 파일을 확인하세요."
            )

        self.model_id = model_id

        self.client = InferenceClient(
            provider=provider,
            api_key=token,
        )

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> T:
        # Pydantic 모델을 JSON Schema로 변환
        schema = response_model.model_json_schema()
        schema_text = json.dumps(
            schema,
            ensure_ascii=False,
            indent=2,
        )

        # json_object 모드는 JSON 형식만 보장하기 때문에
        # 실제 필드 구조를 system prompt에 직접 전달해야 함
        full_system_prompt = f"""
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

        response = self.client.chat_completion(
            model=self.model_id,
            messages=[
                {
                    "role": "system",
                    "content": full_system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            response_format={
                "type": "json_object",
            },
            temperature=temperature,
            max_tokens=max_tokens,
        )

        content = response.choices[0].message.content

        if not content:
            raise RuntimeError(
                "LLM 응답이 비어 있습니다."
            )

        try:
            return response_model.model_validate_json(content)

        except ValidationError as error:
            raise ValueError(
                "LLM 응답이 SituationAnalysis 스키마와 일치하지 않습니다.\n"
                f"원본 응답:\n{content}\n\n"
                f"검증 오류:\n{error}"
            ) from error