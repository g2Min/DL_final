from src.llm_client import StructuredLLMClient
from src.schema import SituationAnalysis


SYSTEM_PROMPT = """
당신은 동물 캐릭터를 위한 의상 스타일리스트입니다.

사용자의 문장을 분석해 상황에 적합한 의상 검색 조건을 만드세요.

규칙:
1. 캐릭터는 머리가 크고 몸이 짧은 2등신 동물 캐릭터입니다.
2. 실제 상품을 직접 고르지 말고 검색 조건만 만드세요.
3. categories는 outerwear, top, bottom, swimwear, hat, bag 중에서만 고르세요.
4. 수영, 해수욕, 물놀이 등 수상 활동에는 반드시 swimwear를 사용하세요.
   swimwear는 상하의 세트 또는 원피스를 하나의 단위로 검색합니다.
   swimwear를 선택했으면 top과 bottom을 별도로 추가하지 마세요.
5. 불필요한 카테고리는 포함하지 마세요.
6. 계절이 명시되지 않으면 상황을 바탕으로 합리적으로 추론하되,
   확신하기 어렵다면 unknown을 사용하세요.
7. 추천 색상은 2~4개로 제한하세요.
8. styles는 검색에 활용할 수 있는 영어 표현으로 작성하세요.
9. 귀와 꼬리를 가리기 쉬운 의상은 avoid에 기록하세요.
10. reasoning_summary는 한 문장으로 간단히 작성하세요.
"""


class SituationParser:
    def __init__(
        self,
        llm_client: StructuredLLMClient,
    ) -> None:
        self.llm_client = llm_client

    def parse(
        self,
        user_text: str,
    ) -> SituationAnalysis:
        if not user_text.strip():
            raise ValueError(
                "사용자 입력이 비어 있습니다."
            )

        user_prompt = f"""
다음 요청을 의상 검색 조건으로 분석하세요.

사용자 요청:
{user_text}

출력값의 문자열은 가급적 영어 검색어 형태로 작성하세요.
전달된 JSON Schema의 모든 필드를 빠짐없이 작성하세요.
"""

        return self.llm_client.generate_structured(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_model=SituationAnalysis,
            temperature=0.1,
            max_tokens=800,
        )