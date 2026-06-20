import json

from src.llm_client import StructuredLLMClient
from src.schema import CharacterOutfitSpec, SituationAnalysis


SYSTEM_PROMPT = """
당신은 실제 패션 상품을 2등신 동물 캐릭터용 의상으로
재해석하는 캐릭터 의상 디자이너입니다.

규칙:
1. 실제 옷의 핵심 색상, 패턴, 종류, 주요 디테일만 유지하세요.
2. 브랜드명, 성별, 사람 체형 표현은 제거하세요.
3. 캐릭터는 큰 머리와 매우 짧고 둥근 몸을 가집니다.
4. 의상 길이는 짧게 조정하세요.
5. 실사 질감 대신 단순하고 부드러운 2D 일러스트 질감으로 바꾸세요.
6. 귀, 얼굴, 꼬리를 가리지 마세요.
7. 실루엣은 둥글고 단순하게 만드세요.
8. 원본 정보에 없는 로고나 복잡한 디테일을 임의로 추가하지 마세요.
"""


class GarmentAdapter:
    def __init__(
        self,
        llm_client: StructuredLLMClient,
    ) -> None:
        self.llm_client = llm_client

    def adapt(
        self,
        *,
        analysis: SituationAnalysis,
        selected_items: dict[str, dict],
    ) -> CharacterOutfitSpec:
        clean_items = []

        for category, item in selected_items.items():
            clean_items.append(
                {
                    "source_id": str(item["id"]),
                    "category": category,
                    "article_type": item.get(
                        "article_type",
                        "",
                    ),
                    "color": item.get(
                        "color",
                        "",
                    ),
                    "season": item.get(
                        "season",
                        "",
                    ),
                    "usage": item.get(
                        "usage",
                        "",
                    ),
                    "product_name": item.get(
                        "product_name",
                        "",
                    ),
                    "description": item.get(
                        "description",
                        "",
                    ),
                }
            )

        user_prompt = f"""
캐릭터 정보:
- species: {analysis.character_species}
- activity: {analysis.activity}
- season: {analysis.season}
- styles: {analysis.styles}
- avoid: {analysis.avoid}

선택된 실제 의류:
{json.dumps(clean_items, ensure_ascii=False, indent=2)}

각 실제 의류를 2등신 동물 캐릭터용 디자인으로 변환하세요.
모든 source_id와 category는 원본 값을 유지하세요.
"""

        return self.llm_client.generate_structured(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_model=CharacterOutfitSpec,
            temperature=0.2,
            max_tokens=1600,
        )