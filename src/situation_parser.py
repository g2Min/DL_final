from src.schema import SituationAnalysis


def parse_situation_rule_based(text: str) -> SituationAnalysis:
    text_lower = text.lower()

    if "캠핑" in text_lower:
        return SituationAnalysis(
            activity="camping",
            season=(
                "autumn"
                if "가을" in text_lower
                else None
            ),
            weather="cool",
            requirements=[
                "warm",
                "comfortable",
                "easy to move",
                "outdoor",
            ],
            categories=[
                "outerwear",
                "top",
                "bottom",
                "hat",
                "bag",
            ],
            colors=[
                "khaki",
                "brown",
                "beige",
            ],
            styles=[
                "cute",
                "casual",
                "outdoor",
            ],
        )

    raise ValueError(
        "현재 규칙 기반 parser에서는 캠핑 상황만 지원합니다."
    )