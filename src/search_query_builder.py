from src.schema import SituationAnalysis


CATEGORY_LABELS = {
    "outerwear": "outerwear jacket or vest",
    "top": "shirt, sweater, or sweatshirt",
    "bottom": "pants or shorts",
    "hat": "hat or cap",
    "bag": "small backpack or bag",
}


def build_search_query(
    analysis: SituationAnalysis,
    category: str,
) -> str:
    category_text = CATEGORY_LABELS.get(
        category,
        category,
    )

    parts = [
        category_text,
        analysis.season,
        analysis.activity,
        analysis.weather,
        *analysis.requirements,
        *analysis.colors,
        *analysis.styles,
    ]

    normalized_parts = [
        part.strip()
        for part in parts
        if part
        and part not in {"unknown", "all"}
    ]

    return ", ".join(normalized_parts)