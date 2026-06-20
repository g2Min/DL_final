from pprint import pprint

from src.garment_adapter import GarmentAdapter
from src.llm_client import StructuredLLMClient
from src.schema import SituationAnalysis


def main() -> None:
    client = StructuredLLMClient()

    adapter = GarmentAdapter(client)

    analysis = SituationAnalysis(
        character_species="cat",
        activity="camping",
        season="autumn",
        weather="cool",
        requirements=[
            "warm",
            "comfortable",
            "easy to move",
        ],
        categories=[
            "outerwear",
            "top",
            "bottom",
            "bag",
        ],
        colors=[
            "khaki",
            "brown",
            "beige",
        ],
        styles=[
            "cute outdoor",
            "casual camping",
        ],
        avoid=[
            "covering ears",
            "covering tail",
        ],
        reasoning_summary=(
            "Warm outdoor clothing is suitable "
            "for autumn camping."
        ),
    )

    selected_items = {
        "outerwear": {
            "id": "1001",
            "article_type": "Jackets",
            "color": "Khaki",
            "season": "Fall",
            "usage": "Casual",
            "product_name": (
                "Men Khaki Casual Jacket"
            ),
            "description": (
                "Khaki, Jackets, Fall, Casual"
            ),
        },
        "top": {
            "id": "1002",
            "article_type": "Shirts",
            "color": "Brown",
            "season": "Fall",
            "usage": "Casual",
            "product_name": (
                "Men Brown Checked Shirt"
            ),
            "description": (
                "Brown checked shirt"
            ),
        },
    }

    result = adapter.adapt(
        analysis=analysis,
        selected_items=selected_items,
    )

    pprint(result.model_dump())


if __name__ == "__main__":
    main()