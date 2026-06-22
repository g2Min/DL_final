"""
articleType 데이터셋은 분류
"""


CATEGORY_MAPPING = {
    "Tops": "top",
    "Tshirts": "top",
    "Shirts": "top",
    "Sweaters": "top",
    "Sweatshirts": "top",
    "Lounge Tshirts": "top",

    "Jackets": "outerwear",
    "Rain Jacket": "outerwear",
    "Nehru Jackets": "outerwear",
    "Waistcoat": "outerwear",
    "Tracksuits": "outerwear",
    "Night suits": "outerwear",

    # 수영복: 상하의 세트 또는 원피스로 전신 처리
    "Swimwear": "swimwear",
    "Trunk": "swimwear",

    "Jeans": "bottom",
    "Trousers": "bottom",
    "Shorts": "bottom",
    "Lounge Shorts": "bottom",
    "Track Pants": "bottom",
    "Lounge Pants": "bottom",

    "Caps": "hat",

    "Backpacks": "bag",
    "Handbags": "bag",
    "Messenger Bag": "bag",
    "Duffel Bag": "bag",

    "Salwar": "dress",
    "Nightdress": "dress",
}


def get_project_category(article_type: str) -> str | None:
    return CATEGORY_MAPPING.get(article_type)