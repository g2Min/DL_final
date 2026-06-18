CATEGORY_MAPPING = {
    "Tops": "top",
    "Tshirts": "top",
    "Shirts": "top",
    "Sweaters": "top",
    "Sweatshirts": "top",
    "Lounge Tshirts": "top",

    "Jackets": "outerwear",
    "Rain Jacket": "outwear",
    "Nehru Jackets": "outwear",
    "Waistcoat": "outerwear",
    "Swimwear": "outerwear",
    "Tracksuits": "outerwear",
    "Night suits": "outerwear",

    "Jeans": "bottom",
    "Trousers": "bottom",
    "Shorts": "bottom",
    "Lounge Shorts": "bottom",
    "Track Pants": "bottom",
    "Lounge Pants": "bottom",
    "Trunk": "bottom",

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