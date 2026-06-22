import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

"""
캐릭터 옷 입히기에 필요한 항목만 사용함
"""

OUTPUT_ROOT = Path("datasets/garments")
IMAGE_DIR = OUTPUT_ROOT / "images"
METADATA_PATH = OUTPUT_ROOT / "metadata.jsonl"

ALLOWED_ARTICLE_TYPES = {
    "Tops",
    "Tshirts",
    "Shirts",
    "Lounge Tshirts",
    "Sweaters",
    "Sweatshirts",
    "Jackets",
    "Rain Jacket",
    "Nehru Jackets",
    "Waistcoat",
    "Jeans",
    "Trousers",
    "Shorts",
    "Lounge Shorts",
    "Track Pants",
    "Lounge Pants",
    "Trunk",
    "Caps",
    "Backpacks",
    "Handbags",
    "Messenger Bag",
    "Duffel Bag",
    "Salwar",
    "Swimwear",
    "Tracksuits",
    "Nightdress",
    "Night suits"
}

MAX_ITEMS = 2000


def normalize_text(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def convert_row(row: dict, image_filename: str) -> dict:
    article_type = normalize_text(row.get("articleType"))
    color = normalize_text(row.get("baseColour"))
    season = normalize_text(row.get("season"))
    usage = normalize_text(row.get("usage"))
    product_name = normalize_text(row.get("productDisplayName"))

    description_parts = [
        color,
        article_type,
        season,
        usage,
        product_name,
    ]

    description = ", ".join(
        part for part in description_parts if part
    )

    return {
        "id": str(row["id"]),
        "image_path": str(Path("datasets/garments/images") / image_filename),
        "article_type": article_type,
        "color": color,
        "season": season,
        "usage": usage,
        "product_name": product_name,
        "description": description,
    }


def main() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        "./datasets/garments",
        split="train",
    )

    filtered = dataset.filter(
        lambda row: row.get("articleType") in ALLOWED_ARTICLE_TYPES
    )

    filtered = filtered.shuffle(seed=42)

    max_items = min(MAX_ITEMS, len(filtered))
    filtered = filtered.select(range(max_items))

    with METADATA_PATH.open("w", encoding="utf-8") as file:
        for row in tqdm(filtered, desc="Preparing garments"):
            image = row["image"].convert("RGB")

            image_filename = f"{row['id']}.jpg"
            image_path = IMAGE_DIR / image_filename

            image.save(image_path, quality=95)

            metadata = convert_row(row, image_filename)

            file.write(
                json.dumps(metadata, ensure_ascii=False) + "\n"
            )

    print(f"Saved {max_items} garments")
    print(f"Metadata: {METADATA_PATH}")


if __name__ == "__main__":
    main()