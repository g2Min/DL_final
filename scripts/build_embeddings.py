import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.fashion_clip_encoder import FashionClipEncoder


METADATA_PATH = Path("data/garments/metadata.jsonl")
OUTPUT_PATH = Path("data/garments/embeddings.npy")

BATCH_SIZE = 16


def load_metadata() -> list[dict]:
    records = []

    with METADATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            records.append(json.loads(line))

    return records


def main() -> None:
    records = load_metadata()

    image_paths = [
        record["image_path"]
        for record in records
    ]

    encoder = FashionClipEncoder()

    all_embeddings = []

    for start in tqdm(
        range(0, len(image_paths), BATCH_SIZE),
        desc="Encoding images",
    ):
        batch_paths = image_paths[
            start : start + BATCH_SIZE
        ]

        embeddings = encoder.encode_images(batch_paths)

        all_embeddings.append(embeddings)

    embeddings = np.concatenate(
        all_embeddings,
        axis=0,
    )

    np.save(OUTPUT_PATH, embeddings)

    print("shape:", embeddings.shape)
    print("saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()