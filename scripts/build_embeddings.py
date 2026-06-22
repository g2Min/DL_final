import json
from pathlib import Path

import numpy as np
from fashion_clip.fashion_clip import FashionCLIP
from tqdm import tqdm


METADATA_PATH = Path("datasets/garments/metadata.jsonl")
OUTPUT_PATH = Path("datasets/garments/embeddings.npy")

BATCH_SIZE = 32


def load_metadata() -> list[dict]:
    records = []

    with METADATA_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            records.append(json.loads(line))

    return records


def main() -> None:
    records = load_metadata()
    image_paths = [record["image_path"] for record in records]

    model = FashionCLIP("fashion-clip")

    all_embeddings = []

    for start in tqdm(
        range(0, len(image_paths), BATCH_SIZE),
        desc="Encoding garment images",
    ):
        batch_paths = image_paths[start : start + BATCH_SIZE]

        embeddings = model.encode_images(
            batch_paths,
            batch_size=len(batch_paths),
        )

        all_embeddings.append(embeddings)

    embeddings = np.concatenate(all_embeddings, axis=0)

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    embeddings = embeddings / np.clip(norms, 1e-12, None)

    np.save(OUTPUT_PATH, embeddings)

    print("Embedding shape:", embeddings.shape)
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()