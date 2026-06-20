import json
from pathlib import Path

import numpy as np
from fashion_clip.fashion_clip import FashionCLIP

from src.dataset_loader import get_project_category


class GarmentRetriever:
    def __init__(
        self,
        metadata_path: str,
        embeddings_path: str,
    ):
        self.model = FashionCLIP("fashion-clip")

        self.metadata = self._load_metadata(metadata_path)
        self.embeddings = np.load(embeddings_path)

    @staticmethod
    def _load_metadata(path: str) -> list[dict]:
        records = []

        with Path(path).open("r", encoding="utf-8") as file:
            for line in file:
                records.append(json.loads(line))

        return records

    def search(
        self,
        query: str,
        category: str,
        season: str | None = None,
        top_k: int = 5,
    ) -> list[dict]:
        query_embedding = self.model.encode_text(
            [query],
            batch_size=1,
        )[0]

        query_embedding = query_embedding / max(
            np.linalg.norm(query_embedding),
            1e-12,
        )

        scores = self.embeddings @ query_embedding
        ranked_indices = np.argsort(scores)[::-1]

        results = []

        for index in ranked_indices:
            item = self.metadata[index]

            project_category = get_project_category(
                item["article_type"]
            )

            if project_category != category:
                continue

            if season:
                item_season = item.get("season", "").lower()
                _season_aliases = {"autumn": "fall", "fall": "autumn"}
                normalized = season.lower()
                valid_seasons = {normalized, _season_aliases.get(normalized, normalized), "all"}

                if item_season not in valid_seasons:
                    continue

            results.append(
                {
                    **item,
                    "score": float(scores[index]),
                }
            )

            if len(results) >= top_k:
                break

        return results