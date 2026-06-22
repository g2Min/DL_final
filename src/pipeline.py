import itertools
from pathlib import Path

from PIL import Image as PILImage

from src.garment_adapter import GarmentAdapter
from src.garment_retriever import GarmentRetriever
from src.outfit_generator import OutfitGenerator
from src.prompt_builder import PromptBuilder
from src.schema import (
    FULL_BODY_CATEGORIES,
    LOWER_BODY_CATEGORIES,
    UPPER_BODY_CATEGORIES,
    SituationAnalysis,
)
from src.search_query_builder import (
    build_search_query,
)
from src.situation_parser import SituationParser


def _load_garment_images(
    selected_items: dict[str, dict],
    categories: frozenset[str],
) -> list[PILImage.Image] | None:
    """선택된 의상 중 해당 카테고리의 원본 이미지를 PIL로 로드."""
    images: list[PILImage.Image] = []
    for cat, item in selected_items.items():
        if cat in categories:
            path = item.get("image_path", "")
            if path:
                try:
                    images.append(PILImage.open(path).convert("RGB"))
                except Exception:
                    pass
    return images if images else None


class AnimalOutfitPipeline:
    def __init__(
        self,
        *,
        situation_parser: SituationParser,
        retriever: GarmentRetriever,
        garment_adapter: GarmentAdapter,
        prompt_builder: PromptBuilder,
        generator: OutfitGenerator,
    ) -> None:
        self.situation_parser = situation_parser
        self.retriever = retriever
        self.garment_adapter = garment_adapter
        self.prompt_builder = prompt_builder
        self.generator = generator

    def retrieve_combinations(
        self,
        *,
        user_text: str,
        top_k: int = 3,
    ) -> tuple[
        SituationAnalysis,
        dict[str, list[dict]],
        list[dict[str, dict]],
    ]:
        """상황 분석 + 카테고리별 검색 후 모든 top-k 조합을 반환."""
        analysis = self.situation_parser.parse(user_text)

        candidates: dict[str, list[dict]] = {}
        for category in analysis.categories:
            query = build_search_query(
                analysis=analysis,
                category=category,
            )
            results = self.retriever.search(
                query=query,
                category=category,
                season=analysis.season,
                top_k=top_k,
            )
            if results:
                candidates[category] = results

        categories = list(candidates)
        combinations = [
            dict(zip(categories, combo))
            for combo in itertools.product(
                *[candidates[c] for c in categories]
            )
        ]

        return analysis, candidates, combinations

    def run(
        self,
        *,
        user_text: str,
        character_path: str | Path,
        top_mask_path: str | Path | None = None,
        bottom_mask_path: str | Path | None = None,
        all_mask_path: str | Path | None = None,
        output_dir: str | Path,
        top_k: int = 3,
        seed: int = 42,
    ) -> dict:
        # 1. 사용자 상황 분석
        analysis = self.situation_parser.parse(
            user_text
        )

        # 2. 카테고리별 의상 검색
        candidates: dict[str, list[dict]] = {}

        for category in analysis.categories:
            query = build_search_query(
                analysis=analysis,
                category=category,
            )

            results = self.retriever.search(
                query=query,
                category=category,
                season=analysis.season,
                top_k=top_k,
            )

            candidates[category] = results

        # 3. 카테고리별 검색 top-1 선택
        selected_items = {
            category: results[0]
            for category, results in candidates.items()
            if results
        }

        if not selected_items:
            raise RuntimeError(
                "검색된 의상이 없습니다."
            )

        # 4. 실제 의상을 캐릭터용 명세로 변환
        outfit_spec = self.garment_adapter.adapt(
            analysis=analysis,
            selected_items=selected_items,
        )

        negative_prompt = (
            self.prompt_builder.build_negative()
        )

        output_dir = Path(output_dir)
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        generated_paths: dict[str, str] = {}
        current_image = None  # 이전 pass 결과를 다음 pass 입력으로 연결

        outfit_categories = {
            g.category for g in outfit_spec.garments
        }

        # 5a. 전신 pass (swimwear 등) — 상하체 구분 없이 전신 마스크로 한 번에
        full_categories = outfit_categories & FULL_BODY_CATEGORIES
        if full_categories and all_mask_path is not None:
            positive_prompt = (
                self.prompt_builder.build_positive_for_categories(
                    outfit_spec,
                    FULL_BODY_CATEGORIES,
                )
            )

            full_path, current_image = (
                self.generator.generate(
                    character_path=character_path,
                    mask_path=all_mask_path,
                    positive_prompt=positive_prompt,
                    negative_prompt=negative_prompt,
                    output_path=(
                        output_dir
                        / f"outfit_full_seed_{seed}.png"
                    ),
                    ip_adapter_images=_load_garment_images(
                        selected_items, FULL_BODY_CATEGORIES
                    ),
                    seed=seed,
                )
            )

            generated_paths["full"] = str(full_path)

        # 5b. 상체 pass (top, outerwear, hat, bag)
        upper_categories = outfit_categories & UPPER_BODY_CATEGORIES
        if upper_categories and top_mask_path is not None:
            positive_prompt = (
                self.prompt_builder.build_positive_for_categories(
                    outfit_spec,
                    UPPER_BODY_CATEGORIES,
                )
            )

            upper_path, current_image = (
                self.generator.generate(
                    character_path=character_path,
                    mask_path=top_mask_path,
                    positive_prompt=positive_prompt,
                    negative_prompt=negative_prompt,
                    output_path=(
                        output_dir
                        / f"outfit_upper_seed_{seed}.png"
                    ),
                    ip_adapter_images=_load_garment_images(
                        selected_items, UPPER_BODY_CATEGORIES
                    ),
                    seed=seed,
                )
            )

            generated_paths["upper"] = str(upper_path)

        # 5c. 하체 pass (bottom) — 상체 결과 이미지를 입력으로 사용
        lower_categories = outfit_categories & LOWER_BODY_CATEGORIES
        if lower_categories and bottom_mask_path is not None:
            positive_prompt = (
                self.prompt_builder.build_positive_for_categories(
                    outfit_spec,
                    LOWER_BODY_CATEGORIES,
                )
            )

            lower_path, current_image = (
                self.generator.generate(
                    character_path=character_path,
                    mask_path=bottom_mask_path,
                    positive_prompt=positive_prompt,
                    negative_prompt=negative_prompt,
                    output_path=(
                        output_dir
                        / f"outfit_lower_seed_{seed}.png"
                    ),
                    base_image=current_image,
                    ip_adapter_images=_load_garment_images(
                        selected_items, LOWER_BODY_CATEGORIES
                    ),
                    seed=seed,
                )
            )

            generated_paths["lower"] = str(lower_path)

        if not generated_paths:
            raise RuntimeError(
                "생성 가능한 마스크와 카테고리 조합이 없습니다."
            )

        # 최종 이미지 경로 = 마지막 pass 결과
        final_path = list(generated_paths.values())[-1]

        return {
            "analysis": analysis.model_dump(),
            "candidates": candidates,
            "selected_items": selected_items,
            "outfit_spec": outfit_spec.model_dump(),
            "negative_prompt": negative_prompt,
            "generated_paths": generated_paths,
            "generated_path": final_path,
        }
