import json
from pathlib import Path

from src.figure_maker import make_figure
from src.garment_adapter import GarmentAdapter
from src.garment_retriever import GarmentRetriever
from src.llm_client import StructuredLLMClient
from src.outfit_generator import OutfitGenerator
from src.pipeline import AnimalOutfitPipeline
from src.prompt_builder import PromptBuilder
from src.situation_parser import SituationParser


def main() -> None:
    llm_client = StructuredLLMClient(model_id="Qwen/Qwen3-32B")
    situation_parser = SituationParser(llm_client)
    retriever = GarmentRetriever(
        metadata_path=(
            "datasets/garments/metadata.jsonl"
        ),
        embeddings_path=(
            "datasets/garments/embeddings.npy"
        ),
    )

    garment_adapter = GarmentAdapter(
        llm_client
    )

    prompt_builder = PromptBuilder()
    generator = OutfitGenerator()
    pipeline = AnimalOutfitPipeline(
        situation_parser=situation_parser,
        retriever=retriever,
        garment_adapter=garment_adapter,
        prompt_builder=prompt_builder,
        generator=generator,
    )

    result = pipeline.run(
        user_text=(
            "다람쥐 캐릭터가 여름에 수영을 하러 가."
        ),
        character_path=(
            "datasets/characters/squirrel.png"
        ),
        top_mask_path=(
            "datasets/characters/masking/squirrel_top.png"
        ),
        bottom_mask_path=(
            "datasets/characters/masking/squirrel_bottom.png"
        ),
        all_mask_path=(
            "datasets/characters/masking/squirrel_all.png"
        ),
        output_dir="datasets/outputs/camping2",
        top_k=3,
        seed=42,
    )

    result_path = Path(
        "datasets/outputs/camping/result.json"
    )

    result_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with result_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2,
        )

    output_dir = Path("datasets/outputs/camping2")
    figure_path = make_figure(
        [result],
        output_dir / "figure.png",
        cols=1,
    )

    for part, path in result["generated_paths"].items():
        print(f"generated ({part}):", path)
    print("figure:", figure_path)
    print("metadata:", result_path)


if __name__ == "__main__":
    main()