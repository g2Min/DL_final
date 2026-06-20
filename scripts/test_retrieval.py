from src.garment_retriever import GarmentRetriever
from src.situation_parser import SituationParser
from src.llm_client import StructuredLLMClient

def build_search_query(analysis, category: str) -> str:
    values = [
        category,
        analysis.season or "",
        analysis.activity,
        *analysis.requirements,
        *analysis.colors,
        *analysis.styles,
    ]

    return ", ".join(value for value in values if value)


def main() -> None:
    text = "고양이 캐릭터가 가을에 캠핑을 가"

    llm_client = StructuredLLMClient(
        model_id="Qwen/Qwen3-32B"
    )

    # analysis = parse_situation_rule_based(text)
    parser = SituationParser(
        llm_client=llm_client,
    )
    
    analysis = parser.parse(text)

    retriever = GarmentRetriever(
        metadata_path="datasets/garments/metadata.jsonl",
        embeddings_path="datasets/garments/embeddings.npy",
    )
    
    candidates = {}

    for category in analysis.categories:
        query = build_search_query(analysis, category)

        candidates = retriever.search(
            query=query,
            category=category,
            season=analysis.season,
            top_k=3,
        )

        print(f"\n[{category}]")
        print("query:", query)

        for result in candidates:
            print(
                f"- {result['description']} "
                f"score={result['score']:.4f}"
            )


if __name__ == "__main__":
    main()