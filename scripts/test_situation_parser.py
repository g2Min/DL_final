from pprint import pprint

from src.llm_client import StructuredLLMClient
from src.situation_parser import SituationParser


def main() -> None:
    client = StructuredLLMClient(
        model_id="Qwen/Qwen3-32B",
    )

    parser = SituationParser(client)

    result = parser.parse(
        "고양이 캐릭터가 가을에 캠핑을 가. "
        "귀와 꼬리는 가리지 않았으면 좋겠어."
    )

    pprint(result.model_dump())


if __name__ == "__main__":
    main()