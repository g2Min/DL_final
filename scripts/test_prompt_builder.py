from pprint import pprint

from src.prompt_builder import PromptBuilder
from src.schema import CharacterOutfitSpec


def main() -> None:
    outfit_spec = CharacterOutfitSpec.model_validate(
        {
            "character_species": "cat",
            "activity": "camping",
            "overall_style": "cute outdoor casual camping",
            "overall_palette": [
                "khaki",
                "brown",
            ],
            "garments": [
                {
                    "source_id": "1001",
                    "category": "outerwear",
                    "garment_type": "short khaki jacket",
                    "color": "khaki",
                    "pattern": "solid",
                    "material_appearance": "thick fabric",
                    "silhouette": "rounded puffer jacket shape",
                    "sleeve_style": "short sleeves",
                    "body_length": "very short",
                    "key_details": [
                        "zipper",
                        "pockets",
                    ],
                    "character_adaptation": [
                        "shortened hemline for tail visibility",
                        "rounded shoulder fit",
                    ],
                    "preserve_ears": True,
                    "preserve_tail": True,
                },
                {
                    "source_id": "1002",
                    "category": "top",
                    "garment_type": "cute brown shirt",
                    "color": "brown",
                    "pattern": "plaid",
                    "material_appearance": "cotton fabric",
                    "silhouette": "rounded casual shirt",
                    "sleeve_style": "short sleeves",
                    "body_length": "short",
                    "key_details": [
                        "buttons",
                        "collar",
                    ],
                    "character_adaptation": [
                        "raised hemline for tail visibility",
                        "adjustable collar",
                    ],
                    "preserve_ears": True,
                    "preserve_tail": True,
                },
            ],
        }
    )

    builder = PromptBuilder()

    positive = builder.build_positive(outfit_spec)
    negative = builder.build_negative()

    print("\n=== OUTFIT SPEC ===")
    pprint(outfit_spec.model_dump())

    print("\n=== POSITIVE PROMPT ===")
    print(positive)

    print("\n=== NEGATIVE PROMPT ===")
    print(negative)


if __name__ == "__main__":
    main()