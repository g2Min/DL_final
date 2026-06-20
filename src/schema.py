from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


Season = Literal[
    "spring",
    "summer",
    "autumn",
    "winter",
    "all",
    "unknown",
]

GarmentCategory = Literal[
    "outerwear",
    "top",
    "bottom",
    "swimwear",
    "hat",
    "bag",
]

UPPER_BODY_CATEGORIES: frozenset[str] = frozenset(
    {"outerwear", "top", "hat", "bag"}
)
LOWER_BODY_CATEGORIES: frozenset[str] = frozenset({"bottom"})
FULL_BODY_CATEGORIES: frozenset[str] = frozenset({"swimwear"})

Weather = Literal[
    "hot",
    "warm",
    "mild",
    "cool",
    "cold",
    "rainy",
    "snowy",
    "unknown",
]


class SituationAnalysis(BaseModel):
    """
    사용자 상황을 의상 검색 조건으로 변환한 결과.
    """

    model_config = ConfigDict(extra="forbid")

    character_species: str = Field(description="캐릭터 동물 종류. 예: cat, rabbit")
    activity: str = Field(description="주요 활동. 예: camping, picnic, party")
    season: Season = Field(description="명시되거나 추론된 계절")
    weather: Weather = Field(description="예상되는 날씨 또는 기온")
    
    requirements: list[str] = Field(
        description=(
            "상황에 필요한 기능적 조건. "
            "예: warm, comfortable, easy to move"
        )
    )
    
    categories: list[GarmentCategory] = Field(description="추천해야 할 의류 카테고리")
    colors: list[str] = Field(description="추천 색상")
    styles: list[str] = Field(description="추천 스타일")

    avoid: list[str] = Field(
        default_factory=list,
        description="피해야 할 의상 특성"
    )

    reasoning_summary: str = Field(description="추천 조건을 정한 짧은 이유")
    
class CharacterGarmentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_id: str = Field(description="원본 상품 ID")

    category: GarmentCategory
    garment_type: str = Field(
        description=(
            "사람 성별 표현이 제거된 의상 종류. "
            "예: short fleece vest"
        )
    )

    color: str
    pattern: str = Field(
        description=(
            "solid, plaid, striped, floral 등"
        )
    )
    material_appearance: str = Field(
        description=(
            "실제 소재가 아닌 일러스트로 표현할 시각적 질감"
        )
    )
    key_details: list[str] = Field(
        description=(
            "주머니, 지퍼, 단추 등 유지할 핵심 디자인"
        )
    )

    silhouette: str = Field(
        description=(
            "2등신 몸에 적합한 의상 실루엣"
        )
    )
    body_length: Literal[
        "very short",
        "short",
        "medium",
        "n/a",
    ]

    sleeve_style: str
    character_adaptation: list[str] = Field(description="캐릭터 체형에 맞추기 위한 변경사항")

    preserve_ears: bool = True
    preserve_tail: bool = True


class CharacterOutfitSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    character_species: str
    activity: str
    garments: list[CharacterGarmentSpec]
    overall_palette: list[str]
    overall_style: str