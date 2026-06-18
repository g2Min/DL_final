from pydantic import BaseModel, Field


class SituationAnalysis(BaseModel):
    activity: str
    season: str | None = None
    weather: str | None = None

    requirements: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    colors: list[str] = Field(default_factory=list)
    styles: list[str] = Field(default_factory=list)


class GarmentRecord(BaseModel):
    id: str
    image_path: str
    article_type: str
    color: str
    season: str
    usage: str
    product_name: str
    description: str


class CharacterGarmentSpec(BaseModel):
    garment_type: str
    color: str
    pattern: str | None = None
    material_appearance: str | None = None
    key_details: list[str] = Field(default_factory=list)

    body_length: str = "short"
    silhouette: str = "rounded"
    preserve_ears: bool = True
    preserve_tail: bool = True