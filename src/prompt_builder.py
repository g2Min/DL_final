from src.schema import CharacterOutfitSpec


class PromptBuilder:
    MAX_DETAILS = 3
    MAX_ADAPTATIONS = 2

    def build_positive_for_categories(
        self,
        outfit: CharacterOutfitSpec,
        categories: frozenset[str],
    ) -> str:
        filtered = CharacterOutfitSpec(
            character_species=outfit.character_species,
            activity=outfit.activity,
            garments=[
                garment
                for garment in outfit.garments
                if garment.category in categories
            ],
            overall_palette=outfit.overall_palette,
            overall_style=outfit.overall_style,
        )

        return self.build_positive(filtered)

    def build_positive(
        self,
        outfit: CharacterOutfitSpec,
    ) -> str:
        garment_prompts = [
            self._build_garment_prompt(garment)
            for garment in outfit.garments
        ]

        garment_prompts = [
            prompt
            for prompt in garment_prompts
            if prompt
        ]

        parts = [
            "Only redraw the masked clothing region",
            (
                f"cute original two-head-tall chibi "
                f"{outfit.character_species}"
            ),
            "large round head, tiny rounded torso, short limbs",
        ]

        if garment_prompts:
            parts.append(
                "wearing " + "; ".join(garment_prompts)
            )

        if outfit.overall_style:
            parts.append(
                f"{outfit.overall_style} outfit style"
            )

        palette = self._clean_values(
            outfit.overall_palette,
            limit=3,
        )
        if palette:
            parts.append(
                f"{', '.join(palette)} color palette"
            )

        parts.extend([
            "simple clean 2D game character illustration",
            "soft flat colors, simplified fabric texture",
            "rounded garment shapes",
            (
                "keep face, eyes, expression, ears, tail, pose, "
                "body proportions, line art, background and "
                "camera angle unchanged"
            ),
        ])

        return ", ".join(parts) + "."

    def _build_garment_prompt(self, garment) -> str:
        parts = []

        # 생성 결과에 영향이 큰 속성부터 배치
        self._append(parts, garment.color)
        self._append(parts, garment.body_length)
        self._append(parts, garment.garment_type)
        self._append(parts, garment.silhouette)
        self._append(parts, garment.pattern)
        self._append(parts, garment.sleeve_style)
        self._append(parts, garment.material_appearance)

        details = self._clean_values(
            garment.key_details,
            limit=self.MAX_DETAILS,
        )
        parts.extend(details)

        adaptations = self._clean_values(
            garment.character_adaptation,
            limit=self.MAX_ADAPTATIONS,
        )
        parts.extend(adaptations)

        return " ".join(parts)

    @staticmethod
    def _append(parts: list[str], value: str | None) -> None:
        if PromptBuilder._is_meaningful(value):
            parts.append(value.strip())

    @staticmethod
    def _clean_values(
        values,
        limit: int,
    ) -> list[str]:
        if not values:
            return []

        cleaned = [
            str(value).strip()
            for value in values
            if PromptBuilder._is_meaningful(value)
        ]

        return cleaned[:limit]

    @staticmethod
    def _is_meaningful(value) -> bool:
        if value is None:
            return False

        text = str(value).strip().lower()

        return text not in {
            "",
            "none",
            "null",
            "n/a",
            "not applicable",
            "unspecified",
            "unknown",
        }

    def build_negative(self) -> str:
        return (
            "human, realistic anatomy, photorealistic, "
            "tall body, long torso, long limbs, "
            "changed face, changed expression, "
            "covered ears, missing ears, missing tail, extra tail, "
            "extra limbs, distorted body, different pose, "
            "different background, different camera angle, "
            "text, logo, watermark"
        )