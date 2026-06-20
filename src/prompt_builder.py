from src.schema import CharacterOutfitSpec


class PromptBuilder:
    def build_positive_for_categories(
        self,
        outfit: CharacterOutfitSpec,
        categories: frozenset[str],
    ) -> str:
        filtered = CharacterOutfitSpec(
            character_species=outfit.character_species,
            activity=outfit.activity,
            garments=[
                g
                for g in outfit.garments
                if g.category in categories
            ],
            overall_palette=outfit.overall_palette,
            overall_style=outfit.overall_style,
        )
        return self.build_positive(filtered)

    def build_positive(
        self,
        outfit: CharacterOutfitSpec,
    ) -> str:
        garment_sentences = []

        for garment in outfit.garments:
            details = ", ".join(
                garment.key_details
            )

            adaptations = ", ".join(
                garment.character_adaptation
            )

            garment_sentences.append(
                (
                    f"a {garment.body_length} "
                    f"{garment.color} "
                    f"{garment.garment_type}, "
                    f"{garment.pattern} pattern, "
                    f"{garment.material_appearance}, "
                    f"{garment.silhouette}, "
                    f"{garment.sleeve_style}, "
                    f"with {details}, "
                    f"adapted by {adaptations}"
                )
            )

        garments_text = "; ".join(
            garment_sentences
        )

        return (
            f"A cute original two-head-tall chibi "
            f"{outfit.character_species} character, "
            f"wearing {garments_text}. "
            f"The outfit has a {outfit.overall_style} style "
            f"with the color palette "
            f"{', '.join(outfit.overall_palette)}. "
            "The character has a very large round head, "
            "a tiny short rounded torso, tiny arms and tiny legs. "
            "Use simple clean 2D game character illustration, "
            "soft flat colors, simplified fabric textures, "
            "and rounded garment shapes. "
            "Preserve the exact original face, eyes, nose, mouth, "
            "ears, tail, pose, body proportions, line art, "
            "background and camera angle. "
            "Only redraw the masked clothing region."
        )

    def build_negative(self) -> str:
        return (
            "human, realistic person, human anatomy, "
            "adult body, realistic body, long torso, "
            "long arms, long legs, tall body, "
            "changed face, changed eyes, changed expression, "
            "missing ears, covered ears, missing tail, "
            "extra tail, extra arms, extra legs, extra limbs, "
            "photorealistic fabric, realistic skin, "
            "different pose, different camera angle, "
            "different background, blurry face, distorted body, "
            "text, logo, watermark"
        )