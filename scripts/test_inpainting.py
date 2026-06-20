from src.outfit_generator import OutfitGenerator


def main() -> None:
    generator = OutfitGenerator()

    positive_prompt = """
A cute original two-head-tall chibi cat character,
wearing a very short rounded khaki camping vest
with two tiny front pockets and a small zipper.
Very large round head, tiny rounded torso,
tiny arms and legs, simple clean 2D game illustration,
flat soft colors, preserve the exact original face,
eyes, ears, tail, pose, body proportions,
line art and background.
Only redraw the masked clothing region.
"""

    negative_prompt = """
human, realistic person, human anatomy,
long torso, long arms, long legs,
changed face, missing ears, missing tail,
extra limbs, photorealistic,
different pose, different background
"""

    output_path = generator.generate(
        character_path=(
            "datasets/characters/squirrel.png"
        ),
        mask_path=(
            "datasets/characters/masking/squirrel_all.png"
        ),
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
        output_path=(
            "datasets/outputs/test_inpainting.png"
        ),
        seed=42,
        width=768,
        height=768,
    )

    print("saved:", output_path)


if __name__ == "__main__":
    main()