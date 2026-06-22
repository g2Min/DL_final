from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def make_figure(
    results: list[dict],
    output_path: str | Path,
    *,
    cols: int = 3,
    cell_w: int = 512,
    cell_h: int = 512,
    gap: int = 8,
    label_h: int = 24,
    bg_color: tuple[int, int, int] = (240, 240, 240),
    font_size: int = 14,
) -> Path:
    """
    각 combo의 upper / final(lower) 이미지를 나란히 배치한 grid figure를 생성.
    가로 2칸을 차지하며 cols 값에 따라 행을 채움.
    """
    # 셀 크기 계산
    thumb_w = cell_w // 2        # upper / final 각각의 너비
    thumb_h = cell_h - label_h  # 라벨 공간 제외

    pair_w = thumb_w * 2 + gap   # 한 combo 쌍의 너비
    pair_h = cell_h              # 한 combo 쌍의 높이 (라벨 포함)

    n = len(results)
    n_cols = min(cols, n)
    n_rows = (n + n_cols - 1) // n_cols

    canvas_w = n_cols * pair_w + (n_cols - 1) * gap
    canvas_h = n_rows * pair_h + (n_rows - 1) * gap

    canvas = Image.new("RGB", (canvas_w, canvas_h), bg_color)
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            font_size,
        )
    except OSError:
        font = ImageFont.load_default()

    # 각 combo 배치
    for i, result in enumerate(results):
        row = i // n_cols
        col = i % n_cols

        x0 = col * (pair_w + gap)
        y0 = row * (pair_h + gap)

        upper_path = result.get("generated_paths", {}).get("upper")
        lower_path = result.get("generated_paths", {}).get("lower")

        # upper 썸네일
        if upper_path and Path(upper_path).exists():
            img = Image.open(upper_path).convert("RGB")
            img = img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        else:
            img = Image.new("RGB", (thumb_w, thumb_h), (200, 200, 200))

        canvas.paste(img, (x0, y0))

        # final (lower) 썸네일
        final_path = lower_path or upper_path  # lower 없으면 upper가 최종
        if final_path and Path(final_path).exists():
            img2 = Image.open(final_path).convert("RGB")
            img2 = img2.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        else:
            img2 = Image.new("RGB", (thumb_w, thumb_h), (200, 200, 200))

        canvas.paste(img2, (x0 + thumb_w + gap, y0))

        # 구분선
        draw.line(
            [(x0 + thumb_w + gap // 2, y0), (x0 + thumb_w + gap // 2, y0 + thumb_h)],
            fill=(180, 180, 180),
            width=1,
        )

        # 라벨
        combo_idx = result.get("combo_idx", i)
        label = f"combo {combo_idx:03d}  upper →  final"
        draw.text(
            (x0 + 4, y0 + thumb_h + 4),
            label,
            fill=(60, 60, 60),
            font=font,
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)

    return output_path
