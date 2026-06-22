"""
fastapi 로 백엔드 구현

엔드포인트:
  GET  /api/characters          - list available characters
  GET  /api/backgrounds         - list background images
  POST /api/generate            - start a pipeline job
  GET  /api/jobs/{job_id}       - poll job status + results

고정 마운트:
  /static/characters/**  → datasets/characters/
  /static/backgrounds/** → datasets/assets/
  /static/outputs/**     → datasets/outputs/
"""

import asyncio
import json
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageFilter
from pydantic import BaseModel

BASE_DIR = Path(__file__).parent.resolve()
_OUTPUTS_DIR = BASE_DIR / "datasets/outputs"

CHARACTER_REGISTRY: dict[str, dict] = {
    "squirrel": {
        "id": "squirrel",
        "name": "다람쥐",
        "image_url": "/static/characters/squirrel.png",
        "character_path": str(BASE_DIR / "datasets/characters/squirrel.png"),
        "transparent_path": str(BASE_DIR / "datasets/characters_tranapent/squirrel.png"),
        "top_mask_path": str(BASE_DIR / "datasets/characters/masking/squirrel/squirrel_top.png"),
        "bottom_mask_path": str(BASE_DIR / "datasets/characters/masking/squirrel/squirrel_bottom.png"),
        "all_mask_path": str(BASE_DIR / "datasets/characters/masking/squirrel/squirrel_all.png"),
        "has_masks": True,
    },
    "raccoon": {
        "id": "raccoon",
        "name": "라쿤",
        "image_url": "/static/characters/raccoon.png",
        "character_path": str(BASE_DIR / "datasets/characters/raccoon.png"),
        "transparent_path": str(BASE_DIR / "datasets/characters_transparent/raccoon.png"),
        "top_mask_path": str(BASE_DIR / "datasets/characters/masking/raccoon/raccoon_top.png"),
        "bottom_mask_path": str(BASE_DIR / "datasets/characters/masking/raccoon/raccoon_bottom.png"),
        "all_mask_path": str(BASE_DIR / "datasets/characters/masking/raccoon/raccoon_all.png"),
        "has_masks": True,
    },
    "bear": {
        "id": "bear",
        "name": "곰",
        "image_url": "/static/characters/bear.png",
        "character_path": str(BASE_DIR / "datasets/characters/bear.png"),
        "transparent_path": str(BASE_DIR / "datasets/characters/bear.png"),
        "top_mask_path": str(BASE_DIR / "datasets/characters/masking/bear/bear_top.png"),
        "bottom_mask_path": str(BASE_DIR / "datasets/characters/masking/bear/bear_bottom.png"),
        "all_mask_path": str(BASE_DIR / "datasets/characters/masking/bear/bear_all.png"),
        "has_masks": True,
    },
    "cat": {
        "id": "cat",
        "name": "고양이",
        "image_url": "/static/characters/cat.png",
        "character_path": str(BASE_DIR / "datasets/characters/cat.png"),
        "transparent_path": str(BASE_DIR / "datasets/characters/cat.png"),
        "top_mask_path": str(BASE_DIR / "datasets/characters/masking/cat/cat_top.png"),
        "bottom_mask_path": str(BASE_DIR / "datasets/characters/masking/cat/cat_bottom.png"),
        "all_mask_path": str(BASE_DIR / "datasets/characters/masking/cat/cat_all.png"),
        "has_masks": True,
    },
    "hamster": {
        "id": "hamster",
        "name": "햄스터",
        "image_url": "/static/characters/hamster.png",
        "character_path": str(BASE_DIR / "datasets/characters/hamster.png"),
        "transparent_path": str(BASE_DIR / "datasets/characters/hamster.png"),
        "top_mask_path": str(BASE_DIR / "datasets/characters/masking/hamster/hamster_top.png"),
        "bottom_mask_path": str(BASE_DIR / "datasets/characters/masking/hamster/hamster_bottom.png"),
        "all_mask_path": str(BASE_DIR / "datasets/characters/masking/hamster/hamster_all.png"),
        "has_masks": True,
    },
    "sheep": {
        "id": "sheep",
        "name": "강아지",
        "image_url": "/static/characters/sheep.png",
        "character_path": str(BASE_DIR / "datasets/characters/sheep.png"),
        "transparent_path": str(BASE_DIR / "datasets/characters/sheep.png"),
        "top_mask_path": str(BASE_DIR / "datasets/characters/masking/sheep/sheep_top.png"),
        "bottom_mask_path": str(BASE_DIR / "datasets/characters/masking/sheep/sheep_bottom.png"),
        "all_mask_path": str(BASE_DIR / "datasets/characters/masking/sheep/sheep_all.png"),
        "has_masks": True,
    },
}

# In-memory job store
jobs: dict[str, dict] = {}

app = FastAPI(title="Animal Outfit Generator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static 파일 마운트
app.mount(
    "/static/characters",
    StaticFiles(directory=str(BASE_DIR / "datasets/characters")),
    name="characters",
)

_bg_dir = BASE_DIR / "datasets/assets"
_bg_dir.mkdir(parents=True, exist_ok=True)
app.mount(
    "/static/backgrounds",
    StaticFiles(directory=str(_bg_dir)),
    name="backgrounds",
)

_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount(
    "/static/outputs",
    StaticFiles(directory=str(_OUTPUTS_DIR)),
    name="outputs",
)

_transparent_dir = BASE_DIR / "datasets/assets/characters_transparent"
_transparent_dir.mkdir(parents=True, exist_ok=True)
app.mount(
    "/static/characters_transparent",
    StaticFiles(directory=str(_transparent_dir)),
    name="characters_transparent",
)

# 배경 제거 custom func (more_masks.ipynb)
def _remove_outer_white_background(
    input_path: Path,
    output_path: Path,
    white_threshold: int = 243,
    feather_radius: int = 2,
) -> None:
    image = Image.open(input_path).convert("RGBA")
    arr = np.array(image)
    height, width = arr.shape[:2]
    rgb = arr[..., :3]

    white_candidate = np.all(rgb >= white_threshold, axis=2)

    background = np.zeros((height, width), dtype=bool)
    visited = np.zeros((height, width), dtype=bool)
    queue: deque = deque()

    for x in range(width):
        queue.append((x, 0))
        queue.append((x, height - 1))
    for y in range(height):
        queue.append((0, y))
        queue.append((width - 1, y))

    while queue:
        x, y = queue.popleft()
        if visited[y, x]:
            continue
        visited[y, x] = True
        if not white_candidate[y, x]:
            continue
        background[y, x] = True
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and not visited[ny, nx]:
                queue.append((nx, ny))

    alpha = np.where(background, 0, 255).astype(np.uint8)

    if feather_radius > 0:
        alpha_image = Image.fromarray(alpha, mode="L")
        alpha_image = alpha_image.filter(ImageFilter.GaussianBlur(radius=feather_radius))
        alpha = np.array(alpha_image)
        alpha[~background & (alpha > 220)] = 255

    result = arr.copy()
    result[..., 3] = alpha
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result, mode="RGBA").save(output_path)


class MakeTransparentRequest(BaseModel):
    image_url: str

# 생성된 의상 배경 투명화 api
@app.post("/api/make-transparent")
async def make_transparent(req: MakeTransparentRequest):
    prefix = "/static/outputs/"
    if not req.image_url.startswith(prefix):
        raise HTTPException(status_code=400, detail="Only /static/outputs/ images are supported")

    rel = req.image_url[len(prefix):]
    input_path = _OUTPUTS_DIR / rel
    if not input_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    output_path = input_path.parent / (input_path.stem + "_transparent.png")

    if not output_path.exists():
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            _io_pool,
            lambda: _remove_outer_white_background(input_path, output_path),
        )

    rel_out = output_path.relative_to(_OUTPUTS_DIR)
    return {"transparent_url": f"/static/outputs/{rel_out}"}


# 생성 api가 올때마다 모델 재로딩되는걸 방지하기 위해 warmup 단계에 모두 로딩
_pipeline = None   # AnimalOutfitPipeline
_generator = None  # OutfitGenerator (SDXL + IP-Adapter)
_io_pool = ThreadPoolExecutor(max_workers=4)   # LLM/IO 병렬
_gpu_pool = ThreadPoolExecutor(max_workers=1)  # GPU 직렬화


@app.on_event("startup")
async def _startup() -> None:
    global _pipeline, _generator

    from src.garment_adapter import GarmentAdapter
    from src.garment_retriever import GarmentRetriever
    from src.llm_client import StructuredLLMClient
    from src.outfit_generator import OutfitGenerator
    from src.pipeline import AnimalOutfitPipeline
    from src.prompt_builder import PromptBuilder
    from src.situation_parser import SituationParser

    llm = StructuredLLMClient(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        base_url="http://localhost:8000/v1",
    )
    _pipeline = AnimalOutfitPipeline(
        situation_parser=SituationParser(llm),
        retriever=GarmentRetriever(
            metadata_path=str(BASE_DIR / "datasets/garments/metadata.jsonl"),
            embeddings_path=str(BASE_DIR / "datasets/garments/embeddings.npy"),
        ),
        garment_adapter=GarmentAdapter(llm),
        prompt_builder=PromptBuilder(),
        generator=None,  # type: ignore[arg-type]
    )

    print("Loading OutfitGenerator (SDXL + IP-Adapter)…")
    loop = asyncio.get_event_loop()
    _generator = await loop.run_in_executor(_gpu_pool, OutfitGenerator)
    print("✓ OutfitGenerator ready")


# 고정 이미지 assets api
@app.get("/api/characters")
def get_characters():
    return [
        {
            "id": c["id"],
            "name": c["name"],
            "image_url": c["transparent_path"],
            "has_masks": c["has_masks"],
        }
        for c in CHARACTER_REGISTRY.values()
    ]


@app.get("/api/backgrounds")
def get_backgrounds():
    bg_dir = BASE_DIR / "datasets/assets"
    valid_ext = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
    for f in sorted(bg_dir.iterdir()):
        if f.is_file() and f.suffix.lower() in valid_ext:
            return {"url": f"/static/backgrounds/{f.name}"}
    return {"url": None}


class GenerateRequest(BaseModel):
    user_text: str
    character_id: str
    top_n: int = 3


# 생성 이미지 api
@app.post("/api/generate")
async def generate(req: GenerateRequest):
    if req.character_id not in CHARACTER_REGISTRY:
        raise HTTPException(status_code=400, detail="Unknown character_id")

    char = CHARACTER_REGISTRY[req.character_id]
    if not char["has_masks"]:
        raise HTTPException(
            status_code=400,
            detail=f"'{char['name']}'은(는) 아직 마스크 데이터가 없어요.",
        )

    top_n = max(1, min(req.top_n, 10))
    job_id = uuid.uuid4().hex[:8]
    output_dir = str(_OUTPUTS_DIR / f"web_{job_id}")

    jobs[job_id] = {
        "status": "pending",
        "progress": [],
        "output_dir": output_dir,
        "character_id": req.character_id,
        "user_text": req.user_text,
        "top_n": top_n,
        "results": None,
        "error": None,
    }

    asyncio.create_task(_run_pipeline(job_id, req.user_text, char, output_dir, top_n))
    return {"job_id": job_id}

# 작업 진행상황 전달 api
@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    j = jobs[job_id]
    return {
        "status": j["status"],
        "progress": j["progress"][-40:],
        "results": j["results"],
        "error": j["error"],
    }


# Pipeline runner (subprocess 없이 직접 실행)
# 비동기 실행
async def _run_pipeline(
    job_id: str,
    user_text: str,
    char: dict,
    output_dir: str,
    top_n: int,
) -> None:
    jobs[job_id]["status"] = "running"
    loop = asyncio.get_event_loop()

    def log(msg: str) -> None:
        jobs[job_id]["progress"].append(msg)
        print(msg)

    try:
        from PIL import Image as PILImage
        from src.schema import (
            FULL_BODY_CATEGORIES,
            LOWER_BODY_CATEGORIES,
            UPPER_BODY_CATEGORIES,
        )

        # 1. 상황 분석 & 의상 검색
        log("▶ 상황 분석 및 의상 검색 중...")
        analysis, candidates, combinations = await loop.run_in_executor(
            _io_pool,
            lambda: _pipeline.retrieve_combinations(user_text=user_text, top_k=top_n),
        )
        n_combos = len(combinations)
        log(f"  {len(candidates)}개 카테고리 × top-{top_n} → 총 {n_combos}개 조합")

        if not combinations:
            raise RuntimeError("검색된 의상이 없습니다.")

        # 2. 의상 명세 변환 (LLM 병렬)
        log("▶ 의상 명세 변환 중...")
        adapted: dict[int, object] = {}

        async def _adapt(idx: int, selected: dict) -> None:
            spec = await loop.run_in_executor(
                _io_pool,
                lambda: _pipeline.garment_adapter.adapt(
                    analysis=analysis, selected_items=selected
                ),
            )
            adapted[idx] = spec
            log(f"  [{len(adapted)}/{n_combos}] 조합 {idx:03d} 변환 완료")

        await asyncio.gather(*[_adapt(i, combo) for i, combo in enumerate(combinations)])

        # 3. 이미지 생성 (GPU 직렬) 
        log("▶ 이미지 생성 시작...")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        negative_prompt = _pipeline.prompt_builder.build_negative()
        all_results = []

        def _load_ip(selected: dict, cats) -> list | None:
            imgs = []
            for cat, item in selected.items():
                if cat in cats:
                    p = item.get("image_path", "")
                    if p:
                        try:
                            imgs.append(PILImage.open(p).convert("RGB"))
                        except Exception:
                            pass
            return imgs or None

        for idx in range(n_combos):
            outfit_spec = adapted[idx]
            selected = combinations[idx]
            outfit_categories = {g.category for g in outfit_spec.garments}
            generated_paths: dict[str, str] = {}
            current_image = None

            # 전체 outfit 생성
            if outfit_categories & FULL_BODY_CATEGORIES and char.get("all_mask_path"):
                pos = _pipeline.prompt_builder.build_positive_for_categories(
                    outfit_spec, FULL_BODY_CATEGORIES
                )
                out_path = str(Path(output_dir) / f"combo_{idx:03d}_full.png")
                _ci, _ip = current_image, _load_ip(selected, FULL_BODY_CATEGORIES)
                path, current_image = await loop.run_in_executor(
                    _gpu_pool,
                    lambda: _generator.generate(
                        character_path=char["character_path"],
                        mask_path=char["all_mask_path"],
                        positive_prompt=pos,
                        negative_prompt=negative_prompt,
                        output_path=out_path,
                        base_image=_ci,
                        ip_adapter_images=_ip,
                        seed=42,
                    ),
                )
                generated_paths["full"] = str(path)

            # 상의 outfit 생성 과정
            if outfit_categories & UPPER_BODY_CATEGORIES and char.get("top_mask_path"):
                pos = _pipeline.prompt_builder.build_positive_for_categories(
                    outfit_spec, UPPER_BODY_CATEGORIES
                )
                out_path = str(Path(output_dir) / f"combo_{idx:03d}_upper.png")
                _ci, _ip = current_image, _load_ip(selected, UPPER_BODY_CATEGORIES)
                path, current_image = await loop.run_in_executor(
                    _gpu_pool,
                    lambda: _generator.generate(
                        character_path=char["character_path"],
                        mask_path=char["top_mask_path"],
                        positive_prompt=pos,
                        negative_prompt=negative_prompt,
                        output_path=out_path,
                        base_image=_ci,
                        ip_adapter_images=_ip,
                        seed=42,
                    ),
                )
                generated_paths["upper"] = str(path)

            # 하의 outfit 생성 과정
            if outfit_categories & LOWER_BODY_CATEGORIES and char.get("bottom_mask_path"):
                pos = _pipeline.prompt_builder.build_positive_for_categories(
                    outfit_spec, LOWER_BODY_CATEGORIES
                )
                out_path = str(Path(output_dir) / f"combo_{idx:03d}_lower.png")
                _ci, _ip = current_image, _load_ip(selected, LOWER_BODY_CATEGORIES)
                path, current_image = await loop.run_in_executor(
                    _gpu_pool,
                    lambda: _generator.generate(
                        character_path=char["character_path"],
                        mask_path=char["bottom_mask_path"],
                        positive_prompt=pos,
                        negative_prompt=negative_prompt,
                        output_path=out_path,
                        base_image=_ci,
                        ip_adapter_images=_ip,
                        seed=42,
                    ),
                )
                generated_paths["lower"] = str(path)

            all_results.append({
                "combo_idx": idx,
                "generated_paths": generated_paths,
                "outfit_spec": outfit_spec.model_dump(),
                "selected_items": selected,
            })
            log(
                f"  [{idx + 1}/{n_combos}] 조합 {idx:03d} 완료"
                f" → {list(generated_paths.values())}"
            )

        # 4. 결과 저장
        results_path = Path(output_dir) / "results.json"
        results_path.write_text(
            json.dumps(
                {
                    "analysis": analysis.model_dump(),
                    "candidates": candidates,
                    "total_combinations": n_combos,
                    "results": all_results,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["results"] = _collect_top_n(output_dir, top_n)
        log(f"✓ 완료: {n_combos}개 조합 생성  →  {output_dir}")

    except Exception as exc:
        import traceback
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(exc)
        log(traceback.format_exc())


# helpers 
def _collect_top_n(output_dir: str, n: int = 3) -> list[dict]:
    results_path = Path(output_dir) / "results.json"
    if not results_path.exists():
        return _scan_combo_images(output_dir, n)

    data = json.loads(results_path.read_text(encoding="utf-8"))
    results = []
    for r in sorted(data.get("results", []), key=lambda x: x["combo_idx"])[:n]:
        paths = r.get("generated_paths", {})
        final = paths.get("lower") or paths.get("upper") or paths.get("full")
        if final:
            try:
                rel = Path(final).relative_to(_OUTPUTS_DIR)
                results.append({
                    "combo_idx": r["combo_idx"],
                    "image_url": f"/static/outputs/{rel}",
                })
            except ValueError:
                pass
    return results


def _scan_combo_images(output_dir: str, n: int = 3) -> list[dict]:
    out = Path(output_dir)
    combos: dict[int, Path] = {}
    for p in sorted(out.glob("combo_*.png")):
        parts = p.stem.split("_")
        if len(parts) >= 2:
            try:
                idx = int(parts[1])
                if idx not in combos or p.stem > combos[idx].stem:
                    combos[idx] = p
            except ValueError:
                pass

    result = []
    for idx in sorted(combos)[:n]:
        p = combos[idx]
        try:
            rel = p.relative_to(_OUTPUTS_DIR)
            result.append({"combo_idx": idx, "image_url": f"/static/outputs/{rel}"})
        except ValueError:
            pass
    return result
