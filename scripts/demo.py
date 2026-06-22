"""
top-k 의상 조합 전체를 N개 GPU에서 병렬 생성하는 스크립트.

흐름:
  1. 메인 프로세스: 상황 분석 + 카테고리별 top-k 검색
  2. 메인 프로세스: itertools.product로 모든 조합 열거
  3. ThreadPoolExecutor: LLM 의상 변환(adapt)을 병렬 수행
  4. GPU 워커 (spawn): 각 GPU에서 2-pass inpainting 병렬 실행
  5. 결과 수집 및 JSON 저장
"""

import itertools
import json
import os
import signal
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Process, Queue
from pathlib import Path

from src.figure_maker import make_figure
from src.garment_adapter import GarmentAdapter
from src.garment_retriever import GarmentRetriever
from src.llm_client import StructuredLLMClient
from src.pipeline import AnimalOutfitPipeline
from src.prompt_builder import PromptBuilder
from src.situation_parser import SituationParser


# ── GPU 워커 ─────────────────────────────────────────────────────────────────

def _gpu_worker(
    gpu_id: int,
    tasks: list[dict],
    character_path: str,
    top_mask_path: str | None,
    bottom_mask_path: str | None,
    all_mask_path: str | None,
    output_dir: str,
    result_queue: Queue,
) -> None:
    """spawn으로 띄워진 워커. CUDA_VISIBLE_DEVICES 설정 후 모델 로드."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # spawn 이후 import — CUDA_VISIBLE_DEVICES가 먼저 설정된 상태
    from PIL import Image as PILImage

    from src.outfit_generator import OutfitGenerator
    from src.prompt_builder import PromptBuilder
    from src.schema import (
        FULL_BODY_CATEGORIES,
        LOWER_BODY_CATEGORIES,
        UPPER_BODY_CATEGORIES,
        CharacterOutfitSpec,
    )

    def _load_ip_images(
        selected: dict,
        categories: frozenset,
    ) -> list | None:
        images = []
        for cat, item in selected.items():
            if cat in categories:
                path = item.get("image_path", "")
                if path:
                    try:
                        images.append(PILImage.open(path).convert("RGB"))
                    except Exception:
                        pass
        return images if images else None

    import torch

    generator = OutfitGenerator()  # cuda:0 = 물리적 gpu_id
    prompt_builder = PromptBuilder()
    negative_prompt = prompt_builder.build_negative()
    out = Path(output_dir)

    for task in tasks:
        combo_idx: int = task["combo_idx"]
        seed: int = task["seed"]
        selected_items: dict = task["selected_items"]
        outfit_spec = CharacterOutfitSpec.model_validate(
            task["outfit_spec"]
        )

        generated_paths: dict[str, str] = {}
        current_image = None
        outfit_categories = {
            g.category for g in outfit_spec.garments
        }

        # 전신 pass (swimwear) — 단일 pass로 처리
        if outfit_categories & FULL_BODY_CATEGORIES and all_mask_path:
            pos = prompt_builder.build_positive_for_categories(
                outfit_spec, FULL_BODY_CATEGORIES
            )
            path, current_image = generator.generate(
                character_path=character_path,
                mask_path=all_mask_path,
                positive_prompt=pos,
                negative_prompt=negative_prompt,
                output_path=out / f"combo_{combo_idx:03d}_full.png",
                ip_adapter_images=_load_ip_images(
                    selected_items, FULL_BODY_CATEGORIES
                ),
                seed=seed,
            )
            generated_paths["full"] = str(path)

        # 상체 pass
        if outfit_categories & UPPER_BODY_CATEGORIES and top_mask_path:
            pos = prompt_builder.build_positive_for_categories(
                outfit_spec, UPPER_BODY_CATEGORIES
            )
            path, current_image = generator.generate(
                character_path=character_path,
                mask_path=top_mask_path,
                positive_prompt=pos,
                negative_prompt=negative_prompt,
                output_path=out / f"combo_{combo_idx:03d}_upper.png",
                base_image=current_image,
                ip_adapter_images=_load_ip_images(
                    selected_items, UPPER_BODY_CATEGORIES
                ),
                seed=seed,
            )
            generated_paths["upper"] = str(path)

        # 하체 pass (상체/전신 결과를 base_image로)
        if outfit_categories & LOWER_BODY_CATEGORIES and bottom_mask_path:
            pos = prompt_builder.build_positive_for_categories(
                outfit_spec, LOWER_BODY_CATEGORIES
            )
            path, current_image = generator.generate(
                character_path=character_path,
                mask_path=bottom_mask_path,
                positive_prompt=pos,
                negative_prompt=negative_prompt,
                output_path=out / f"combo_{combo_idx:03d}_lower.png",
                base_image=current_image,
                ip_adapter_images=_load_ip_images(
                    selected_items, LOWER_BODY_CATEGORIES
                ),
                seed=seed,
            )
            generated_paths["lower"] = str(path)

        result_queue.put(
            {
                "combo_idx": combo_idx,
                "gpu_id": gpu_id,
                "generated_paths": generated_paths,
                "outfit_spec": task["outfit_spec"],
                "selected_items": task["selected_items"],
            }
        )
        torch.cuda.empty_cache()

    result_queue.put(None)  # sentinel: 이 워커 완료


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Animal outfit pipeline")
    parser.add_argument("--user-text", type=str, default=(
        "다람쥐 캐릭터가 가을에 캠핑을 가. "
        "따뜻하고 움직이기 편한 옷을 추천해서 "
        "입혀줘. 귀와 꼬리는 가리지 마."
    ))
    parser.add_argument("--character-path", type=str,
                        default="datasets/characters/squirrel.png")
    parser.add_argument("--top-mask-path", type=str,
                        default="datasets/characters/masking/squirrel/squirrel_top.png")
    parser.add_argument("--bottom-mask-path", type=str,
                        default="datasets/characters/masking/squirrel/squirrel_bottom.png")
    parser.add_argument("--all-mask-path", type=str,
                        default="datasets/characters/masking/squirrel/squirrel_all.png")
    parser.add_argument("--output-dir", type=str,
                        default="datasets/outputs/camping_parallel2")
    parser.add_argument("--num-gpus", type=int, default=7)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    NUM_GPUS = args.num_gpus
    TOP_K = args.top_k
    SEED = args.seed

    USER_TEXT = args.user_text
    CHARACTER_PATH = args.character_path
    TOP_MASK_PATH = args.top_mask_path or None
    BOTTOM_MASK_PATH = args.bottom_mask_path or None
    ALL_MASK_PATH = args.all_mask_path or None
    OUTPUT_DIR = args.output_dir

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # 1. 상황 분석(SituationParser) + 의상 검색 (GarmentRetriever)
    print("▶ 상황 분석 및 의상 검색 중...")
    llm_client = StructuredLLMClient(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        base_url="http://localhost:8000/v1",
    )
    pipeline = AnimalOutfitPipeline(
        situation_parser=SituationParser(llm_client),
        # 카테고리별 top-k 검색
        retriever=GarmentRetriever( 
            metadata_path="datasets/garments/metadata.jsonl",
            embeddings_path="datasets/garments/embeddings.npy",
        ),
        garment_adapter=GarmentAdapter(llm_client),
        prompt_builder=PromptBuilder(),
        generator=None,  # type: ignore[arg-type]  # 메인에선 생성 안 함
    )

    analysis, candidates, combinations = pipeline.retrieve_combinations(
        user_text=USER_TEXT,
        top_k=TOP_K,
    )

    n_cats = len(candidates)
    n_combos = len(combinations)
    print(
        f"  카테고리 {n_cats}개 × top-{TOP_K} → "
        f"총 {n_combos}개 조합"
    )

    if not combinations:
        raise RuntimeError("검색된 의상 조합이 없습니다.")

    # 2. 조합별 의상 명세 변환 (LLM, I/O bound → 스레드 병렬) 
    print(f"\n▶ 의상 명세 변환 중 (ThreadPoolExecutor)...")
    adapter = GarmentAdapter(llm_client)
    adapted: dict[int, dict] = {}

    def _adapt(idx_items: tuple[int, dict]) -> tuple[int, dict]:
        idx, selected = idx_items
        spec = adapter.adapt(
            analysis=analysis,
            selected_items=selected,
        )
        return idx, spec.model_dump()

    with ThreadPoolExecutor(
        max_workers=min(8, n_combos)
    ) as pool:
        futures = {
            pool.submit(_adapt, (idx, combo)): idx
            for idx, combo in enumerate(combinations)
        }
        for fut in as_completed(futures):
            idx, spec_dict = fut.result()
            adapted[idx] = spec_dict
            print(f"  [{len(adapted):>{len(str(n_combos))}}/{n_combos}] 조합 {idx:03d} 변환 완료")

    # 3. 태스크 리스트 구성 (top-k) + 여러 조합인 경우 GPU별 분배
    tasks = [
        {
            "combo_idx": idx,
            "outfit_spec": adapted[idx],
            "selected_items": combinations[idx],
            "seed": SEED,
        }
        for idx in range(n_combos)
    ]

    num_workers = min(NUM_GPUS, n_combos)
    # 라운드로빈으로 분배: tasks[0]→gpu0, tasks[1]→gpu1, ...
    chunks: list[list[dict]] = [[] for _ in range(num_workers)]
    for idx, task in enumerate(tasks):
        chunks[idx % num_workers].append(task)

    # 4. GPU 워커 프로세스 실행
    print(f"\n▶ {num_workers}개 GPU에서 이미지 생성 시작...")
    result_queue: Queue = Queue()
    processes: list[Process] = []

    def _terminate_all(signum=None, frame=None):
        print("\n중단 신호 수신 — 워커 프로세스 종료 중...")
        for p in processes:
            if p.is_alive():
                p.terminate()
        for p in processes:
            p.join(timeout=5)
        sys.exit(1)

    signal.signal(signal.SIGINT, _terminate_all)
    signal.signal(signal.SIGTERM, _terminate_all)

    for gpu_id, chunk in enumerate(chunks):
        if not chunk:
            continue
        p = Process(
            target=_gpu_worker,
            args=(
                gpu_id,
                chunk,
                CHARACTER_PATH,
                TOP_MASK_PATH,
                BOTTOM_MASK_PATH,
                ALL_MASK_PATH,
                OUTPUT_DIR,
                result_queue,
            ),
        )
        p.start()
        processes.append(p)
        print(f"  GPU {gpu_id}: {len(chunk)}개 조합 할당")

    # 5. 결과 수집
    all_results: list[dict] = []
    sentinels = 0

    while sentinels < len(processes):
        item = result_queue.get()
        if item is None:
            sentinels += 1
        else:
            all_results.append(item)
            done = len(all_results)
            gpu = item["gpu_id"]
            paths = list(item["generated_paths"].values())
            print(
                f"  [{done:>{len(str(n_combos))}}/{n_combos}] "
                f"조합 {item['combo_idx']:03d} 완료 (GPU {gpu}) → {paths}"
            )

    for p in processes:
        p.join()

    all_results.sort(key=lambda r: r["combo_idx"])

    # 6. 결과 저장
    result_data = {
        "analysis": analysis.model_dump(),
        "candidates": candidates,
        "total_combinations": n_combos,
        "results": all_results,
    }

    result_path = Path(OUTPUT_DIR) / "results.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    figure_path = make_figure(
        all_results,
        Path(OUTPUT_DIR) / "figure.png",
        cols=TOP_K,
    )

    print(f"\n✓ 완료: {len(all_results)}/{n_combos}개 조합 생성")
    print(f"  결과 저장: {result_path}")
    print(f"  figure  : {figure_path}")


if __name__ == "__main__":
    # CUDA + fork 충돌 방지 → 반드시 spawn 사용
    from multiprocessing import set_start_method
    set_start_method("spawn", force=True)
    main()
