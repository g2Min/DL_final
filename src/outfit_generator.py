from pathlib import Path

import torch
# from diffusers import AutoPipelineForInpainting
from diffusers import StableDiffusionXLInpaintPipeline
from PIL import Image


class OutfitGenerator:
    def __init__(
        self,
        model_id: str = (
            "diffusers/"
            "stable-diffusion-xl-1.0-inpainting-0.1"
        ),
        device: str | None = None,
    ) -> None:
        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        if self.device == "cuda":
            dtype = torch.float16
        else:
            dtype = torch.float32

        load_kwargs = {
            "torch_dtype": dtype,
        }

        if self.device == "cuda":
            load_kwargs["variant"] = "fp16"

        self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            model_id,
            **load_kwargs,
        )

        if self.device == "cuda":
            # VRAM이 부족한 경우 전체 모델을 GPU에
            # 고정하지 않고 필요할 때 올린다.
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe = self.pipe.to("cpu")

    @staticmethod
    def load_and_resize(
        image_path: str | Path,
        *,
        mode: str,
        size: tuple[int, int],
    ) -> Image.Image:
        image = Image.open(image_path).convert(mode)

        return image.resize(
            size,
            Image.Resampling.LANCZOS,
        )

    def _inpaint(
        self,
        *,
        init_image: Image.Image,
        mask_path: str | Path,
        positive_prompt: str,
        negative_prompt: str,
        seed: int,
        width: int,
        height: int,
        strength: float,
        guidance_scale: float,
        num_inference_steps: int,
        mask_blur: int,
    ) -> Image.Image:
        mask_image = self.load_and_resize(
            mask_path,
            mode="L",
            size=(width, height),
        )

        if mask_blur > 0:
            mask_image = self.pipe.mask_processor.blur(
                mask_image,
                blur_factor=mask_blur,
            )

        generator_device = (
            "cuda"
            if self.device == "cuda"
            else "cpu"
        )

        generator = torch.Generator(
            device=generator_device
        ).manual_seed(seed)

        return self.pipe(
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            mask_image=mask_image,
            width=width,
            height=height,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        ).images[0]

    def generate(
        self,
        *,
        character_path: str | Path,
        mask_path: str | Path,
        positive_prompt: str,
        negative_prompt: str,
        output_path: str | Path,
        base_image: Image.Image | None = None,
        seed: int = 42,
        width: int = 768,
        height: int = 768,
        strength: float = 0.85,
        guidance_scale: float = 7.0,
        num_inference_steps: int = 35,
        mask_blur: int = 8,
    ) -> tuple[Path, Image.Image]:
        if base_image is not None:
            init_image = base_image.resize(
                (width, height),
                Image.Resampling.LANCZOS,
            )
        else:
            init_image = self.load_and_resize(
                character_path,
                mode="RGB",
                size=(width, height),
            )

        result = self._inpaint(
            init_image=init_image,
            mask_path=mask_path,
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            width=width,
            height=height,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            mask_blur=mask_blur,
        )

        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        result.save(output_path)

        return output_path, result
