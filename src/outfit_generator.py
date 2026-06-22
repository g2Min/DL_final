from pathlib import Path

import torch
from diffusers import StableDiffusionXLInpaintPipeline
from PIL import Image


def _composite_garment_images(
    images: list[Image.Image],
    size: int = 512,
) -> Image.Image:
    """여러 의상 이미지를 가로로 이어붙여 단일 IP-Adapter 참조 이미지 생성."""
    if len(images) == 1:
        return images[0].resize((size, size), Image.Resampling.LANCZOS)
    n = len(images)
    w = size // n
    canvas = Image.new("RGB", (size, size), (200, 200, 200))
    for i, img in enumerate(images):
        thumb = img.resize((w, size), Image.Resampling.LANCZOS)
        canvas.paste(thumb, (i * w, 0))
    return canvas


class OutfitGenerator:
    def __init__(
        self,
        model_id: str = (
            "diffusers/"
            "stable-diffusion-xl-1.0-inpainting-0.1"
        ),
        device: str | None = None,
        ip_adapter_repo: str = "h94/IP-Adapter",
        ip_adapter_weight: str = "ip-adapter_sdxl.bin",
        ip_adapter_scale: float = 0.5,
    ) -> None:
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        dtype = (
            torch.float16 if self.device == "cuda" else torch.float32
        )

        load_kwargs: dict = {"torch_dtype": dtype}
        if self.device == "cuda":
            load_kwargs["variant"] = "fp16"

        self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            model_id, **load_kwargs
        )

        # IP-Adapter는 CPU offload 설정 전에 로드해야 가중치가 올바르게 등록됨
        self.pipe.load_ip_adapter(
            ip_adapter_repo,
            subfolder="sdxl_models",
            weight_name=ip_adapter_weight,
        )
        self._ip_adapter_scale = ip_adapter_scale
        self.pipe.set_ip_adapter_scale(ip_adapter_scale)

        if self.device == "cuda":
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
        return (
            Image.open(image_path)
            .convert(mode)
            .resize(size, Image.Resampling.LANCZOS)
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
        ip_adapter_images: list[Image.Image] | None = None,
    ) -> Image.Image:
        mask_image = self.load_and_resize(
            mask_path, mode="L", size=(width, height)
        )
        if mask_blur > 0:
            mask_image = self.pipe.mask_processor.blur(
                mask_image, blur_factor=mask_blur
            )

        generator_device = "cuda" if self.device == "cuda" else "cpu"
        generator = (
            torch.Generator(device=generator_device).manual_seed(seed)
        )

        pipe_kwargs: dict = dict(
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
        )

        if ip_adapter_images:
            self.pipe.set_ip_adapter_scale(self._ip_adapter_scale)
            pipe_kwargs["ip_adapter_image"] = _composite_garment_images(
                ip_adapter_images
            )
        else:
            # 이미지가 없을 때 scale=0으로 IP-Adapter 영향을 무력화
            self.pipe.set_ip_adapter_scale(0.0)

        return self.pipe(**pipe_kwargs).images[0]

    def generate(
        self,
        *,
        character_path: str | Path,
        mask_path: str | Path,
        positive_prompt: str,
        negative_prompt: str,
        output_path: str | Path,
        base_image: Image.Image | None = None,
        ip_adapter_images: list[Image.Image] | None = None,
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
                (width, height), Image.Resampling.LANCZOS
            )
        else:
            init_image = self.load_and_resize(
                character_path, mode="RGB", size=(width, height)
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
            ip_adapter_images=ip_adapter_images,
        )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(output_path)
        return output_path, result
