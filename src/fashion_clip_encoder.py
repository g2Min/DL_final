from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoModelForZeroShotImageClassification,
    AutoProcessor,
)


class FashionClipEncoder:
    def __init__(
        self,
        model_id: str = "patrickjohncyh/fashion-clip",
        device: str | None = None,
    ):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.processor = AutoProcessor.from_pretrained(model_id)

        self.model = (
            AutoModelForZeroShotImageClassification
            .from_pretrained(model_id)
            .to(self.device)
            .eval()
        )

    @torch.inference_mode()
    def encode_images(
        self,
        image_paths: list[str],
    ) -> np.ndarray:
        images = [
            Image.open(path).convert("RGB")
            for path in image_paths
        ]

        inputs = self.processor(
            images=images,
            return_tensors="pt",
            padding=True,
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        embeddings = self.model.get_image_features(**inputs)

        embeddings = torch.nn.functional.normalize(
            embeddings,
            dim=-1,
        )

        return embeddings.cpu().numpy()

    @torch.inference_mode()
    def encode_texts(
        self,
        texts: list[str],
    ) -> np.ndarray:
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        embeddings = self.model.get_text_features(**inputs)

        embeddings = torch.nn.functional.normalize(
            embeddings,
            dim=-1,
        )

        return embeddings.cpu().numpy()