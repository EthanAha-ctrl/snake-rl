import importlib.util
import os
from typing import Optional

import numpy as np
import torch
import torchvision.transforms as transforms


def _load_project2_get_model():
    module_path = os.path.join(os.path.dirname(__file__), "..", "project2", "train.py")
    spec = importlib.util.spec_from_file_location("project2_train_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load project2 train module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_model


_PROJECT2_GET_MODEL = None


def _get_project2_get_model():
    global _PROJECT2_GET_MODEL
    if _PROJECT2_GET_MODEL is None:
        _PROJECT2_GET_MODEL = _load_project2_get_model()
    return _PROJECT2_GET_MODEL


class VisionStateExtractor:
    """
    Uses the project2 vision model to predict (cart_position, pole_angle) from RGB frames.
    """

    def __init__(self, model_path: str = "position_detection.pth", device: Optional[torch.device] = None):
        self.model_path = model_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[torch.nn.Module] = None
        self.transform: Optional[transforms.Compose] = None

    def _ensure_initialized(self, frame: np.ndarray):
        if self.model is not None:
            return
        if frame is None:
            raise ValueError("Cannot initialize vision extractor without a frame.")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Vision model weights not found at {self.model_path}")

        img_h, img_w = frame.shape[:2]
        crop_size = min(img_h, img_w)
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(crop_size)])

        get_model = _get_project2_get_model()
        self.model = get_model(input_size=crop_size)
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, frame: np.ndarray) -> np.ndarray:
        self._ensure_initialized(frame)
        assert self.transform is not None
        assert self.model is not None

        input_tensor = self.transform(frame).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prediction = self.model(input_tensor).cpu().numpy()[0]
        return prediction.astype(np.float32)
