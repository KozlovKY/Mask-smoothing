import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from torchvision import transforms


class SemanticSegmentation:
    def __init__(
        self, model_name="deeplabv3_resnet50", output_dir="masks_raw", device=None
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.model = None
        self.preprocess = None
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

    def load_model(self):
        self.model = torch.hub.load(
            "pytorch/vision:v0.10.0", self.model_name, pretrained=True
        )
        self.model.eval()
        self.model.to(self.device)

        self.preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def process_frames(self, frames_dir):
        if self.model is None:
            self.load_model()

        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])

        existing_masks = len(
            [f for f in os.listdir(self.output_dir) if f.endswith(".png")]
        )
        if existing_masks == len(frame_files):
            print(
                " Masks already exist. Skipping segmentation (delete folder to re-run)."
            )
            return len(frame_files)

        for frame_file in tqdm(frame_files, desc="Segmentation"):
            input_image = Image.open(os.path.join(frames_dir, frame_file))
            input_tensor = self.preprocess(input_image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(input_tensor)["out"][0]

            output_predictions = output.argmax(0)
            mask = (output_predictions > 0).cpu().numpy().astype(np.uint8) * 255

            output_path = os.path.join(
                self.output_dir, frame_file.replace(".jpg", ".png")
            )
            cv2.imwrite(output_path, mask)

        return len(frame_files)
