import os
import cv2
from tqdm import tqdm


class VideoCreator:
    def __init__(self, masks_dir, frames_dir, output_path="output/output.mp4", fps=20):
        self.masks_dir = masks_dir
        self.frames_dir = frames_dir
        self.output_path = output_path
        self.fps = fps

    def create_video(self, alpha_original=0.6, alpha_mask=0.4):
        print(f"Creating video: {self.output_path}, {self.fps} FPS")

        mask_files = sorted(
            [f for f in os.listdir(self.masks_dir) if f.endswith(".png")]
        )
        frame_files = sorted(
            [f for f in os.listdir(self.frames_dir) if f.endswith(".jpg")]
        )

        first_mask = cv2.imread(
            os.path.join(self.masks_dir, mask_files[0]), cv2.IMREAD_GRAYSCALE
        )
        h, w = first_mask.shape

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(self.output_path, fourcc, self.fps, (w, h))

        for i, (mask_file, frame_file) in enumerate(
            tqdm(
                zip(mask_files, frame_files),
                total=len(mask_files),
                desc="Creating video",
            )
        ):
            frame = cv2.imread(os.path.join(self.frames_dir, frame_file))
            mask = cv2.imread(
                os.path.join(self.masks_dir, mask_file), cv2.IMREAD_GRAYSCALE
            )

            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask_colored = mask_3ch.copy()
            mask_colored[mask > 127] = [0, 255, 0]

            combined = cv2.addWeighted(
                frame, alpha_original, mask_colored, alpha_mask, 0
            )
            out.write(combined)

        out.release()


class VideoExtractor:
    def __init__(self, target_fps=None, output_dir="frames_extracted"):
        self.target_fps = target_fps
        self.output_dir = output_dir

    def extract(self, video_path):
        cap = cv2.VideoCapture(video_path)

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(
            f"Original: {width}x{height}, {original_fps:.2f} FPS ({total_frames} frames)"
        )

        fps_to_use = self.target_fps if self.target_fps else original_fps
        frame_step = max(1, int(round(original_fps / fps_to_use)))

        print(f"Extraction Step: {frame_step} (Target FPS: {fps_to_use})")

        extracted_count = 0
        frame_idx = 0

        with tqdm(total=total_frames, desc="Extracting") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_step == 0:
                    filename = f"{extracted_count:06d}.jpg"
                    filepath = os.path.join(self.output_dir, filename)
                    cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    extracted_count += 1

                frame_idx += 1
                pbar.update(1)

        cap.release()
        return extracted_count, fps_to_use
