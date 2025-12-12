import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm


class MaskSmoother:
    def __init__(
        self,
        input_dir,
        output_dir="masks_smoothed",
        method="temporal_median",
        window_size=5,
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.method = method
        self.window_size = window_size
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def load_masks(self):
        mask_files = sorted(
            [f for f in os.listdir(self.input_dir) if f.endswith((".png", ".jpg"))]
        )
        masks = []
        for mask_file in tqdm(mask_files, desc="Loading masks"):
            mask_path = os.path.join(self.input_dir, mask_file)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = mask.astype(np.float32) / 255.0
            masks.append(mask)

        return masks, mask_files[: len(masks)]

    def median(self, masks, window_size=5):
        n = len(masks)
        half_window = window_size // 2
        smoothed = []

        for i in tqdm(range(n), desc="Median smoothing"):
            start = max(0, i - half_window)
            end = min(n, i + half_window + 1)

            window_masks = np.array([masks[j] for j in range(start, end)])
            smoothed_mask = np.median(window_masks, axis=0)
            smoothed_mask = np.where(smoothed_mask > 0.5, 1.0, 0.0)
            smoothed.append(smoothed_mask)

        return smoothed

    def gaussian(self, masks, window_size=5, sigma=1.0):
        n = len(masks)
        half_window = window_size // 2
        smoothed = []

        for i in tqdm(range(n), desc="Gaussian local smoothing"):
            start = max(0, i - half_window)
            end = min(n, i + half_window + 1)

            window_masks = [masks[j] for j in range(start, end)]
            local_indices = np.arange(len(window_masks))

            center = i - start
            distances = np.abs(local_indices - center)
            weights = np.exp(-(distances**2) / (2 * sigma**2))
            weights /= weights.sum()

            smoothed_mask = np.zeros_like(masks[0])
            for j, mask in enumerate(window_masks):
                smoothed_mask += mask * weights[j]

            smoothed_mask = np.where(smoothed_mask > 0.5, 1.0, 0.0)
            smoothed.append(smoothed_mask)

        return smoothed

    def save_masks(self, masks, mask_files, output_dir):
        for mask, mask_file in tqdm(
            zip(masks, mask_files), total=len(masks), desc="Saving"
        ):
            mask_uint8 = (mask * 255).astype(np.uint8)
            mask_binary = cv2.threshold(mask_uint8, 127, 255, cv2.THRESH_BINARY)[1]

            output_path = os.path.join(output_dir, mask_file)
            cv2.imwrite(output_path, mask_binary)

    def smooth(self, **kwargs):
        masks, mask_files = self.load_masks()

        if self.method == "median":
            smoothed = self.median(masks, window_size=self.window_size)

        elif self.method == "gaussian":
            sigma = kwargs["sigma"]
            smoothed = self.gaussian(masks, window_size=self.window_size, sigma=sigma)
        else:
            print(f"Not released: {self.method}")
            return False

        self.save_masks(smoothed, mask_files, self.output_dir)


class MaskComparator:
    def __init__(self, raw_dir, smoothed_dir, analysis_dir="analysis"):
        self.raw_dir = raw_dir
        self.smoothed_dir = smoothed_dir
        self.analysis_dir = analysis_dir

    def compare(self):
        raw_files = sorted([f for f in os.listdir(self.raw_dir) if f.endswith(".png")])
        smoothed_files = sorted(
            [f for f in os.listdir(self.smoothed_dir) if f.endswith(".png")]
        )

        iou_scores_smoothed = []

        for i in range(len(raw_files) - 1):
            smooth1 = (
                cv2.imread(
                    os.path.join(self.smoothed_dir, smoothed_files[i]),
                    cv2.IMREAD_GRAYSCALE,
                )
                > 127
            )
            smooth2 = (
                cv2.imread(
                    os.path.join(self.smoothed_dir, smoothed_files[i + 1]),
                    cv2.IMREAD_GRAYSCALE,
                )
                > 127
            )

            iou_smooth = self._compute_iou(smooth1, smooth2)
            iou_scores_smoothed.append(iou_smooth)

        return iou_scores_smoothed

    def _compute_iou(self, mask1, mask2):
        mask1_flat = mask1.flatten()
        mask2_flat = mask2.flatten()
        intersection = np.logical_and(mask1_flat, mask2_flat).sum()
        union = np.logical_or(mask1_flat, mask2_flat).sum()
        if union == 0:
            return 1.0
        return intersection / union

    def plot_comparison(self, iou_raw, iou_smoothed):
        _, ax = plt.subplots(figsize=(12, 5))

        ax.plot(iou_raw, linewidth=1.5, label="Raw masks", alpha=0.7)
        ax.plot(iou_smoothed, linewidth=1.5, label="Smoothed masks", alpha=0.7)
        ax.axhline(
            np.mean(iou_raw),
            color="blue",
            linestyle="--",
            alpha=0.5,
            label=f"Raw mean: {np.mean(iou_raw):.3f}",
        )
        ax.axhline(
            np.mean(iou_smoothed),
            color="orange",
            linestyle="--",
            alpha=0.5,
            label=f"Smoothed mean: {np.mean(iou_smoothed):.3f}",
        )

        ax.set_xlabel("Frame pair")
        ax.set_ylabel("IoU")
        ax.set_title("Сравнение стабильности: До и После сглаживания")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.analysis_dir, "comparison_before_after.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()


class MaskAnalyzer:
    def __init__(self, masks_dir, analysis_dir="analysis"):
        self.masks_dir = masks_dir
        self.analysis_dir = analysis_dir
        self.iou_scores = []
        self.frame_changes = []

    def compute_iou(self, mask1, mask2):
        mask1_flat = mask1.flatten()
        mask2_flat = mask2.flatten()
        intersection = np.logical_and(mask1_flat, mask2_flat).sum()
        union = np.logical_or(mask1_flat, mask2_flat).sum()
        if union == 0:
            return 1.0
        return intersection / union

    def analyze_stability(self):
        mask_files = sorted(
            [f for f in os.listdir(self.masks_dir) if f.endswith(".png")]
        )

        for i in range(len(mask_files) - 1):
            mask1 = (
                cv2.imread(
                    os.path.join(self.masks_dir, mask_files[i]), cv2.IMREAD_GRAYSCALE
                )
                > 127
            )
            mask2 = (
                cv2.imread(
                    os.path.join(self.masks_dir, mask_files[i + 1]),
                    cv2.IMREAD_GRAYSCALE,
                )
                > 127
            )

            iou = self.compute_iou(mask1, mask2)
            self.iou_scores.append(iou)

            change = np.abs(mask1.astype(float) - mask2.astype(float)).sum()
            self.frame_changes.append(change)

        print(
            f"IoU mean: {np.mean(self.iou_scores):.3f}, std: {np.std(self.iou_scores):.3f}"
        )
        print(
            f"Frame changes mean: {np.mean(self.frame_changes):.1f}, std: {np.std(self.frame_changes):.1f}"
        )

        return self.iou_scores, self.frame_changes

    def plot_stability(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

        ax1.plot(self.iou_scores, linewidth=1.5, label="IoU между кадрами")
        ax1.axhline(
            np.mean(self.iou_scores),
            color="r",
            linestyle="--",
            label=f"Mean: {np.mean(self.iou_scores):.3f}",
        )
        ax1.set_xlabel("Frame pair")
        ax1.set_ylabel("IoU")
        ax1.set_title("Нестабильность масок")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(self.frame_changes, linewidth=1.5, label="Pixel changes")
        ax2.axhline(
            np.mean(self.frame_changes),
            color="r",
            linestyle="--",
            label=f"Mean: {np.mean(self.frame_changes):.1f}",
        )
        ax2.set_xlabel("Frame pair")
        ax2.set_ylabel("Pixels changed")
        ax2.set_title("Покадровые изменения маски")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.analysis_dir, "stability_before.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
