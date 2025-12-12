import os
import numpy as np
from pathlib import Path
from mask_utils import MaskAnalyzer, MaskSmoother, MaskComparator

from net_utils import SemanticSegmentation
from video_utils import VideoCreator, VideoExtractor

DIRS = {
    "frames": "./../output/frames_extracted",
    "masks_raw": "./../output/masks_raw",
    "masks_smoothed": "./../output/masks_smoothed",
    "analysis": "./../output/analysis",
    "output": "./../output/videos",
}


def setup_directories():
    for d in DIRS.values():
        Path(d).mkdir(parents=True, exist_ok=True)


def main(config):
    setup_directories()

    extractor = VideoExtractor(
        target_fps=config["target_fps"], output_dir=DIRS["frames"]
    )
    _, real_fps = extractor.extract(config["video_path"])

    seg = SemanticSegmentation(output_dir=DIRS["masks_raw"])
    seg.process_frames(DIRS["frames"])

    analyzer = MaskAnalyzer(DIRS["masks_raw"], DIRS["analysis"])
    iou_raw, _ = analyzer.analyze_stability()
    analyzer.plot_stability()

    smoother = MaskSmoother(
        input_dir=DIRS["masks_raw"],
        output_dir=DIRS["masks_smoothed"],
        method=config["smooth_method"],
        window_size=config["window_size"],
    )
    smoother.smooth(window_size=config["window_size"], sigma=config["sigma"])

    comparator = MaskComparator(
        DIRS["masks_raw"], DIRS["masks_smoothed"], DIRS["analysis"]
    )
    iou_smooth = comparator.compare()
    comparator.plot_comparison(iou_raw, iou_smooth)

    output_full_path = os.path.join(DIRS["output"], config["output_filename"])
    creator = VideoCreator(
        masks_dir=DIRS["masks_smoothed"],
        frames_dir=DIRS["frames"],
        output_path=output_full_path,
        fps=real_fps,
    )
    creator.create_video()

    improvement = (np.mean(iou_smooth) - np.mean(iou_raw)) / np.mean(iou_raw) * 100
    print(f"REPORT: {config['smooth_method']} (Window={config['window_size']})")
    print(f"Improvement: {improvement:+.2f}%")
    print(f"Saved to: {output_full_path}")


if __name__ == "__main__":
    CONFIG = {
        "video_path": "./../test.mp4",
        "target_fps": None,
        "smooth_method": "gaussian",  # median
        "window_size": 8,
        "sigma": 8 / 3,
        "output_filename": "median.mp4",
    }
    main(CONFIG)
