import re
import cv2
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    "--path", type=str, default="out/events_*.png", help="Folder containing images"
)
parser.add_argument(
    "--output", type=str, default="output.mp4", help="Output video file name"
)
parser.add_argument(
    "--og_fps", type=int, default=1000, help="Original frames per second"
)
parser.add_argument(
    "--video_fps", type=int, default=50, help="Output video frames per second"
)
parser.add_argument(
    "--drop", action="store_true", help="Drop frames to match video FPS"
)

args = parser.parse_args()


def make_video(path, output, og_fps, video_fps):
    accum = og_fps / video_fps
    if accum < 1:
        raise ValueError("Video FPS must be less than or equal to original FPS.")

    images = sorted(
        glob(path),
        key=lambda f: tuple(map(int, re.findall(r"\d+", f))),
    )

    if not images:
        print("No images found in the specified folder.")
        return

    frame = cv2.imread(str(images[0]))
    height, width, layers = frame.shape

    # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # non compressed
    fourcc = cv2.VideoWriter_fourcc(*"HFYU")
    out = cv2.VideoWriter(output, fourcc, video_fps, (width, height))

    bucketed_img = np.zeros_like(frame, dtype=np.uint8)
    for i, image in tqdm(
        enumerate(images), total=len(images), desc="Processing images"
    ):
        img = cv2.imread(str(image))
        if not args.drop:
            bucketed_img[img == 255] = 255
            if (i + 1) % accum == 0:
                out.write(bucketed_img)
                bucketed_img = np.zeros_like(frame, dtype=np.uint8)
        else:
            if (i + 1) % accum == 0:
                out.write(img)
    out.release()
    print(f"Video saved as {output}")


if __name__ == "__main__":
    make_video(args.path, args.output, args.og_fps, args.video_fps)
