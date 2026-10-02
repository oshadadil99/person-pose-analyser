"""Manual fallback: click the bed corners if automatic detection got it wrong.

Opens the middle frame of the video. Left-click to add points, right-click to
undo the last one, Enter to save, Esc to quit without saving. If the scene
file already exists, only its bed is replaced and the detected seats are kept.

    python tools/draw_bed_polygon.py --video data/videos/room1.mp4 --scene outputs/room1/scene.json
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.scene import Scene, SceneObject, load_scene, save_scene
from src.video_io import get_video_info, read_frame_at, resize_to_width

WINDOW = "bed polygon: L-click add, R-click undo, Enter save, Esc quit"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--scene", required=True, help="scene.json to create or update")
    ap.add_argument("--max-width", type=int, default=1280, help="display size only")
    args = ap.parse_args()

    info = get_video_info(args.video)
    frame = read_frame_at(args.video, info.duration_sec / 2)
    if frame is None:
        sys.exit("Could not read a frame from the video")
    shown, scale = resize_to_width(frame, args.max_width)
    points: list[tuple[int, int]] = []   # display pixels

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)
    while True:
        img = shown.copy()
        if points:
            cv2.polylines(img, [np.array(points, np.int32)], len(points) > 2, (255, 120, 0), 2)
            for p in points:
                cv2.circle(img, p, 4, (0, 255, 255), -1)
        cv2.imshow(WINDOW, img)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            cv2.destroyAllWindows()
            sys.exit("Cancelled, nothing saved")
        if key == 13 and len(points) >= 3:
            break
    cv2.destroyAllWindows()

    polygon = [[round(x / scale, 1), round(y / scale, 1)] for x, y in points]   # back to original pixels
    bed = SceneObject("bed", "bed", polygon, conf=1.0, seen_ratio=1.0)
    path = Path(args.scene)
    if path.exists():
        scene = load_scene(path)
        scene.bed, scene.source = bed, "manual"
    else:
        scene = Scene(info.width, info.height, bed, [], source="manual")
    save_scene(path, scene)
    print(f"Saved bed polygon with {len(polygon)} points to {path}")


if __name__ == "__main__":
    main()
