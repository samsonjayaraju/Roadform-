import hashlib
import math
import os
import shutil
import tempfile
from contextlib import closing
from functools import lru_cache
from pathlib import Path
from threading import Lock

import cv2
import imageio_ffmpeg
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file
from PIL import Image, ImageOps, UnidentifiedImageError
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024
inference_lock = Lock()


@lru_cache(maxsize=1)
def detector():
    path = ROOT / "models/pothole.pt"
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    if checksum != "b19b75299ab48adc41929e9aa9e5d0f8f4cf2d119c6c6ae4a506d6b3b609cd69":
        raise RuntimeError("The bundled model failed its integrity check.")
    return YOLO(path)


def predict(source, confidence=0.1):
    with inference_lock:
        result = detector().predict(source, imgsz=800, conf=confidence, iou=0.45,
                                    device="cpu", verbose=False)[0]
        return result.boxes.data.cpu().tolist()


def image_result(stream):
    try:
        with Image.open(stream, formats=["JPEG", "PNG", "WEBP"]) as image:
            if image.width * image.height > 32_000_000:
                raise ValueError("Choose an image smaller than 32 megapixels.")
            image = ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("This image could not be read. Try a JPEG, PNG, or WebP.") from exc
    return {"width": image.width, "height": image.height, "boxes": predict(image)}


@lru_cache(maxsize=1)
def sample_result():
    return image_result(ROOT / "static/sample-road.jpg")


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/api/sample")
def sample():
    return jsonify(sample_result())


def draw_frame(frame, boxes):
    # FFmpeg supplies RGB frames. Only the inference copy is converted to BGR.
    height, width = frame.shape[:2]
    red = (230, 31, 50)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_height = max(12, round(max(width / 72, height / 39)))
    weight = max(1, round(text_height / 10))
    scale = cv2.getFontScaleFromHeight(font, text_height, weight)
    stroke = max(2, round(text_height / 8))
    pad = max(3, round(text_height / 5))
    labels = []
    for x1, y1, x2, y2, score, _ in boxes:
        left, top = max(0, int(x1)), max(0, int(y1))
        right, bottom = min(width, math.ceil(x2)), min(height, math.ceil(y2))
        if right <= left or bottom <= top:
            continue
        area = frame[top:bottom, left:right]
        cv2.addWeighted(area, 0.86, np.full_like(area, red), 0.14, 0, dst=area)
        cv2.rectangle(frame, (left, top), (right - 1, bottom - 1), red, stroke)
        labels.append((left, top, bottom, f"{score:.0%}"))

    occupied = []
    for left, top, bottom, label in labels:
        (tw, th), baseline = cv2.getTextSize(label, font, scale, weight)
        w, h = tw + 2 * pad, th + baseline + 2 * pad
        x = max(0, min(left, width - w))
        # Prefer the top edge, then try below or inside a crowded box.
        for candidate in (top - h, bottom, top):
            y = max(0, min(candidate, height - h))
            if all(x + w <= x0 or x >= x1 or y + h <= y0 or y >= y1
                   for x0, y0, x1, y1 in occupied):
                break
        occupied.append((x, y, x + w, y + h))
        cv2.rectangle(frame, (x, y), (x + w - 1, y + h - 1), (156, 17, 37), cv2.FILLED)
        cv2.putText(frame, label, (x + pad, y + pad + th), font, scale,
                    (255, 255, 255), weight, cv2.LINE_AA)


def video_result(upload, confidence):
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "source.mov"
        output = Path(directory) / "inspection.mp4"
        upload.save(source)
        reader = imageio_ffmpeg.read_frames(str(source), input_params=[
            "-protocol_whitelist", "file,pipe", "-f", "mov"
        ])
        with closing(reader):
            try:
                metadata = next(reader)
            except (OSError, RuntimeError, StopIteration) as exc:
                raise ValueError("This clip could not be read. Try an MP4 or MOV video.") from exc
            width, height = metadata["size"]
            fps = metadata["fps"]
            if not math.isfinite(fps) or fps <= 0:
                raise ValueError("The video frame rate could not be read.")
            writer = imageio_ffmpeg.write_frames(
                str(output), (width, height), fps=fps, codec="libx264",
                macro_block_size=1, output_params=["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                                                  "-movflags", "+faststart"]
            )
            frames = count = 0
            confidence_sum = 0.0
            with closing(writer):
                writer.send(None)
                for pixels in reader:
                    frame = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 3).copy()
                    boxes = predict(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), confidence)
                    draw_frame(frame, boxes)
                    writer.send(frame.tobytes())
                    count += len(boxes)
                    confidence_sum += sum(box[4] for box in boxes)
                    frames += 1
        if not frames:
            raise ValueError("The clip contains no readable frames.")
        # The response owns this file and closes it when playback data has been sent.
        media = tempfile.TemporaryFile()
        with output.open("rb") as encoded:
            shutil.copyfileobj(encoded, media)
        media.seek(0)
        response = send_file(media, mimetype="video/mp4", download_name="inspection.mp4")
        response.headers.update({"X-Frames": frames, "X-Detections": count,
                                 "X-Mean-Confidence": confidence_sum / count if count else 0,
                                 "X-Width": width, "X-Height": height})
        return response


@app.post("/api/inspect")
def inspect():
    upload = request.files.get("media")
    if not upload or not upload.filename:
        return jsonify(error="Choose a road image or video."), 400
    try:
        if Path(upload.filename).suffix.lower() in {".mp4", ".mov"}:
            confidence = float(request.form.get("confidence", "0.4"))
            if not 0.1 <= confidence <= 1:
                raise ValueError("Choose a confidence between 10% and 100%.")
            return video_result(upload, confidence)
        return jsonify(image_result(upload.stream))
    except ValueError as exc:
        return jsonify(error=str(exc)), 400


@app.errorhandler(413)
def too_large(_error):
    return jsonify(error="Choose a file smaller than 100 MB."), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "8080")), threaded=True)
