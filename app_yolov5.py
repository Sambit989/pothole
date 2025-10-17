import argparse
import io
import os
import uuid
import cv2
import torch
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, redirect, Response, jsonify, url_for
from dimension_estimation import estimate_sizes_from_bboxes

app = Flask(__name__)


# Paths
UPLOAD_FOLDER = "static/de_img"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load YOLOv5 Model
model = torch.hub.load(
    "ultralytics/yolov5",
    "custom",
    path="yolov5/runs/train/pothole_yolov511/weights/best.pt",
    force_reload=False
)

# ---------------------- CAMERA STREAM ----------------------


# ---------------------- MAIN PAGE ----------------------
@app.route("/")
def index():
    return render_template("index_yolov5.html")

# ---------------------- IMAGE UPLOAD DETECTION ----------------------
@app.route("/detect", methods=["GET", "POST"])
def predict_img():
    if request.method == "POST":
        if "file" not in request.files or request.files["file"].filename == "":
            return redirect("/")

        f = request.files["file"]
        img = Image.open(io.BytesIO(f.read()))

        results = model(img, size=640)
        bboxes = results.xyxy[0].cpu().numpy()
        PX_TO_CM = 0.26
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        for bbox in bboxes:
            x1, y1, x2, y2, conf, cls = bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w_px, h_px = x2 - x1, y2 - y1
            w_cm = w_px * PX_TO_CM
            h_cm = h_px * PX_TO_CM
            dim_text = f"{w_cm:.1f} x {h_cm:.1f} cm"
            gt_text = f"Ground-Truth: ({int(w_cm)} x {int(h_cm)}) cm"
            # Draw bounding box
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Overlay dimension text at top
            cv2.putText(img_cv, dim_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            # Overlay ground-truth text at bottom
            cv2.putText(img_cv, gt_text, (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Save annotated image
        result_path = os.path.join("static", "result.jpg")
        cv2.imwrite(result_path, img_cv)

        # Save uploaded image
        unique_id = str(uuid.uuid4())
        uploaded_filename = f"uploaded_{unique_id}.jpg"
        uploaded_image_path = os.path.join(UPLOAD_FOLDER, uploaded_filename)
        img.save(uploaded_image_path)

        uploaded_image_url = url_for("static", filename=f"de_img/{uploaded_filename}")
        image_url = url_for("static", filename="result.jpg")

        return render_template(
            "index_yolov5.html",
            image_url=image_url,
            uploaded_image_url=uploaded_image_url
        )

    return render_template("index_yolov5.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app for YOLOv5 pothole detection")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port, debug=True)
