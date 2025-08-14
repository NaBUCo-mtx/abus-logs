import cv2
import os
import re
from datetime import datetime
from open_image_models import LicensePlateDetector
from fast_plate_ocr import LicensePlateRecognizer
from flask import Flask, render_template_string, send_from_directory, send_file

# --- ADD PICAMERA2 IMPORT ---
from picamera2 import Picamera2

# --- CONFIGURATION ---
IMAGE_FOLDER = "detected_plates"
CSV_FILE = "placas.csv"

# --- SETUP ---
os.makedirs(IMAGE_FOLDER, exist_ok=True)
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w") as f:
        f.write("plate,date,time,filename\n")

# --- PLATE FORMATS ---
def clean_plate(text):
    return re.sub(r'[^A-Za-z0-9]', '', text)

def is_valid_plate(plate):
    return (
        re.fullmatch(r'[A-Za-z]{3}\d{3}', plate) or
        re.fullmatch(r'[A-Za-z]{3}\d{4}', plate)
    )

# --- LOAD MODELS ---
plate_detector = LicensePlateDetector(detection_model="yolo-v9-t-256-license-plate-end2end")
plate_recognizer = LicensePlateRecognizer('cct-xs-v1-global-model')

# --- FLASK APP ---
app = Flask(__name__)

@app.route("/")
def index():
    # Read last 5 entries from CSV
    with open(CSV_FILE, "r") as f:
        lines = f.readlines()[1:]  # skip header
    last5 = lines[-5:] if len(lines) >= 5 else lines
    last5 = [line.strip().split(",") for line in last5]
    # List all images
    images = sorted(os.listdir(IMAGE_FOLDER))
    html = """
    <h2>Últimas 5 placas detectadas</h2>
    <table border="1">
        <tr><th>Placa</th><th>Fecha</th><th>Hora</th><th>Imagen</th></tr>
        {% for row in last5 %}
        <tr>
            <td>{{row[0]}}</td>
            <td>{{row[1]}}</td>
            <td>{{row[2]}}</td>
            <td><a href="/images/{{row[3]}}">{{row[3]}}</a></td>
        </tr>
        {% endfor %}
    </table>
    <h2>Descargar archivos</h2>
    <a href="/download/csv">Descargar CSV</a><br>
    <h2>Imágenes detectadas</h2>
    {% for img in images %}
        <a href="/images/{{img}}">{{img}}</a><br>
    {% endfor %}
    """
    return render_template_string(html, last5=last5, images=images)

@app.route("/images/<filename>")
def images(filename):
    return send_from_directory(IMAGE_FOLDER, filename)

@app.route("/download/csv")
def download_csv():
    return send_file(CSV_FILE, as_attachment=True)

# --- MAIN LOOP ---
def main_loop():
    picam2 = Picamera2()
    picam2.start()
    import time
    time.sleep(2)  # Give camera time to warm up

    while True:
        frame = picam2.capture_array()
        # Convert to BGR if needed (picamera2 returns RGB by default)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        predictions = plate_detector.predict(frame_bgr)
        if not predictions:
            cv2.waitKey(100)
            continue

        for pred in predictions:
            bbox = pred.bounding_box
            x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
            plate_roi = frame_bgr[y1:y2, x1:x2]
            if plate_roi.size == 0:
                continue
            # Resize and convert for recognizer
            plate_roi_resized = cv2.resize(plate_roi, (128, 64))
            plate_roi_rgb = cv2.cvtColor(plate_roi_resized, cv2.COLOR_BGR2RGB)
            texts = plate_recognizer.run(plate_roi_rgb)
            for text in texts:
                plate = clean_plate(text)
                if is_valid_plate(plate):
                    now = datetime.now()
                    date_str = now.strftime("%Y-%m-%d")
                    time_str = now.strftime("%H-%M-%S")
                    filename = f"{plate}_{date_str}_{time_str}.jpg"
                    filepath = os.path.join(IMAGE_FOLDER, filename)
                    cv2.imwrite(filepath, frame_bgr)
                    # Log to CSV
                    with open(CSV_FILE, "a") as f:
                        f.write(f"{plate},{date_str},{time_str},{filename}\n")
                    print(f"Detected and saved: {plate}")
        # Optional: slow down loop for Pi
        cv2.waitKey(100)

if __name__ == "__main__":
    import threading
    # Start main loop in a thread
    t = threading.Thread(target=main_loop, daemon=True)
    t.start()
    # Start Flask app, accessible as http://abus-logs:5000/ if hostname is set
    app.run(host="0.0.0.0", port=5000, debug=False)