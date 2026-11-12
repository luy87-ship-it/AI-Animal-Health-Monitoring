import time
import os
import csv
import re
import warnings
from datetime import datetime
import torch
from PIL import Image
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
import threading
from collections import deque
import sys
import atexit
import queue
from pathlib import Path
from pi_upload import validate_server, read_token, upload_image, UploadError

LOG_PATH = "/home/yiranspi5/camera_project/FinalDecision/Program/system_log.txt"

class TeeLog:
    def __init__(self, console, file_handle, lock):
        self.console = console
        self.file_handle = file_handle
        self.lock = lock

    def write(self, message):
        with self.lock:
            self.console.write(message)
            self.file_handle.write(message)
            self.file_handle.flush()
        return len(message)

    def flush(self):
        with self.lock:
            self.console.flush()
            self.file_handle.flush()

    def isatty(self):
        return getattr(self.console, "isatty", lambda: False)()

    def __getattr__(self, name):
        return getattr(self.console, name)


def setup_logging():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    logfile = open(LOG_PATH, "a", encoding="utf-8", buffering=1)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    lock = threading.RLock()
    sys.stdout = TeeLog(original_stdout, logfile, lock)
    sys.stderr = TeeLog(original_stderr, logfile, lock)

    def close_log():
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        logfile.flush()
        logfile.close()

    atexit.register(close_log)
    print(f"\n{'=' * 60}")
    print(f"[SESSION START] {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"[LOG] Appending output to {LOG_PATH}")
    return close_log

setup_logging()

import cv2
import numpy as np
import tkinter as tk
import base64
from flask import Flask, Response
from gpiozero import DistanceSensor, RGBLED, MCP3008
from picamera2 import Picamera2
from libcamera import controls
from RPLCD.i2c import CharLCD


# ============================================================
# 1. CONFIGURATION
# ============================================================

TRIGGER_PIN = 23
ECHO_PIN = 24

START_DISTANCE = 20.0
STOP_DISTANCE = 40.0
SENSOR_INTERVAL = 0.2

SAMPLE_SIZE = 20
REQUIRED_COUNT = 19
START_ALTERNATIVE_COUNT = 18
START_CONSECUTIVE = 3
STOP_ALTERNATIVE_COUNT = 15
STOP_CONSECUTIVE = 3

RED_PIN = 17
GREEN_PIN = 27
BLUE_PIN = 22

LED_FADE_DURATION = 8.0
LED_UPDATE_INTERVAL = 0.02

LDR_CHANNEL = 1
LIGHT_SENSOR_INTERVAL = 0.5

ADC_DARK = 0.10
ADC_BRIGHT = 0.90
MIN_LED_BRIGHTNESS = 0.0
MAX_LED_BRIGHTNESS = 1.0

LCD_ADDRESS = 0x27
LCD_RETRY_SECONDS = 5.0
LCD_WRITE_RETRIES = 2

# Display rule only: not the website's injury/sickness verdict.
# This percentage is separate from ANIMAL_CONFIDENCE_THRESHOLD.
HEALTHY_CONFIDENCE_THRESHOLD = 50.0

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
WEB_PORT = 5000


PROGRAM_FOLDER = "/home/yiranspi5/camera_project/FinalMerge/Programs"
PICS_FOLDER = "/home/yiranspi5/camera_project/FinalMerge/Pics"
CSV_PATH = os.path.join(PROGRAM_FOLDER, "mobilenetv3_detections.csv")

ANIMAL_CONFIDENCE_THRESHOLD = 0.30
ANIMAL_CLASS_IDS = list(range(398))
BLUR_THRESHOLD = 35.0
TENENGRAD_THRESHOLD = 80.0
LIVE_DETECTION_INTERVAL = 1.0
SAVE_INTERVAL = 2.0

MAX_TEMP = 70.0
TEMP_CHECK_INTERVAL = 4.0
TEMP_RESUME = 65.0

os.makedirs(PROGRAM_FOLDER, exist_ok=True)
os.makedirs(PICS_FOLDER, exist_ok=True)

# Upload configuration
UPLOAD_SERVER_URL = os.environ.get(
    "PET_SERVER_URL",
    "https://ece-pet-health.onrender.com"
)
UPLOAD_TOKEN_PATH = Path.home() / ".config/pet-image/device-token"
UPLOAD_PET_ID = None
UPLOAD_QUEUE_MAX = 30
UPLOAD_RESULTS_CSV = os.path.join(
    PROGRAM_FOLDER, "upload_results.csv"
)
upload_queue = queue.Queue(maxsize=UPLOAD_QUEUE_MAX)

warnings.filterwarnings("ignore", category=UserWarning)
torch.set_num_threads(2)

print("Loading MobileNetV3 Large...")
weights = MobileNet_V3_Large_Weights.DEFAULT
model = mobilenet_v3_large(weights=weights).cpu().eval()
preprocess = weights.transforms()
print("MobileNetV3 ready")


# ============================================================
# 2. GLOBAL VARIABLES
# ============================================================

app = Flask(__name__)

running = True
camera_active = False
overheated = False

latest_frame = None
frame_lock = threading.Lock()

state_lock = threading.Lock()
led_lock = threading.Lock()

ambient_light = 0.5

current_led_brightness = 0.0
target_led_brightness = 0.0
fade_start_brightness = 0.0
fade_start_time = time.monotonic()

camera_lock = threading.Lock()
ai_frame_lock = threading.Lock()

ai_frame = None
ai_frame_sequence = 0
camera_session = 0
camera_on_time = None
first_saved_session = -1

# Protected by state_lock. Keep the latest completed result during a session.
health_status = "Unknown"
upload_available = None  # None = initializing, False = unavailable


# ============================================================
# 3. INITIALIZE HARDWARE
# ============================================================

print("Initializing hardware...")

distance_sensor = DistanceSensor(
    echo=ECHO_PIN,
    trigger=TRIGGER_PIN,
    max_distance=2.0
)

rgb_led = RGBLED(
    red=RED_PIN,
    green=GREEN_PIN,
    blue=BLUE_PIN,
    active_high=True,
    pwm=True
)

rgb_led.off()
ldr = MCP3008(channel=LDR_CHANNEL)

lcd = None
lcd_lock = threading.RLock()
lcd_last_attempt = 0.0
lcd_last_lines = None

picam2 = Picamera2()

camera_config = picam2.create_video_configuration(
    main={
        "size": (CAMERA_WIDTH, CAMERA_HEIGHT),
        "format": "RGB888"
    }
)
picam2.configure(camera_config)

try:
    picam2.set_controls({
        "AfMode": controls.AfModeEnum.Continuous
    })
except Exception as e:
    print("Autofocus not available:", e)


# ============================================================
# 4. LCD FUNCTIONS
# ============================================================

def update_lcd(camera_on):
    global lcd, lcd_last_attempt, lcd_last_lines

    # Snapshot inside the display lock so concurrent callers cannot restore
    # stale text. camera_on is retained for compatibility with existing calls.
    with lcd_lock:
        with state_lock:
            active = camera_active
            status = health_status
        health_line = (
            "Analyzing..." if status == "Analyzing..."
            else f"Animal:{status}"
        )
        # "Animal:Unhealthy" is exactly 16 characters.
        lines = (
            ("Camera: ON" if active else "Camera: OFF").ljust(16),
            health_line[:16].ljust(16)
        )
        if lcd is not None and lcd_last_lines == lines:
            return

        if lcd is None:
            now = time.monotonic()
            if now - lcd_last_attempt < LCD_RETRY_SECONDS:
                return

            lcd_last_attempt = now

            try:
                lcd = CharLCD(
                    i2c_expander="PCF8574",
                    address=LCD_ADDRESS,
                    port=1,
                    cols=16,
                    rows=2,
                    charmap="A02"
                )
                print("[LCD] I2C connection established")
            except Exception as e:
                print("LCD error:", e)
                return

        try:
            lcd.clear()
            lcd.cursor_pos = (0, 0)
            lcd.write_string(lines[0])
            lcd.cursor_pos = (1, 0)
            lcd.write_string(lines[1])
            lcd_last_lines = lines

        except Exception as e:
            print("LCD error:", e)
            try:
                lcd.close(clear=False)
            except Exception:
                pass
            lcd = None
            lcd_last_lines = None
            lcd_last_attempt = time.monotonic()


def classify_health(answer):
    """Validate the percentage returned by pi_upload.py, then apply our rule."""
    if not isinstance(answer, str) or not re.fullmatch(
        r"(?:100(?:\.0+)?|(?:[0-9]|[1-9][0-9])(?:\.[0-9]+)?)%",
        answer.strip()
    ):
        raise ValueError("Invalid healthy-confidence percentage")
    percentage = float(answer.strip()[:-1])
    status = (
        "Healthy" if percentage >= HEALTHY_CONFIDENCE_THRESHOLD
        else "Unhealthy"
    )
    return percentage, status


def set_health_status(session, status):
    """ Ignore replies from photos captured before the current camera session."""
    global health_status
    with state_lock:
        if session != camera_session:
            return False
        health_status = status
    # The existing LCD thread refreshes within about one second.
    return True


def lcd_recovery_loop():
    while running:
        with state_lock:
            on = camera_active
        update_lcd(on)
        time.sleep(1.0)


# ============================================================
# 5. LIGHT SENSOR
# ============================================================

def read_ambient_light():
    """
    MCP3008 CH1 reading from 0.0 to 1.0.
    Brighter environment -> higher ADC reading.
    """
    return ldr.value


def calculate_led_brightness(light_value):
    """
    Dark environment -> bright LED.
    Bright environment -> dim LED.
    """

    if light_value <= ADC_DARK:
        return MAX_LED_BRIGHTNESS

    if light_value >= ADC_BRIGHT:
        return MIN_LED_BRIGHTNESS

    normalized = (
        (light_value - ADC_DARK)
        / (ADC_BRIGHT - ADC_DARK)
    )

    brightness = 1.0 - normalized

    return max(
        MIN_LED_BRIGHTNESS,
        min(MAX_LED_BRIGHTNESS, brightness)
    )


# ============================================================
# 6. PWM LED CONTROL
# ============================================================

def set_led_target(brightness):
    global target_led_brightness
    global fade_start_brightness
    global fade_start_time

    brightness = max(0.0, min(1.0, brightness))

    with led_lock:
        if abs(brightness - target_led_brightness) < 0.02:
            return

        fade_start_brightness = current_led_brightness
        target_led_brightness = brightness
        fade_start_time = time.monotonic()


def led_control_loop():
    global current_led_brightness

    while running:
        with led_lock:
            elapsed = time.monotonic() - fade_start_time

            progress = min(
                elapsed / LED_FADE_DURATION,
                1.0
            )

            current_led_brightness = (
                fade_start_brightness
                + (
                    target_led_brightness
                    - fade_start_brightness
                ) * progress
            )

            brightness = current_led_brightness

        try:
            rgb_led.color = (
                brightness,
                brightness,
                brightness
            )
        except Exception as e:
            print("LED error:", e)

        time.sleep(LED_UPDATE_INTERVAL)

    rgb_led.off()


# ============================================================
# 7. LIGHT SENSOR THREAD
# ============================================================

def light_sensor_loop():
    global ambient_light

    while running:
        try:
            value = read_ambient_light()

            with state_lock:
                ambient_light = value
                active = camera_active

            desired_brightness = calculate_led_brightness(value)

            if active:
                set_led_target(desired_brightness)
            else:
                set_led_target(0.0)

            print(
                f"[LIGHT] "
                f"ADC={round(value * 1023):4d} "
                f"Ambient={value:.3f} "
                f"LED Target={desired_brightness:.2f}"
            )

        except Exception as e:
            print("Light sensor error:", e)

        time.sleep(LIGHT_SENSOR_INTERVAL)

# ============================================================
# 8. CAMERA CONTROL
# ============================================================

def start_camera(trigger_distance_cm=None):
    global camera_active, camera_session, camera_on_time, first_saved_session
    global health_status

    with camera_lock:

        with state_lock:
            if camera_active or overheated or not running:
                return

        try:
            picam2.start()

            with state_lock:
                camera_active = True
                camera_session += 1
                health_status = (
                    "Unknown" if upload_available is False else "Analyzing..."
                )
                camera_on_time = time.monotonic()
                first_saved_session = -1
                light_value = ambient_light

            update_lcd(True)

            brightness = calculate_led_brightness(light_value)
            set_led_target(brightness)

            print(
                f"[CAMERA] ON | Distance: {trigger_distance_cm:.1f} cm"
                if trigger_distance_cm is not None
                else "[CAMERA] ON | Trigger: ultrasonic"
            )

        except Exception as e:
            print("Camera start error:", e)


def stop_camera():
    global camera_active
    global latest_frame, ai_frame

    with camera_lock:

        with state_lock:
            if not camera_active:
                return

            camera_active = False

        # Start 8-second LED fade-out.
        set_led_target(0.0)

        try:
            picam2.stop()

        except Exception as e:
            print("Camera stop error:", e)

        with frame_lock:
            latest_frame = None

        with ai_frame_lock:
            ai_frame = None

        update_lcd(False)

        print("[CAMERA] OFF")


# ============================================================
# 9. DISTANCE SENSOR CONTROL
# ============================================================

def distance_control_loop():

    # Sliding window:
    # Used ONLY for normal startup (19/20).
    sliding_history = deque(maxlen=SAMPLE_SIZE)

    # Independent batches:
    # Used for alternative startup and shutdown.
    batch_history = []

    start_consecutive_count = 0
    stop_consecutive_count = 0

    # Track camera state changes.
    previous_active = False

    while running:

        try:
            # Read distance in centimeters.
            distance_cm = distance_sensor.distance * 100.0

            with state_lock:
                active = camera_active

            # Reset histories if camera state changed.
            if active != previous_active:

                sliding_history.clear()
                batch_history.clear()

                start_consecutive_count = 0
                stop_consecutive_count = 0

                previous_active = active

            # =================================================
            # CAMERA OFF
            # =================================================

            if not active:

                is_near = distance_cm <= START_DISTANCE

                # 1. Normal startup: sliding window 19/20.
                sliding_history.append(is_near)

                # 2. Alternative startup:
                # Independent batches 18/20 x 3.
                batch_history.append(is_near)

                # Check sliding window every new reading.
                if len(sliding_history) == SAMPLE_SIZE:

                    sliding_count = sum(sliding_history)

                    if sliding_count >= REQUIRED_COUNT:

                        print(
                            f"[DISTANCE] Sliding START: "
                            f"{sliding_count}/{SAMPLE_SIZE}"
                        )

                        start_camera()

                        sliding_history.clear()
                        batch_history.clear()

                        start_consecutive_count = 0
                        stop_consecutive_count = 0

                        with state_lock:
                            previous_active = camera_active

                        time.sleep(SENSOR_INTERVAL)
                        continue

                # Check alternative startup every 20 readings.
                if len(batch_history) == SAMPLE_SIZE:

                    batch_count = sum(batch_history)

                    if batch_count >= START_ALTERNATIVE_COUNT:
                        start_consecutive_count += 1
                    else:
                        start_consecutive_count = 0

                    print(
                        f"[DISTANCE] Start batch: "
                        f"{batch_count}/{SAMPLE_SIZE} "
                        f"| Consecutive="
                        f"{start_consecutive_count}/"
                        f"{START_CONSECUTIVE}"
                    )

                    # Start a NEW independent batch.
                    batch_history.clear()

                    if start_consecutive_count >= START_CONSECUTIVE:

                        print(
                            "[DISTANCE] Alternative START triggered"
                        )

                        start_camera()

                        sliding_history.clear()
                        batch_history.clear()

                        start_consecutive_count = 0
                        stop_consecutive_count = 0

                        with state_lock:
                            previous_active = camera_active

                        time.sleep(SENSOR_INTERVAL)
                        continue

            # =================================================
            # CAMERA ON
            # =================================================

            else:

                is_far = distance_cm >= STOP_DISTANCE

                # Shutdown:
                # Independent batches 15/20 x 3.
                batch_history.append(is_far)

                if len(batch_history) == SAMPLE_SIZE:

                    batch_count = sum(batch_history)

                    if batch_count >= STOP_ALTERNATIVE_COUNT:
                        stop_consecutive_count += 1
                    else:
                        stop_consecutive_count = 0

                    print(
                        f"[DISTANCE] Stop batch: "
                        f"{batch_count}/{SAMPLE_SIZE} "
                        f"| Consecutive="
                        f"{stop_consecutive_count}/"
                        f"{STOP_CONSECUTIVE}"
                    )

                    # Start a NEW independent batch.
                    batch_history.clear()

                    if stop_consecutive_count >= STOP_CONSECUTIVE:

                        print(
                            "[DISTANCE] STOP triggered"
                        )

                        stop_camera()

                        sliding_history.clear()
                        batch_history.clear()

                        start_consecutive_count = 0

                        with state_lock:
                            previous_active = camera_active

                        time.sleep(SENSOR_INTERVAL)
                        continue

            # =================================================
            # STATUS OUTPUT
            # =================================================

            with state_lock:
                current_state = camera_active

            print(
                f"[DISTANCE] {distance_cm:.1f} cm "
                f"| Camera={'ON' if current_state else 'OFF'}"
            )

        except Exception as e:
            print("Distance sensor error:", e)

        time.sleep(SENSOR_INTERVAL)


# ============================================================
# 10. CAMERA FRAME CAPTURE
# ============================================================

def camera_capture_loop():
    global latest_frame, ai_frame, ai_frame_sequence

    while running:

        with state_lock:
            active = camera_active

        if not active:
            time.sleep(0.1)
            continue

        try:
            with camera_lock:

                with state_lock:
                    if not camera_active:
                        continue

                frame = picam2.capture_array()

            # Share a copy with AI.
            # Never open a second camera.
            with ai_frame_lock:
                ai_frame = frame.copy()
                ai_frame_sequence += 1

            # Picamera2 RGB888 capture_array()
            # is BGR ordered for OpenCV.
            frame_bgr = frame

            success, encoded = cv2.imencode(
                ".jpg",
                frame_bgr,
                [cv2.IMWRITE_JPEG_QUALITY, 80]
            )

            if success:

                with frame_lock:
                    latest_frame = encoded.tobytes()

        except Exception as e:
            print("Camera capture error:", e)
            time.sleep(0.1)


# ============================================================
# 11. FLASK VIDEO STREAM
# ============================================================

def generate_frames():

    while running:

        with frame_lock:
            frame = latest_frame

        if frame is None:
            time.sleep(0.1)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame
            + b"\r\n"
        )

        time.sleep(0.05)


@app.route("/")
def index():

    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Animal Monitoring System</title>
        <style>
            body {
                background: #111;
                color: white;
                font-family: Arial;
                text-align: center;
            }

            img {
                width: 90%;
                max-width: 1280px;
                border: 2px solid #555;
            }
        </style>
    </head>

    <body>
        <h1>Animal Monitoring System</h1>
        <p>Raspberry Pi 5 Camera Monitoring</p>
        <img src="/video_feed">
    </body>
    </html>
    """


@app.route("/video_feed")
def video_feed():

    return Response(
        generate_frames(),
        mimetype=(
            "multipart/x-mixed-replace; boundary=frame"
        )
    )

# ============================================================
# 12. MOBILENETV3 ANALYSIS
# ============================================================

@torch.inference_mode()
def detect_animal(frame_rgb):
    tensor = preprocess(
        Image.fromarray(
            cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
        )
    ).unsqueeze(0)

    probabilities = torch.softmax(model(tensor)[0], dim=0)
    confidence = float(probabilities[ANIMAL_CLASS_IDS].sum().item())
    top_id = int(probabilities.argmax().item())
    top_confidence = float(probabilities[top_id].item())

    return (
        confidence >= ANIMAL_CONFIDENCE_THRESHOLD,
        confidence,
        top_id,
        top_confidence
    )


def calculate_sharpness(frame_rgb):
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def calculate_tenengrad(frame_rgb):
    small = cv2.resize(
        frame_rgb, (320, 180),
        interpolation=cv2.INTER_AREA
    )
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return float(cv2.mean(gx * gx + gy * gy)[0])


# ============================================================
# 13. CPU TEMPERATURE SAFETY
# ============================================================

def cpu_temperature():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return float(f.read().strip()) / 1000.0
    except (OSError, ValueError):
        return None


def temperature_monitor_loop():
    global overheated

    while running:
        temp = cpu_temperature()

        if temp is None:
            print("[TEMP] Unable to read CPU temperature")
        else:
            with state_lock:
                was_overheated = overheated

                if temp >= MAX_TEMP:
                    overheated = True
                elif temp <= TEMP_RESUME:
                    overheated = False

                now_overheated = overheated

            status = (
                "OVERHEAT - CAMERA DISABLED"
                if now_overheated else "NORMAL"
            )

            print(
                f"[TEMP] Raspberry Pi CPU: "
                f"{temp:.1f} C | {status}"
            )

            if now_overheated and not was_overheated:
                print(
                    f"[SAFETY] {MAX_TEMP:.0f} C reached: "
                    "camera stopping."
                )
                stop_camera()

            elif was_overheated and not now_overheated:
                print(
                    "[SAFETY] Temperature safe again; "
                    "ultrasonic triggering enabled"
                )

        time.sleep(TEMP_CHECK_INTERVAL)


# ============================================================
# 14. AI ANALYSIS AND IMAGE SAVING
# ============================================================

def ai_analysis_loop():
    global first_saved_session

    last_detection = 0.0
    last_save = 0.0
    last_sequence = -1
    last_processed_session = -1
    frame_number = 0

    print("[AI] Analysis thread ready")

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "Frame", "Timestamp", "Animal_Detected",
            "Animal_Confidence", "Top_Class_ID",
            "Top_Class_Confidence", "Inference_Time_s",
            "Sharpness", "Tenengrad", "Picture_Saved",
            "Reason", "Recognition_To_Save_s",
            "File_Path", "Camera_On_To_First_Save_s"
        ])

        while running:
            now = time.monotonic()

            if now - last_detection < LIVE_DETECTION_INTERVAL:
                time.sleep(0.05)
                continue

            with state_lock:
                active = camera_active
                session = camera_session

            if not active:
                time.sleep(0.1)
                continue

            if session != last_processed_session:
                last_save = 0.0
                last_processed_session = session

            with ai_frame_lock:
                sequence = ai_frame_sequence
                frame = (
                    None if ai_frame is None
                    else ai_frame.copy()
                )

            if frame is None or sequence == last_sequence:
                time.sleep(0.05)
                continue

            last_sequence = sequence
            last_detection = time.monotonic()
            frame_number += 1

            timestamp = datetime.now().isoformat(
                timespec="milliseconds"
            )

            sharpness = None
            tenengrad = None
            saved_path = ""
            recognition_to_save = None
            first_picture_elapsed = None

            try:
                t0 = time.monotonic()

                detected, confidence, top_id, top_confidence = (
                    detect_animal(frame)
                )

                inference_time = time.monotonic() - t0
                recognition_time = time.monotonic()

                reason = "No animal detected"

                if detected:
                    sharpness = calculate_sharpness(frame)

                    if sharpness < BLUR_THRESHOLD:
                        reason = "Image too blurry (Laplacian)"

                    else:
                        tenengrad = calculate_tenengrad(frame)

                        if tenengrad < TENENGRAD_THRESHOLD:
                            reason = "Image too blurry (Tenengrad)"

                        elif recognition_time - last_save < SAVE_INTERVAL:
                            reason = "Save interval"

                        else:
                            with state_lock:
                                still_active = (
                                    camera_active
                                    and camera_session == session
                                )

                            if not still_active:
                                reason = "Camera stopped"

                            else:
                                filename = (
                                    f"mobilenet_pet_"
                                    f"{datetime.now():%Y%m%d_%H%M%S_%f}"
                                    f"_conf_{confidence:.2f}"
                                    f"_sharp_{sharpness:.1f}.jpg"
                                )

                                path = os.path.join(
                                    PICS_FOLDER, filename
                                )

                                success = cv2.imwrite(
                                    path, frame,
                                    [cv2.IMWRITE_JPEG_QUALITY, 95]
                                )

                                if success:
                                    saved_path = path

                                    # Queue the photo for HTTPS upload.
                                    queue_saved_photo(path, session)

                                    last_save = time.monotonic()
                                    recognition_to_save = last_save - t0

                                    with state_lock:
                                        if first_saved_session != session:
                                            first_saved_session = session

                                            if camera_on_time is not None:
                                                first_picture_elapsed = (
                                                    last_save - camera_on_time
                                                )

                                    reason = "Saved"

                                    if first_picture_elapsed is not None:
                                        print(
                                            "[SPEED] Camera ON -> first "
                                            "animal photo saved: "
                                            f"{first_picture_elapsed:.3f} s"
                                        )
                                    else:
                                        print(
                                            "[SPEED] Animal recognition -> "
                                            "photo saved: "
                                            f"{recognition_to_save:.3f} s"
                                        )

                                else:
                                    reason = "Save error"

                writer.writerow([
                    frame_number,
                    timestamp,
                    "YES" if detected else "NO",
                    f"{confidence:.4f}",
                    top_id,
                    f"{top_confidence:.4f}",
                    f"{inference_time:.4f}",
                    "" if sharpness is None else f"{sharpness:.2f}",
                    "" if tenengrad is None else f"{tenengrad:.2f}",
                    "YES" if saved_path else "NO",
                    reason,
                    "" if recognition_to_save is None
                    else f"{recognition_to_save:.4f}",
                    saved_path,
                    "" if first_picture_elapsed is None
                    else f"{first_picture_elapsed:.4f}"
                ])

                f.flush()

            except Exception as e:
                print("[AI] Analysis error:", e)
                time.sleep(0.2)


# ============================================================
# 15. AUTO IMAGE UPLOAD
# ============================================================

def upload_worker_loop():
    """
    Upload saved photos sequentially.
    No automatic retry is performed.
    """
    global upload_available, health_status

    try:
        server = validate_server(UPLOAD_SERVER_URL)
        token = read_token(UPLOAD_TOKEN_PATH)

    except (UploadError, OSError, ValueError) as exc:
        with state_lock:
            upload_available = False
            health_status = "Unknown"
        print(f"[UPLOAD] Disabled: {exc}")
        print(
            "[UPLOAD] Photos still save locally; "
            "configure a private device token to enable uploads."
        )
        return

    with state_lock:
        upload_available = True
    print("[UPLOAD] Worker ready; using HTTPS server:", server)

    try:
        with open(
            UPLOAD_RESULTS_CSV, "a", newline="", encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            if file.tell() == 0:
                writer.writerow([
                    "Timestamp",
                    "Image_Path",
                    "Status",
                    "Healthy_Confidence",
                    "Detail"
                ])
                file.flush()

            while running or not upload_queue.empty():

                try:
                    path, session = upload_queue.get(timeout=0.5)

                except queue.Empty:
                    continue

                try:
                    answer = upload_image(
                        server,
                        token,
                        Path(path),
                        UPLOAD_PET_ID
                    )

                    percentage, status = classify_health(answer)
                    displayed = set_health_status(session, status)
                    print(
                        f"[HEALTH] {percentage:g}% -> {status} "
                        f"| threshold={HEALTHY_CONFIDENCE_THRESHOLD:g}% "
                        f"| {'LCD state updated' if displayed else 'old session; LCD unchanged'}"
                    )

                    print(
                        f"[UPLOAD] Success: {Path(path).name} "
                        f"| Healthy confidence: {answer}"
                    )

                    writer.writerow([
                        datetime.now().isoformat(timespec="seconds"),
                        path,
                        "SUCCESS",
                        answer,
                        ""
                    ])

                except (UploadError, OSError, ValueError) as exc:
                    set_health_status(session, "Unknown")
                    print(
                        f"[UPLOAD] Failed: {Path(path).name}: {exc}"
                    )

                    writer.writerow([
                        datetime.now().isoformat(timespec="seconds"),
                        path,
                        "FAILED",
                        "",
                        str(exc)
                    ])

                finally:
                    try:
                        file.flush()
                    finally:
                        upload_queue.task_done()
    except OSError as exc:
        with state_lock:
            upload_available = False
            health_status = "Unknown"
        print(f"[UPLOAD] Worker stopped: {exc}")


def queue_saved_photo(path, session):
    with state_lock:
        available = upload_available
    if available is False:
        set_health_status(session, "Unknown")
        print(f"[UPLOAD] Disabled; photo retained locally: {path}")
        return
    try:
        # Carry the capture session with the path through the upload queue.
        upload_queue.put_nowait((path, session))
        print(f"[UPLOAD] Queued: {Path(path).name}")

    except queue.Full:
        set_health_status(session, "Unknown")
        print(
            "[UPLOAD] Queue full; photo retained locally, "
            f"NOT uploaded: {path}"
        )


# ============================================================
# 16. MAIN
# ============================================================

def main():
    global running

    print("====================================")
    print(" Animal Monitoring System")
    print(" Raspberry Pi 5")
    print(" MCP3008 LDR Channel: CH1")
    print(" LED PWM Fade: 8 seconds")
    print(" MobileNetV3 + Laplacian + Tenengrad enabled")
    print(" Pictures:", PICS_FOLDER)
    print(" Upload results:", UPLOAD_RESULTS_CSV)
    print(f" LCD healthy-confidence threshold: {HEALTHY_CONFIDENCE_THRESHOLD:g}%")
    print(" CSV:", CSV_PATH)
    print(
        f" CPU safety: stop at {MAX_TEMP:.0f} C; "
        f"resume at {TEMP_RESUME:.0f} C"
    )
    print(" Preview: Tkinter")
    print("====================================")

    print(" Distance Detection Configuration:")
    print(" Sliding START: 19/20 <= 20 cm")
    print(" Batch START: 18/20 x 3 <= 20 cm")
    print(" Batch STOP: 15/20 x 3 >= 40 cm")
    print(" Sensor Interval: 0.2 seconds")

    update_lcd(False)

    threads = [
        threading.Thread(
            target=lcd_recovery_loop, daemon=True
        ),
        threading.Thread(
            target=led_control_loop, daemon=True
        ),
        threading.Thread(
            target=light_sensor_loop, daemon=True
        ),
        threading.Thread(
            target=distance_control_loop, daemon=True
        ),
        threading.Thread(
            target=camera_capture_loop, daemon=True
        ),
        threading.Thread(
            target=ai_analysis_loop, daemon=True
        ),
        threading.Thread(
            target=upload_worker_loop, daemon=True
        ),
        threading.Thread(
            target=temperature_monitor_loop, daemon=True
        )
    ]

    for thread in threads:
        thread.start()

    # Flask runs in a background thread.
    web_thread = threading.Thread(
        target=lambda: app.run(
            host="0.0.0.0",
            port=WEB_PORT,
            threaded=True,
            use_reloader=False
        ),
        daemon=True
    )

    web_thread.start()

    # Tkinter preview runs on the main thread.
    window = None

    try:
        window = tk.Tk()
        window.title("Animal Monitoring - Live Camera")
        window.geometry("900x590")

        status_label = tk.Label(
            window,
            text="Waiting for ultrasonic trigger...",
            font=("Arial", 12)
        )
        status_label.pack(pady=8)

        image_label = tk.Label(
            window,
            bg="black",
            width=800,
            height=450
        )
        image_label.pack(padx=10, pady=5)

        def request_stop(event=None):
            global running

            print("[CONTROL] E / Stop button: shutting down")
            running = False
            window.quit()

        tk.Button(
            window,
            text="STOP (E)",
            command=request_stop,
            font=("Arial", 12)
        ).pack(pady=6)

        window.bind("<KeyPress-e>", request_stop)
        window.bind("<KeyPress-E>", request_stop)
        window.protocol("WM_DELETE_WINDOW", request_stop)

        print(
            "[PREVIEW] Tkinter preview opened. "
            "Focus window and press E, or click STOP (E)."
        )

        def refresh_preview():

            if not running:
                window.quit()
                return

            with frame_lock:
                jpeg = latest_frame

            if jpeg is not None:
                frame = cv2.imdecode(
                    np.frombuffer(jpeg, dtype=np.uint8),
                    cv2.IMREAD_COLOR
                )

                if frame is not None:
                    h, w = frame.shape[:2]
                    new_w = 800
                    new_h = max(1, round(h * new_w / w))

                    frame = cv2.resize(
                        frame, (new_w, new_h)
                    )

                    ok, png = cv2.imencode(".png", frame)

                    if ok:
                        photo = tk.PhotoImage(
                            data=base64.b64encode(
                                png.tobytes()
                            ).decode("ascii"),
                            format="png"
                        )

                        image_label.configure(image=photo)
                        image_label.image = photo

                        status_label.configure(
                            text="Camera: ON | Live Preview"
                        )

            else:
                status_label.configure(
                    text="Camera: OFF | Waiting for ultrasonic trigger..."
                )
                image_label.configure(image="")
                image_label.image = None

            window.after(150, refresh_preview)

        window.after(150, refresh_preview)
        window.mainloop()

    except KeyboardInterrupt:
        print("\nStopping system...")

    except tk.TclError as exc:
        print("[PREVIEW] Desktop display unavailable:", exc)
        print(
            "[PREVIEW] Flask stream may still be accessible "
            "while program runs."
        )

    except Exception as exc:
        print("[PREVIEW] Unexpected error:", repr(exc))
        raise

    finally:
        running = False
        stop_camera()

        if window is not None:
            try:
                window.destroy()
            except tk.TclError:
                pass

        set_led_target(0.0)

        for thread in threads:
            thread.join(timeout=2.0)

        try:
            picam2.stop()
        except Exception:
            pass

        try:
            picam2.close()
        except Exception:
            pass

        rgb_led.off()
        rgb_led.close()

        ldr.close()
        distance_sensor.close()

        with lcd_lock:
            if lcd is not None:
                try:
                    lcd.close(clear=False)
                except Exception:
                    pass

        print(
            f"[SESSION END] "
            f"{datetime.now():%Y-%m-%d %H:%M:%S}"
        )
        print("System stopped.")


if __name__ == "__main__":
    main()
