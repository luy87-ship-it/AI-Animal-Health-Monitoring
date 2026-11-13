# AI Animal Health Monitoring

A Raspberry Pi 5 prototype for monitoring pet and livestock health. The system uses distance sensing to trigger the camera, identifies animals with MobileNetV3 Large, checks image sharpness, and uploads selected photos to a website for health analysis. An LCD displays camera status and a health label based on the returned healthy-confidence percentage.

This program uses one local animal-recognition model. The website runs the health analysis; MobileNetV3 on the Pi identifies animals rather than diagnosing injuries or sickness.

## Main Features

1. Automatic camera activation and shutdown using an ultrasonic sensor.

2. MobileNetV3 Large animal recognition with Laplacian and Tenengrad sharpness checks.

3. Ambient-light sensing and smooth PWM LED brightness control.

4. Local photo storage and automatic HTTPS uploads in a background thread.

5. Dynamic Healthy, Unhealthy, Analyzing..., and Unknown LCD messages.

6. Detection-session tracking to prevent old upload results from changing a new session's LCD status.

7. Tkinter live preview and a Flask video stream.

8. CSV records, system logs, LCD connection recovery, and CPU temperature protection.

## Hardware

1. **Raspberry Pi 5:** runs image processing and hardware control.

2. **Raspberry Pi Camera Module 3:** captures animal images.

3. **HC-SR04 ultrasonic sensor:** measures distance to trigger camera operation.

4. **16×2 I2C LCD:** displays camera and health status.

5. **LDR and MCP3008 ADC:** measure ambient light.

6. **RGB LED:** provides adjustable illumination.

7. **3D-printed structure:** supports and positions the components.


## Workflow

1. Load the model, initialize the hardware, and start the background threads.

2. Measure distance every 0.2 seconds. Start the camera when 19 of the latest 20 readings are at or below 20 cm, or when three consecutive batches each contain at least 18 qualifying readings out of 20.

3. Begin a new detection session and display `Analyzing...` if uploading is available.

4. Read ambient light and adjust the LED target brightness. In darker conditions, increase illumination with an 8-second brightness transition.

5. Capture 1280×720 images and share frames with the live preview and recognition thread.

6. Analyze a new frame approximately once per second. Continue image selection when the animal-confidence score reaches 0.30.

7. Check sharpness with both methods: **Laplacian** measures rapid changes in image detail; **Tenengrad** measures edge strength using horizontal and vertical gradients. Reject images below either configured threshold.

8. Save passing images, allowing at least 2 seconds between saves, and queue their paths for upload.

9. The upload worker reads the device token and sends each image in an HTTPS POST request using `multipart/form-data`.

10. The website analyzes the image and returns a health-confidence percentage as plain text, such as `20%`.

11. The Pi validates that response, applies the 50% display threshold, and updates the shared health status. The LCD thread reads this status and writes it to the display over I2C, normally within about one second.

12. Record detection details and upload outcomes. Log results from an earlier camera session, but do not overwrite the current session's LCD status.

13. Stop the camera when three consecutive batches each contain at least 15 of 20 readings at or above 40 cm. Fade out the LED and wait for the next approach.

14. Distance sensing, lighting, capture, recognition, uploading, LCD recovery, and temperature monitoring run in separate threads. Uploading therefore does not block image capture.

## LCD Health Display

1. **New detection session awaiting a result:** the LCD displays `Analyzing...`.

2. **Healthy confidence ≥ 50%:** the LCD displays `Animal:Healthy`.

3. **Healthy confidence < 50%:** the LCD displays `Animal:Unhealthy`.

4. **Startup, unavailable uploads, failed upload, or invalid result:** The LCD displays `Animal: Unknown`.

5. The first line shows `Camera: ON` or `Camera: OFF`. Within a session, the most recent completed result remains visible while the system processes subsequent images. A result may arrive after the camera stops, as long as a newer session has not started.

6. The 50% cutoff is a configurable prototype display rule, not a validated medical threshold. It does not directly reproduce the website's injury/sickness verdict. The returned percentage is an uncalibrated model estimate; this prototype is not a veterinary diagnostic tool.

## Files

1. **`animal_health_monitor.py`:** Integrated monitoring program with dynamic LCD status.

2. **`pi_upload.py`:** HTTPS upload client and response validation.

3. **`README.md`:** project overview and operating instructions.

4. Keep the two Python files in the same directory. If you saved the updated main program under a different filename, use that filename in the launch command below.

## Setup and Run

1. Use a Raspberry Pi OS desktop environment with the camera, I2C, and SPI interfaces configured for the hardware.

2. Provide the program's dependencies: PyTorch, Torchvision, OpenCV, NumPy, Pillow, Flask, gpiozero, Picamera2/libcamera, RPLCD, and Tkinter. The upload client uses Python 3.10+ and its standard library.

3. Review `PROGRAM_FOLDER`, `PICS_FOLDER`, and `LOG_PATH` in the main program. The supplied paths belong to the original prototype; change them to writable locations on your Pi. `LOG_PATH` is configured separately from `PROGRAM_FOLDER`.

4. Create a device credential for the selected pet on the analysis website and save it privately at `~/.config/pet-image/device-token`. Restrict the token file to its owner with `chmod 600 ~/.config/pet-image/device-token`.

5. Check `UPLOAD_SERVER_URL`, or set the `PET_SERVER_URL` environment variable to your server's HTTPS origin. The server must support the healthy-confidence upload API expected by `pi_upload.py`.

6. Open the main program in Thonny and run it, or launch it from its directory: `python3 animal_health_monitor.py`

7. Focus the Tkinter preview and press **E**, click **STOP**, or close the window to stop the program.

8. The Flask preview listens on port `5000`; on a trusted local network, open `http://<PI_IP_ADDRESS>:5000`. This local preview is separate from the HTTPS connection used for health-analysis uploads. The current program expects a desktop session for its Tkinter window.

9. Keep device tokens, passwords, and API keys out of the repository. The Pi uses a device token and does not need the website's OpenAI API key.

## Default Settings

1. **Animal-recognition threshold:** 0.30.

2. **Laplacian sharpness threshold:** 35.0.

3. **Tenengrad sharpness threshold:** 80.0.

4. **Healthy-confidence display threshold:** 50%.

5. **Maximum queued uploads:** 30.

6. **CPU temperature check interval:** 4 seconds.

7. **Temperature shutdown / restart eligibility:** 70°C / 65°C.

8. Animal confidence and healthy confidence serve different purposes. The first selects frames containing animals; the second controls the LCD label after website analysis. Sharpness and lighting settings may need adjustment for the physical setup.

## Saved Output and Error Handling

1. **Photos:** saved under `PICS_FOLDER` with timestamps and image-selection scores in their filenames.

2. **Detection CSV:** `mobilenetv3_detections.csv` under `PROGRAM_FOLDER`; recreated at each program start.

3. **Upload CSV:** `upload_results.csv` under `PROGRAM_FOLDER`; upload results are appended.

4. **System log:** appended at `LOG_PATH`.

5. Uploads are processed sequentially without automatic retries. Failed uploads retain the local photo; a full queue also leaves the photo on disk. LCD connection failures trigger reconnection attempts. At 70°C, camera operation stops; at 65°C or below, distance-triggered operation becomes eligible again.

## Validation
The LCD update passed Python syntax checking and 14 simulated tests covering threshold boundaries, invalid responses, upload failures, session changes, display length, and LCD reconnection. These checks do not replace testing the complete system on the Raspberry Pi hardware.
