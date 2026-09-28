# Driver Alertness Monitor

This workspace now contains the first software implementation of the research project.

## Desktop MVP

The desktop app is in `desktop/`. It supports:

- Live webcam monitoring with OpenCV
- Optional YOLO weights selected from the UI or with `--model`
- Image testing without a camera
- Seven behavior labels from the research project
- Temporal alerting after sustained unsafe detections
- A simple monitoring dashboard

Run it from PowerShell:

```powershell
cd desktop
py -m pip install -r requirements.txt
py app.py
```

Without a trained weights file, the interface runs in demo mode. The supplied archives do not include model weights or training labels, so the trained `.pt` file must be added separately before real detections are possible.

Run the alert logic tests with:

```powershell
py -m unittest test_behavior_logic.py
```

## Mobile version

The Android project is in `mobile-android/`. It provides the mobile UI shell and camera permission flow. The model runtime should be added after exporting the trained YOLO model to an Android-compatible ONNX or TFLite format.
