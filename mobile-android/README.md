# Android Mobile Version

This is the Android application shell for the Driver Alertness Monitor.

It includes:

- Kotlin Android application structure
- Front-camera permission flow
- CameraX live preview
- Mobile monitoring state and alert display
- Live ONNX Runtime inference using `app/src/main/assets/best.onnx`
- Sustained unsafe-behavior alerts matching the desktop logic

Open this folder in Android Studio and run it on an Android 8.0+ device. The Android model is exported from `desktop/weights/best.pt`. Re-export it after replacing the trained weights with:

```powershell
cd desktop
py -c "from ultralytics import YOLO; YOLO('weights/best.pt').export(format='onnx', imgsz=640, nms=True, opset=12)"
```

Then copy the generated `desktop/weights/best.onnx` to `mobile-android/app/src/main/assets/best.onnx`.
