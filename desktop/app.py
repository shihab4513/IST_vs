import argparse
import base64
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2

from behavior_logic import AlertEngine


COLORS = {
    "background": "#10151b",
    "panel": "#18212b",
    "text": "#f3f6f8",
    "muted": "#9aa8b5",
    "accent": "#35c9a5",
    "warning": "#ffb547",
    "danger": "#ff6b6b",
}

MODEL_LABELS = {
    "Normal_Focused": "Normal Focused",
    "Normal_Non_Focused": "Normal Non-Focused",
    "Using_Phone": "Using Phone",
    "Yawning_Drowsiness": "Yawning/Drowsiness",
}


class DriverMonitor:
    def __init__(self, root, model_path=None):
        self.root = root
        self.root.title("Driver Alertness Monitor")
        self.root.geometry("1120x720")
        self.root.minsize(860, 560)
        self.root.configure(bg=COLORS["background"])
        self.capture = None
        self.model = None
        self.running = False
        self.engine = AlertEngine()
        self.last_frame = None
        self.model_path = model_path
        self.status = tk.StringVar(value="Ready to start")
        self.behavior = tk.StringVar(value="Normal Focused")
        self.confidence = tk.StringVar(value="--")
        self.alert = tk.StringVar(value="Monitoring inactive")
        self._build_ui()

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", padding=8, font=("Segoe UI", 10))
        style.configure("TLabel", background=COLORS["panel"], foreground=COLORS["text"])

        header = tk.Frame(self.root, bg=COLORS["background"], padx=24, pady=18)
        header.pack(fill="x")
        tk.Label(header, text="DRIVER / ALERTNESS", bg=COLORS["background"], fg=COLORS["accent"], font=("Segoe UI", 10, "bold")).pack(anchor="w")
        tk.Label(header, text="Camera-based safety monitor", bg=COLORS["background"], fg=COLORS["text"], font=("Segoe UI", 24, "bold")).pack(anchor="w")

        body = tk.Frame(self.root, bg=COLORS["background"], padx=24, pady=4)
        body.pack(fill="both", expand=True)
        viewer = tk.Frame(body, bg="#070a0d")
        viewer.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Label(viewer, text="Camera preview\n\nChoose Start Camera to begin", bg="#070a0d", fg=COLORS["muted"], font=("Segoe UI", 15))
        self.canvas.pack(fill="both", expand=True)

        panel = tk.Frame(body, bg=COLORS["panel"], width=280, padx=20, pady=20)
        panel.pack(side="right", fill="y", padx=(16, 0))
        panel.pack_propagate(False)
        self._metric(panel, "CURRENT STATE", self.behavior)
        self._metric(panel, "CONFIDENCE", self.confidence)
        self._metric(panel, "SYSTEM", self.status)
        tk.Label(panel, textvariable=self.alert, bg=COLORS["panel"], fg=COLORS["muted"], font=("Segoe UI", 11), wraplength=230, justify="left").pack(anchor="w", pady=(24, 14))
        ttk.Button(panel, text="Start Camera", command=self.start).pack(fill="x", pady=4)
        ttk.Button(panel, text="Stop Monitoring", command=self.stop).pack(fill="x", pady=4)
        ttk.Button(panel, text="Open Image", command=self.open_image).pack(fill="x", pady=4)
        ttk.Button(panel, text="Choose Model", command=self.choose_model).pack(fill="x", pady=4)

        footer = tk.Label(self.root, text="The alert is raised after sustained unsafe detections, not a single frame.", bg=COLORS["background"], fg=COLORS["muted"], font=("Segoe UI", 9), padx=24, pady=12)
        footer.pack(fill="x")

    def _metric(self, parent, title, variable):
        tk.Label(parent, text=title, bg=COLORS["panel"], fg=COLORS["muted"], font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(4, 2))
        tk.Label(parent, textvariable=variable, bg=COLORS["panel"], fg=COLORS["text"], font=("Segoe UI", 16, "bold"), wraplength=230, justify="left").pack(anchor="w", pady=(0, 12))

    def _load_model(self):
        if self.model is not None:
            return True
        if not self.model_path:
            self.status.set("Demo mode: model not selected")
            return False
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            self.status.set("Model loaded")
            return True
        except Exception as error:
            messagebox.showerror("Model error", str(error))
            self.status.set("Model could not load")
            return False

    def start(self):
        if self.running:
            return
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            messagebox.showerror("Camera error", "Could not open the default camera.")
            self.capture.release()
            self.capture = None
            return
        self.running = True
        self.status.set("Monitoring")
        self._tick()

    def stop(self):
        self.running = False
        if self.capture:
            self.capture.release()
            self.capture = None
        self.status.set("Ready to start")
        self.alert.set("Monitoring inactive")
        self.engine.reset()

    def _tick(self):
        if not self.running or not self.capture:
            return
        success, frame = self.capture.read()
        if success:
            self.last_frame = frame
            behavior, confidence = self._detect(frame)
            self._update_state(behavior, confidence)
            self._show_frame(frame)
        self.root.after(30, self._tick)

    def _detect(self, frame):
        if not self._load_model():
            return "Normal Focused", 0.0
        result = self.model.predict(frame, verbose=False)[0]
        if not result.boxes:
            return "Normal Focused", 0.0
        best = int(result.probs.top1) if result.probs else int(result.boxes.conf.argmax())
        confidence = float(result.boxes.conf[best])
        class_id = int(result.boxes.cls[best])
        label = str(result.names[class_id])
        return MODEL_LABELS.get(label, label), confidence

    def _update_state(self, behavior, confidence):
        triggered = self.engine.update(behavior)
        self.behavior.set(behavior)
        self.confidence.set(f"{confidence:.0%}" if confidence else "Demo")
        if triggered:
            self.alert.set(f"WARNING\nSustained {behavior} detected")
        else:
            self.alert.set("No active warning")

    def _show_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ok, encoded = cv2.imencode(".png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if ok:
            image_data = base64.b64encode(encoded).decode("ascii")
            image = tk.PhotoImage(data=image_data)
            image = image.subsample(max(1, image.width() // max(1, self.canvas.winfo_width())), max(1, image.height() // max(1, self.canvas.winfo_height())))
            self.canvas.configure(image=image, text="")
            self.canvas.image = image

    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if not path:
            return
        frame = cv2.imread(path)
        if frame is None:
            messagebox.showerror("Image error", "Could not read that image.")
            return
        self.last_frame = frame
        behavior, confidence = self._detect(frame)
        self._update_state(behavior, confidence)
        self._show_frame(frame)

    def choose_model(self):
        path = filedialog.askopenfilename(filetypes=[("YOLO weights", "*.pt *.onnx *.tflite")])
        if path:
            self.model_path = Path(path)
            self.model = None
            self.status.set(f"Model selected: {Path(path).name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Path to trained YOLO weights")
    args = parser.parse_args()
    default_model = Path(__file__).parent / "weights" / "best.pt"
    model_path = Path(args.model) if args.model else (default_model if default_model.exists() else None)
    root = tk.Tk()
    DriverMonitor(root, model_path)
    root.mainloop()


if __name__ == "__main__":
    main()
