from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # downloads model on first run

source = 1

if isinstance(source, int):
    # Webcam source
    model.track(source=source, tracker="bytetrack.yaml", conf=0.35, show=True, save=True)
elif source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
    # Image file
    model.predict(source=source, conf=0.35, show=True, save=True)
else:
    # Video file
    model.track(source=source, tracker="bytetrack.yaml", conf=0.35, show=True, save=True)

print("Done. Check runs/ folder.")