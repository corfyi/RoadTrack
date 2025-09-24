import cv2
import os
from ultralytics import YOLO

# 路径设置
WEIGHTS = os.path.join(os.path.dirname(__file__), 'yolo11m.pt')
VIDEO_PATH = os.path.join(os.path.dirname(__file__), 'video.mp4')

# 加载YOLO模型
model = YOLO(WEIGHTS)

# 打开视频
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
	print(f"Cannot open video: {VIDEO_PATH}")
	exit(1)

while True:
	ret, frame = cap.read()
	if not ret:
		break

	# YOLO推理
	results = model(frame,verbose=False)
	boxes = results[0].boxes
	for box in boxes:
		x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
		conf = float(box.conf[0])
		cls = int(box.cls[0])
		label = f"{model.names[cls]} {conf:.2f}"
		color = (0, 255, 0)
		cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
		cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

	cv2.imshow('YOLO Detection', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
