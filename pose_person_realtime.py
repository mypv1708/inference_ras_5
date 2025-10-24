import cv2
from ultralytics import YOLO
def get_keypoint_position(results, keypoint_num, axis='x'):
"""
Keypoint reference:
0: nose 5: left_shoulder 10: right_wrist 15: left_ankle
1: left_eye 6: right_shoulder 11: left_hip 16: right_ankle
2: right_eye 7: left_elbow 12: right_hip
3: left_ear 8: right_elbow 13: left_knee
4: right_ear 9: left_wrist 14: right_knee
"""
if not 0 <= keypoint_num <= 16:
raise ValueError("Keypoint number must be between 0 and 16")
if axis.lower() not in ['x', 'y']:
raise ValueError("Axis must be 'x' or 'y'")
# Get the keypoint data
keypoint = results[0].keypoints.xyn[0][keypoint_num]
# Return x or y coordinate
return keypoint[0].item() if axis.lower() == 'x' else keypoint[1].item()
# --- Setup webcam ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
# --- Load YOLO Pose model ---
model = YOLO("yolo11s-pose_ncnn_model") # hoặc "yolo11s-pose.pt" nếu muốn chính xác hơn
while True:
ret, frame = cap.read()
if not ret:
print("Không đọc được khung hình từ webcam.")
break
# Run YOLO Pose prediction
results = model.predict(frame, imgsz=416, verbose=False)
try:
# Lấy vị trí mũi (keypoint 0)
nose_x = get_keypoint_position(results, 0, 'x')
nose_y = get_keypoint_position(results, 0, 'y')
print(f"Position - X: {nose_x:.3f}, Y: {nose_y:.3f}")
except (IndexError, AttributeError):
print("Không phát hiện người trong khung hình.")
# Vẽ khung và keypoints
annotated_frame = results[0].plot()
# Tính FPS từ inference time
inference_time = results[0].speed['inference']
fps = 1000 / inference_time if inference_time > 0 else 0
text = f'FPS: {fps:.1f}'
# Hiển thị FPS
cv2.putText(annotated_frame, text, (20, 40),
cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
# Hiển thị hình ảnh
cv2.imshow("YOLO Pose - Webcam", annotated_frame)
# Nhấn 'q' để thoát
if cv2.waitKey(1) & 0xFF == ord('q'):
break
# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
