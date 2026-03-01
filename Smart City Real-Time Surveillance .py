!pip install ultralytics
import cv2
from ultralytics import YOLO
from google.colab.patches import cv2_imshow

# Load model
model = YOLO("yolov8n.pt")

# Video path
video_path = "/content/kolkata traffic #india #youtubeshorts.mp4"
cap = cv2.VideoCapture(video_path)

# Get frame height to auto set middle line
ret, frame = cap.read()
if not ret:
    print("Video not found!")
    exit()

frame_height = frame.shape[0]
line_y = frame_height // 2   # middle of frame

# Reset video to first frame
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

count = 0
counted_ids = set()
prev_positions = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True)

    for r in results:
        boxes = r.boxes

        if boxes.id is not None:
            for box, track_id in zip(boxes, boxes.id):

                cls = int(box.cls[0])

                # Only cars (COCO class 2)
                if cls == 2:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    track_id = int(track_id)

                    # Draw bounding box
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.circle(frame, (cx,cy), 5, (0,0,255), -1)
                    cv2.putText(frame, f"ID:{track_id}", (x1,y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                    # Check previous position
                    if track_id in prev_positions:
                        prev_cy = prev_positions[track_id]

                        # Detect crossing (bottom → top)
                        if prev_cy > line_y and cy <= line_y:
                            if track_id not in counted_ids:
                                count += 1
                                counted_ids.add(track_id)
                                print("Car crossed! Count:", count)

                        # Detect crossing (top → bottom)
                        if prev_cy < line_y and cy >= line_y:
                            if track_id not in counted_ids:
                                count += 1
                                counted_ids.add(track_id)
                                print("Car crossed! Count:", count)

                    prev_positions[track_id] = cy

    # Draw counting line
    cv2.line(frame, (0,line_y), (frame.shape[1],line_y), (255,0,0), 3)

    # Show count
    cv2.putText(frame, f"Total Cars: {count}", (50,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)

    cv2_imshow(frame)

cap.release()
cv2.destroyAllWindows()

print("Final Count:", count)
