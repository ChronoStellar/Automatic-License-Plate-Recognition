import cv2
from ultralytics import YOLO

# Load the YOLOv8n model
model = YOLO('./best.pt')  # Replace with 'best.pt' if using a custom-trained model

# Initialize the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream")
    exit()

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture image")
        break

    # Perform object detection
    results = model(frame)

    # Loop over the detections in the frame
    for result in results:  # Each result corresponds to a frame
        boxes = result.boxes  # Bounding boxes
        for box in boxes:
            # Extract coordinates of the bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get box corners as integers
            
            # Crop the image based on the bounding box
            cropped_object = frame[y1:y2, x1:x2]
            
            # Resize the cropped image to a fixed size for better display (optional)
            cropped_object = cv2.resize(cropped_object, (400, 400))

            # Display the cropped (zoomed) object
            # cv2.imshow('Cropped Object Detection', cropped_object)
            
            # Optional: Draw bounding boxes and labels on the original frame for reference
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'Object {int(box.cls[0])}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the original frame with bounding boxes
    cv2.imshow('YOLOv8 Real-time Object Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
