import socket
import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture(2)

# Get width and height
camWidth  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
camHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# communication to UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort = ("127.0.0.1", 5052)

boxesToSend = []

model = YOLO('yolov8n.pt') #use 8n or 8s for performance

while True:
    success, webcam = cap.read()
    if not success:
        print("ERROR: Problem with webcam.")
    
    # clear each loop so not adding while holding old data
    boxesToSend.clear()

    results = model.predict(source=webcam, classes=[0], conf=0.4, verbose=False)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            # calculate the center point and width and height on the
            # webcam as well as translating to be able to be used by
            # the Raw Image in Unity
            centerX = float((x1 + x2) / 2)
            rawImageX = float(centerX / camWidth)
            rawImageX = round(rawImageX, 3)
            centerY = float((y1 + y2) / 2)
            rawImageY = float(centerY / camHeight)
            rawImageY = round(rawImageY, 3)
            width = float(x2 - x1)
            rawImageWidth = float(width / camWidth)
            rawImageWidth = round(rawImageWidth, 3)
            height = float(y2 - y1)
            rawImageHeight = float(height / camHeight)
            rawImageHeight = round(rawImageHeight, 3)

            cv2.rectangle(webcam, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(webcam, (int(centerX), int(centerY)), 5, (255, 0, 0), -1)
            # Format as CSV string for each box: "centerX,centerY,width,height"
            boxesToSend.append(f"{rawImageX},{rawImageY},{rawImageWidth},{rawImageHeight}")

    # Join all boxes by semicolon or another delimiter
    csv_string = ";".join(boxesToSend)

    print(csv_string)
    sock.sendto(csv_string.encode(), serverAddressPort)

    cv2.imshow("Webcam", webcam)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
socket.close()