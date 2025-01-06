from ultralytics import YOLO as yl
import cv2 as cv
import numpy as np
import cvzone as cz
import sort as st

model = yl("yolov8n.pt")
capture = cv.VideoCapture("Cars.mp4")


classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
    "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]


mask = cv.imread("mask.jpg", cv.IMREAD_GRAYSCALE)
tracker = st.Sort(max_age = 20 , min_hits = 3 , iou_threshold = 0.3 )
limits = [924,497,373,497]
totalCount = []

while True:
    isTrue, frame = capture.read()
    if not isTrue:
        break  

    if mask.shape[:2] != frame.shape[:2]:
        mask = cv.resize(mask, (frame.shape[1], frame.shape[0]))

    masked_frame = cv.bitwise_and(frame, frame, mask=mask)
    results = model(masked_frame)
    detections = np.empty((0,5))

    cv.line(frame,(limits[0],limits[1]),(limits[2],limits[3]),(255,0,0),thickness=4)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            conf = round(box.conf[0].item(), 2)
            cls = int(box.cls[0].item())
            currentclass = classNames[cls]

            if currentclass == "car" and conf >= 0.3:
                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)            
                currentArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections,currentArray))


    resultsTracker = tracker.update(detections)

    finalCount = []
    for result in resultsTracker:
        x1 , y1 , x2 , y2 , id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cz.putTextRect(frame, f"{id}", (x1, max(35, y1)),
                               scale=0.6, thickness=1, offset=4)
        cx,cy = (x1 + x2 )/2,(y1 + y2)/2

        cx = int(cx)
        cy = int(cy)
        id = int(id)

        if 924 > cx > 373 and 497 - 20 < cy < 497 + 20:
            if totalCount.count(id) == 0:
                totalCount.append(id)

        cz.putTextRect(frame, f"Total Cars Passed: {len(totalCount)}", (50,50),
                               scale=2, thickness=4, offset=3)
        cv.rectangle(frame, (cx,cy), (cx+10, cy+10), (0, 255, 0), 2)
    
        print(cx," ",cy," ",id,"")

    cv.imshow("Video with Car Detections", frame)
    if cv.waitKey(1) & 0xFF == ord("d"):
        break

cv.waitKey(0)
capture.release()
cv.destroyAllWindows()