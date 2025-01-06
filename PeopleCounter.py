""" 
Hi Everyone, I am Lakshya Bakshi ,CSE 1st year student 
I created this code after learning from this youtube video.

Youtube Link -"https://youtu.be/WgPbbWmnXJ8?si=CEiUgAlfiJ1eRDra"

you can check out this video for further details and any other 
project you make . i have created a basic tracker which tell you 
whether a object has passed the



The Bounding Box is the box that surronds the objects detected 
"""


from ultralytics import YOLO as yl
import cv2 as cv
import numpy as np
import sort as st
import cvzone as cz


model = yl("yolov8n.pt")    # Load the YOLO model
capture = cv.VideoCapture("Your_video.mp4") # Load the video file
mask = cv.imread("Your_mask.jpg", cv.IMREAD_GRAYSCALE) # Load the mask image , Create a Custom mask image for your video
tracker = st.Sort(max_age=20, min_hits=3, iou_threshold=0.3)
limit = [110, 150, 230, 133] # These limits should be adjusted accourding to your  video as these values are use to define whether the object has crossed a perticular line or not
totalCount = [] #Initializing a empty list to store number of objects passed
classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
    "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
] #These are the default class names for YOLOv8 model which YOLOv8 use to classify object object

while True: #loop should run untill the video is finished
    ret, frame = capture.read() # Read a frame from the video
    detections = np.empty((0, 5)) # Initialize an empty array to store the detection results

    if mask.shape[:2] != frame.shape[:2]: # Check if the mask image is of the same size as the frame if not then make the resilution of mask and the image same
        mask = cv.resize(mask, (frame.shape[1], frame.shape[0])) 

    masked_frame = cv.bitwise_and(frame, frame, mask=mask) # Apply the mask to the frame Using bitwise and so that it should only take the mask region
    results = model(masked_frame, stream=True)

    cv.line(frame, (limit[0], limit[1]), (limit[2], limit[3]), (0, 0, 255), 2) # Draw a line on the frame at the specified limits where you want to count the object after it passses

    for result in results: # Loop through the detection results
        for box in result.boxes: # Loop through the bounding boxes of the detection result
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy() # Extract the bounding box coordinates from the result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # Convert the coordinates to integers

            conf = box.conf[0].item() # Extract the confidence score from the result (confidence is the value by YOLOv8 model which tell you how sure is the model about a particualr class)
            cls = int(box.cls[0].item())  # Extract the class index from the result
            current_class = classNames[cls] # Convert the class index to the class name

            if current_class == "(Class You want to identify)" and conf > threshold: #this if statement allows you to only how the bound the object when its in the class "person" and the confidence is greater than threshold(float value between (1 > x > 0)) choose the current class from the classNames list you want to identify through your code
                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # Draw a green rectangle around the object
                cv.putText(frame, f"{current_class} {conf:.2f}", (x1, y1 - 10),cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) # Display the class name and confidence score on the frame
                currentArray = np.array([[x1, y1, x2, y2, conf]]) # Convert the detection result to a numpy array
                detections = np.vstack((detections, currentArray)) # Stack the detection result to the existing array

    numbers = tracker.update(detections) # Create a tracker object and Update the tracker with the detection results and store it in numbers cariable

    for number in numbers: # Loop through the tracker results
        x1, y1, x2, y2, id = map(int, number) # Extract the bounding box coordinates and id from the tracker result
        cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2) # Draw a red rectangle around the object
        cz.putTextRect(frame, f"ID {id}", (x1, max(35, y1)),scale=0.6, thickness=1, offset=4) # Display the id on the frame
        
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2   # Calculate the center of the bounding box

        if 110 < cx < 230 and 150 - 20 < cy < 150 + 20: # Check if the object is within the specified limits
            if totalCount.count(id) == 0: # Check if the id is not in the total count list
                totalCount.append(id) # Add the id to the total count list

        cz.putTextRect(frame, f"Total People Upstair: {len(totalCount)}", (50,50), 
                               scale=2, thickness=4, offset=3) # Display the total count on the frame
        cv.rectangle(frame, (cx,cy), (cx+10, cy+10), (0, 255, 0), 2) # Draw a green rectangle around the center of the bounding box

    cv.imshow("Frame", frame) # Display the frame

    if cv.waitKey(1) & 0xFF == ord("d"): # Press 'd' to exit the loop
        break

capture.release() # Release the video capture object
cv.destroyAllWindows() # Destroy all windows
