import cv2
import numpy as np
import imutils

# Load the pre-trained gun detection Haar Cascade XML file
gun_cascade = cv2.CascadeClassifier('cascade.xml')

# Open the webcam
camera = cv2.VideoCapture(0)

# Variable to track whether a gun has been detected
gun_detected = False

while True:
    # Capture frame from the camera
    ret, frame = camera.read()
    
    # Resize the frame for faster processing
    frame = imutils.resize(frame, width=500)
    
    # Convert the frame to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect guns in the frame
    guns = gun_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(100, 100))
    
    # If a gun is detected, display a warning message
    if len(guns) > 0:
        gun_detected = True
        # Draw rectangles around detected guns
        for (x, y, w, h) in guns:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red rectangle
        
        # Display the danger message on the frame
        cv2.putText(frame, "DANGER! GUN DETECTED!", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Show the video feed with the detection boxes and the danger message
    cv2.imshow("Gun Detection Feed", frame)
    
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Print whether any guns were detected during the session
if gun_detected:
    print("Gun(s) detected.")
else:
    print("No guns detected.")

# Release the webcam and close OpenCV windows
camera.release()
cv2.destroyAllWindows()
