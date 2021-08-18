# Author : Subhro Mukherjee
# Last Modified : 23-06-2021


# Import opencv
import cv2


# Load pre-trained data from opencv
trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Pump the image into the variable
#img = cv2.imread('face.jpg')

# Capture video from webcam
webcam = cv2.VideoCapture(0)

# Loop to iterate through all the frames in the video
while True:

    # Read the current frame (1st var is boolean type to indicate if it was sucessful, frame stores the actual image frame ; read func returns 2 things)
    successful_frame_read, frame = webcam.read()
    
    #Convert ot grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    
    # Detect Faces (multiscale checks for all sizes of face using the pretrained data -- technical term = sliding window)
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    # Draw rectangle (1st parameter is the colored image,middle two are the face coordinates,then there is the rgb color currently set to green, the last 2 is thickness of the rectangle)
    for[x, y, w, h] in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,256,0), 2)

    # Testing the terminal
    cv2.imshow('Face Detector Running', frame)
    key = cv2.waitKey(1)

    # Stop is Q key is pressed
    if key == 81 or key == 113:
        break
    
# Release the video capture object
webcam.release()





