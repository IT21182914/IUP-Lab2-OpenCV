import cv2 as cv
import numpy as np

# Load the video file
video = cv.VideoCapture('small_laptop_connections.mov')

# Process each frame in the video
while video.isOpened():
    ret, frame = video.read()  # Read frame from video
    if not ret:
        break

    # Convert from BGR to HSV color-space
    hsvimg = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Define blue color range in HSV
    lower_blue = np.array([110, 50, 50])  # Lower bound for blue
    upper_blue = np.array([130, 255, 255])  # Upper bound for blue

    # Threshold the HSV image to detect blue color
    blueMask = cv.inRange(hsvimg, lower_blue, upper_blue)

    # Extract the blue object using bitwise_and
    result = cv.bitwise_and(frame, frame, mask=blueMask)

    # Resize for better viewing
    originalvideo = cv.resize(frame, (400, 400))
    blueMaskVideo = cv.resize(blueMask, (400, 400))
    blueobject = cv.resize(result, (400, 400))

    # Display the original video
    cv.imshow('Original Video', originalvideo)

    # Display the binary mask (black & white)
    cv.imshow('Binary Mask Video', blueMaskVideo)

    # Display only the extracted blue object
    cv.imshow('Blue Object', blueobject)

    # Press 'q' to exit
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release video and close all OpenCV windows
video.release()
cv.destroyAllWindows()
