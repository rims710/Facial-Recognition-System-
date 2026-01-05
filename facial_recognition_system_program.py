# ______________________________________________________FACIAL RECOGNITION SYSTEM ~ RIMJHIM JAIN__________________________________________________________
# ____________________________________________________________________START OF PROGRAM____________________________________________________________________

from deepface import DeepFace # Used for facial recognition and verification.
import cv2 # OpenCV library for video capture and image processing.
import numpy as np # Fundamental package for numerical operations, especially with arrays.
from PIL import Image # Python Imaging Library, used for image manipulation (though not directly used in the main loop here).
import pandas as pd # Data analysis and manipulation library (not directly used in the main loop here).
from tqdm import tqdm # Library for progress bars (not directly used in the main loop here).
import os # Provides a way of using operating system dependent functionality.
import tensorflow as tf # Open-source machine learning framework (DeepFace uses it internally).
import threading # For running tasks concurrently.
import time # Used for time-related functions, like adding pauses for debugging.

# ______________________________________________________________________Model Initialization_____________________________________________________________

print("Starting model build...")

# This line builds the VGG-Face model, which is a pre-trained deep learning model.
# specifically designed for face recognition tasks.

model = DeepFace.build_model("VGG-Face")
print("Model built")


# __________________________________________________________________________Webcam Setup_________________________________________________________________

print("Starting video capture")

# Initializes the webcam. '0' usually refers to the default webcam.
# 'cv2.CAP_DSHOW' is a specific backend that can help with webcam issues on Windows.

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
print("Video capture started")

# Set webcam resolution.
# These lines attempt to set the width and height of the video frame.

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Check if the webcam opened successfully.
# It's crucial to verify if the webcam is accessible.

if not cap.isOpened():
    print("Error: Cannot open webcam. Please check if it's connected and not in use by another application.")

    # If the webcam can't be opened, the script exits to prevent further errors.
    
    exit()

# ____________________________________________________________________Face Matching Variables____________________________________________________________

# This boolean variable will store the result of whether a face match is found.

face_match = False

# __________________________________________________________________Reference Image Path_________________________________________________________________

# IMPORTANT: This is the path to the image that DeepFace will compare against the webcam feed.
# Make sure this path is correct and the image actually exists on your system.

reference_img = r"C:\Users\Dell\Pictures\Camera Roll\WIN_20250525_13_15_16_Pro.jpg"

# __________________________________________________________________Verification Counter_________________________________________________________________

# This counter helps control how often the computationally intensive face verification runs.
# We don't need to check every single frame, as that can slow down the program.

counter = 0

# __________________________________________________________________Main Video Processing Loop___________________________________________________________

# Main loop for video processing.
# This loop continuously reads frames from the webcam and performs face verification.

while True:

    # Read a single frame from the webcam.
    # 'ret' is a boolean indicating if the frame was read successfully.
    # 'frame' is the actual image frame.
    
    ret, frame = cap.read() 
    if not ret:

        # If a frame cannot be read, it indicates an issue with the webcam, so the loop breaks.
        
        print("Failed to grab frame. Exiting...")
        break 
# __________________________________________________________________Face Verification Logic______________________________________________________________

# We perform face verification only every 30 frames to save processing power.
    
    if counter % 30 == 0:
        print("Attempting face verification directly...")
        try:
            
            # DeepFace.verify compares the 'frame' from the webcam with the 'reference_img'.
            # 'model_name="VGG-Face"' specifies which model to use for comparison.
            # 'enforce_detection=False' means it will try to verify even if a perfect.
            # face detection isn't made, which can be useful in varying conditions.

            result = DeepFace.verify(frame, reference_img, model_name="VGG-Face", enforce_detection=False)

            # The 'verified' key in the result dictionary tells us if a match was found.
            
            face_match = result["verified"] 
            print(f"Verification result: {face_match}") 
        except Exception as e:

            # If an error occurs during verification (e.g., no face detected in either image).
            # It will be caught here, and 'face_match' is set to False.
            
            print(f"!!! CRITICAL FACE VERIFICATION ERROR: {e}")
            face_match = False

    # Increment the counter for the next iteration.

    counter += 1

    # Get the current face match status.
    
    match = face_match

# _______________________________________________________________Displaying Results on the Frame_________________________________________________________

    # This section draws text on the video frame to indicate whether a match was found.
    
    if match:

        # If 'match' is True, display "MATCH!!" in green.
        
        cv2.putText(frame, "MATCH!!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    else:
        
        # If 'match' is False, display "NO MATCH!!" in red.
        
        cv2.putText(frame, "NO MATCH!!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)


# _______________________________________________________________Displaying Results on the Frame_________________________________________________________

    # Show the modified frame in a window named "video".
    
    cv2.imshow("video", frame)

    # These lines try to ensure the video window stays on top and at a specific position.
    
    cv2.setWindowProperty("video", cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow("video", 0, 0) 

# _______________________________________________________________Displaying Results on the Frame_________________________________________________________

    # Wait for 1 millisecond for a key press.
    
    key = cv2.waitKey(1)
    
     # If the 'q' key is pressed, break out of the loop, ending the program.
     
    if key == ord("q"):
        print(" 'q' pressed. Exiting...")
        break

# ____________________________________________________________________________Cleanup____________________________________________________________________


# These actions are performed after the main loop finishes (e.g., when 'q' is pressed or an error occurs).

print("Releasing webcam...")

# Release the webcam resource to free it up for other applications.

cap.release()
# Close all OpenCV windows.
print("Destroying all OpenCV windows...")

# Close all open OpenCV windows.

cv2.destroyAllWindows()
print("Script finished.")
# ______________________________________________________________________END OF PROGRAM___________________________________________________________________
