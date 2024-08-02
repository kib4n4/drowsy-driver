import cv2  # OpenCV library for computer vision tasks
import os  # Library for operating system dependent functionality
from keras.models import load_model  # Keras model loading function
import numpy as np  # Library for numerical operations
from pygame import mixer  # Pygame mixer for sound playback
import time  # Library for time-related functions

# Initialize the mixer for playing sound
mixer.init()
# Load the alarm sound
sound = mixer.Sound('alarm.wav')

# Load Haar Cascade classifiers for face and eye detection
face = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')

# Labels for eye states
lbl = ['Close', 'Open']

# Load the pre-trained drowsiness detection model
model = load_model('dl_model/drowsiness_cnn.h5')

# Get the current working directory
path = os.getcwd()

# Start capturing video from the default camera (0)
cap = cv2.VideoCapture(0)

# Set font for displaying text on the video frame
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

# Initialize counters and variables
count = 0  # Frame count
closed_eye_frames = 0  # Counter for consecutive closed eye frames
closed_eye_threshold = 30  # Threshold for how many frames both eyes need to be closed
thicc = 2  # Thickness of the rectangle for alert
rpred = [99]  # Right eye prediction
lpred = [99]  # Left eye prediction

# Main loop to continuously capture frames from the camera
while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    
    # Get the height and width of the frame
    height, width = frame.shape[:2]

    # Convert the frame color from BGR to RGB
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the frame
    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    # Detect left and right eyes in the frame
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    # Draw a black rectangle at the bottom of the frame for displaying text
    cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

    # Process right eye detection
    for (x, y, w, h) in right_eye:
        # Extract the right eye region from the frame
        r_eye = frame[y:y + h, x:x + w]
        count += 1  # Increment the frame count

        # Convert the right eye image to RGB
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2RGB)
        # Resize the image to 100x100 pixels
        r_eye = cv2.resize(r_eye, (100, 100))
        # Reshape for model input (batch size, height, width, channels)
        r_eye = r_eye.reshape((-1, 100, 100, 1))

        # Make a prediction using the model for the right eye
        rpred = np.argmax(model.predict(r_eye), axis=-1)
        break  # Break after processing the first detected right eye

    # Process left eye detection
    for (x, y, w, h) in left_eye:
        # Extract the left eye region from the frame
        l_eye = frame[y:y + h, x:x + w]
        count += 1  # Increment the frame count

        # Convert the left eye image to RGB
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2RGB)
        # Resize the image to 100x100 pixels
        l_eye = cv2.resize(l_eye, (100, 100))
        # Reshape for model input (batch size, height, width, channels)
        l_eye = l_eye.reshape((-1, 100, 100, 1))

        # Make a prediction using the model for the left eye
        lpred = np.argmax(model.predict(l_eye), axis=-1)
        break  # Break after processing the first detected left eye

    # Check if both eyes are closed
    if rpred[0] == 0 and lpred[0] == 0:
        closed_eye_frames += 1  # Increment if both eyes are closed
    else:
        closed_eye_frames = 0  # Reset if either eye is open

    # Trigger the alarm if the threshold is met
    if closed_eye_frames > closed_eye_threshold:
        cv2.imwrite(os.path.join(path, 'image.jpg'), frame)  # Save the current frame
        sound.play()  # Play the alarm sound
        # Visual alert logic remains the same
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)

   # Display the current frame with any annotations
    cv2.putText(frame, 'Left: ' + lbl[0] + ', Right: ' + lbl[1], (10, height - 10), font, 1, (255, 255, 255), 2)

    # Show the video frame
    cv2.imshow('Drowsiness Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
