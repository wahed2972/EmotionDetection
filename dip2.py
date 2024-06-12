import cv2
import numpy as np
from keras.models import load_model

# Load the pre-trained model
model = load_model('EmotionDetectionModel.h5')

# Define a dictionary mapping emotion labels to BGR color codes for drawing bounding boxes
emotion_colors = {'Angry': (0, 0, 255), 'Disgust': (255, 0, 0), 'Fear': (128, 0, 128), 'Happy': (255, 255, 0), 'Sad': (0, 0, 128), 'Surprise': (0, 255, 255), 'Neutral': (0, 255, 0)}

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frameq
    ret, frame = video_capture.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame using OpenCV's Haar Cascade classifier
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Make predictions on each face in the frame
    for (x, y, w, h) in faces:
        # Extract the face region from the frame and preprocess it
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face.reshape((1, 48, 48, 1)) / 255.0

        # Make a prediction on the face using the pre-trained model
        prediction = model.predict(face)[0]

        # Convert the prediction to an emotion label
        emotion_label = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'][np.argmax(prediction)]

        # Draw a bounding box around the face and label it with the predicted emotion
        cv2.rectangle(frame, (x, y), (x+w, y+h), emotion_colors[emotion_label], 2)
        cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, emotion_colors[emotion_label], 2)

    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)

    # Exit the program if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
video_capture.release()
cv2.destroyAllWindows()
