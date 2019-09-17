from keras.models import load_model
from keras.preprocessing import image
import dlib, cv2
import numpy as np


COLOR = (255,255,255)
LINE_WIDTH = 1
IMG_SIZE = 48


#Load model for emotion detection.
model = load_model('model.h5')

#Define dlib face detector.
detector = dlib.get_frontal_face_detector()

#Capture webcam.
cam = cv2.VideoCapture(0)

while True:
    #Get RGB frame from camera and use to get face coords.
    _, img = cam.read()
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rects = detector(rgb_image)
    
    for rect in rects:

        #Get face boundaries.
        x, y, w, h = [rect.left(), rect.top(), rect.right(), rect.bottom()]
        ul, lr = [(x, y), (w, h)]

        #Crop face.
        face = img[int(y):int(y+h), int(x):int(x+w)]
        
        #Make grayscale and resize for model.
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face = image.img_to_array(face)
        face = np.expand_dims(face, axis = 0)
 
        #Make predictions based off of this face in this frame.
        predictions = model.predict(face)
 
        #Find max indexed array and get the label for best guess.
        guess_idx = np.argmax(predictions[0])
        labels = ["Angry", "Sad", "Happy", "Neutral"]
        emotion = labels[guess_idx]

        if emotion == "Angry":
            COLOR = (0,0,255)
        elif emotion == "Sad":
            COLOR = (255,0,0)
        elif emotion == "Happy":
            COLOR = (0,255,255)
        else:
            COLOR = (255,255,255)
 
        #Write text over image.
        cv2.putText(img, emotion, (int(x), int(y) - 20),
                cv2.FONT_HERSHEY_TRIPLEX, 1, COLOR, 2)

    #Draw rectangle around face.
    #cv2.rectangle(img, ul, lr, COLOR, LINE_WIDTH)

    #Update frame in video feed.
    cv2.imshow('Emotion Detection', img)

    #Press ESC to quit.
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
