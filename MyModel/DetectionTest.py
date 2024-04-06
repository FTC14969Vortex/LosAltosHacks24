import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
from cvzone.ClassificationModule import Classifier
import cv2

#Update these file paths here as the model and label files are added
#Copy file paths by right clicking on the file, pressing option and selecting "Copy as Pathname"
filePath = '/Users/shash/Documents/Vortex/LosAltosHacks24/MyModel/keras_model.h5'
labelPath = '/Users/shash/Documents/Vortex/LosAltosHacks24/MyModel/labels.txt'

#Initialize the classifier with the file and label paths
classifier = Classifier(filePath, labelPath)

#Get the camera
camera = cv2.VideoCapture(0)  # Initialize video capture

#font
font = cv2.FONT_HERSHEY_SIMPLEX  
# position 
org = (550, 100)
# fontScale 
fontScale = 2
# Blue color in BGR 
color = (255, 0, 0) 
# Line thickness of 2 px 
thickness = 2

run = True
while run:
    _, img = camera.read()  # Capture frame-by-frame
    prediction = classifier.getPrediction(img, False)
    
    output = str(prediction[2])
    
    img = cv2.putText(img, output, org, font, fontScale, color, thickness, cv2.LINE_AA)
    
    print(prediction)  # Print prediction result

    cv2.imshow("Image", img)

    
    # Wait for a key press to close window
    if cv2.waitKey(1) == 27:
        run = False
            
camera.release()
cv2.destroyAllWindows()