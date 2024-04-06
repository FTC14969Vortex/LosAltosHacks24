import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
from cvzone.ClassificationModule import Classifier
import cv2

#Update these file paths here as the model and label files are added
#Copy file paths by right clicking on the file, pressing option and selecting "Copy as Pathname"
filePath = '/Users/shash/Documents/Vortex/LosAltosHacks24/MyModel/keras_model.h5'
labelPath = '/Users/shash/Documents/Vortex/LosAltosHacks24/MyModel/labels.txt'

file = open("/Users/shash/Documents/Vortex/LosAltosHacks24/output.txt", "w")
#Initialize the classifier with the file and label paths
classifier = Classifier(filePath, labelPath)

#Get the camera
camera = cv2.VideoCapture(0)  # Initialize video capture

#Text on the screen
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
    
    file.write(output)
    print("Output written to Output.txt")
    
    img = cv2.putText(img, output, org, font, fontScale, color, thickness, cv2.LINE_AA)
    
    #Flip the image
    cv2.flip(img, 1)
    
    #Display the image
    cv2.imshow("Image", img)

    
    # Wait for a key press to close window
    if cv2.waitKey(1) == 27:
        run = False
        file.close()
            
camera.release()
cv2.destroyAllWindows()