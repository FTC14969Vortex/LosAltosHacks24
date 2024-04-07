import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
from cvzone.ClassificationModule import Classifier
import cv2
from PIL import ImageFont, ImageDraw, Image  

#Update these file paths here as the model and label files are added
#Copy file paths by right clicking on the file, pressing option and selecting "Copy as Pathname"
filePath = '/Users/shash/Documents/Vortex/LosAltosHacks24/MyModel/keras_model.h5'
labelPath = '/Users/shash/Documents/Vortex/LosAltosHacks24/MyModel/labels.txt'
outputFilePath = "/Users/shash/Documents/Vortex/LosAltosHacks24/output.txt"

#Initialize the classifier with the file and label paths
classifier = Classifier(filePath, labelPath)

#Get the camera
camera = cv2.VideoCapture(0)  # Initialize video capture
_, img = camera.read()  # Capture frame-by-frame

#Text on the screen
#font
font = cv2.FONT_HERSHEY_COMPLEX  
# position 
org = (int((img.shape[0]/2)), int((img.shape[1]/2)))

# fontScale 
fontScale = 2
# Blue color in BGR 
color = (255, 100, 0) 
# Line thickness of 2 px 
thickness = 2

#Reset Text File
f = open(outputFilePath, 'w')
f.write('')
f.close()

run = True
while run:
    _, img = camera.read()  # Capture frame-by-frame
    prediction = classifier.getPrediction(img, False)
        
    height, width, _ = img.shape
    size = min(height, width)
    start_x = (width - size) // 2
    start_y = (height - size) // 2
    img = img[start_y:start_y+size, start_x:start_x+size]
    
    output = str(prediction[2])
    print(output.split()[1])
    
    #Write Current Letter
    if cv2.waitKey(1) == 13:
        file = open(outputFilePath, 'a')
        file.write(output.split()[1])
        print('Wrote to file')
        file.close()
    
        
    if cv2.waitKey(1) == 32:
        file = open(outputFilePath, 'a')
        file.write(' ')
        print('Wrote space to file')
        file.close()

        
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    

    img = cv2.flip(img, 1)

    outputFile = open(outputFilePath, 'r')
    image = cv2.imread('/Users/shash/Documents/Vortex/LosAltosHacks24/MyModel/black-370118_1280.png')
    img = cv2.putText(img, output.split()[1], org, font, fontScale, color, thickness, cv2.LINE_AA)
    image = cv2.putText(image, outputFile.read(), org, font, fontScale, (255,255,255), thickness, cv2.LINE_AA)
    cv2.imshow("Image", img)
    cv2.imshow("Output", image)

    # Wait for a key press to close window
    if cv2.waitKey(1) == 27:
        run = False
    
camera.release()
cv2.destroyAllWindows()