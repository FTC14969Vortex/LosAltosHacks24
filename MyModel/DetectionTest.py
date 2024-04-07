import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
from cvzone.ClassificationModule import Classifier
import cv2
import pygame

#Set up pygame
pygame.init()

#Update these file paths here as the model and label files are added
#Copy file paths by right clicking on the file, pressing option and selecting "Copy as Pathname"
filePath = '/Users/shash/Documents/Vortex/LosAltosHacks24/MyModel/keras_model.h5'
labelPath = '/Users/shash/Documents/Vortex/LosAltosHacks24/MyModel/labels.txt'
outputFilePath = "/Users/shash/Documents/Vortex/LosAltosHacks24/output.txt"
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

#Get the inital image
_, img = camera.read()  # Capture frame-by-frame

#Set up the pygame window
X, Y, _ = img.shape
screen = pygame.display.set_mode((X, Y))
pygame.display.set_caption('ASL Translator')

run = True
while run:
    _, img = camera.read()  # Capture frame-by-frame

    
    prediction = classifier.getPrediction(img, False)
    
    output = str(prediction[2])
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.putText(img, output, org, font, fontScale, color, thickness, cv2.LINE_AA)

    
    # Convert the image to Pygame surface
    height, width, _ = img.shape
    pygame_img = pygame.image.fromstring(img.tobytes(), (width, height), 'RGB')
        
    pressed = pygame.key.get_pressed()
    if pressed[pygame.K_RETURN]:
        print('Enter key pressed')
        file = open(outputFilePath, "w")
        file.write(output)
        file.close()
        
    # Display the image
    screen.blit(pygame_img, (0, 0))
    pygame.display.flip()
    
    # img = cv2.putText(img, output, org, font, fontScale, color, thickness, cv2.LINE_AA)
    
    #Flip the image
    cv2.flip(img, 1)
    
        
    # Wait for a key press to close window
    pressed = pygame.key.get_pressed()
    if pressed[pygame.K_ESCAPE]:
        run = False
            
camera.release()
cv2.destroyAllWindows()