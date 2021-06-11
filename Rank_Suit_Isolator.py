### Takes a card picture and creates a top-down 200x300 flattened image
### of it. Isolates the suit and rank and saves the isolated images.
### Runs through A - K ranks and then the 4 suits.

# Import necessary packages
import cv2
import numpy as np
import time
import Cards
import os

img_path = os.path.dirname(os.path.abspath(__file__)) + '/Card_Imgs/'

IM_WIDTH = 1280
IM_HEIGHT = 720

#Groesse der Ecken zur Identifikation in Pixeln
RANK_WIDTH = 70
RANK_HEIGHT = 125

SUIT_WIDTH = 70
SUIT_HEIGHT = 100

# If using a USB Camera instead of a PiCamera, change PiOrUSB to 2
PiOrUSB = 2

if PiOrUSB == 1:
    # Import packages from picamera library
    from picamera.array import PiRGBArray
    from picamera import PiCamera

    # Initialize PiCamera and grab reference to the raw capture
    camera = PiCamera()
    camera.resolution = (IM_WIDTH,IM_HEIGHT)
    camera.framerate = 10
    rawCapture = PiRGBArray(camera, size=(IM_WIDTH,IM_HEIGHT))

if PiOrUSB == 2:
    # Initialize USB camera
    cap = cv2.VideoCapture(0)

# Use counter variable to switch from isolating Rank to isolating Suit


for farbname in ['Eichel','Schellen','Herz','Gras']:
    for schlagname in ['As','Koenig','Ober','Unter','Zehn','Neun','Acht','Sieben']:
        filename = farbname + '_' + schlagname + '.jpg'
    print('Druecke "p" um ein Bild von ' + farbname + ' ' + schlagname + ' aufzunehmen')
    
    
#bei PiKamera muss so das Video eingelesen werden
    if PiOrUSB == 1: # PiCamera
        rawCapture.truncate(0)
        # Press 'p' to take a picture
        for frame in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):

            image = frame.array
            cv2.imshow("Card",image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("p"):
                break

            rawCapture.truncate(0)
#bei USB
    if PiOrUSB == 2: # USB camera
        # Press 'p' to take a picture
        while(True):

            ret, frame = cap.read()
            cv2.imshow("Card",frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("p"):
                image = frame
                break

    # Pre-process image
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    retval, thresh = cv2.threshold(blur,100,255,cv2.THRESH_BINARY)

    # Find contours and sort them by size
    cnts,hier = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea,reverse=True)

    # Assume largest contour is the card. If there are no contours, print an error
    flag = 0
    image2 = image.copy()

    if len(cnts) == 0:
        print('No contours found!')
        quit()

    card = cnts[0]

    # Approximate the corner points of the card
    peri = cv2.arcLength(card,True)
    approx = cv2.approxPolyDP(card,0.01*peri,True)
    pts = np.float32(approx)

    x,y,w,h = cv2.boundingRect(card)

    # Flatten the card and convert it to 200x300 (import aus cards.py)
    warp = Cards.flattener(image,pts,w,h)
    cv2.imshow('ecke',warp)

    # Grab corners of card image
    corner_rechts_oben = warp[15:85, 182:225]
    corner_links_oben = warp[0:75, 0:38]
    height,width = corner_rechts_oben.shape[:2]

    #Resize faktor 4, sodass eine fixe große Maske machbar
    corner_zoom = cv2.resize(corner_rechts_oben, (4*width,4*height),cv2.INTER_CUBIC)
    corner_blur = cv2.GaussianBlur(corner_zoom,(5,5),0)
    corner_bilateral = cv2.bilateralFilter(corner_blur,9,75,75)
    retval, corner_thresh = cv2.threshold(corner_bilateral, 155, 255, cv2. THRESH_BINARY_INV)

    #retval, corner_thresh = cv2.threshold(corner_blur, 155, 255, cv2. THRESH_BINARY_INV)
    #zu Debug zwecken : Testecke anzeigen
    #cv2.imshow('ecke',corner)

    # Isolate Schlag oder Farbe
    rank = corner_thresh
    cv2.imshow('rank',rank)
    #bessere Konturenfindung in unzugeschnittener Ecke
    rank_cnts, hier = cv2.findContours(corner_thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #falls rank_cnts nicht ausreichen: CHAIN_APPROX_NONE probieren
    cv2.drawContours(corner_thresh, rank_cnts, -1, (0,255,0), 3)

    # sortieren der Konturen in unserer bisherigen Maske
    rank_cnts = sorted(rank_cnts, key=cv2.contourArea,reverse=True)

    #richtigen Teil auswählen -- experimentieren!
    x0,y0,w0,h0 = cv2.boundingRect(rank_cnts[0])
    rank_roi = rank[y0:y0+h0, x0:x0+w0]
    # größe fürs resizen festlegen - schlag und farbe sind unterschiedlich groß
    #rank_sized = cv2.resize(rank_roi, (RANK_WIDTH, RANK_HEIGHT), 0, 0)
    #suit_sized = cv2.resize(suit_roi, (SUIT_WIDTH, SUIT_HEIGHT), 0, 0)
    final_img = rank_roi

    cv2.imshow("Image",final_img)

    # Save image
    print('Um zu speichern Taste "c" druecken.')
    key = cv2.waitKey(0) & 0xFF
    if key == ord('c'):
        cv2.imwrite(img_path+filename,final_img)



cv2.destroyAllWindows()
camera.close()
