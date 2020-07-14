# -*- coding: utf-8 -*-

from PIL import ImageFont, ImageDraw, Image  
from datetime import datetime
from scipy import signal
import numpy as np
import cv2, time

# cap = cv2.VideoCapture(0)

# load font
font1 = ImageFont.truetype("Roboto-Regular.ttf", 80)
font2 = ImageFont.truetype("Roboto-Regular.ttf", 60)
font3 = ImageFont.truetype("Roboto-Regular.ttf", 35)

while(True):
    
    # get the current time
    now = datetime.now()
    current_date = now.strftime("%d %b %Y")
    current_time = now.strftime("%H:%M %p")
    current_day = now.strftime("%A")

    # get bpm and temperature
    bpm = "70 bpm"
    degree = u'\u00B0C'
    temperature = "29.5" + degree

    # capture frame by frame
    # ret, frame = cap.read()
    
    # Create a black image
    img = np.zeros((768,1366,3), np.uint8)

    # convert to pil for custom font
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    draw.text((10, 50), current_day, font=font1)
    draw.text((25, 130), current_date, font=font3)
    draw.text((15, 180), current_time, font=font2)
    draw.text((1150, 120), bpm, font=font2)
    draw.text((1150, 180), temperature, font=font2)
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)  
   
    # put the string in the image
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # cv2.putText(img, current_day,(10,100), font, 2.9,(255,255,255),3,cv2.LINE_AA)
    # cv2.putText(img, current_date,(20,150), font, 1.5,(255,255,255),1,cv2.LINE_AA)
    # cv2.putText(img, current_time,(15,220), font, 2,(255,255,255),2,cv2.LINE_AA)
    # cv2.putText(img, bpm,(1000,150), font, 2,(255,255,255),2,cv2.LINE_AA)
    # cv2.putText(img, temperature,(1000,220), font, 2,(255,255,255),2,cv2.LINE_AA)

    # create a window name frame and set it as fullscreen
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # display resulting frame
    # cv2.imshow('frame', frame)

    # display the result
    cv2.imshow('frame', img)

    # press 'Esc' key to quit
    k = cv2.waitKey(27)
    if k == 27:
        break

# when everything is done, release the capture and close the windows
cv2.destroyAllWindows()