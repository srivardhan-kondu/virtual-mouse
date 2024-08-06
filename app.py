import cv2
import numpy as np
import time
import track as tk
import autopy   


pTime = 0               
width = 1080             
heigtk = 1920            
frameR = 100            
smoothening = 8         
prev_x, prev_y = 0, 0   
curr_x, curr_y = 0, 0   

cap = cv2.VideoCapture(0)   
cap.set(3, width)           
cap.set(4, heigtk)

detector = tk.handDetector(maxHands=1)                  
screen_width, screen_heigtk = autopy.screen.size()
while True:
    success, img = cap.read()
    img = detector.findHands(img)                       
    lmlist, bbox = detector.findPosition(img)           

    if len(lmlist)!=0:
        x1, y1 = lmlist[8][1:]
        x2, y2 = lmlist[12][1:]

        fingers = detector.fingersUp()      
        cv2.rectangle(img, (frameR, frameR), (width - frameR, heigtk - frameR), (255, 0, 255), 2)   # Creating boundary box
        if fingers[1] == 1 and fingers[2] == 0:     
            x3 = np.interp(x1, (frameR,width-frameR), (0,screen_width))
            y3 = np.interp(y1, (frameR, heigtk-frameR), (0, screen_heigtk))

            curr_x = prev_x + (x3 - prev_x)/smoothening
            curr_y = prev_y + (y3 - prev_y) / smoothening

            autopy.mouse.move(screen_width - curr_x, curr_y)    
            cv2.circle(img, (x1, y1), 7, (255, 0, 255), cv2.FILLED)
            prev_x, prev_y = curr_x, curr_y

        if fingers[1] == 1 and fingers[2] == 1:    
            length, img, lineInfo = detector.findDistance(8, 12, img)

            if length < 40:    
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()    

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
