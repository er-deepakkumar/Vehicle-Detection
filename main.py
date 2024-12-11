import cv2 as  cv

import numpy as np

# Open video file
cap=cv.VideoCapture('video.mp4')
count_line_position=360 # Position of the counting line
counter=0

# Minimum width and height of detected rectangles
min_width_react=100
min_height_react=100

# Create a background subtractor object
algo=cv.bgsegm.createBackgroundSubtractorMOG()
def center_handle(x,y,w,h):
    x1=int(w/2)
    y1=int(h/2)
    cx=x+x1
    cy=y+y1
    return cx,cy

detect=[]
offset =6 # allowable error between pixel
counter=0





while True:
    ret,frame1=cap.read()
    if not ret:
        print("End of video or failed to capture frame.")
        break
    # Convert frame to grayscale and apply Gaussian blur
    grey=cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
    blur=cv.GaussianBlur(grey,(3,3),5)
    
    # Apply background subtraction and morphological transformations
    img_sub=algo.apply(blur)
    dilat = cv.dilate(img_sub,np.ones((5,5)))
    kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
    dilatada=cv.morphologyEx(dilat,cv.MORPH_CROSS,kernel)
    dilatada=cv.morphologyEx(dilatada,cv.MORPH_CROSS,kernel)
     # Find contours on the processed image
    counterShape,h=cv.findContours(dilatada,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    cv.line(frame1,(25,count_line_position),(1200,count_line_position),(255,127,0),3)
    
    # draw the rectangle box for  the detected object
    for (i,c ) in enumerate(counterShape): 
        (x,y,w,h)=cv.boundingRect(c)
        validate_counter=(w>=min_width_react) and (h>=min_height_react)
        if not validate_counter:
            continue
        
        cv.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)
        center=center_handle(x,y,w,h)
        detect.append(center)
        cv.circle(frame1,center,4,(0,0,255),-1)
        # Counting logic: Check if center crosses the line
        for (cx,cy) in detect:
            if  cy <(count_line_position+offset) and cy>(count_line_position-offset):
                
                counter+=1
                cv.line(frame1,(25,count_line_position),(400,count_line_position),(0,127,255),3)
                detect.remove((cx,cy))
                print('vehicle counter :'+str(counter))
        
 # Display counter on frame
    cv.putText(frame1,"vehicle counter :" + str(counter),(100,70),cv.FONT_HERSHEY_COMPLEX_SMALL,2,(0,0,255),5)
    
    #  it is sample video to show how  to use background subtraction
    # cv.imshow('detector',dilatada)
        
     # Display the video frame
    cv.imshow('video Original',frame1)
 
      # Exit on pressing Enter (ASCII code 13)
    if cv.waitKey(15)==13:
        break
    
# Release the resources
cap.release()
cv.destroyAllWindows()
