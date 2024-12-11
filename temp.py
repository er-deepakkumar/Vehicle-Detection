import cv2 as cv
import numpy as np

# Open video file
cap = cv.VideoCapture('video.mp4')
count_line_position = 550

# Create a background subtractor object
algo = cv.bgsegm.createBackgroundSubtractorMOG()

vehicle_count = 0  # Initialize vehicle counter

while True:
    ret, frame1 = cap.read()
    
    # Check if frame is read correctly
    if not ret:
        break
    
    # Convert frame to grayscale
    grey = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blur = cv.GaussianBlur(grey, (3, 3), 5)
    
    # Apply background subtraction on blurred frame
    img_sub = algo.apply(blur)
    
    # Dilate the image to fill in gaps
    dilat = cv.dilate(img_sub, np.ones((5, 5), np.uint8))
    
    # Define the structuring element for morphological transformations
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    
    # Perform morphological operations to remove noise
    dilatada = cv.morphologyEx(dilat, cv.MORPH_CLOSE, kernel)
    dilatada = cv.morphologyEx(dilatada, cv.MORPH_CLOSE, kernel)
    
    # Find contours on the processed frame
    contours, _ = cv.findContours(dilatada, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    # Draw the counting line
    cv.line(frame1, (25, count_line_position), (1200, count_line_position), (255, 127, 0), 3)
    
    # Process each contour
    for contour in contours:
        area = cv.contourArea(contour)
        if area > 500:  # Filter based on area
            x, y, w, h = cv.boundingRect(contour)  # Get bounding box
            cv.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw bounding box
            
            # Check if the center of the bounding box crosses the line
            center_y = y + h // 2
            if center_y > count_line_position:  # Adjust this condition based on your logic
                vehicle_count += 1  # Increment vehicle count

    # Display the vehicle count
    cv.putText(frame1, f'Count: {vehicle_count}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the original video frame
    cv.imshow('Original Video', frame1)
    
    # Break the loop on pressing 'Enter'
    if cv.waitKey(15) == 13:
        break

# Release the resources
cap.release()
cv.destroyAllWindows()
