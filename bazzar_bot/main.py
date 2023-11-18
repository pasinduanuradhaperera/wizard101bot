import os
from time import time 
from PIL import ImageGrab
import numpy as np
import cv2 as cv 
import pygetwindow

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Select the Wizard101 window
activeWindow = pygetwindow.getWindowsWithTitle('Wizard101')[0]

left, top = activeWindow.topleft
right, bottom = activeWindow.bottomright

# Read the template image and set template-related variables
needle_img = cv.imread('nextpage.png', cv.IMREAD_UNCHANGED)
needle_img_gray = cv.cvtColor(needle_img, cv.COLOR_BGR2GRAY)
needle_w = needle_img.shape[1]
needle_h = needle_img.shape[0]
method = cv.TM_CCOEFF_NORMED

# Initialize the start time
startTime = time()

while True:
    try:
        # Capture the image
        screenCap = ImageGrab.grab(bbox=(left + 400, top + 350, right - 400, bottom - 150))
        screenCap = np.array(screenCap)  # Convert image into a numpy array

        # Check if the captured image is not empty
        if screenCap.size == 0:
            raise ValueError("Empty image captured")

        screenCap = cv.cvtColor(screenCap, cv.COLOR_RGB2BGR)
        screenCap_gray = cv.cvtColor(screenCap, cv.COLOR_BGR2GRAY)

        # Perform template matching
        result = cv.matchTemplate(screenCap_gray, needle_img_gray, method)

        locations = np.where(result >= 0.1)
        locations = list(zip(*locations[::-1]))

        rectangles = []
        for loc in locations:
            rect = [int(loc[0]), int(loc[1]), needle_w, needle_h]
            # Add every box to the list twice in order to retain single (non-overlapping) boxes
            rectangles.append(rect)
            rectangles.append(rect)


        rectangles, weights = cv.groupRectangles(rectangles, groupThreshold=1, eps=0.5)
        #print(rectangles)

        points = []
        if len(rectangles):
            #print('Found needle.')

            line_color = (0, 255, 0)
            line_type = cv.LINE_4
            marker_color = (255, 0, 255)
            marker_type = cv.MARKER_CROSS

            # Loop over all the rectangles
            for (x, y, w, h) in rectangles:

                # Determine the center position
                center_x = x + int(w/2)
                center_y = y + int(h/2)
                # Save the points
                points.append((center_x, center_y))

                
            top_left = (x, y)
            bottom_right = (x + w, y + h)
            # Draw the box
            cv.rectangle(screenCap, top_left, bottom_right, color=line_color, 
                                lineType=line_type, thickness=2)
                
        
        # Display the game window
        cv.imshow('game window', screenCap)

        print('FPS: {}'.format(1 / (time() - startTime)))
        startTime = time()  # Reset timer

        if cv.waitKey(1) == ord('q'):
            cv.destroyAllWindows()
            break

    except Exception as e:
        # Handle the exception
        print(f"Error: {e}")

print('finished')
