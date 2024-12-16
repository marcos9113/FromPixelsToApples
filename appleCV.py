#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 11:23:45 2023
@author: marcos_007
"""

import cv2
import numpy as np

# Read the image
image = cv2.imread('tomatoApple.jpg')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Laplacian filter
laplacian_image = cv2.Laplacian(gray_image, cv2.CV_64F)
laplacian_image = np.uint8(np.absolute(laplacian_image) * 255)

# Apply thresholding
thresholded_image = cv2.threshold(laplacian_image, 127, 255, cv2.THRESH_BINARY)[1]

# Find contours
contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw bounding rectangles around contours
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Show the image with rectangles
cv2.imshow('Original Image', image)
key = cv2.waitKey(0)

# Close windows on pressing 'q'
if key == ord('q'):
    cv2.destroyAllWindows()

# Save images for apple and tomato
for i, contour in enumerate(contours):
    if i == 1:
        x2, y2, w2, h2 = cv2.boundingRect(contour)
        apple_image = image[y:y + h, x:x + w]
        tomato_image = image[y2:y2 + h2, x2:x2 + w2]
        cv2.imwrite('apple.jpg', apple_image)
        cv2.imwrite('tomato.jpg', tomato_image)

        # Analyze the texture of the apple image
        apple_texture = cv2.textureFeatures(apple_image, cv2.TEXTURE_ENERGY_MEAN)

        # Analyze the texture of the tomato image
        tomato_texture = cv2.textureFeatures(tomato_image, cv2.TEXTURE_ENERGY_MEAN)

        # Print texture comparison results
        print("Apple texture:", apple_texture)
        print("Tomato texture:", tomato_texture)

        if apple_texture < tomato_texture:
            print("The apple is smoother than the tomato.")
        elif apple_texture > tomato_texture:
            print("The apple is rougher than the tomato.")
        else:
            print("The apple and tomato have the same texture.")
