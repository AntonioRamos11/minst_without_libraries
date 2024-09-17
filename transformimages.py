import cv2
import numpy as np

# Load the image
image_path = 'ejemplos/'
#search for images in the folder
for i in range(10):
    image = cv2.imread(image_path + str(i) + '.jpeg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to remove noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Thresholding to create a binary image
    _, thresholded = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY_INV)

    # Optionally, apply morphology to clean small areas
    kernel = np.ones((3, 3), np.uint8)
    morphed = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)

    # Save the result to check the binary mask
    image_paths = (image_path + str(i) + '.jpeg')
    cv2.imwrite(image_paths, morphed)
