import numpy as np
import cv2
import glob
import imutils

# Get the dimensions of the screen
screen_width, screen_height = 1020, 500  # Set the dimensions of your screen

image_path = glob.glob('images/*.jpg')
images = []

for image in image_path:
    img = cv2.imread(image)
    if img is not None:
        # Resize the image to fit within the screen dimensions
        img = imutils.resize(img, width=screen_width, height=screen_height)

        images.append(img)
        cv2.imshow("Resized Image", img)
        cv2.waitKey(0)
    else:
        print(f"Error loading image: {image}")

if len(images) < 2:
    print("Insufficient images for stitching.")
else:
    imageStitcher = cv2.Stitcher_create()
    status, stitched_img = imageStitcher.stitch(images)

    if status == cv2.Stitcher_OK:
        cv2.imwrite("images/stitchedOutput.png", stitched_img)
        cv2.imshow("Stitched Image", stitched_img)
        cv2.waitKey(0)
    else:
        print(f"Stitching failed with error code: {status}")
