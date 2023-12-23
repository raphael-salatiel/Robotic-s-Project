import cv2
import numpy as np
import imutils

# Function to stitch two frames together
def stitch_frames(frame1, frame2):
    # Convert frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Use ORB feature detector and descriptor
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # Use Brute Force matcher with Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort matches based on their distances
    matches = sorted(matches, key=lambda x: x.distance)

    # Get matching points
    points1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    points2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    # Find the homography matrix
    H, _ = cv2.findHomography(points1, points2, cv2.RANSAC, 4.0)

    # Warp the first frame
    result = cv2.warpPerspective(frame1, H, (frame1.shape[1] + frame2.shape[1], frame1.shape[0]))

    # Combine the two frames
    result[:, 0:frame2.shape[1]] = frame2

    return result

# Get the dimensions of the screen
screen_width, screen_height = 1020, 500

video1_path = 'videos/video1.mp4'
video2_path = 'videos/video2.mp4'

# Initialize video captures
cap1 = cv2.VideoCapture(video1_path)
cap2 = cv2.VideoCapture(video2_path)

# Check if video captures were successful
if not cap1.isOpened() or not cap2.isOpened():
    print("Error opening video files.")
    exit()

# Read the first frame from each video
ret1, frame1 = cap1.read()
ret2, frame2 = cap2.read()

# Check if frames were read successfully
if not ret1 or not ret2:
    print("Error reading frames.")
    exit()

# Resize frames
frame1 = imutils.resize(frame1, width=screen_width, height=screen_height)
frame2 = imutils.resize(frame2, width=screen_width, height=screen_height)

# Display original frames for debugging
cv2.imshow("Video 1", frame1)
cv2.imshow("Video 2", frame2)
cv2.waitKey(0)

# Stitch the first two frames
result = stitch_frames(frame1, frame2)

# Loop through the rest of the frames
while True:
    # Read the next frame from the second video
    ret2, frame2 = cap2.read()

    # Check if the frame was read successfully
    if not ret2:
        break

    # Resize the frame
    frame2 = imutils.resize(frame2, width=screen_width, height=screen_height)

    # Display the original frame for debugging
    cv2.imshow("Video 2", frame2)
    cv2.waitKey(0)

    # Stitch the current frame with the result
    result = stitch_frames(result, frame2)

# Display the stitched video
cv2.imshow("Stitched Video", result)
cv2.waitKey(0)

# Release video captures
cap1.release()
cap2.release()
cv2.destroyAllWindows()
