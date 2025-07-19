import cv2
import os

# Folder containing the images
image_folder = 'pics/temp'
video_name = 'videos/hmpcc.mp4'
fps = 10  # frames per second

# Get sorted list of image files
images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png") or img.endswith(".jpg")])
images = images[:100]
# Read the first image to get dimensions
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# Define the video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' for .avi
video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

# Add images to the video
for image in images:
    img_path = os.path.join(image_folder, image)
    video.write(cv2.imread(img_path))

video.release()
print(f"Video saved as {video_name}")