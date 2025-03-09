import cv2
import numpy as np

# Load OpenCV's pre-trained colorization model
proto_file = "https://github.com/richzhang/colorization/blob/master/colorization_deploy_v2.prototxt"
model_file = "https://github.com/richzhang/colorization/blob/master/models/colorization_release_v2.caffemodel"
points_file = "https://github.com/richzhang/colorization/blob/master/resources/pts_in_hull.npy"

# Load model
net = cv2.dnn.readNetFromCaffe(proto_file, model_file)
pts_in_hull = np.load(points_file)

# Load image
gray_image = cv2.imread("black_and_white_image.jpg", cv2.IMREAD_GRAYSCALE)
gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)  # Convert to 3 channels

# Convert image to LAB color space
lab_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2LAB)

# Resize and preprocess
h, w = lab_image.shape[:2]
lab_image = cv2.resize(lab_image, (224, 224))
L = lab_image[:, :, 0] - 50  # Normalize

# Pass through the model
net.setInput(cv2.dnn.blobFromImage(L))
ab_channels = net.forward()[0].transpose((1, 2, 0))

# Resize back to original image size
ab_channels = cv2.resize(ab_channels, (w, h))

# Merge L channel with predicted AB channels
colorized_image = np.concatenate((lab_image[:, :, 0][:, :, np.newaxis], ab_channels), axis=2)

# Convert back to BGR
colorized_image = cv2.cvtColor(colorized_image, cv2.COLOR_LAB2BGR)
colorized_image = np.clip(colorized_image, 0, 255).astype(np.uint8)

# Show images
cv2.imshow("Original Black & White", gray_image)
cv2.imshow("Colorized Image", colorized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
