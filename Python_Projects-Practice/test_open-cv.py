import cv2
import matplotlib.pyplot as plt

image_path = r'C:\Users\himub\OneDrive\Desktop\python\All_Image\test_cv.jpeg'
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found.")
else:
    # Convert from BGR to RGB for correct display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.imshow(image_rgb)
    plt.axis("off")  # Hide axis
    plt.show()
