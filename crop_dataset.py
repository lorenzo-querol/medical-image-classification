import cv2
import os


def crop_and_resize(image, target_size=(224, 224), negative_space_color=(0, 0, 0)):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold to get a binary mask of the content
    _, thresh = cv2.threshold(
        gray, 10, 255, cv2.THRESH_BINARY
    )  # Assuming black background

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If no contours are detected, just return the resized image
    if not contours:
        return cv2.resize(image, target_size)

    # Find the bounding box of the content
    x, y, w, h = cv2.boundingRect(contours[0])
    for contour in contours:
        x_, y_, w_, h_ = cv2.boundingRect(contour)
        x = min(x, x_)
        y = min(y, y_)
        w = max(x + w, x_ + w_) - x
        h = max(y + h, y_ + h_) - y

    # Crop the image to this bounding box
    cropped = image[y : y + h, x : x + w]

    # Resize the cropped image to the target size
    resized = cv2.resize(cropped, target_size)

    return resized


# Base directory
base_dir = "Dataset/extracted"
save_base_dir = "Dataset/cropped"

# Ensure the save directory exists
if not os.path.exists(save_base_dir):
    os.makedirs(save_base_dir)

# Iterate through each sub-directory: train, val, test
for subset in ["train", "val", "test"]:
    subset_dir = os.path.join(base_dir, subset)
    save_subset_dir = os.path.join(save_base_dir, subset)

    # Ensure the subset save directory exists
    if not os.path.exists(save_subset_dir):
        os.makedirs(save_subset_dir)

    # Iterate through class folders: 0 and 1
    for class_label in ["0", "1"]:
        class_dir = os.path.join(subset_dir, class_label)
        save_class_dir = os.path.join(save_subset_dir, class_label)

        # Ensure the class save directory exists
        if not os.path.exists(save_class_dir):
            os.makedirs(save_class_dir)

        # List all image files in the directory
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            save_img_path = os.path.join(save_class_dir, img_name)

            # Load the image
            img = cv2.imread(img_path)

            # Process the image
            processed_img = crop_and_resize(img)

            # Save the processed image
            cv2.imwrite(save_img_path, processed_img)
