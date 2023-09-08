import cv2
import os


def crop_and_resize(image, target_size=(224, 224), negative_space_color=(0, 0, 0)):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return cv2.resize(image, target_size)

    x, y, w, h = cv2.boundingRect(contours[0])
    for contour in contours:
        x_, y_, w_, h_ = cv2.boundingRect(contour)
        x = min(x, x_)
        y = min(y, y_)
        w = max(x + w, x_ + w_) - x
        h = max(y + h, y_ + h_) - y

    cropped = image[y : y + h, x : x + w]

    resized = cv2.resize(cropped, target_size)

    return resized


base_dir = "raw_dataset"
save_base_dir = "cropped_dataset"

if not os.path.exists(save_base_dir):
    os.makedirs(save_base_dir)

for class_label in ["0", "1"]:
    class_dir = os.path.join(base_dir, class_label)
    save_class_dir = os.path.join(save_base_dir, class_label)

    if not os.path.exists(save_class_dir):
        os.makedirs(save_class_dir)

    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        save_img_path = os.path.join(save_class_dir, img_name)

        img = cv2.imread(img_path)

        processed_img = crop_and_resize(img)

        cv2.imwrite(save_img_path, processed_img)
