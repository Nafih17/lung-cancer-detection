import os
import cv2
import numpy as np
from tqdm import tqdm
from albumentations import HorizontalFlip, VerticalFlip, Rotate

def augment_data(images, masks, save_path, augment=True):
    size = (512, 512)
    os.makedirs(os.path.join(save_path, "image"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "mask"), exist_ok=True)

    for idx, (x, y) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        name = os.path.basename(x).split(".")[0]

        # Read images and masks
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = cv2.imread(y, cv2.IMREAD_COLOR)

        # Check if images and masks are read correctly
        if x is None:
            print(f"Error reading image: {x}")
            continue
        if y is None:
            print(f"Error reading mask: {y}")
            continue

        X, Y = [x], [y]

        if augment:
            augmentations = [
                HorizontalFlip(p=1.0),
                VerticalFlip(p=1.0),
                Rotate(limit=45, p=1.0),
            ]
            for aug in augmentations:
                augmented = aug(image=x, mask=y)
                X.append(augmented["image"])
                Y.append(augmented["mask"])

        for aug_idx, (i, m) in enumerate(zip(X, Y)):
            i = cv2.resize(i, size)
            m = cv2.resize(m, size)

            cv2.imwrite(os.path.join(save_path, "image", f"{name}_{aug_idx}.png"), i)
            cv2.imwrite(os.path.join(save_path, "mask", f"{name}_{aug_idx}.png"), m)

# Example usage:
from glob import glob
train_x = sorted(glob(r"C:\Users\Admin\PycharmProjects\mushfiq\new_data\train\image\*.jpg"))
train_y = sorted(glob(r"C:\Users\Admin\PycharmProjects\mushfiq\new_data\train\mask\*.jpg"))
augment_data(train_x, train_y, "output/train", augment=True)
