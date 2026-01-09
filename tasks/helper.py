import numpy as np
import cv2
import matplotlib.pyplot as plt
# Use this file to store functions that you'll use in the rest of the tasks in this class!

def load_image(image_path: str) -> np.ndarray:
    return cv2.imread(image_path)

def save_image(image: np.ndarray, image_path: str) -> None:
    cv2.imwrite(image_path, image)
    pass

def display_image(image: np.ndarray, greyscale=False) -> None:
    if greyscale:
        plt.imshow(image, cmap='grey')
    else:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cmap='grey')
    plt.axis('off')
    plt.show()
    pass

def crop_image(image: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
    return image[y:(y+height), x:(x+width)]

def edit_brightness(image: np.ndarray, brightness_delta: int) -> np.ndarray:
    return np.clip(image.astype(np.float32) + brightness_delta, 0, 255).astype(np.uint8)

def edit_contrast(image: np.ndarray, contrast: float) -> np.ndarray:
    contrasted = (image.copy().astype(np.float32) - 128) * contrast + 128
    return np.clip(contrasted, 0, 255).astype(np.uint8)

def make_greyscale(image: np.ndarray) -> np.ndarray:
    return np.dot(image[..., :], [0.114, 0.587, 0.299])

def tint_image(image: np.ndarray, color: tuple, tint_weight=0.3) -> np.ndarray:
    image_tinted = image.copy().astype(np.float32)
    image_tinted = (1 - tint_weight) * image_tinted + tint_weight * np.array(color)    
    return np.clip(image_tinted, 0, 255).astype(np.uint8)