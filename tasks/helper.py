import numpy as np
import cv2
import matplotlib.pyplot as plt
# Use this file to store functions that you'll use in the rest of the tasks in this class!

def load_image(image_path: str) -> np.ndarray:
    """
    Use OpenCV to load an image from a file.
    """
    pass

def save_image(image: np.ndarray, image_path: str) -> None:
    """
    Use OpenCV to save an image to a file.
    """
    pass

def display_image(image: np.ndarray) -> None:
    """
    Use matplotlib to display an image.
    """
    pass

def crop_image(image: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
    """
    Crop an image to a given rectangle.
    """
    pass

def edit_brightness(image: np.ndarray, brightness: int) -> np.ndarray:
    """
    Edit the brightness of an image.
    """
    pass

def edit_contrast(image: np.ndarray, contrast: int) -> np.ndarray:
    """
    Edit the contrast of an image.
    """
    pass

def make_greyscale(image: np.ndarray) -> np.ndarray:
    """
    Make an image greyscale.
    """
    pass

def tint_image(image: np.ndarray, color: tuple) -> np.ndarray:
    """
    Tint an image with a given color. Hint: look at the make_redder function!
    """
    pass
