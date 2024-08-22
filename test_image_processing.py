import cv2
import numpy as np
from faceREC import TOGRAY, face_detect  # Replace with the actual module name

def test_TOGRAY():
    # Create a dummy color image (3x3 pixels)
    color_image = np.array([
        [[255, 0, 0], [0, 255, 0], [0, 0, 255]],  # Red, Green, Blue
        [[255, 255, 0], [0, 255, 255], [255, 0, 255]],  # Yellow, Cyan, Magenta
        [[0, 0, 0], [127, 127, 127], [255, 255, 255]]  # Black, Gray, White
    ], dtype=np.uint8)

    # Convert to grayscale
    gray_image = TOGRAY(color_image)

    # Check if the output is a 2D array (grayscale)
    assert len(gray_image.shape) == 2
    assert gray_image.shape == (3, 3)

def test_face_detect():
    # Create a dummy image with no faces
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)

    # Run face detection
    output_image = face_detect(dummy_image)

    # Check if the output image has the same shape as the input
    assert output_image.shape == dummy_image.shape

    # Since there are no faces, the output should be the same as the input
    np.testing.assert_array_equal(dummy_image, output_image)
