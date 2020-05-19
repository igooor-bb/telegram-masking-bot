"""
Module contains the function that generates an image using in the bot.

Author: Igor Belov (igooor.bb@gmail.com)
"""

import img.frame as frame
import numpy as np
import cv2
from io import BytesIO
from PIL import ImageFont, ImageDraw, Image

# Images directory
IMG_DIR = 'img/'
FONT_DIR = 'fonts/'
# Main image (on which the transformed image and text will be placed).
template = cv2.imread(IMG_DIR + 'template.jpg', cv2.IMREAD_COLOR)
# Mask for the main image.
mask = cv2.imread(IMG_DIR + 'mask.jpg', cv2.IMREAD_GRAYSCALE)

def generate_image(input_bytes : BytesIO, caption : str) -> BytesIO:
    """ 
    Transforms given picture with respect to four points points on the main image declared in the frame module
    and then combine main image with the transformed one.
    Args:
        input_bytes (BytesIO): Image as bytes in the BytesIO object.
    Returns:
        BytesIO: Processed image as bytes in the BytesIO object.
    """
    # Convert given bytes to cv2 object.
    uploaded = cv2.imdecode(np.frombuffer(input_bytes, np.uint8), cv2.IMREAD_COLOR)
    
    # Shape of given image.
    h1, w1, _ = uploaded.shape
    # Shape of main image.
    h2, w2, _ = template.shape

    # The points on the main image that the received image will be transformed to.
    # Declared in the frame module.
    expected_points = np.array(frame.points, dtype=np.float64)
    # The frame of the received image.
    original_points = np.array([[0, 0], [w1, 0], [0, h1], [w1, h1]], dtype=np.float64)
    
    # Calculate coefficients and apply Homography Transformation.
    hg, _ = cv2.findHomography(original_points, expected_points, cv2.RANSAC, 5.0)
    transformed = cv2.warpPerspective(uploaded, hg, (w2, h2))
    
    # Apply mask
    mask_inv = cv2.bitwise_not(mask)
    transformed = cv2.bitwise_and(transformed, transformed, mask=mask_inv)
    result = cv2.bitwise_and(template, template, mask=mask)
    result = cv2.bitwise_or(result, transformed)

    if caption is not None:
         result = insert_text(result, caption)
    
    _, byte_image = cv2.imencode(".jpg", result)
    iobuff = BytesIO(byte_image)

    return iobuff

def insert_text(image : np.ndarray, text : str) -> np.ndarray:
    """
    Put text on an image with automatic scaling of a font size
    Converts opencv object to pillow one to work with truetype font.
    Args:
        image (np.ndarray): Image object of opencv
        text (str): Text to insert
    Returns:
        np.ndarray: Image object of opencv
    """

    def normalize_text_length(text : str, max_length = 50) -> str:
        """
        Normalizes the length of the text. 
        If there is only one word, it just cuts it off by characters, 
        otherwise just finds the nearest end of a word.

        Args:
            text(str): String to normalaize.
            max_length(int): The maximum length of the trimmed string.
        Return: 
            str: Processed string.
        """
        if len(text) >= max_length:
            index = text.rfind(' ', 0, max_length)
            if (index != -1):
                text = text[:index]
            else:
                text = text[:max_length]
            text += '...'
        return text

    def normalize_font_size(text : str, image_width : int, ratio : float) -> ImageFont:
        """
        Adjust the font size with respect to length of the text and image width.

        Args:
            text(str): Given text.
            image_width(int): Width of the image where the text will be placed.
            ratio(float): A percentage of the length of the image allocated to the text.
        Returns:
            ImageFont: Adjusted font.
        """
        font_size = 80 if len(text) < 40 else 50

        font = ImageFont.truetype(FONT_DIR + "Lobster-Regular.ttf", font_size)
        while (font.getsize(text)[0] > image_width * ratio):
            font_size -= 2
            font = ImageFont.truetype(FONT_DIR + "Lobster-Regular.ttf", font_size)
        
        return font

    height, width, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    canvas = ImageDraw.Draw(pil_image)

    text_norm = normalize_text_length(text)
    font = normalize_font_size(text_norm, width, 0.7)
    wt, ht = canvas.textsize(text_norm, font=font)
    
    canvas.text(((width-wt)/2, height-ht-20), text, font=font)

    # Convert pillow object back to opencv.
    result = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    return result

