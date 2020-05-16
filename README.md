# telegram-masking-bot
A simple bot that uses Homography transformation to change an arbitrary image.

# Configuration
`config.py` must contain the configuration in the following way:
```python
API_KEY = 'Place your API key here'
greeting = 'Hello there! Send me a photo or a sticker'
caption = 'Let me think'
```
# Usage
**Don't forget install python requirements**

Place the template image `template.jpg` and its mask `mask.jpg` in the folder `img/`.
Then describe the points on the main image that the received image will be transformed to in the `img/frame.py` in the following way:
```python
# Order: TOP-LEFT, TOP-RIGHT, BOTTOM-LEFT, BOTTOM-RIGHT
points = [[216, 70], [1024, 0], [224, 678], [1006, 700]]
```
