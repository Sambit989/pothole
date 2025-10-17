import cv2
import numpy as np
blank = 255 * np.ones((480, 640, 3), np.uint8)
cv2.imwrite('static/blank.jpg', blank)
print('Created static/blank.jpg')
