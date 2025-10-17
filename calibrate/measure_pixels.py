import cv2
import numpy as np
# Simple interactive tool: open image, click left and right pixel positions along known object,
# it prints pixel distance.
pts = []
def click(e,x,y,f,p):
    if e==1:
        pts.append((x,y))
        print('Clicked', x,y)
img = cv2.imread('calibrate/sample_calib.jpg')
cv2.namedWindow('img')
cv2.setMouseCallback('img', click)
while True:
    disp = img.copy()
    for pt in pts:
        cv2.circle(disp, pt, 3, (0,255,0), -1)
    cv2.imshow('img', disp)
    if cv2.waitKey(1)&0xFF==27:
        break
if len(pts)>=2:
    dx = pts[-1][0]-pts[-2][0]
    dy = pts[-1][1]-pts[-2][1]
    print('Pixel distance:', (dx*dx+dy*dy)**0.5)
cv2.destroyAllWindows()
