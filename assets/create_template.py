import cv2, numpy as np, os
os.makedirs('assets', exist_ok=True)
path = 'assets/template_fleet.png'
w,h = 80,60
img = np.zeros((h,w,3), dtype=np.uint8)
pts_left = np.array([[8,55],[28,10],[36,10],[16,55]], dtype=np.int32)
pts_right = np.array([[w-8,55],[w-28,10],[w-36,10],[w-16,55]], dtype=np.int32)
cv2.fillPoly(img, [pts_left], (255,255,255))
cv2.fillPoly(img, [pts_right], (255,255,255))
cv2.imwrite(path, img)
print('Wrote', path)
