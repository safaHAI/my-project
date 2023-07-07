import cv2
import numpy as np

img = cv2.imread('vig.png')
org = cv2.imread('orgsw.png')
height, width = img.shape[:2]
print("height, width =", height, width)

ker_x = cv2.getGaussianKernel(width,  180)
ker_y = cv2.getGaussianKernel(height, 1000)
ker = ker_y.T * ker_x
msk = 255 * ker  / np.linalg.norm(ker)
msk = msk * 2.2
img1 = img.copy()
for i in range(3):
    img1[:, :, i] = img1[:, :, i] / msk

hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
hsv = np.array(hsv, dtype = np.float64)
hsv[:,:,1] = hsv[:,:,1]* 1.8 ## scale pixel up or down (Lightness)
hsv[:,:,1][hsv[:,:,1]>255]  = 255
hsv[:,:,2] = hsv[:,:,2]* 1.8 ## scale pixel up or down (Lightness)
hsv[:,:,2][hsv[:,:,2]>255]  = 255
hsv = np.array(hsv, dtype = np.uint8)
img2 = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


cv2.imshow('bright', org)
cv2.imshow('dark', img)
cv2.imshow('output', img2)
cv2.imwrite('pic1after_removal.jpg', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()