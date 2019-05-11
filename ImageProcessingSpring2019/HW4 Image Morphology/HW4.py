import numpy as np
import cv2
from matplotlib import pyplot as plt

def explosive(x,y,img):
     left_bound = y
     right_bound = y
     while (x<img.shape[0]):
          cur_left_bound = left_bound
          cur_right_bound = right_bound
          while (left_bound > 0) & (img[x,left_bound-1] == 1):
               left_bound = left_bound - 1
          while (left_bound < cur_right_bound) & (img[x,left_bound] == 0):
               left_bound = left_bound + 1
          while (right_bound > cur_left_bound) & (img[x,right_bound] == 0):
               right_bound = right_bound - 1
          while (right_bound + 1 < img.shape[1]-1) & (img[x,right_bound+1] == 1):
               right_bound = right_bound + 1
          if ((img[x,left_bound] == 0) & (img[x,right_bound] == 0)):
               break
          img[x, left_bound:right_bound+1] = 0
          x = x  + 1

def main():
     img_tmp = cv2.imread('coins.png', cv2.IMREAD_GRAYSCALE)
     img = 255 - img_tmp
     ret, img = cv2.threshold(img, 100, 1, cv2.THRESH_BINARY)

     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (100, 100))
     img = cv2.morphologyEx(img, cv2.MORPH_HITMISS, kernel)

     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
     img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
     img_tmp1 = img*1

     numCircle = 0
     for x in range(img.shape[0]):
          for y in range(img.shape[1]):
               if  (img[x,y] == 1):
                    numCircle = numCircle + 1
                    explosive(x,y,img)

     print('There are', numCircle,'circles in this image')
     images = [img_tmp,img_tmp1]
     for i in range(2):
          plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray')
          plt.xticks([]), plt.yticks([])
     plt.show()

if __name__ == '__main__':
    main()