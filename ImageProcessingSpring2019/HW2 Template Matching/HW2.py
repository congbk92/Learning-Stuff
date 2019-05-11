import numpy as np
import cv2
from matplotlib import pyplot as plt


def main():
    img_src = cv2.imread('face.png', cv2.IMREAD_GRAYSCALE)
    img_template = cv2.imread('eye.png', cv2.IMREAD_GRAYSCALE)
    [k, l] = img_template.shape

    ''' Normalized Template'''
    img_template = img_template - np.mean(img_template)
    sum_square_img_template = np.sum(img_template**2)

    img_des = np.empty((img_src.shape[0] - k + 1, img_src.shape[1] - l + 1))

    '''Normalized cross-correlation'''
    for m in range(img_des.shape[0]):
        for n in range(img_des.shape[1]):
            sub_arr = img_src[m:m+k, n:n+l]
            sub_arr = sub_arr - np.mean(sub_arr)
            img_des[m, n] = np.sum(img_template*sub_arr)/np.sqrt(sum_square_img_template*np.sum(sub_arr**2))

    '''Convert to gray scale img'''
    img_des = ((img_des - img_des.min()) / (img_des.max() - img_des.min()) * 255).astype('uint8')

    '''Get threshold img'''
    ret, img_threshold = cv2.threshold(img_des, 170, 255, cv2.THRESH_BINARY)

    titles = ['Input', 'Normalized X-Correlation', 'Thresholded Image']
    images = [img_src, img_des, img_threshold]
    for i in range(3):
        plt.subplot(1, 3, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()

if __name__ == '__main__':
    main()
