import numpy as np
import cv2
from matplotlib import pyplot as plt


def get_cbf(img):
    b_in, g_in, r_in = cv2.split(img)

    '''Get Histograms of RGB'''
    histr_b = cv2.calcHist([img], [0], None, [256], [0, 256])
    histr_g = cv2.calcHist([img], [1], None, [256], [0, 256])
    histr_r = cv2.calcHist([img], [2], None, [256], [0, 256])

    '''Calculate Cumulative Histograms '''
    cdf_b = histr_b.cumsum()
    cdf_g = histr_r.cumsum()
    cdf_r = histr_g.cumsum()

    cdf_m_b = np.ma.masked_equal(cdf_b, 0)
    cdf_m_b = (cdf_m_b - cdf_m_b.min()) * 255 / (cdf_m_b.max() - cdf_m_b.min())
    cdf_b = np.ma.filled(cdf_m_b, 0).astype('uint8')

    cdf_m_g = np.ma.masked_equal(cdf_g, 0)
    cdf_m_g = (cdf_m_g - cdf_m_g.min()) * 255 / (cdf_m_g.max() - cdf_m_g.min())
    cdf_g = np.ma.filled(cdf_m_g, 0).astype('uint8')

    cdf_m_r = np.ma.masked_equal(cdf_r, 0)
    cdf_m_r = (cdf_m_r - cdf_m_r.min()) * 255 / (cdf_m_r.max() - cdf_m_r.min())
    cdf_r = np.ma.filled(cdf_m_r, 0).astype('uint8')

    return cdf_b,cdf_g,cdf_r

def show_histr(cdf_b,cdf_g,cdf_r):
    plt.plot(cdf_b, color='b')
    plt.plot(cdf_g, color='g')
    plt.plot(cdf_r, color='r')
    plt.xlim([0, 256])
    plt.show()

def main():
    img = cv2.imread('B.jpg', cv2.IMREAD_COLOR)
    b_in, g_in, r_in = cv2.split(img)
    cdf_b, cdf_g, cdf_r = get_cbf(img)
    '''Histogram Equalization'''
    b_equal = cdf_b[b_in]
    g_equal = cdf_g[g_in]
    r_equal = cdf_r[r_in]

    img2 = cv2.imread('1.jpg', cv2.IMREAD_COLOR)
    b_in, g_in, r_in = cv2.split(img)
    cdf_b, cdf_g, cdf_r = get_cbf(img2)

    arr = np.arange(256)
    reversed_cdf_b = arr[cdf_b]
    reversed_cdf_g = arr[cdf_g]
    reversed_cdf_r = arr[cdf_r]

    b_out = reversed_cdf_b[b_equal]
    g_out = reversed_cdf_g[g_equal]
    r_out = reversed_cdf_r[r_equal]

    print(b_out.shape)
    print(g_out.shape)
    print(r_out.shape)
    out_image = cv2.merge((b_out,g_out,r_out))
    cv2.imwrite('Out.jpg', out_image)

    show_histr(reversed_cdf_b, reversed_cdf_g, reversed_cdf_r)

if __name__ == '__main__':
    main()
