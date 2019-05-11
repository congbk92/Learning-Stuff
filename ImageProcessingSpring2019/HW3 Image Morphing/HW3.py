import numpy as np
import cv2
from matplotlib import pyplot as plt


def drawTrianglesInImg(img,list_Triangles,coor_pnt):
    for i in range(0,len(list_Triangles)):
        pntA = tuple(coor_pnt[list_Triangles[i][0]][:])
        pntB = tuple(coor_pnt[list_Triangles[i][1]][:])
        img = cv2.line(img, pntA, pntB ,(0,0,255),2)

        pntA = tuple(coor_pnt[list_Triangles[i][1]][:])
        pntB = tuple(coor_pnt[list_Triangles[i][2]][:])
        img = cv2.line(img, pntA, pntB ,(0,0,255),2)

        pntA = tuple(coor_pnt[list_Triangles[i][0]][:])
        pntB = tuple(coor_pnt[list_Triangles[i][2]][:])
        img = cv2.line(img, pntA, pntB ,(0,0,255),2)

def drawIndexPointInImg(img,coor_pnt):
    for i in range(1,len(coor_pnt)):
        pntA = tuple(coor_pnt[i][:])
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(img,str(i),pntA, font, 1,(0,255,0),1,cv2.LINE_AA)
    return img

def drawCirclePointInImg(img,coor_pnt):
    for i in range(1,len(coor_pnt)):
        pntA = tuple(coor_pnt[i][:])
        img = cv2.circle(img, pntA, 2 ,(255,0,0),2)
    return img

def warpTriangleInImgToOther(img_s, img_d, triangle_s, triangle_d):
    bounding_s = cv2.boundingRect(triangle_s)
    bounding_d = cv2.boundingRect(triangle_d)
    # Offset points by left top corner of the
    # respective rectangles
    tri_s_cropped = []
    tri_d_cropped = []
    for i in range(0,3):
        tri_s_cropped.append(((triangle_s[i][0] - bounding_s[0]),(triangle_s[i][1] - bounding_s[1])))
        tri_d_cropped.append(((triangle_d[i][0] - bounding_d[0]),(triangle_d[i][1] - bounding_d[1])))
    # Apply warpImage to small rectangular patches
    img_s_cropped = img_s[bounding_s[1]:bounding_s[1] + bounding_s[3], bounding_s[0]:bounding_s[0] + bounding_s[2]]
    #img_d_cropped = img_d[bounding_d[1]:bounding_d[1] + bounding_d[3], bounding_d[0]:bounding_d[0] + bounding_d[2]]
    warpMat = cv2.getAffineTransform( np.float32(tri_s_cropped), np.float32(tri_d_cropped))
    img_d_cropped = cv2.warpAffine( img_s_cropped, warpMat, (bounding_d[2], bounding_d[3]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )
    return img_d_cropped

def warp2TriangleIn2ImgToOther(img_s, img_d, img_m, triangle_s, triangle_d, triangle_m, alpha):
    img_s_to_m_cropped = warpTriangleInImgToOther(img_s, img_m, triangle_s, triangle_m)
    img_d_to_m_cropped = warpTriangleInImgToOther(img_d, img_m, triangle_d, triangle_m)
    img_m_cropped = img_s_to_m_cropped*(1-alpha) + img_d_to_m_cropped*alpha

    bounding_m = cv2.boundingRect(triangle_m)
    tri_m_cropped = []
    for i in range(0,3):
        tri_m_cropped.append(((triangle_m[i][0] - bounding_m[0]),(triangle_m[i][1] - bounding_m[1])))

    mask = np.zeros((bounding_m[3], bounding_m[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(tri_m_cropped), (1.0, 1.0, 1.0), 16, 0)

    img_m[bounding_m[1]:bounding_m[1] + bounding_m[3], bounding_m[0]:bounding_m[0] + bounding_m[2]] = img_m[bounding_m[1]:bounding_m[1] + bounding_m[3], bounding_m[0]:bounding_m[0] + bounding_m[2]] * ((1.0, 1.0, 1.0) - mask) + img_m_cropped*mask
    return img_m

def main():
    img_src = cv2.imread('source.png')
    img_des = cv2.imread('target.png')

    coor_corresp_pnt_src = [[0, 0], [211, 455], [150, 433], [99, 392], [71,314], [65, 234], [84, 146], [155, 115], [243, 98], [298, 100], [331, 245], [332,316], [305, 392], [266, 432], [46, 317], [33, 288], [33,227], [25, 155], [47, 75], [105, 26], [253, 9], [308, 48], [344, 100], [354, 152], [347,254], [343, 305], [111, 211], [280, 198], [92, 238], [103, 203], [185, 188], [229, 203], [267, 184], [319, 230], [208, 201], [168, 316], [217, 323], [252, 308], [145,344], [209,341], [266,344], [182,363], [210,374], [242,364], [319, 140], [134, 189], [153, 188], [191,204], [169, 202], [239,186], [291, 187], [310, 208], [250, 198], [145,218], [104,239], [145,243], [178,242], [274,213], [237,237], [270,241], [305,234], [125,297], [294,292], [210,353], [211,360], [170,0], [23,237], [356,239], [20,258], [60,324], [358,250], [341,316], [0,0], [379,0], [379,501], [0,501], [0,345], [379,345]]
    coor_corresp_pnt_des = [[0, 0], [185, 452], [126, 421], [89,371], [71, 326], [63, 223], [96, 100], [140, 99], [203,90], [271,92], [306,225], [299,328], [266, 400], [235, 429], [49,320], [46,294], [45,212], [52, 150], [80, 92], [118, 57], [235, 48], [270, 80], [304, 104], [324, 151], [323,223], [332, 282], [105, 168], [270, 171], [83, 202], [98, 144], [168, 171], [209,197], [255, 149], [292,214], [186,193], [150, 275], [187, 284], [226, 284], [130,328], [184,329], [240,327], [158,364], [185,368], [213,364], [290,136], [105,140], [132,144], [165,193], [127,171], [224,163], [282,150], [289,164], [244,175], [117,190], [93,215], [121,225], [155,211], [249,194], [214,216], [251,227], [277,216], [98,272], [270,287], [185,341], [187,353], [179,32], [27,209], [342,216], [17,224], [56,329], [352,234], [312,334], [0, 0], [376,0], [376,501], [0,501], [0,345], [376,345]]
    index_triangle       = [[19,65,7], [65,7,8], [65,8,20], [18,19,6], [17,6,5], [19,6,7], [17,18,6], [17,5,16], [15,16,4], [16,5,4], [15,14,4], [4,3,38], [4,35,38], [7,30,8], [20,8,21], [8,9,21], [8,49,30], [34,35,36], [34,36,37], [36,37,39], [35,36,39], [35,38,39], [38,41,2], [38,2,3], [41,2,1], [41,42,1], [42,43,1], [43,1,13], [43,13,40], [39,40,37], [40,13,12], [40,11,12], [37,40,11], [10,11,24], [11,24,25], [10,23,24], [9,21,22], [9,44,22], [44,23,22], [5,28,29], [6,5,29], [6,29,45], [29,26,28], [8,9,32], [9,32,44], [6,7,45], [7,45,46], [7,46,30], [30,47,48], [48,30,46], [48,45,46], [48,45,26], [29,26,45], [49,31,52], [49,32,52], [32,44,50], [51,33,10], [33,51,27], [50,51,27], [32,50,27], [32,52,27], [30,34,47], [30,49,34], [49,34,31], [28,53,54], [28,26,53], [53,26,48], [53,48,47], [53,47,56], [47,34,56], [56,34,35], [53,56,55], [53,55,54], [34,58,37], [34,31,58], [58,31,57], [31,57,52], [57,52,27], [57,27,33], [57,33,60], [57,60,59], [57,59,58], [38,39,63], [38,63,64], [38,64,41], [64,41,42], [64,42,43], [64,43,40], [64,63,40], [63,40,39], [5,28,54], [5,61,4], [4,61,35], [61,35,56], [61,56,55], [61,54,55], [61,54,5], [60,33,10], [33,10,62], [10,62,11], [62,11,37], [62,58,59], [62,59,60], [62,60,33], [66,16,15], [24,67,25], [58,62,37], [67,70,25], [11,25,71], [66,68,15], [14,4,69], [44,51,50], [72,18,19], [72,17,18], [72,73,65], [72,76,68], [73,20,21], [73,21,22], [73,22,23], [74,12,13], [74,12,71], [74,13,1], [74,75,1], [75,2,3], [75,2,1], [75,3,69], [75,69,14], [76,68,15], [77,70,25], [77,74,25], [71,74,25], [23,24,67], [73,23,67], [72,19,65], [73,20,65], [75,76,14], [3,4,69], [11,12,71], [23,44,10], [8,49,32], [76,14,15], [72,17,66], [72,66,68], [16,17,66], [73,77,70], [73,67,70], [44,10,51]]

    coor_triangle_src = np.zeros((len(index_triangle),3,2))
    coor_triangle_des = np.zeros((len(index_triangle),3,2))
    coor_triangle_mid = np.zeros((len(index_triangle),3,2))
    for i in range(0,len(index_triangle)):
        coor_triangle_src[i][:][:] = [coor_corresp_pnt_src[index_triangle[i][0]][:],coor_corresp_pnt_src[index_triangle[i][1]][:],coor_corresp_pnt_src[index_triangle[i][2]][:]]
        coor_triangle_des[i][:][:] = [coor_corresp_pnt_des[index_triangle[i][0]][:],coor_corresp_pnt_des[index_triangle[i][1]][:],coor_corresp_pnt_des[index_triangle[i][2]][:]]

    numberIter = 11

    for i in range(0,numberIter):
        img_mid = np.zeros(img_src.shape)
        alpha = i/(numberIter-1)
        coor_triangle_mid = coor_triangle_src*(1-alpha) + coor_triangle_des*alpha
        for j in range(0,len(index_triangle)):
            warp2TriangleIn2ImgToOther(img_src, img_des, img_mid, np.float32(coor_triangle_src[j][:][:]), np.float32(coor_triangle_des[j][:][:]), np.float32(coor_triangle_mid[j][:][:]), alpha)
        cv2.imwrite( str(i) + '.jpg',img_mid)


    drawTrianglesInImg(img_src, index_triangle, coor_corresp_pnt_src)
    drawTrianglesInImg(img_des, index_triangle, coor_corresp_pnt_des)

    #drawIndexPointInImg(img_src, coor_corresp_pnt_src)
    #drawIndexPointInImg(img_des, coor_corresp_pnt_des)

    drawCirclePointInImg(img_src, coor_corresp_pnt_src)
    drawCirclePointInImg(img_des, coor_corresp_pnt_des)

    images = [img_src, img_des]
    for i in range(2):
        plt.subplot(1, 2, i + 1), plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        plt.xticks([]), plt.yticks([])
    plt.show()

if __name__ == '__main__':
    main()
