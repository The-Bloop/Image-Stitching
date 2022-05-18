#Only add your code inside the function (including newly improted packages)
# You can design a new function and call the new function in the given functions. 
# Not following the project guidelines will result in a 10% reduction in grades

import cv2
import numpy as np
import matplotlib.pyplot as plt


def stitch_background(img1, img2, savepath=''):
    "The output image should be saved in the savepath."
    "Do NOT modify the code provided."

    "Sift"
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, f1 = sift.detectAndCompute(img1, None)
    kp2, f2 = sift.detectAndCompute(img2, None)
    match_list = []
    for i in range(len(kp1)):
        h1 = f1[i, 0:128]
        match = compute_ssd(h1, f2)
        if (match >= 0):
            match_list.append([i, match])
    kp1_new = []
    kp2_new = []
    src_pts = []
    dst_pts = []
    for k in range(len(match_list)):
        kp1_new.append(kp1[match_list[k][0]])
        kp2_new.append(kp2[match_list[k][1]])
        src_pts.append(np.float32(kp1[match_list[k][0]].pt))
        dst_pts.append(np.float32(kp2[match_list[k][1]].pt))

    src_pts = np.array(src_pts)
    src_pts = src_pts.reshape(-1, 1, 2)
    dst_pts = np.array(dst_pts)
    dst_pts = dst_pts.reshape(-1, 1, 2)

    M1, mask1 = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    M1_det = 0
    if (np.any(M1) != None):
        M1_det = round(np.linalg.det(M1), 3)
    img_stc = img2
    if(abs(M1_det) >= 0.001):
        x = img1.shape
        y = img2.shape
        bb1 = np.float32([[0, 0], [0, x[0] - 1], [x[1] - 1, x[0] - 1], [x[1] - 1, 0]]).reshape(-1, 1, 2)

        dst = cv2.perspectiveTransform(bb1, M1)

        dst = dst.reshape(4, 2)

        bb_max = np.amax(dst, 0)
        bb_min = np.amin(dst, 0)

        bb_l = [bb_max[0], bb_min[1]]
        bb_w = [bb_min[0], bb_max[1]]

        l_sum = (bb_max[0] - bb_l[0]) ** 2 + (bb_max[1] - bb_l[1]) ** 2
        w_sum = (bb_max[0] - bb_w[0]) ** 2 + (bb_max[1] - bb_w[1]) ** 2

        l = np.sqrt(l_sum)
        w = np.sqrt(w_sum)
        area = l*w
        # print('Area : ', area)

        if(area>=1):

            a = [1, 1]
            if(bb_min[0] > 0):
                bb_min[0] = 0
                a[0] = 0
            if (bb_min[1] > 0):
                bb_min[1] = 0
                a[1] = 0

            bb_l = [bb_max[0], bb_min[1]]
            bb_r = [bb_min[0], bb_max[1]]

            bb_max = bb_max - bb_min

            if (int(abs(bb_max[0])) < img2.shape[1] + int(abs(bb_min[0]))):
                bb_max[0] = img2.shape[1] + int(abs(bb_min[0]))
            if (int(abs(bb_max[1])) < img2.shape[0] + int(abs(bb_min[1]))):
                bb_max[1] = img2.shape[0] + int(abs(bb_min[1]))

            T = np.float32([[1, 0, -bb_min[0]], [0, 1, -bb_min[1]], [0, 0, 1]])
            M = np.dot(T, M1)

            img_stc = cv2.warpPerspective(img1, M, (int(abs(bb_max[0])), int(abs(bb_max[1]))))

            g_stc = cv2.cvtColor(img_stc, cv2.COLOR_BGR2GRAY)
            g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            match_array = g_stc[int(abs(bb_min[1])):(img2.shape[0] + int(abs(bb_min[1]))), (int(abs(bb_min[0]))):(img2.shape[1] + int(abs(bb_min[0])))]

            log_array = np.less(match_array, g2)
            ind_array = np.where(log_array == 1)
            ind_array = np.transpose(np.array(ind_array))
            for i in range(len(ind_array)):
                img_stc[(ind_array[i][0] + int(abs(bb_min[1]))), (ind_array[i][1] + int(abs(bb_min[0])))] = img2[ind_array[i][0], ind_array[i][1]]

    cv2.imwrite(savepath, img_stc)
    print('Done')
    return

def compute_ncc(h1,h2):
    h1_mean = np.sum(h1) / 128
    h2_mean = np.sum(h2) / 128

    h1_diff = h1[:] - h1_mean
    h2_diff = h2[:] - h2_mean

    h_diff = np.dot(h1_diff, h2_diff)
    numerator = np.sum(h_diff)

    h1_d2 = np.power(h1_diff, 2)
    h2_d2 = np.power(h2_diff, 2)
    h1_d2s = np.sum(h1_d2)
    h2_d2s = np.sum(h2_d2)
    denominator = pow(h1_d2s, (1 / 2)) * pow(h2_d2s, (1 / 2))

    value = numerator / denominator
    return value

def compute_ssd(h1,h2):
    ssd_arr = np.sum(np.power((h2[:] - h1), 2), 1)
    ssd_unq = np.unique(ssd_arr)
    ssd_unq = sorted(ssd_unq)
    ssd_sort = sorted(ssd_arr)
    s1 = ssd_sort[0]
    s2 = ssd_sort[1]
    value = round(s1/s2)
    index = -1
    if(value < 0.6):
        ind = np.where(ssd_arr == s1)
        ind = np.array(ind)
        index = ind[0][0]
    return index

if __name__ == "__main__":
    img1 = cv2.imread('./images/t1_1.png')
    img2 = cv2.imread('./images/t1_2.png')
    savepath = 'task1.png'
    stitch_background(img1, img2, savepath=savepath)

