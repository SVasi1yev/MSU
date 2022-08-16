import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt
import ftdetect.features
import ftdetect.masks
from sklearn.neighbors import KDTree


def kirsch_filter(img):
    kernels = [0] * 8
    kernels[0] = np.array(
        [[5, 5, 5],
         [-3, 0, -3],
         [-3, -3, -3]]
    )
    kernels[1] = np.array(
        [[-3, 5, 5],
         [-3, 0, 5],
         [-3, -3, -3]]
    )
    kernels[2] = np.array(
        [[-3, -3, 5],
         [-3, 0, 5],
         [-3, -3, 5]]
    )
    kernels[3] = np.array(
        [[-3, -3, -3],
         [-3, 0, 5],
         [-3, 5, 5]]
    )
    kernels[4] = np.array(
        [[-3, -3, -3],
         [-3, 0, -3],
         [5, 5, 5]]
    )
    kernels[5] = np.array(
        [[-3, -3, -3],
         [5, 0, -3],
         [5, 5, -3]]
    )
    kernels[6] = np.array(
        [[5, -3, -3],
         [5, 0, -3],
         [5, -3, -3]]
    )
    kernels[7] = np.array(
        [[5, 5, -3],
         [5, 0, -3],
         [-3, -3, -3]]
    )

    kirsch = cv2.filter2D(img, -1, kernels[0]) / 8
    for i in range(1, 8):
        kirsch += cv2.filter2D(img, -1, kernels[i]) / 8

    return kirsch


def change_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v,value)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def count_figs(img):
    cart_mask = img[:, :, ::-1]

    cart_mask = change_brightness(cart_mask, value=255)
    cart_mask = cart_mask[:, :, 2] - cart_mask[:, :, 0] / 2 - cart_mask[:, :, 1] / 2
    t, cart_mask = cv2.threshold(cart_mask, 60, 255, 0)

    for i in range(5):  # 5
        cart_mask = cv2.medianBlur(cart_mask.astype(np.float32), 5)

    cart_mask = cv2.erode(cart_mask, np.ones((3, 3), 'uint8'), iterations=4)  # 4

    q1, q2 = cv2.findContours(
        cart_mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )
    for k, e1 in enumerate(q1):
        if q2[0][k][-1] == -1:
            continue
        cv2.fillPoly(cart_mask, pts=[e1], color=(255, 255, 255))

    hsv = cv2.cvtColor(img[:, :, ::-1], cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    filter_ = kirsch_filter
    borders_mask = filter_(img[:, :, ::-1][:, :, 0])

    a = cart_mask * borders_mask / 255
    conters = a.copy()
    b = (255 - s)
    for i in range(2):
        b = cv2.GaussianBlur(b, (3, 3), 0)
    conters = b * conters / 255
    t, conters = cv2.threshold(conters, 12, 255, 0)

    q1, q2 = cv2.findContours(
        conters.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    r = np.zeros(conters.shape)
    n = 0
    for e1 in q1:
        if len(e1) < 100 or cv2.contourArea(e1) < 800:
            continue
        n += 1
        for e2 in e1:
            e2 = e2.reshape(-1)
            r[e2[1], e2[0]] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    r1 = np.array(r)
    r1 = cv2.dilate(r1, kernel, iterations=1)
    r1 = cv2.morphologyEx(
        r1, cv2.MORPH_CLOSE, kernel, iterations=1
    )

    r2 = r1.copy()
    n2 = n
    for i in range(0):
        q = r2.copy()
        q1, q2 = cv2.findContours(
            q.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
        )
        r2 = np.zeros(conters.shape)
        n2 = 0
        for k, e1 in enumerate(q1):
            if q2[0][k][-1] == -1 or len(e1) < 100 or cv2.contourArea(e1) < 800:
                continue
            n2 += 1
            for e2 in e1:
                e2 = e2.reshape(-1)
                r2[e2[1], e2[0]] = 255

        r2 = cv2.dilate(r2, kernel, iterations=1)
        r2 = cv2.morphologyEx(
            r2, cv2.MORPH_CLOSE, kernel, iterations=1
        )

    q1, q2 = cv2.findContours(
        r2.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )
    r3 = np.zeros(conters.shape)
    for k, e1 in enumerate(q1):
        if q2[0][k][-1] == -1 or len(e1) < 100 or cv2.contourArea(e1) < 800:
            continue
        for e2 in e1:
            e2 = e2.reshape(-1)
            r3[e2[1], e2[0]] = 255

    r4 = r3.copy()
    for k, e1 in enumerate(q1):
        if q2[0][k][-1] == -1:
            continue
        cv2.fillPoly(r4, pts=[e1], color=(255, 255, 255))

    c = ftdetect.features.susanCorner(r4, t=25, radius=10)

    c1 = c.copy()
    corners = []
    for y in range(c1.shape[0]):
        for x in range(c1.shape[1]):
            if c1[y, x] > 0:
                corners.append([x, y])

    t = []
    for corner in corners:
        fl = True
        for e in t:
            if ((corner[0] - e[0]) ** 2 + (corner[1] - e[1]) ** 2) ** 0.5 < 10:
                fl = False
                break
        if fl:
            t.append(corner.copy())
    corners = t

    for corner in corners:
        cv2.circle(c1, (corner[0], corner[1]), 3, 255, -1)

    r5 = c1 + r3

    figs = {}
    points = []
    points_conter = []

    q1, q2 = cv2.findContours(
        r3.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )

    count = 0
    for k, e1 in enumerate(q1):
        if q2[0][k][-1] == -1:
            continue
        x, y, w, h = cv2.boundingRect(e1)
        figs[count] = {
            'corners': 0,
            'convex': False,  # cv2.isContourConvex(e1),
            'coords': (int(x), int(y + h / 2))
        }
        for l, e2 in enumerate(e1):
            e2 = e2.reshape(-1)
            points.append([e2[0], e2[1]])
            points_conter.append(count)
        count += 1

    count = 0
    for k, e1 in enumerate(q1):
        if q2[0][k][-1] == -1:
            continue
        figs[count]['convex'] = cv2.isContourConvex(e1)
        count += 1

    points = np.array(points)
    kd_tree = KDTree(points, leaf_size=2)
    idx = kd_tree.query(corners, k=1, return_distance=False)
    for i in idx:
        i = i[0]
        figs[points_conter[i]]['corners'] += 1

    res = img[:, :, ::-1].astype(np.uint8)
    for k in figs:
        text = f'{k + 1}: P{figs[k]["corners"]}'
        if figs[k]['convex']:
            text += 'C'
        cv2.putText(
            res, text, figs[k]['coords'], cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 255, 0), 2
        )

    return cart_mask, borders_mask, a, b, conters, r, r1, r2, r3, r4, r5, res, n, n2, q1, q2


if __name__ == '__main__':
    img_name = sys.argv[1]

    img = cv2.imread(img_name)
    cart_mask, borders_mask, a, b, conters, r, r1, r2, r3, r4, r5, res, n, n2, q1, q2 = count_figs(img)

    font = {'size': 16}
    plt.rc('font', **font)
    fig = plt.figure(figsize=(20, 20))
    plt.imshow(res, cmap='gist_gray')
    fig.savefig('result.jpg')

    font = {'size': 16}
    plt.rc('font', **font)
    fig = plt.figure(figsize=(20, 20))
    plt.imshow(r5, cmap='gist_gray')
    fig.savefig('conters&corners.jpg')