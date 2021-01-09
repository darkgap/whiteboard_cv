import cv2
import os
import numpy as np

from typing import List, Tuple, Optional


FILENAMES = [f'images/{name}' for name in os.listdir('images')]


def biggest_component(num_labels: int, labels_im: np.ndarray) -> np.ndarray:
    sizes = {}
    for i in range(num_labels):
        sizes[i] = np.count_nonzero(labels_im == i)

    biggest_i = max(range(num_labels), key=lambda i: sizes[i])

    mask = np.zeros(labels_im.shape, dtype=np.uint8)
    mask[labels_im == biggest_i] = 255

    return mask

def angle_diff(a, b) -> float:
    diff = a - b
    
    while diff < -np.pi:
        diff += 2 * np.pi
    while diff > np.pi:
        diff -= 2 * np.pi
    
    return diff

# rho, theta -> m, q
# FIXME working with m, q messes up with vertical lines!
def rt_to_mq(rho, theta) -> Tuple[float, float]:
    x0 = rho * np.cos(theta)
    y0 = rho * np.sin(theta)

    m = - np.cos(theta) / np.sin(theta)
    q = y0 - m * x0

    return m, q

def line_intersection(line_1, line_2) -> Optional[Tuple[int, int]]:
    m1, q1 = rt_to_mq(line_1[0], line_1[1])
    m2, q2 = rt_to_mq(line_2[0], line_2[1])

    if abs(m1 - m2) > 0.5:
        x = (q2 - q1) / (m1 - m2)
        y = m1 * x + q1
        if np.isnan(x) or np.isnan(y):
            return None
        else:
            return x, y
    else:
        return None

def all_intersections(lines, theta_threshold = np.pi / 180 * 45) -> List[Tuple[int, int]]:
    points = []
    for a in lines:
        for b in lines:
            line_a = a[0]
            line_b = b[0]

            if angle_diff(line_a[1], line_b[1]) >= theta_threshold:
                point = line_intersection(line_a, line_b)
                if point is not None:
                    # print(point)
                    points.append(point)

    return points

def find_corners(img: np.ndarray, debug = False) -> List[Tuple[float, float]]:
    RESCALE_FACTOR_X = img.shape[1] / 576
    RESCALE_FACTOR_Y = img.shape[0] / 432

    img = cv2.resize(img, dsize=(0, 0), fx=1 / RESCALE_FACTOR_X, fy=1 / RESCALE_FACTOR_Y)

    print(img.shape)

    assert(img.shape[:2] == (432, 576))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mean_value = hsv[:, :, 2].mean()

    # print(f"mean_value = {mean_value}")

    lower = np.array([0, 0, int(mean_value)])
    upper = np.array([255, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    mask = cv2.erode(mask,  np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
    mask = cv2.erode(mask,  np.ones((5, 5), np.uint8), iterations=2)

    num_labels, labels_im = cv2.connectedComponents(mask)

    mask = biggest_component(num_labels, labels_im)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours_img = np.zeros(mask.shape, dtype=np.uint8)

    cv2.drawContours(contours_img, contours, -1, 255, 1)

    lines = cv2.HoughLines(contours_img, 1, np.pi / 180, 50)

    intersections = np.array(all_intersections(lines))

    if debug:
        debug_img = cv2.cvtColor(contours_img, cv2.COLOR_GRAY2BGR)
        for item in lines:
            rho, theta = item[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

            cv2.line(debug_img, (x1,y1), (x2,y2), (0,0,255), 2)
        
        for item in intersections:
            x, y = item
            cv2.circle(debug_img, (x, y), 4, (0, 255, 0))

        cv2.imshow(f'debug', debug_img)
    
    return intersections * np.array([RESCALE_FACTOR_X, RESCALE_FACTOR_Y])
