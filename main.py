from matplotlib import image
from matplotlib import pyplot as plt
import numpy as np
from img_process import get_whiteboard_from_points
from camboard import find_corners
import cv2
from sklearn.cluster import KMeans
import os


def main():
    FILENAMES = [f'images/{name}' for name in os.listdir('images')]
    [process_image(fn) for fn in FILENAMES]


def filter_points(points, width, height, margin=0.2):
    return np.array([
        (x, y) for x, y in points
        if x > -width*margin and x < width*(1+margin) and y > -height*margin and y < height*(1+margin)
    ])


def process_image(img_name):    
    img_cv2 = cv2.imread(img_name)
    img = image.imread(img_name)

    pts = find_corners(img_cv2)

    height, width = img.shape[:2]

    pts = filter_points(pts, width, height)
    
    [cv2.circle(img_cv2, (int(x), int(y)), 100, (255, 0, 0), 10) for x, y in pts]

    kmeans = KMeans(n_clusters = 4, random_state = 0).fit(pts)
    
    means = np.float32(kmeans.cluster_centers_)
    print(means)
    [cv2.circle(img_cv2, (int(x), int(y)), 100, (0, 255, 0), 10) for x, y in means]

    target_dims = (1920, 1080)
    
    dst = get_whiteboard_from_points(img, means, target_dims)
    
    cv2.imshow('input', cv2.resize(img_cv2, (0,0), fx=1/8, fy=1/8))
    cv2.imshow('output', cv2.resize(dst, (0,0), fx=1/8, fy=1/8))
    cv2.waitKey(0)
    

if __name__ == "__main__":
    # execute only if run as a script
    main()
    cv2.destroyAllWindows()
