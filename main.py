from sklearn.cluster import KMeans
import numpy as np
import cv2
import os

from img_process import get_whiteboard_from_points
from camboard import find_corners

def main():
    FILENAMES = [f'images/{name}' for name in os.listdir('images')]
    [process_image(fn) for fn in FILENAMES]

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def filter_points(points, width, height, margin=0.2):
    return np.array([
        (x, y) for x, y in points
        if x > -width*margin and x < width*(1+margin) and y > -height*margin and y < height*(1+margin)
    ])


def process_image(img_name):
    img = cv2.imread(img_name)

    pts = find_corners(img)

    height, width = img.shape[:2]

    pts = filter_points(pts, width, height)

    kmeans = KMeans(n_clusters = 4, random_state = 0).fit(pts)
    
    means = np.float32(kmeans.cluster_centers_)

    out_width, out_height = 1920, 1080
    out_dims = (out_width, out_height)
    dst = get_whiteboard_from_points(img, means, out_dims)
    
    [cv2.circle(img,(ptx,pty),100,(0,255,0),10) for [ptx, pty] in np.round(means)]
    images = np.hstack((cv2.resize(img,(300,300)),cv2.resize(dst,(300,300))))
    cv2.imshow(img_name,images)

if __name__ == "__main__":
    main()
