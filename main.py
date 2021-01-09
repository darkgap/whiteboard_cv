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

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_image(img_name):

    img_cv2 = cv2.imread(img_name)
    img = cv2.imread(img_name)

    pts = find_corners(img_cv2)

    img_dims = img.shape[:2]
    width,height = img_dims[0],img_dims[1]
    crop_limit = max(width,height)

    valid_pts = np.argwhere(pts[:,0] < crop_limit*1.20).reshape(-1,) # and pts < width
    pts = pts[valid_pts]
    valid_pts = np.argwhere(pts[:,1] < crop_limit*1.20).reshape(-1,) # and pts < width
    pts = pts[valid_pts]

    kmeans = KMeans(n_clusters = 4, random_state = 0).fit(pts)
    
    means = np.float32(kmeans.cluster_centers_)

    out_width, out_height = 1920,1080
    out_dims = (out_width, out_height)
    dst = get_whiteboard_from_points(img, means, out_dims)
    
    [cv2.circle(img,(ptx,pty),100,(0,255,0),10) for [ptx, pty] in np.round(means)]

    cv2.imshow(img_name,np.hstack((cv2.resize(debug,(300,300)),cv2.resize(dst,(300,300)))))

if __name__ == "__main__":
    main()