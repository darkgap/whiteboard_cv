from sklearn.cluster import KMeans
import numpy as np
import cv2
import os
import time

from img_process import get_whiteboard_from_points
from camboard import find_corners

def main():
    WORK_ON_FILES = False

    if WORK_ON_FILES:
        FILENAMES = [f'images/{name}' for name in os.listdir('images')]
        [process_image(fn, cv2.imread(fn)) for fn in FILENAMES]
        cv2.waitKey(0)
    else:
        cap = cv2.VideoCapture(0)

        last_time = time.time()

        while True:
            ret, frame = cap.read()
            assert(ret)
            print(frame.shape)

            now = time.time()
            frame_time = now - last_time
            last_time = now
            fps = 1 / frame_time

            print(f"FPS = {fps}")

            start = time.time()
            process_image('webcam', frame)
            elapsed = time.time() - start

            print(f"alg time = {elapsed}")
            print(f"max fps = {1 / elapsed}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()

    cv2.destroyAllWindows()


def filter_points(points, width, height, margin=0.2):
    return np.array([
        (x, y) for x, y in points
        if x > -width*margin and x < width*(1+margin) and y > -height*margin and y < height*(1+margin)
    ])


def process_image(img_name, img):
    start = time.time()
    pts = find_corners(img)
    elapsed = time.time() - start
    print(f"find corners: {elapsed}")

    height, width = img.shape[:2]

    pts = filter_points(pts, width, height)

    start = time.time()
    kmeans = KMeans(n_clusters = 4, random_state = 0).fit(pts)
    elapsed = time.time() - start
    print(f"kmeans: {elapsed}")
    
    means = np.float32(kmeans.cluster_centers_)

    out_width, out_height = 1920, 1080
    out_dims = (out_width, out_height)
    start = time.time()
    dst = get_whiteboard_from_points(img, means, out_dims)
    elapsed = time.time() - start
    print(f"get_whiteboard_from_points: {elapsed}")
    
    [cv2.circle(img,(ptx,pty),100,(0,255,0),10) for [ptx, pty] in np.round(means)]
    images = np.hstack((cv2.resize(img,(300,300)),cv2.resize(dst,(300,300))))
    cv2.imshow(img_name, images)

if __name__ == "__main__":
    main()
