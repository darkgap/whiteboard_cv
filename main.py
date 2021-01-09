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
    [heh(fn) for fn in FILENAMES]

def process_image(img_name):

    
    # pts = np.float32([
    #     [621.5, 699.5],
    #     [3153.5, 1353.5],
    #     [3123.5, 3093.5],
    #     [723.5,	4275.5]])
    
    img_cv2 = cv2.imread(img_name)
    img = image.imread(img_name)

    pts = find_corners(img_cv2)

    # print(type(pts))
    # print(pts.shape)

    # plt.scatter(pts[:,0],pts[:,1])
    # plt.show()
    
    # width,height = 1920,1080
    # filter height
    # valid_pts_width = np.argwhere(pts[:,[0]] < width) # and pts < width
    # pts = pts[valid_pts_width]
    # # filter width
    # valid_pts_height = np.argwhere(pts[:,[1]] < height) # and pts < width
    # pts = pts[valid_pts_height]
    # print(max(height,width))

    img_dims = img.shape[:2]
    width,height = img_dims[0],img_dims[1]
    # print(width,height)

    limit = max(width,height)
    valid_pts = np.argwhere(pts[:,0] < crop_limit*1.20).reshape(-1,) # and pts < width
    pts = pts[valid_pts]
    valid_pts = np.argwhere(pts[:,1] < crop_limit*1.20).reshape(-1,) # and pts < width
    pts = pts[valid_pts]
    # print(pts.shape)
    valid_pts = np.argwhere(pts[:,0] > -crop_limit*0.2).reshape(-1,) # and pts < width
    pts = pts[valid_pts]
    valid_pts = np.argwhere(pts[:1] > -crop_limit*0.2).reshape(-1,) # and pts < width
    pts = pts[valid_pts]
    # print(pts.shape)
    
    # plt.scatter(pts[:,0],pts[:,1])
    # plt.show()

    # print(pts.shape)
    kmeans = KMeans(n_clusters = 4, random_state = 0).fit(pts)
    
    means = np.float32(kmeans.cluster_centers_)


  
    # print(pts)
    # debug = img
    # [cv2.circle(debug,(ptx,pty),100,(255,0,0),10) for [ptx, pty] in np.round(means)]
    # plt.imshow(debug)
    # plt.show()

    # means = np.sort(means,1)
    width,height = 1920,1080
    dims = (width,height)
    means = ordered_corners(means)
    print(means)
    dst = get_whiteboard_from_points(img, means, dims)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # plt.subplot(121),plt.imshow(img),plt.title('Input')
    # plt.subplot(122),plt.imshow(dst),plt.title('Output')
    # plt.show()

    

if __name__ == "__main__":
    # execute only if run as a script
    main()