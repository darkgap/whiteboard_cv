import numpy as np
import cv2

def get_whiteboard_from_points(img,pts,resolution = (1920,1080)):
    width,height = resolution 

    # A = get_mat_A_for_dim(*resolution)
    # # print(pts.shape)
    # B = basis_to_point(pts)

    # C = A @ np.linalg.inv(B)
    # C = C.T/C[-1,-1] # back to homogeneus transform matrix
    
    C = cv2.getPerspectiveTransform(pts,np.float32([[0,width,width,0],[0,0,height,height]]).T)

    dst = cv2.warpPerspective(img,C,resolution)
    return dst

    
def get_mat_A_for_dim(width: int = 1920, height: int = 1080):
    return basis_to_point([[0,width,width,0],[0,0,height,height]])

def basis_to_point(points):
    pts = np.vstack((points, np.ones((1,4))))
    left_pts = pts[:, :3]
    right_pts = pts[:, [-1]]

    # print(left_pts.shape)

    # print(destination_pt_matrix)
    # print(np.linalg.inv(destinatio n_pt_matrix).shape," | ",np.array([0,height,1]).reshape(-1,1).shape )
    coeffs = np.linalg.inv(left_pts) @ np.array(right_pts).reshape(-1,1)
    # print(coeffs)
    return left_pts @ np.diag(coeffs.reshape(-1,))
