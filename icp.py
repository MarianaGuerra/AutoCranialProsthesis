# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>


import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# fonte: https://github.com/ClayFlannigan/icp/blob/master/icp.py


def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform between corresponding 3D points A->B
    Input:
      A: Nx3 numpy array of corresponding 3D points
      B: Nx3 numpy array of corresponding 3D points
    Returns:
      T: 4x4 homogeneous transformation matrix
      R: 3x3 rotation matrix
      t: 3x1 column vector
    '''

    assert len(A) == len(B)

    # translate points to their centroids
    # centroid_A = np.mean(A, axis=0)
    # centroid_B = np.mean(B, axis=0)
    AA = A  # - centroid_A
    BB = B  # - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[2,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation - not wanted
    # t = centroid_B.T - np.dot(R,centroid_A.T)
    t = [0, 0, 0]

    # homogeneous transformation - the rotation only because t was set to zero
    T = np.identity(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = t

    return T, R, t


def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nx3 array of points
        dst: Nx3 array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    all_dists = cdist(src, dst, 'euclidean')
    indices = all_dists.argmin(axis=1)
    distances = all_dists[np.arange(all_dists.shape[0]), indices]
    return distances, indices


def icp(A, B, init_pose=None, max_iterations=20, tolerance=0.001):
    '''
    The Iterative Closest Point method
    Input:
        A: Nx3 numpy array of source 3D points
        B: Nx3 numpy array of destination 3D point
        init_pose: 4x4 homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation
        distances: Euclidean distances (errors) of the nearest neighbor
    '''

    # make points homogeneous, copy them so as to maintain the originals
    src = np.ones((4, A.shape[0]))
    dst = np.ones((4, B.shape[0]))
    src[0:3, :] = np.copy(A.T)
    dst[0:3, :] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbours between the current source and destination points
        distances, indices = nearest_neighbor(src[0:3, :].T, dst[0:3, :].T)

        # compute the transformation between the current source and nearest destination points
        T, _, _ = best_fit_transform(src[0:3, :].T, dst[0:3, indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.sum(distances) / distances.size
        if abs(mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T, _, _ = best_fit_transform(A, src[0:3, :].T)

    return T, distances


def icp_wrap(moving_points, fixed_points, debug=False):
    if debug:
        fig = plt.figure()
        plt.scatter(moving_points[:, 0], moving_points[:, 1], c='red', s=2)
        plt.scatter(fixed_points[:, 0], fixed_points[:, 1], c='blue', s=2)

    # Convert to 3D
    if moving_points.shape[1] != 3 or fixed_points.shape[1] != 3:
        new_moving_points = np.zeros([moving_points.shape[0], 3])
        new_fixed_points = np.zeros([fixed_points.shape[0], 3])

        new_moving_points[:, 0:2] = moving_points
        new_fixed_points[:, 0:2] = fixed_points

        moving_points = new_moving_points
        fixed_points = new_fixed_points

    T, x = icp(moving_points, fixed_points, init_pose=None, max_iterations=500, tolerance=0.00001)

    if debug:
        print("transform matrix = \n" + str(T))

    # Convert to 4D
    moving_points_4d = np.zeros([moving_points.shape[0], 4])

    moving_points_4d[:, 0:3] = moving_points
    moving_points_4d[:, 3] = 1

    moving_points_4d = np.matmul(T, np.transpose(moving_points_4d))

    if debug:
        print("points matrix = \n" + str(np.transpose(moving_points_4d)))
        # fig2 = plt.figure()
        plt.scatter(moving_points_4d[0, :], moving_points_4d[1, :], c='black', s=2)
        print("result = \n" + str(moving_points_4d))
        plt.show()

    return T


def main():
    # A = np.array([[1.0, 1.0, 0], [1.1, 1.1, 0], [1.2, 1.2, 0], [1.3, 1.31, 0], [1.4, 1.4, 0], [1.51, 1.5, 0],
    # [1.6, 1.6, 0]])
    # B = np.array([[0.3, 1.0, 0], [0.3, 1.1, 0], [0.3, 1.2, 0], [0.31, 1.3, 0], [0.3, 1.4, 0], [0.3, 1.5, 0],
    # [0.3, 1.6, 0]])

    # A = np.array([[1, 1, 0], [1.5, 2.5, 0], [3, 2.75, 0], [4, 2.5, 0], [5, 1.5, 0]])
    # B = np.array([[1.28171276,  0.59767248, 0], [2.26458929,  1.83620134, 0], [3.75963326,  1.55809428, 0],
    # [4.61382084, 0.98115098, 0], [5.21149332, -0.30056179, 0]])

    # side_a = np.loadtxt("a_cont40.txt", delimiter=' ')
    # side_b = np.loadtxt("b_mirror_cont40.txt", delimiter=' ')

    side_a = np.loadtxt("a_cont2.txt", delimiter=' ')
    side_b = np.loadtxt("b_mirror_cont2.txt", delimiter=' ')

    # side_a += [100, 0]

    icp_wrap(side_b, side_a, debug=True)


if __name__ == '__main__':
    main()
