import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from scipy.sparse import coo_matrix
import scipy.sparse as ss
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold
from sklearn.decomposition import PCA
import time

def plot_coo_matrix(m):
    if not isinstance(m, coo_matrix):
        m = coo_matrix(m)
    print("Created coo matrix..")
    fig = plt.figure()
    ax = fig.add_subplot(111, axisbg='black')
    print("Created ax..")
    ax.plot(m.col, m.row, 's', color='white', ms=1)
    ax.set_xlim(0, m.shape[1])
    ax.set_ylim(0, m.shape[0])
    ax.set_aspect('equal')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    return ax

def main():
    X = np.load('train_reduced.npy')
    print("Loaded data..")
    ax = plot_coo_matrix(X)
    ax.figure.show()
    time.sleep(999999999)

def main2():
    X = np.load('train_reduced.npy')
    print("Loaded data..")
    plt.spy(ss.csr_matrix(X))
    plt.show()

def main3():
    X_t = np.load('lmrd_train.npy')
    y_t = np.load('lmrd_train_y.npy')
    print("Finished loading data..")
    X_centred = X_t - X_t.mean(axis=0)
    U, s, V = np.linalg.svd(X_centred)
    print("Finished SVD Decomp..")
    X_3d = X_centred.dot(V.T[:, :3])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(X_3d.shape[0]):

        if y_t[i] == 1:
            c = 'r'
            m = 'o'
        else:
            c = 'b'
            m = '^'
        ax.scatter(X_3d[i, 0], X_3d[i, 1], X_3d[i, 2], c=c, marker=m)

    ax.set_xlabel('1st Principle Component')
    ax.set_ylabel('2nd Principle Component')
    ax.set_zlabel('3rd Principle Component')
    plt.show()


if __name__ == "__main__":
    # main()
    # main2()
    main3()
