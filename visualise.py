import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from scipy.sparse import coo_matrix
import scipy.sparse as ss
import matplotlib.pyplot as plt
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
    pass


if __name__ == "__main__":
    # main()
    # main2()
    main3()
