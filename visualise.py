import numpy as np
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

def GetEigenvectors():
    X_t = np.load('lmrd_train.npy')
    y_t = np.load('lmrd_train_y.npy')
    print("Finished loading data..")
    X_centred = X_t - X_t.mean(axis=0)
    U, s, V = np.linalg.svd(X_centred)
    print("Finished SVD Decomp..")
    np.save('lmrd_eigens.npy', V)

def PlotLMRDTrain():
    X_t = np.load('lmrd_train.npy')
    y_t = np.load('lmrd_train_y.npy')
    V = np.load('lmrd_eigens.npy')
    print("Finished loading..")
    X_t = (X_t - X_t.mean(axis=0)).dot(V.T[:, :3])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(X_t.shape[0]):
        print(i)
        if y_t[i] == 1:
            c = 'r'
            m = 'o'
        else:
            c = 'b'
            m = '^'
        ax.scatter(X_t[i, 0], X_t[i, 1], X_t[i, 2], c=c, marker=m)

    ax.set_xlabel('1st Principle Component')
    ax.set_ylabel('2nd Principle Component')
    ax.set_zlabel('3rd Principle Component')

    plt.savefig('DGFGFGF.png')
    plt.show()

def SaveLMRD95ExplainedVariance():
    X = np.load('lmrd_train.npy')
    X_cv = np.load('lmrd_cv.npy')
    print("Loaded data..")
    X = np.concatenate((X, X_cv))
    X_cv = None
    
    pca = PCA(n_components=0.95)
    X_reduced = pca.fit_transform(X)
    print("PCA finished..")
    np.save('lmrd_train_0-95_components.npy', pca.components_)
    #np.save('lmrd_train_0-95.npy', X_reduced)
    print(X_reduced.shape)
    print(pca.components_.shape)


def PlotOBCWTrain():
    X = np.load('obcw_train.npy')
    y = np.load('obcw_train_y.npy')
    
    pca = PCA(n_components=0.95)
    X_reduced = pca.fit_transform(X)
    print(X_reduced.shape)
    print(pca.components_.shape)
    np.save('obcw_train_0-95.npy', X_reduced)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(X_reduced.shape[0]):
        if y[i] == 1:
            c = 'r'
            m = 'o'
        else:
            c = 'b'
            m = '^'
        ax.scatter(X_reduced[i, 0], X_reduced[i, 1], X_reduced[i, 2], c=c, marker=m)

    ax.set_xlabel('1st Principle Component')
    ax.set_ylabel('2nd Principle Component')
    ax.set_zlabel('3rd Principle Component')

    plt.savefig('obcw.png')
    plt.show()


if __name__ == "__main__":
    # GetEigenvectors()
    # PlotLMRDTrain()
    # SaveLMRD95ExplainedVariance()
    PlotOBCWTrain()
