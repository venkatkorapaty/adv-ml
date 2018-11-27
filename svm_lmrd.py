import numpy as np
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import KFold


def main():
    X_t = np.load('lmrd_train.npy')
    X_cv = np.load('lmrd_cv.npy')
    y_t = np.load('lmrd_train_y.npy')
    y_cv = np.load('lmrd_cv_y.npy')

    X = np.concatenate((X_t, X_cv))
    y = np.concatenate((y_t, y_cv))
    X_t = None
    X_cv = None

    cv_err_avg = KFoldSvm(X, y, n_splits=5, C=100)
    print("CV_avg error: ", cv_err_avg)
    """ RESULTS:
    Split : 1
    C:  100
    Train error:  1.0
    CV error:  0.8526
    Split : 2
    C:  100
    Train error:  1.0
    CV error:  0.8492
    Split : 3
    C:  100
    Train error:  1.0
    CV error:  0.8478
    Split : 4
    C:  100
    Train error:  1.0
    CV error:  0.8432
    Split : 5
    C:  100
    Train error:  1.0
    CV error:  0.8452
    CV_avg error:  0.8475999999999999
    """
    cv_err_avg = KFoldSvm(X, y, n_splits=5, C=50)
    print("CV_avg error: ", cv_err_avg)
    """ RESULTS:
    Split : 1
    C:  50
    Train error:  1.0
    CV error:  0.8526
    Split : 2
    C:  50
    Train error:  1.0
    CV error:  0.8492
    Split : 3
    C:  50
    Train error:  1.0
    CV error:  0.8478
    Split : 4
    C:  50
    Train error:  1.0
    CV error:  0.8432
    Split : 5
    C:  50
    Train error:  1.0
    CV error:  0.8452
    CV_avg error:  0.8475999999999999
    """
    cv_err_avg = KFoldSvm(X, y, n_splits=5, C=0.5)
    print("CV_avg error: ", cv_err_avg)
    """ RESULTS:
    Split : 1
    C:  0.5
    Train error:  0.9995
    CV error:  0.857
    Split : 2
    C:  0.5
    Train error:  0.9994
    CV error:  0.8516
    Split : 3
    C:  0.5
    Train error:  0.9995
    CV error:  0.851
    Split : 4
    C:  0.5
    Train error:  0.99935
    CV error:  0.848
    Split : 5
    C:  0.5
    Train error:  0.9992
    CV error:  0.8506
    CV_avg error:  0.8516400000000001
    """
    cv_err_avg = KFoldSvm(X, y, n_splits=5, C=0.1)
    print("CV_avg error: ", cv_err_avg)
    """Split : 1
    C:  0.1
    Train error:  0.9874
    CV error:  0.8758
    Split : 2
    C:  0.1
    Train error:  0.98815
    CV error:  0.8628
    Split : 3
    C:  0.1
    Train error:  0.98825
    CV error:  0.8686
    Split : 4
    C:  0.1
    Train error:  0.98905
    CV error:  0.8632
    Split : 5
    C:  0.1
    Train error:  0.98805
    CV error:  0.8668
    CV_avg error:  0.8674399999999999
    """
    cv_err_avg = KFoldSvm(X, y, n_splits=5, C=0.05)
    print("CV_avg error: ", cv_err_avg)
    """ RESULTS
    Split : 1
    C:  0.05
    Train error:  0.9758
    CV error:  0.8832
    Split : 2
    C:  0.05
    Train error:  0.97765
    CV error:  0.8678
    Split : 3
    C:  0.05
    Train error:  0.97705
    CV error:  0.8748
    Split : 4
    C:  0.05
    Train error:  0.9784
    CV error:  0.8686
    Split : 5
    C:  0.05
    Train error:  0.9773
    CV error:  0.8746
    CV_avg error:  0.8737999999999999
    """


def main2():
    X = np.load('lmrd_train_0-95.npy')
    y_t = np.load('lmrd_train_y.npy')
    y_cv = np.load('lmrd_cv_y.npy')
    y = np.concatenate((y_t, y_cv))

    cv_err_avg = KFoldSvm(X, y, n_splits=5, C=100)
    print("CV_avg error: ", cv_err_avg)
    """ RESULTS:
    Split : 1
    C:  100
    Train error:  0.99725
    CV error:  0.8312
    Split : 2
    C:  100
    Train error:  0.9973
    CV error:  0.8304
    Split : 3
    C:  100
    Train error:  0.99715
    CV error:  0.8284
    Split : 4
    C:  100
    Train error:  0.9977
    CV error:  0.8228
    Split : 5
    C:  100
    Train error:  0.9971
    CV error:  0.8262
    CV_avg error:  0.8278000000000001
    """
    cv_err_avg = KFoldSvm(X, y, n_splits=5, C=50)
    print("CV_avg error: ", cv_err_avg)
    """ RESULTS:
    Split : 1
    C:  50
    Train error:  0.9974
    CV error:  0.831
    Split : 2
    C:  50
    Train error:  0.9968
    CV error:  0.8318
    Split : 3
    C:  50
    Train error:  0.99695
    CV error:  0.829
    Split : 4
    C:  50
    Train error:  0.9979
    CV error:  0.824
    Split : 5
    C:  50
    Train error:  0.9967
    CV error:  0.826
    CV_avg error:  0.8283599999999998
    """
    cv_err_avg = KFoldSvm(X, y, n_splits=5, C=1)
    print("CV_avg error: ", cv_err_avg)
    """ RESULTS:
    Split : 1
    C:  1
    Train error:  0.9858
    CV error:  0.8494
    Split : 2
    C:  1
    Train error:  0.9878
    CV error:  0.838
    Split : 3
    C:  1
    Train error:  0.98785
    CV error:  0.8446
    Split : 4
    C:  1
    Train error:  0.99005
    CV error:  0.8368
    Split : 5
    C:  1
    Train error:  0.9876
    CV error:  0.8404
    CV_avg error:  0.84184
    """
    cv_err_avg = KFoldSvm(X, y, n_splits=5, C=0.5)
    print("CV_avg error: ", cv_err_avg)
    """ RESULTS:
    Split : 1
    C:  0.5
    Train error:  0.98025
    CV error:  0.8604
    Split : 2
    C:  0.5
    Train error:  0.98205
    CV error:  0.845
    Split : 3
    C:  0.5
    Train error:  0.9819
    CV error:  0.8534
    Split : 4
    C:  0.5
    Train error:  0.98325
    CV error:  0.8406
    Split : 5
    C:  0.5
    Train error:  0.98255
    CV error:  0.8476
    CV_avg error:  0.8493999999999999
    """
    cv_err_avg = KFoldSvm(X, y, n_splits=5, C=0.1)
    print("CV_avg error: ", cv_err_avg)
    """ RESULTS:
    Split : 1
    C:  0.1
    Train error:  0.96155
    CV error:  0.8766
    Split : 2
    C:  0.1
    Train error:  0.9647
    CV error:  0.8662
    Split : 3
    C:  0.1
    Train error:  0.9647
    CV error:  0.8702
    Split : 4
    """
    cv_err_avg = KFoldSvm(X, y, n_splits=5, C=0.05)
    print("CV_avg error: ", cv_err_avg)
    """ RESULTS:
    Split : 1
    C:  0.05
    Train error:  0.9526
    CV error:  0.8826
    Split : 2
    C:  0.05
    Train error:  0.9554
    CV error:  0.8706
    Split : 3
    C:  0.05
    Train error:  0.95505
    CV error:  0.878
    Split : 4
    C:  0.05
    Train error:  0.957
    CV error:  0.8688
    Split : 5
    C:  0.05
    Train error:  0.9556
    CV error:  0.876
    C:  0.05
    Train error:  0.95004
    CV Avg: 0.8752000000000001
    """

def CurrentHypotheses():
    X_t = np.load('lmrd_train.npy')
    X_cv = np.load('lmrd_cv.npy')
    y_t = np.load('lmrd_train_y.npy')
    y_cv = np.load('lmrd_cv_y.npy')

    X = np.concatenate((X_t, X_cv))
    y = np.concatenate((y_t, y_cv))
    X_t = None
    X_cv = None

    print("15236 features")
    cv_err_avg = KFoldSvm(X, y, n_splits=5, C=0.05)
    Svm(X, y, None, None, C=0.05)
    print("CV Avg:", cv_err_avg)
    print("")
    """ RESULTS:
    15236 features
    Split : 1
    C:  0.05
    Train error:  0.9758
    CV error:  0.8832
    Split : 2
    C:  0.05
    Train error:  0.97765
    CV error:  0.8678
    Split : 3
    C:  0.05
    Train error:  0.97705
    CV error:  0.8748
    Split : 4
    C:  0.05
    Train error:  0.9784
    CV error:  0.8686
    Split : 5
    C:  0.05
    Train error:  0.9773
    CV error:  0.8746
    C:  0.05
    Train error:  0.97344
    CV Avg: 0.8737999999999999
    """
    print("5021 principle components")
    X = np.load('lmrd_train_0-95.npy')
    cv_err_avg = KFoldSvm(X, y, 5, C=0.05)
    Svm(X, y, None, None, C=0.05)
    print("CV Avg:", cv_err_avg)
    """ RESULTS:
    5021 principle components
    Split : 1
    C:  0.05
    Train error:  0.9526
    CV error:  0.8826
    Split : 2
    C:  0.05
    Train error:  0.9554
    CV error:  0.8706
    Split : 3
    C:  0.05
    Train error:  0.95505
    CV error:  0.878
    Split : 4
    C:  0.05
    Train error:  0.957
    CV error:  0.8688
    Split : 5
    C:  0.05
    Train error:  0.9556
    CV error:  0.876
    C:  0.05
    Train error:  0.95004
    CV Avg: 0.8752000000000001
    """

def TestError():
    X_t = np.load('lmrd_train.npy')
    X_cv = np.load('lmrd_cv.npy')
    y_t = np.load('lmrd_train_y.npy')
    y_cv = np.load('lmrd_cv_y.npy')
    X_test = np.load('test_15236.npy')
    ones = np.ones((12500, 1))
    zeros = np.zeros((12500, 1))
    y_test = np.atleast_2d(np.append(ones, zeros)).T

    X = np.concatenate((X_t, X_cv))
    y = np.concatenate((y_t, y_cv))
    X_t = None
    X_cv = None

    svm = Svm(X, y, None, None, C=0.05)
    test_pred = svm.predict(X_test)
    correct_preds = np.array(test_pred) == y_test[:, 0]
    print("Test error: ", np.sum(correct_preds) / y_test[:, 0].shape[0])
    """ Results:
    C:  0.05
    Train error:  0.97344
    Test error:  0.86584
    """

    X = np.load('lmrd_train_0-95.npy')    
    prin_comps = np.load('lmrd_train_0-95_components.npy')
    X_test = (X_test - X_test.mean(axis=0)).dot(prin_comps.T)
    svm = Svm(X, y, None, None, C=0.05)

    test_pred = svm.predict(X_test)
    correct_preds = np.array(test_pred) == y_test[:, 0]
    print("Test error: ", np.sum(correct_preds) / y_test[:, 0].shape[0])    
    """ Results:
    C:  0.05
    Train error:  0.95004
    Test error:  0.87256
    """

def NonLinKernelSvms():
    X_t = np.load('lmrd_train.npy')
    X_cv = np.load('lmrd_cv.npy')
    y_t = np.load('lmrd_train_y.npy')
    y_cv = np.load('lmrd_cv_y.npy')

    X = np.concatenate((X_t, X_cv))
    y = np.concatenate((y_t, y_cv))
    X_t = None
    X_cv = None

    cv_err_avg = KFoldSvm(X, y, n_splits=5, C=0.05, svm=NonLinSvm, kernel=[
        "poly", 2, 1])
    print("CV_avg error: ", cv_err_avg)


def KFoldSvm(X, y, n_splits=2, C=1, svm=Svm, kernel=None):
    kf = KFold(n_splits=n_splits)
    split = 1
    cv_err_avg = 0
    for train_index, cv_index in kf.split(X):
        print("Split :", split)
        X_t, X_cv = X[train_index], X[cv_index]
        y_t, y_cv = y[train_index], y[cv_index]
        if kernel is None:
            cv_err_avg += svm(X_t, y_t, X_cv, y_cv, C=C)
        else:
            cv_err_avg += svm(X_t, y_t, X_cv, y_cv, C=C, kernel=kernel)
        split += 1
    return cv_err_avg / n_splits


def Svm(X_t, y_t, X_cv, y_cv, C=1):
    svm = LinearSVC(loss="hinge", C=C)
    svm.fit(X_t, y_t[:, 0])
    train_pred = svm.predict(X_t)
    print("C: ", C)
    correct_preds = np.array(train_pred) == y_t[:, 0]
    print("Train error: ", np.sum(correct_preds) / y_t[:, 0].shape[0])
    
    if X_cv is not None:
        cv_pred = svm.predict(X_cv)
        correct_preds = np.array(cv_pred) == y_cv[:, 0]
        cv_err = np.sum(correct_preds) / y_cv[:, 0].shape[0]
        print("CV error: ", cv_err)
        return cv_err
    return svm


def NonLinSvm(X_t, y_t, X_cv, y_cv, C=1, kernel = ["poly", 1, 1]):
    # kernel parameter is kernal type, followed by parameters for it
    if kernel[0] == "poly":
        # type of kernel, degree, coef0 controls how much lower degree polynomials
        # influence the model vs higher degree
        svm = SVC(kernel=kernel[0], degree=kernel[1], coef0=kernel[2], C=C)
    elif kernel[0] == "rbf":
        svm = SVC(kernel=kernel[0], gamma=kernel[1], C=C)
    else:
        return
    svm.fit(X_t, y_t[:, 0])
    train_pred = svm.predict(X_t)
    print("C: ", C)
    correct_preds = np.array(train_pred) == y_t[:, 0]
    print("Train error: ", np.sum(correct_preds) / y_t[:, 0].shape[0])
    
    if X_cv is not None:
        cv_pred = svm.predict(X_cv)
        correct_preds = np.array(cv_pred) == y_cv[:, 0]
        cv_err = np.sum(correct_preds) / y_cv[:, 0].shape[0]
        print("CV error: ", cv_err)
        return cv_err
    return svm


if __name__ == '__main__':
    #main()
    #main2()
    #CurrentHypotheses()
    #TestError()

    # Non-linear SVMs
    NonLinKernelSvms()
