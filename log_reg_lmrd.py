import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import KFold

def main():
    X_t = np.load('lmrd_train.npy')
    y_t = np.load('lmrd_train_y.npy')
    X_cv = np.load('lmrd_cv.npy')
    y_cv = np.load('lmrd_cv_y.npy')

    LogisReg(X_t, y_t, X_cv, y_cv, p='l1', C=1)
    recursive_feature_elimination(X_t, y_t, X_cv, y_cv, p='l1', C=0.75)
    recursive_feature_elimination(X_t, y_t, X_cv, y_cv, p='l1', C=0.5)
    recursive_feature_elimination(X_t, y_t, X_cv, y_cv, p='l1', C=0.3)
    #LogisReg(X_t, y_t, X_cv, y_cv, p='l1', C=0.95)
    #LogisReg(X_t, y_t, X_cv, y_cv, p='l1', C=0.9)
    #LogisReg(X_t, y_t, X_cv, y_cv, p='l1', C=0.85)
    #LogisReg(X_t, y_t, X_cv, y_cv, p='l1', C=0.8)
    #LogisReg(X_t, y_t, X_cv, y_cv, p='l1', C=0.75)
    #LogisReg(X_t, y_t, X_cv, y_cv, p='l1', C=0.7)
    #LogisReg(X_t, y_t, X_cv, y_cv, p='l1', C=0.65)
    #LogisReg(X_t, y_t, X_cv, y_cv, p='l1', C=0.6)
    #LogisReg(X_t, y_t, X_cv, y_cv, p='l1', C=0.55)
    #LogisReg(X_t, y_t, X_cv, y_cv, p='l1', C=0.5)
    #LogisReg(X_t, y_t, X_cv, y_cv, p='l1', C=0.3)
    #LogisReg(X_t, y_t, X_cv, y_cv, p='l1', C=0.1)
    #LogisReg(X_t, y_t, X_cv, y_cv, p='l1', C=0.007)

def main2():
    X = np.load('lmrd_train_0-95.npy')
    y_t = np.load('lmrd_train_y.npy')
    y_cv = np.load('lmrd_cv_y.npy')
    y = np.concatenate((y_t, y_cv))

    # KFoldNoRegLogisticRegression(X, y, 5, C=1)
    """ RESULTS:
    Split : 1
    C:  1
    Train error:  0.9677
    CV error:  0.8762
    Split : 2
    C:  1
    Train error:  0.96915
    CV error:  0.861
    Split : 3
    C:  1
    Train error:  0.97075
    CV error:  0.8706
    Split : 4
    C:  1
    Train error:  0.97065
    CV error:  0.8616
    Split : 5
    C:  1
    Train error:  0.9698
    CV error:  0.8714
    """
    #KFoldNoRegLogisticRegression(X, y, 5, C=0.75)
    """ RESULTS:
    Split : 1
    C:  0.75
    Train error:  0.9602
    CV error:  0.8776
    Split : 2
    C:  0.75
    Train error:  0.96305
    CV error:  0.8674
    Split : 3
    C:  0.75
    Train error:  0.9639
    CV error:  0.8742
    Split : 4
    C:  0.75
    Train error:  0.964
    CV error:  0.8672
    Split : 5
    C:  0.75
    Train error:  0.964
    CV error:  0.8754
    """
    KFoldNoRegLogisticRegression(X, y, 5, C=0.5)
    """RESULTS:
    Split : 1
    C:  0.5
    Train error:  0.95055
    CV error:  0.8816
    Split : 2
    C:  0.5
    Train error:  0.95375
    CV error:  0.8706
    Split : 3
    C:  0.5
    Train error:  0.9524
    CV error:  0.88
    Split : 4
    C:  0.5
    Train error:  0.9539
    CV error:  0.8736
    Split : 5
    C:  0.5
    Train error:  0.95445
    CV error:  0.8806
    """

def CurrentHypothesis():
    X_t = np.load('lmrd_train.npy')
    X_cv = np.load('lmrd_cv.npy')
    y_t = np.load('lmrd_train_y.npy')
    y_cv = np.load('lmrd_cv_y.npy')

    X = np.concatenate((X_t, X_cv))
    y = np.concatenate((y_t, y_cv))
    X_t = None
    X_cv = None

    print("15236 features")
    cv_err_avg = KFoldNoRegLogisticRegression(X, y, 5, C=0.5)
    LogisReg(X, y, None, None, 'l1', C=0.5)
    print("CV Avg:", cv_err_avg)
    print("")
    """ RESULTS:
    15236 features
    Split : 1
    C:  0.5
    Train error:  0.9592
    CV error:  0.8794
    Split : 2
    C:  0.5
    Train error:  0.9623
    CV error:  0.8708
    Split : 3
    C:  0.5
    Train error:  0.96205
    CV error:  0.8828
    Split : 4
    C:  0.5
    Train error:  0.96145
    CV error:  0.8756
    Split : 5
    C:  0.5
    Train error:  0.96265
    CV error:  0.8772
    C:  0.5
    Train error:  0.95972
    CV Avg: 0.8771599999999999
    """
    print("5021 principle components")
    X = np.load('lmrd_train_0-95.npy')
    cv_err_avg = KFoldNoRegLogisticRegression(X, y, 5, C=0.5)
    LogisReg(X, y, None, None, 'l1', C=0.5)
    print("CV Avg:", cv_err_avg)
    """ RESULTS:
    5021 principle components
    Split : 1
    C:  0.5
    Train error:  0.95055
    CV error:  0.8816
    Split : 2
    C:  0.5
    Train error:  0.95375
    CV error:  0.8706
    Split : 3
    C:  0.5
    Train error:  0.9524
    CV error:  0.88
    Split : 4
    C:  0.5
    Train error:  0.9539
    CV error:  0.8736
    Split : 5
    C:  0.5
    Train error:  0.95445
    CV error:  0.8806
    C:  0.5
    Train error:  0.9484
    CV Avg: 0.8772800000000001
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

    lr = LogisReg(X, y, None, None, 'l1', C=0.5)
    test_pred = lr.predict(X_test)
    correct_preds = np.array(test_pred) == y_test[:, 0]
    print("Test error: ", np.sum(correct_preds) / y_test[:, 0].shape[0])
    """ RESULTS:
    C:  0.5
    Train error:  0.95972
    Test error:  0.87376
    """

    X = np.load('lmrd_train_0-95.npy')    
    prin_comps = np.load('lmrd_train_0-95_components.npy')
    X_test = (X_test - X_test.mean(axis=0)).dot(prin_comps.T)
    lr = LogisReg(X, y, None, None, 'l1', C=0.5)

    test_pred = lr.predict(X_test)
    correct_preds = np.array(test_pred) == y_test[:, 0]
    print("Test error: ", np.sum(correct_preds) / y_test[:, 0].shape[0])
    """ RESULTS:
    C:  0.5
    Train error:  0.9484
    Test error:  0.87072
    """


def KFoldNoRegLogisticRegression(X, y, n_splits=2, C=1):
    kf = KFold(n_splits=n_splits)
    split = 1
    cv_err_avg = 0
    for train_index, cv_index in kf.split(X):
        print("Split :", split)
        X_t, X_cv = X[train_index], X[cv_index]
        y_t, y_cv = y[train_index], y[cv_index]
        cv_err_avg += LogisReg(X_t, y_t, X_cv, y_cv, p='l1', C=C)
        split += 1
    return cv_err_avg / n_splits

def LogisReg(X_t, y_t, X_cv, y_cv, p, C=1):
    lr = LogisticRegression(penalty=p, C=C)
    lr.fit(X_t, y_t[:, 0])
    train_pred = lr.predict(X_t)
    print("C: ", C)
    correct_preds = np.array(train_pred) == y_t[:, 0]
    print("Train error: ", np.sum(correct_preds) / y_t[:, 0].shape[0])
    
    if X_cv is not None:
        cv_pred = lr.predict(X_cv)
        correct_preds = np.array(cv_pred) == y_cv[:, 0]
        cv_err = np.sum(correct_preds) / y_cv[:, 0].shape[0]
        print("CV error: ", cv_err)
        return cv_err
    return lr


def recursive_feature_elimination(X_t, y_t, X_cv, y_cv, p, C=1):
    lr = LogisticRegression(penalty=p, C=C)
    selector = RFE(lr, 10000, step=50).fit(X_t, y_t[:, 0])

    train_pred = selector.predict(X_t)
    cv_pred = selector.predict(X_cv)
    correct_preds = np.array(train_pred) == y_t[:, 0]
    print("Train error: ", np.sum(correct_preds) / y_t[:, 0].shape[0])
    
    correct_preds = np.array(cv_pred) == y_cv[:, 0]
    print("CV error: ", np.sum(correct_preds) / y_cv[:, 0].shape[0])    


if __name__ == '__main__':
    #main()
    #main2()
    #CurrentHypothesis()
    TestError()
