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

def KFoldNoRegLogisticRegression(X, y, n_splits=2, C=1):
    kf = KFold(n_splits=n_splits)
    split = 1
    for train_index, cv_index in kf.split(X):
        print("Split :", split)
        X_t, X_cv = X[train_index], X[cv_index]
        y_t, y_cv = y[train_index], y[cv_index]
        LogisReg(X_t, y_t, X_cv, y_cv, p='l1', C=C)
        split += 1

def LogisReg(X_t, y_t, X_cv, y_cv, p, C=1):
    lr = LogisticRegression(penalty=p, C=C)
    lr.fit(X_t, y_t[:, 0])
    train_pred = lr.predict(X_t)
    cv_pred = lr.predict(X_cv)
    print("C: ", C)
    correct_preds = np.array(train_pred) == y_t[:, 0]
    print("Train error: ", np.sum(correct_preds) / y_t[:, 0].shape[0])
    
    correct_preds = np.array(cv_pred) == y_cv[:, 0]
    print("CV error: ", np.sum(correct_preds) / y_cv[:, 0].shape[0])

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
    main2()
