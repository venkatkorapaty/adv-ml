import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures


def SelectInitialModels():
    X = np.load('obcw_train.npy')
    y = np.load('obcw_train_y.npy')

    #cv_err_avg = KFoldNoRegLogisticRegression(X, y, 5, C=1)
    #print("CV Avg:", cv_err_avg)
    """
    Split : 1
    C:  1
    Train error:  0.9724770642201835
    CV error:  0.9636363636363636
    Split : 2
    C:  1
    Train error:  0.9702517162471396
    CV error:  0.9724770642201835
    Split : 3
    C:  1
    Train error:  0.9702517162471396
    CV error:  0.981651376146789
    Split : 4
    C:  1
    Train error:  0.9679633867276888
    CV error:  0.9724770642201835
    Split : 5
    C:  1
    Train error:  0.977116704805492
    CV error:  0.9541284403669725
    CV Avg: 0.9688740617180984
    """
    #cv_err_avg = KFoldNoRegLogisticRegression(X, y, 5, C=0.9)
    #print("CV Avg:", cv_err_avg)
    """
    Split : 1
    C:  0.75
    Train error:  0.9724770642201835
    CV error:  0.9636363636363636
    Split : 2
    C:  0.75
    Train error:  0.9702517162471396
    CV error:  0.9724770642201835
    Split : 3
    C:  0.75
    Train error:  0.9702517162471396
    CV error:  0.981651376146789
    Split : 4
    C:  0.75
    Train error:  0.9679633867276888
    CV error:  0.9724770642201835
    Split : 5
    C:  0.75
    Train error:  0.977116704805492
    CV error:  0.944954128440367
    CV Avg: 0.9670391993327774
    """
    #cv_err_avg = KFoldNoRegLogisticRegression(X, y, 5, C=0.75)
    #print("CV Avg:", cv_err_avg)
    """
    Split : 1
    C:  0.75
    Train error:  0.9724770642201835
    CV error:  0.9636363636363636
    Split : 2
    C:  0.75
    Train error:  0.9702517162471396
    CV error:  0.9724770642201835
    Split : 3
    C:  0.75
    Train error:  0.9702517162471396
    CV error:  0.981651376146789
    Split : 4
    C:  0.75
    Train error:  0.9679633867276888
    CV error:  0.9724770642201835
    Split : 5
    C:  0.75
    Train error:  0.977116704805492
    CV error:  0.944954128440367
    CV Avg: 0.9670391993327774
    """
    #cv_err_avg = KFoldNoRegLogisticRegression(X, y, 5, C=0.5)
    #print("CV Avg:", cv_err_avg)
    """
    Split : 1
    C:  0.5
    Train error:  0.9724770642201835
    CV error:  0.9636363636363636
    Split : 2
    C:  0.5
    Train error:  0.9679633867276888
    CV error:  0.981651376146789
    Split : 3
    C:  0.5
    Train error:  0.9679633867276888
    CV error:  0.9724770642201835
    Split : 4
    C:  0.5
    Train error:  0.9702517162471396
    CV error:  0.9724770642201835
    Split : 5
    C:  0.5
    Train error:  0.977116704805492
    CV error:  0.944954128440367
    CV Avg: 0.9670391993327773
    """
    pf = PolynomialFeatures(degree=2, include_bias=False)
    X_p = pf.fit_transform(X)
    print(X_p.shape)
    #cv_err_avg = KFoldNoRegLogisticRegression(X_p, y, 5, C=0.9)
    #print("CV Avg:", cv_err_avg)
    """
    Split : 1
    C:  0.9
    Train error:  0.9977064220183486
    CV error:  0.9545454545454546
    Split : 2
    C:  0.9
    Train error:  0.9954233409610984
    CV error:  0.9357798165137615
    Split : 3
    C:  0.9
    Train error:  0.9977116704805492
    CV error:  0.926605504587156
    Split : 4
    C:  0.9
    Train error:  0.9954233409610984
    CV error:  0.944954128440367
    Split : 5
    C:  0.9
    Train error:  0.9977116704805492
    CV error:  0.9357798165137615
    CV Avg: 0.9395329441201001
    """

    #cv_err_avg = KFoldNoRegLogisticRegression(X_p, y, 5, C=0.5)
    #print("CV Avg:", cv_err_avg)
    """
    Split : 1
    C:  0.5
    Train error:  0.9931192660550459
    CV error:  0.9454545454545454
    Split : 2
    C:  0.5
    Train error:  0.9908466819221968
    CV error:  0.9541284403669725
    Split : 3
    C:  0.5
    Train error:  0.9954233409610984
    CV error:  0.926605504587156
    Split : 4
    C:  0.5
    Train error:  0.9954233409610984
    CV error:  0.9541284403669725
    Split : 5
    C:  0.5
    Train error:  0.9954233409610984
    CV error:  0.9174311926605505
    CV Avg: 0.9395496246872395
    """
    
    #cv_err_avg = KFoldRFELogisticRegression(X_p, y, 5, C=0.9, reduce=30)
    #print("CV Avg:", cv_err_avg)
    """
    Split : 1
    Train error:  0.9977064220183486
    CV error:  0.9545454545454546
    Split : 2
    Train error:  0.9931350114416476
    CV error:  0.9541284403669725
    Split : 3
    Train error:  1.0
    CV error:  0.926605504587156
    Split : 4
    Train error:  0.9954233409610984
    CV error:  0.944954128440367
    Split : 5
    Train error:  0.9977116704805492
    CV error:  0.9357798165137615
    CV Avg: 0.9432026688907422
    """
    #cv_err_avg = KFoldRFELogisticRegression(X_p, y, 5, C=0.9, reduce=25)
    #print("CV Avg:", cv_err_avg)
    #selector = recursive_feature_elimination(X_p, y, None, None, 'l1', C=0.9, reduce=25)
    #print(selector.support_)
    """
    Split : 1
    Train error:  0.9954128440366973
    CV error:  0.9363636363636364
    Split : 2
    Train error:  0.9931350114416476
    CV error:  0.944954128440367
    Split : 3
    Train error:  0.9954233409610984
    CV error:  0.926605504587156
    Split : 4
    Train error:  0.9954233409610984
    CV error:  0.944954128440367
    Split : 5
    Train error:  0.9977116704805492
    CV error:  0.9357798165137615
    CV Avg: 0.9377314428690575
    Train error:  0.989010989010989
    [False False  True False  True False False False False False False False
      True  True  True False  True False  True  True  True False  True  True
      True  True False  True  True False  True False False  True  True False
     False False False  True False False False False False  True False  True
      True  True False  True  True False]
    """

    #cv_err_avg = KFoldRFELogisticRegression(X_p, y, 5, C=0.9, reduce=20)
    #print("CV Avg:", cv_err_avg)
    #selector = recursive_feature_elimination(X_p, y, None, None, 'l1', C=0.9, reduce=20)
    #print(selector.support_)
    """
    Split : 1
    Train error:  0.9862385321100917
    CV error:  0.9363636363636364
    Split : 2
    Train error:  0.9862700228832952
    CV error:  0.9724770642201835
    Split : 3
    Train error:  0.988558352402746
    CV error:  0.926605504587156
    Split : 4
    Train error:  0.9931350114416476
    CV error:  0.944954128440367
    Split : 5
    Train error:  0.9954233409610984
    CV error:  0.9357798165137615
    CV Avg: 0.9432360300250208
    Train error:  0.9835164835164835
    [ True False False False  True False False False False False False False
      True  True  True False False False  True False  True False  True  True
     False  True False False False False  True False False  True  True False
     False False False  True False False False False False  True False  True
      True  True False  True  True False]
    """

    #cv_err_avg = KFoldRFELogisticRegression(X_p, y, 5, C=0.1, reduce=15)
    #print("CV Avg:", cv_err_avg)
    #selector = recursive_feature_elimination(X_p, y, None, None, 'l1', C=0.1, reduce=15)
    #print(selector.support_)


def SelectDataReducedModels():
    X = np.load('obcw_train_0-95.npy')
    y = np.load('obcw_train_y.npy') 
    cv_err_avg = KFoldNoRegLogisticRegression(X, y, 5, C=1)
    print("CV Avg:", cv_err_avg)
    """
    Split : 1
    C:  1
    Train error:  0.9747706422018348
    CV error:  0.9636363636363636
    Split : 2
    C:  1
    Train error:  0.9725400457665904
    CV error:  0.9724770642201835
    Split : 3
    C:  1
    Train error:  0.9702517162471396
    CV error:  0.981651376146789
    Split : 4
    C:  1
    Train error:  0.9702517162471396
    CV error:  0.9724770642201835
    Split : 5
    C:  1
    Train error:  0.9794050343249427
    CV error:  0.9541284403669725
    CV Avg: 0.9688740617180984
    """


def ChosenHypotheses():
    X = np.load('obcw_train.npy')
    y = np.load('obcw_train_y.npy') 
    cv_err_avg = KFoldNoRegLogisticRegression(X, y, 5, C=1)
    LogisReg(X, y, None, None, 'l1', C=1)
    print("CV Avg:", cv_err_avg)
    """
    Split : 1
    C:  1
    Train error:  0.9724770642201835
    CV error:  0.9636363636363636
    Split : 2
    C:  1
    Train error:  0.9702517162471396
    CV error:  0.9724770642201835
    Split : 3
    C:  1
    Train error:  0.9702517162471396
    CV error:  0.981651376146789
    Split : 4
    C:  1
    Train error:  0.9679633867276888
    CV error:  0.9724770642201835
    Split : 5
    C:  1
    Train error:  0.977116704805492
    CV error:  0.9541284403669725
    C:  1
    Train error:  0.9725274725274725
    CV Avg: 0.9688740617180984
    """
    X = np.load('obcw_train_0-95.npy')
    cv_err_avg = KFoldNoRegLogisticRegression(X, y, 5, C=1)
    LogisReg(X, y, None, None, 'l1', C=1)
    print("CV Avg:", cv_err_avg)
    """
    Split : 1
    C:  1
    Train error:  0.9747706422018348
    CV error:  0.9636363636363636
    Split : 2
    C:  1
    Train error:  0.9725400457665904
    CV error:  0.9724770642201835
    Split : 3
    C:  1
    Train error:  0.9702517162471396
    CV error:  0.981651376146789
    Split : 4
    C:  1
    Train error:  0.9702517162471396
    CV error:  0.9724770642201835
    Split : 5
    C:  1
    Train error:  0.9794050343249427
    CV error:  0.9541284403669725
    C:  1
    Train error:  0.9725274725274725
    CV Avg: 0.9688740617180984
    """

def TestError():
    X = np.load('obcw_train.npy')
    y = np.load('obcw_train_y.npy')
    X_test = np.load('obcw_test.npy')
    y_test = np.load('obcw_test_y.npy')


    lr = LogisReg(X, y, None, None, 'l1', C=1)
    test_pred = lr.predict(X_test)
    correct_preds = np.array(test_pred) == y_test
    print("Test error: ", np.sum(correct_preds) / y_test.shape[0])
    """
    C:  1
    Train error:  0.9725274725274725
    Test error:  0.9635036496350365
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
    lr.fit(X_t, y_t)
    train_pred = lr.predict(X_t)
    print("C: ", C)
    correct_preds = np.array(train_pred) == y_t
    print("Train error: ", np.sum(correct_preds) / y_t.shape[0])
    
    if X_cv is not None:
        cv_pred = lr.predict(X_cv)
        correct_preds = np.array(cv_pred) == y_cv
        cv_err = np.sum(correct_preds) / y_cv.shape[0]
        print("CV error: ", cv_err)
        return cv_err
    return lr


def KFoldRFELogisticRegression(X, y, n_splits=2, C=1, reduce=10):
    kf = KFold(n_splits=n_splits)
    split = 1
    cv_err_avg = 0
    for train_index, cv_index in kf.split(X):
        print("Split :", split)
        X_t, X_cv = X[train_index], X[cv_index]
        y_t, y_cv = y[train_index], y[cv_index]
        cv_err_avg += recursive_feature_elimination(X_t, y_t, X_cv, y_cv, p='l1', C=C, reduce=reduce)
        split += 1
    return cv_err_avg / n_splits


def recursive_feature_elimination(X_t, y_t, X_cv, y_cv, p, C=1, reduce=10):
    lr = LogisticRegression(penalty=p, C=C)
    selector = RFE(lr, reduce, step=1).fit(X_t, y_t)

    train_pred = selector.predict(X_t)
    correct_preds = np.array(train_pred) == y_t
    print("Train error: ", np.sum(correct_preds) / y_t.shape[0])
    
    if X_cv is not None:
        cv_pred = selector.predict(X_cv)    
        correct_preds = np.array(cv_pred) == y_cv
        cv_err = np.sum(correct_preds) / y_cv.shape[0]
        print("CV error: ", cv_err)
        return cv_err
    return selector

if __name__ == '__main__':
    SelectInitialModels()
    SelectDataReducedModels()
    ChosenHypotheses()
    TestError()
