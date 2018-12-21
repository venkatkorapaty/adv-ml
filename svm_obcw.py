import numpy as np
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures


def SelectInitialModels():
    X = np.load('obcw_train.npy')
    y = np.load('obcw_train_y.npy')

    #cv_err_avg = KFoldSvm(X, y, n_splits=5, C=100)
    #print("CV_avg error: ", cv_err_avg)
    """
    Split : 1
    C:  100
    Train error:  0.9701834862385321
    CV error:  0.9727272727272728
    Split : 2
    C:  100
    Train error:  0.9725400457665904
    CV error:  0.963302752293578
    Split : 3
    C:  100
    Train error:  0.9725400457665904
    CV error:  0.981651376146789
    Split : 4
    C:  100
    Train error:  0.9679633867276888
    CV error:  0.9724770642201835
    Split : 5
    C:  100
    Train error:  0.9633867276887872
    CV error:  0.944954128440367
    CV_avg error:  0.9670225187656382
    """
    #cv_err_avg = KFoldSvm(X, y, n_splits=5, C=50)
    #print("CV_avg error: ", cv_err_avg)
    """
    Split : 1
    C:  50
    Train error:  0.9701834862385321
    CV error:  0.9636363636363636
    Split : 2
    C:  50
    Train error:  0.9702517162471396
    CV error:  0.9724770642201835
    Split : 3
    C:  50
    Train error:  0.9473684210526315
    CV error:  0.9541284403669725
    Split : 4
    C:  50
    Train error:  0.9679633867276888
    CV error:  0.9724770642201835
    Split : 5
    C:  50
    Train error:  0.9748283752860412
    CV error:  0.963302752293578
    CV_avg error:  0.9652043369474562
    """
    #cv_err_avg = KFoldSvm(X, y, n_splits=5, C=1)
    #print("CV_avg error: ", cv_err_avg)
    """
    Split : 1
    C:  1
    Train error:  0.9724770642201835
    CV error:  0.9636363636363636
    Split : 2
    C:  1
    Train error:  0.9725400457665904
    CV error:  0.9724770642201835
    Split : 3
    C:  1
    Train error:  0.9702517162471396
    CV error:  0.9724770642201835
    Split : 4
    C:  1
    Train error:  0.9748283752860412
    CV error:  0.9724770642201835
    Split : 5
    C:  1
    Train error:  0.9794050343249427
    CV error:  0.9541284403669725
    CV_avg error:  0.9670391993327773
    """
    #cv_err_avg = KFoldSvm(X, y, n_splits=5, C=0.5)
    #print("CV_avg error: ", cv_err_avg)
    """
    Split : 1
    C:  0.5
    Train error:  0.9747706422018348
    CV error:  0.9454545454545454
    Split : 2
    C:  0.5
    Train error:  0.9725400457665904
    CV error:  0.981651376146789
    Split : 3
    C:  0.5
    Train error:  0.9725400457665904
    CV error:  0.9724770642201835
    Split : 4
    C:  0.5
    Train error:  0.9748283752860412
    CV error:  0.9724770642201835
    Split : 5
    C:  0.5
    Train error:  0.9794050343249427
    CV error:  0.9541284403669725
    CV_avg error:  0.9652376980817348
    """

    #cv_err_avg = KFoldSvm(X, y, n_splits=5, C=1, kernel=[
        #"poly", 2, 1])
    #print("CV_avg error: ", cv_err_avg)
    """
    Split : 1
    C:  1
    Train error:  0.981651376146789
    CV error:  0.9090909090909091
    Split : 2
    C:  1
    Train error:  0.9816933638443935
    CV error:  0.926605504587156
    Split : 3
    C:  1
    Train error:  0.9862700228832952
    CV error:  0.8990825688073395
    Split : 4
    C:  1
    Train error:  0.9610983981693364
    CV error:  0.9174311926605505
    Split : 5
    C:  1
    Train error:  0.9908466819221968
    CV error:  0.926605504587156
    CV_avg error:  0.9157631359466223
    Split : 1
    C:  1
    Train error:  0.9931192660550459
    CV error:  0.9363636363636364
    Split : 2
    C:  1
    Train error:  0.9931350114416476
    CV error:  0.9174311926605505
    Split : 3
    C:  1
    Train error:  0.9908466819221968
    CV error:  0.908256880733945
    Split : 4
    C:  1
    Train error:  0.9977116704805492
    CV error:  0.9357798165137615
    Split : 5
    C:  1
    Train error:  0.9977116704805492
    CV error:  0.9357798165137615
    CV_avg error:  0.926722268557131
    """

    #cv_err_avg = KFoldSvm(X, y, n_splits=5, C=0.3, kernel=[
        #"poly", 2, 1])
    #print("CV_avg error: ", cv_err_avg)
    """
    Split : 1
    C:  0.3
    Train error:  0.9862385321100917
    CV error:  0.9454545454545454
    Split : 2
    C:  0.3
    Train error:  0.9862700228832952
    CV error:  0.963302752293578
    Split : 3
    C:  0.3
    Train error:  0.9862700228832952
    CV error:  0.926605504587156
    Split : 4
    C:  0.3
    Train error:  0.9931350114416476
    CV error:  0.944954128440367
    Split : 5
    C:  0.3
    Train error:  0.9908466819221968
    CV error:  0.9174311926605505
    CV_avg error:  0.9395496246872395
    """

    #cv_err_avg = KFoldSvm(X, y, n_splits=5, C=0.1, kernel=[
        #"poly", 2, 1])
    #print("CV_avg error: ", cv_err_avg)
    """
    Split : 1
    C:  0.1
    Train error:  0.981651376146789
    CV error:  0.9545454545454546
    Split : 2
    C:  0.1
    Train error:  0.9816933638443935
    CV error:  0.963302752293578
    Split : 3
    C:  0.1
    Train error:  0.977116704805492
    CV error:  0.944954128440367
    Split : 4
    C:  0.1
    Train error:  0.9862700228832952
    CV error:  0.9541284403669725
    Split : 5
    C:  0.1
    Train error:  0.988558352402746
    CV error:  0.9174311926605505
    CV_avg error:  0.9468723936613845
    """

    #cv_err_avg = KFoldSvm(X, y, n_splits=5, C=0.1, kernel=[
        #"rbf", 5])
    #print("CV_avg error: ", cv_err_avg)
    """
    Split : 1
    C:  0.1
    Train error:  0.6559633027522935
    CV error:  0.6636363636363637
    Split : 2
    C:  0.1
    Train error:  0.6521739130434783
    CV error:  0.6788990825688074
    Split : 3
    C:  0.1
    Train error:  0.6613272311212814
    CV error:  0.6422018348623854
    Split : 4
    C:  0.1
    Train error:  0.6704805491990846
    CV error:  0.6055045871559633
    Split : 5
    C:  0.1
    Train error:  0.6475972540045767
    CV error:  0.6972477064220184
    CV_avg error:  0.6574979149291076
    """
    #cv_err_avg = KFoldSvm(X, y, n_splits=5, C=1, kernel=[
        #"rbf", 5])
    #print("CV_avg error: ", cv_err_avg)
    """
    Split : 1
    C:  1
    Train error:  1.0
    CV error:  0.6909090909090909
    Split : 2
    C:  1
    Train error:  1.0
    CV error:  0.8073394495412844
    Split : 3
    C:  1
    Train error:  1.0
    CV error:  0.6513761467889908
    Split : 4
    C:  1
    Train error:  1.0
    CV error:  0.6146788990825688
    Split : 5
    C:  1
    Train error:  1.0
    CV error:  0.7064220183486238
    CV_avg error:  0.6941451209341117
    """

    #cv_err_avg = KFoldSvm(X, y, n_splits=5, C=0.5, kernel=[
        #"rbf", 5])
    #print("CV_avg error: ", cv_err_avg)
    """
    Split : 1
    C:  0.5
    Train error:  0.6559633027522935
    CV error:  0.6636363636363637
    Split : 2
    C:  0.5
    Train error:  1.0
    CV error:  0.6788990825688074
    Split : 3
    C:  0.5
    Train error:  0.6704805491990846
    CV error:  0.6422018348623854
    Split : 4
    C:  0.5
    Train error:  0.6796338672768879
    CV error:  0.6055045871559633
    Split : 5
    C:  0.5
    Train error:  1.0
    CV error:  0.7064220183486238
    CV_avg error:  0.6593327773144286
    """

    #cv_err_avg = KFoldSvm(X, y, n_splits=5, C=0.5, kernel=[
        #"rbf", 1/9])
    #print("CV_avg error: ", cv_err_avg)
    """
    Split : 1
    C:  0.5
    Train error:  0.9678899082568807
    CV error:  0.9545454545454546
    Split : 2
    C:  0.5
    Train error:  0.9679633867276888
    CV error:  0.926605504587156
    Split : 3
    C:  0.5
    Train error:  0.9702517162471396
    CV error:  0.926605504587156
    Split : 4
    C:  0.5
    Train error:  0.965675057208238
    CV error:  0.9724770642201835
    Split : 5
    C:  0.5
    Train error:  0.9725400457665904
    CV error:  0.9541284403669725
    CV_avg error:  0.9468723936613845
    """

    #cv_err_avg = KFoldSvm(X, y, n_splits=5, C=0.5, kernel=[
        #"rbf", 1/10])
    #print("CV_avg error: ", cv_err_avg)
    """
    Split : 1
    C:  0.5
    Train error:  0.9724770642201835
    CV error:  0.9545454545454546
    Split : 2
    C:  0.5
    Train error:  0.9679633867276888
    CV error:  0.926605504587156
    Split : 3
    C:  0.5
    Train error:  0.9748283752860412
    CV error:  0.9357798165137615
    Split : 4
    C:  0.5
    Train error:  0.965675057208238
    CV error:  0.981651376146789
    Split : 5
    C:  0.5
    Train error:  0.9725400457665904
    CV error:  0.9541284403669725
    CV_avg error:  0.9505421184320267
    """
    #cv_err_avg = KFoldSvm(X, y, n_splits=5, C=0.5, kernel=[
        #"rbf", 1/11])
    #print("CV_avg error: ", cv_err_avg)
    """
    Split : 1
    C:  0.5
    Train error:  0.9701834862385321
    CV error:  0.9545454545454546
    Split : 2
    C:  0.5
    Train error:  0.9702517162471396
    CV error:  0.9357798165137615
    Split : 3
    C:  0.5
    Train error:  0.9748283752860412
    CV error:  0.9357798165137615
    Split : 4
    C:  0.5
    Train error:  0.965675057208238
    CV error:  0.9724770642201835
    Split : 5
    C:  0.5
    Train error:  0.9748283752860412
    CV error:  0.9541284403669725
    CV_avg error:  0.9505421184320267
    """
    #cv_err_avg = KFoldSvm(X, y, n_splits=5, C=0.5, kernel=[
        #"rbf", 1/12])
    #print("CV_avg error: ", cv_err_avg)
    """
    Split : 1
    C:  0.5
    Train error:  0.9701834862385321
    CV error:  0.9636363636363636
    Split : 2
    C:  0.5
    Train error:  0.9702517162471396
    CV error:  0.944954128440367
    Split : 3
    C:  0.5
    Train error:  0.977116704805492
    CV error:  0.9357798165137615
    Split : 4
    C:  0.5
    Train error:  0.965675057208238
    CV error:  0.9724770642201835
    Split : 5
    C:  0.5
    Train error:  0.9748283752860412
    CV error:  0.963302752293578
    CV_avg error:  0.9560300250208508
    """

    #cv_err_avg = KFoldSvm(X, y, n_splits=5, C=0.5, kernel=[
        #"rbf", 1/15])
    #print("CV_avg error: ", cv_err_avg)
    """
    Split : 1
    C:  0.5
    Train error:  0.9724770642201835
    CV error:  0.9636363636363636
    Split : 2
    C:  0.5
    Train error:  0.9748283752860412
    CV error:  0.944954128440367
    Split : 3
    C:  0.5
    Train error:  0.977116704805492
    CV error:  0.944954128440367
    Split : 4
    C:  0.5
    Train error:  0.9679633867276888
    CV error:  0.963302752293578
    Split : 5
    C:  0.5
    Train error:  0.9748283752860412
    CV error:  0.963302752293578
    CV_avg error:  0.9560300250208508
    """

    #cv_err_avg = KFoldSvm(X, y, n_splits=5, C=1, kernel=[
        #"rbf", 1/15])
    #print("CV_avg error: ", cv_err_avg)
    """
    Split : 1
    C:  1
    Train error:  0.9931192660550459
    CV error:  0.9727272727272728
    Split : 2
    C:  1
    Train error:  0.988558352402746
    CV error:  0.963302752293578
    Split : 3
    C:  1
    Train error:  0.9816933638443935
    CV error:  0.944954128440367
    Split : 4
    C:  1
    Train error:  0.9908466819221968
    CV error:  0.963302752293578
    Split : 5
    C:  1
    Train error:  0.9931350114416476
    CV error:  0.963302752293578
    CV_avg error:  0.9615179316096748
    """
    #cv_err_avg = KFoldSvm(X, y, n_splits=5, C=10, kernel=[
        #"rbf", 1/15])
    #print("CV_avg error: ", cv_err_avg)
    """
    Split : 1
    C:  10
    Train error:  1.0
    CV error:  0.9636363636363636
    Split : 2
    C:  10
    Train error:  1.0
    CV error:  0.944954128440367
    Split : 3
    C:  10
    Train error:  1.0
    CV error:  0.9541284403669725
    Split : 4
    C:  10
    Train error:  1.0
    CV error:  0.963302752293578
    Split : 5
    C:  10
    Train error:  1.0
    CV error:  0.9541284403669725
    CV_avg error:  0.9560300250208507
    """


def ChosenHypotheses():
    X = np.load('obcw_train.npy')
    y = np.load('obcw_train_y.npy')

    cv_err_avg = KFoldSvm(X, y, n_splits=5, C=100)
    Svm(X, y, None, None, C=100)
    print("CV_avg error: ", cv_err_avg)
    """
    Split : 1
    C:  100
    Train error:  0.9747706422018348
    CV error:  0.9636363636363636
    Split : 2
    C:  100
    Train error:  0.9725400457665904
    CV error:  0.9724770642201835
    Split : 3
    C:  100
    Train error:  0.9679633867276888
    CV error:  0.9541284403669725
    Split : 4
    C:  100
    Train error:  0.977116704805492
    CV error:  0.9724770642201835
    Split : 5
    C:  100
    Train error:  0.9702517162471396
    CV error:  0.944954128440367
    C:  100
    Train error:  0.9706959706959707
    CV_avg error:  0.9615346121768141
    """
    print("")
    cv_err_avg = KFoldSvm(X, y, n_splits=5, C=1)
    Svm(X, y, None, None, C=1)
    print("CV_avg error: ", cv_err_avg)
    """
    Split : 1
    C:  1
    Train error:  0.9747706422018348
    CV error:  0.9636363636363636
    Split : 2
    C:  1
    Train error:  0.9748283752860412
    CV error:  0.981651376146789
    Split : 3
    C:  1
    Train error:  0.9702517162471396
    CV error:  0.9724770642201835
    Split : 4
    C:  1
    Train error:  0.9748283752860412
    CV error:  0.9724770642201835
    Split : 5
    C:  1
    Train error:  0.9794050343249427
    CV error:  0.963302752293578
    C:  1
    Train error:  0.9725274725274725
    CV_avg error:  0.9707089241034195
    """
    print("")
    cv_err_avg = KFoldSvm(X, y, n_splits=5, C=1, kernel=[
        "rbf", 1/15])
    NonLinSvm(X, y, None, None, C=1, kernel = ["rbf", 1/15])
    print("CV_avg error: ", cv_err_avg)
    """
    Split : 1
    C:  1
    Train error:  0.9931192660550459
    CV error:  0.9727272727272728
    Split : 2
    C:  1
    Train error:  0.988558352402746
    CV error:  0.963302752293578
    Split : 3
    C:  1
    Train error:  0.9816933638443935
    CV error:  0.944954128440367
    Split : 4
    C:  1
    Train error:  0.9908466819221968
    CV error:  0.963302752293578
    Split : 5
    C:  1
    Train error:  0.9931350114416476
    CV error:  0.963302752293578
    C:  1
    Train error:  0.9853479853479854
    CV_avg error:  0.9615179316096748
    """

def TestError():
    X = np.load('obcw_train.npy')
    y = np.load('obcw_train_y.npy')
    X_test = np.load('obcw_test.npy')
    y_test = np.load('obcw_test_y.npy')

    svm = Svm(X, y, None, None, C=1)
    test_pred = svm.predict(X_test)
    correct_preds = np.array(test_pred) == y_test
    print("Test error: ", np.sum(correct_preds) / y_test.shape[0])
    """
    C:  1
    Train error:  0.9725274725274725
    Test error:  0.9708029197080292
    """
    svm = NonLinSvm(X, y, None, None, C=1, kernel = ["rbf", 1/15])
    test_pred = svm.predict(X_test)
    correct_preds = np.array(test_pred) == y_test
    print("Test error: ", np.sum(correct_preds) / y_test.shape[0])
    """
    C:  1
    Train error:  0.9853479853479854
    Test error:  0.9635036496350365
    """


def KFoldSvm(X, y, n_splits=2, C=1, kernel=None):
    kf = KFold(n_splits=n_splits)
    split = 1
    cv_err_avg = 0
    for train_index, cv_index in kf.split(X):
        print("Split :", split)
        X_t, X_cv = X[train_index], X[cv_index]
        y_t, y_cv = y[train_index], y[cv_index]
        if kernel is None:
            cv_err_avg += Svm(X_t, y_t, X_cv, y_cv, C=C)
        else:
            cv_err_avg += NonLinSvm(X_t, y_t, X_cv, y_cv, C=C, kernel=kernel)
        split += 1
    return cv_err_avg / n_splits


def Svm(X_t, y_t, X_cv, y_cv, C=1):
    svm = LinearSVC(loss="hinge", C=C)
    svm.fit(X_t, y_t)
    train_pred = svm.predict(X_t)
    print("C: ", C)
    correct_preds = np.array(train_pred) == y_t
    print("Train error: ", np.sum(correct_preds) / y_t.shape[0])

    if X_cv is not None:
        cv_pred = svm.predict(X_cv)
        correct_preds = np.array(cv_pred) == y_cv
        cv_err = np.sum(correct_preds) / y_cv.shape[0]
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
    svm.fit(X_t, y_t)
    train_pred = svm.predict(X_t)
    print("C: ", C)
    correct_preds = np.array(train_pred) == y_t
    print("Train error: ", np.sum(correct_preds) / y_t.shape[0])
    
    if X_cv is not None:
        cv_pred = svm.predict(X_cv)
        correct_preds = np.array(cv_pred) == y_cv
        cv_err = np.sum(correct_preds) / y_cv.shape[0]
        print("CV error: ", cv_err)
        return cv_err
    return svm


if __name__ == '__main__':
    SelectInitialModels()
    ChosenHypotheses()
    TestError()
