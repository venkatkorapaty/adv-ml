import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

def main():
    X_t = np.load('lmrd_train.npy')
    y_t = np.load('lmrd_train_y.npy')
    X_cv = np.load('lmrd_cv.npy')
    y_cv = np.load('lmrd_cv_y.npy')

    LogisReg(X_t, y_t, X_cv, y_cv, C=1)
    recursive_feature_elimination(X_t, y_t, X_cv, y_cv)
    #LogisReg(X_t, y_t, X_cv, y_cv, C=0.95)
    #LogisReg(X_t, y_t, X_cv, y_cv, C=0.9)
    #LogisReg(X_t, y_t, X_cv, y_cv, C=0.85)
    #LogisReg(X_t, y_t, X_cv, y_cv, C=0.8)
    #LogisReg(X_t, y_t, X_cv, y_cv, C=0.75)
    #LogisReg(X_t, y_t, X_cv, y_cv, C=0.7)
    #LogisReg(X_t, y_t, X_cv, y_cv, C=0.65)
    #LogisReg(X_t, y_t, X_cv, y_cv, C=0.6)
    #LogisReg(X_t, y_t, X_cv, y_cv, C=0.55)
    #LogisReg(X_t, y_t, X_cv, y_cv, C=0.5)
    #LogisReg(X_t, y_t, X_cv, y_cv, C=0.3)
    #LogisReg(X_t, y_t, X_cv, y_cv, C=0.1)
    #LogisReg(X_t, y_t, X_cv, y_cv, C=0.007)

def LogisReg(X_t, y_t, X_cv, y_cv, C=1):
    lr = LogisticRegression(C=0.75)
    lr.fit(X_t, y_t[:, 0])
    train_pred = lr.predict(X_t)
    cv_pred = lr.predict(X_cv)
    print("C: ", C)
    correct_preds = np.array(train_pred) == y_t[:, 0]
    print("Train error: ", np.sum(correct_preds) / y_t[:, 0].shape[0])
    
    correct_preds = np.array(cv_pred) == y_cv[:, 0]
    print("CV error: ", np.sum(correct_preds) / y_cv[:, 0].shape[0])

def recursive_feature_elimination(X_t, y_t, X_cv, y_cv):
    lr = LogisticRegression()
    selector = RFE(lr, 10000, step=100).fit(X_t, y_t[:, 0])

    train_pred = selector.predict(X_t)
    cv_pred = selector.predict(X_cv)
    correct_preds = np.array(train_pred) == y_t[:, 0]
    print("Train error: ", np.sum(correct_preds) / y_t[:, 0].shape[0])
    
    correct_preds = np.array(cv_pred) == y_cv[:, 0]
    print("CV error: ", np.sum(correct_preds) / y_cv[:, 0].shape[0])    


if __name__ == '__main__':
    main()
