import numpy as np
from sklearn.linear_model import LogisticRegression

def main():
    X_t = np.load('lmrd_train.npy')
    y_t = np.load('lmrd_train_y.npy')
    X_cv = np.load('lmrd_cv.npy')
    y_cv = np.load('lmrd_cv_y.npy')

    lr = LogisticRegression()
    lr.fit(X_t, y_t[:, 0])
    print(lr)
    train_pred = lr.predict(X_t)
    cv_pred = lr.predict(X_cv)
    
    correct_preds = np.array(train_pred) == y_t[:, 0]
    print("Train error: ", np.sum(correct_preds) / y_t[:, 0].shape[0])
    
    correct_preds = np.array(cv_pred) == y_cv[:, 0]
    print("CV error: ", np.sum(correct_preds) / y_cv[:, 0].shape[0])
    


if __name__ == '__main__':
    main()
