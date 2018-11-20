import numpy as np
import random
from sklearn.model_selection import train_test_split

def main():
    X = np.load('train_reduced2.npy')

    amount_of_words = X.shape[1]
    amount_of_data = X.shape[0]

    size_of_cv = 5000
    middle = 12500

    X_cv = np.empty([size_of_cv, amount_of_words])
    y_cv = np.empty([size_of_cv, 1])

    zeroes = 0
    ones = 0

    for i in range(size_of_cv):
        print(i)
        rand_review = random.randint(0, amount_of_data - 1)

        X_cv[i, :] = X[rand_review, :]
        X = np.delete(X, i, 0)

        if (rand_review < middle):
            ones = ones + 1
            y_cv[i, :] = 1
            middle = middle - 1
        else:
            zeroes = zeroes + 1
            y_cv[i, :] = 0

        amount_of_data = amount_of_data - 1

    np.save('./cv_only_reduced.npy', X_cv)
    np.save('./train_only_reduced.npy', X)
    print("Zeroes: ", zeroes, " Ones: ", ones, " middle: ", middle)
    print("X: ", X.shape, " amount of data: ", amount_of_data)
    print("X_cv: ", X_cv.shape)

def main2():
    X = np.load('train_reduced2.npy')
    ones = np.ones((12500, 1))
    zeros = np.zeros((12500, 1))
    y = np.atleast_2d(np.append(ones, zeros)).T
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    print(X_train.shape)
    print(X_val.shape)
    np.save('./lmrd_train.npy', X_train)
    np.save('./lmrd_cv.npy', X_val)
    np.save('./lmrd_train_y.npy', y_train)
    np.save('./lmrd_cv_y.npy', y_val)

if __name__ == '__main__':
    #main()
    main2()
