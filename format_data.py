import os
from sklearn.feature_extraction.text import CountVectorizer

def main():
    path = "./aclImdb/train/pos/"
    reviews = os.listdir(path)
    vectorizer = CountVectorizer()

    for review in reviews:
        print(review)
        text = open(path + review).readlines()
        print(text)

        # need to add each text value into overall array
        x = vectorizer.fit_transform(text)
        print(x)
        # gets the numpy array matrix
        x.toarray()
        # gets what word each feature in the matrix is
        vectorizer.get_feature_names()
        break


if __name__ == '__main__':
    main()
