import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer

def main():
    """
    Gets rid of words that occur at most 5 times in all of the reviews
    """
    X = np.load('train.npy')
    words = np.load('train_words.npy')

    tot_word_freq = np.sum(X, 0)
    most_occ = np.max(np.sum(X, 0))

    print("Most frequent word: ", most_occ, " ", words[tot_word_freq == most_occ])

    five_occ = words[tot_word_freq > 5]
    print(five_occ)

    # gets matrix with words that occur 5 times or less in total, removed
    np.save('./train_reduced.npy', X[:, tot_word_freq > 5])
    np.save('./train_words_reduced.npy', five_occ)


def main2():
    """
    Gets rid of 47 most occuring words in all of the
    articles, asides from good, not and was
    """
    X = np.load('train_reduced.npy')
    words = np.load('train_words_reduced.npy')

    tot_word_freq = np.sum(X, 0)
    most_occ = np.max(np.sum(X, 0))

    print("Most frequent word: ", most_occ, " ", words[tot_word_freq == most_occ])

    fifty_most_freq = words[np.argpartition(tot_word_freq, -50)[-50:]]
    print(fifty_most_freq)

    # removing words based on fifty_most_freq
    for word in fifty_most_freq:
        # three words that may be useful
        if word not in ['good', 'not', 'was']:
            X = X[:, words != word]
            words = words[words != word]
    print(X.shape)
    print(words.shape)
    np.save('./train_reduced2.npy', X)
    np.save('./train_words_reduced2.npy', words)


if __name__ == '__main__':
    # main()
    main2()
