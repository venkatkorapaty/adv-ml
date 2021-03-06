import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer

def RemoveInfrequentWords():
    """
    Gets rid of words that occur at most 15 times in all of the reviews
    """
    X = np.load('train.npy')
    words = np.load('train_words.npy')

    tot_word_freq = np.sum(X, 0)
    most_occ = np.max(np.sum(X, 0))

    print("Most frequent word: ", most_occ, " ", words[tot_word_freq == most_occ])

    fifteen_occ = words[tot_word_freq > 15]
    print(fifteen_occ)

    # gets matrix with words that occur 5 times or less in total, removed
    np.save('./train_reduced15.npy', X[:, tot_word_freq > 15])
    np.save('./train_words_reduced15.npy', fifteen_occ)


def RemoveFrequentWords():
    """
    Gets rid of 48 most occuring words in all of the
    articles, asides from good and bad
    """
    X = np.load('train_reduced15.npy')
    words = np.load('train_words_reduced15.npy')

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
    RemoveInfrequentWords()
    RemoveFrequentWords()
