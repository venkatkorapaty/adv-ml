import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer

def main():
    X = np.load('train.npy')
    words = np.load('train_words.npy')

    tot_word_freq = np.sum(X, 0)
    most_occ = np.max(np.sum(X, 0))

    print("Most frequent word: ", most_occ, " ", words[tot_word_freq == most_occ])

    fifty_most_freq = words[np.argpartition(tot_word_freq, -50)[-50:]]
    print(fifty_most_freq)

    five_occ = words[(tot_word_freq <= 5) == False]
    print(five_occ)

    # gets matrix with words that occur 5 times or less in total, removed
    np.save('./train_reduced.npy', X[:, (tot_word_freq <= 5) == False])
    np.save('./train_words_reduced.npy', five_occ)

if __name__ == '__main__':
    main()
