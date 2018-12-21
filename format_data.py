import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

def CreateLMRTrainData():
#     stop_words = ["a", "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
#     "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are",
#     "around", "as", "at", "back", "be", "became", "because", "become",
#     "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
#     "below", "beside", "besides", "between", "beyond", "bill", "both",
#     "bottom", "br", "but", "by", "call", "can", "cannot", "cant", "co", "con",
#     "could", "couldnt", "de", "describe", "detail", "do", "done",
#     "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else",
#     "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
#     "everything", "everywhere", "except", "few", "fifteen", "fifty", "fill",
#     "find", "fire", "first", "five", "for", "former", "formerly", "forty",
#     "found", "four", "from", "front", "full", "further", "get", "give", "go",
#     "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter",
#     "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his",
#     "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed",
#     "interest", "into", "is", "it", "its", "its", "itself", "keep", "last", "latter",
#     "latterly", "least", "less", "ltd", "made", "many", "may", "me",
#     "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly",
#     "move", "much", "must", "my", "myself", "name", "namely", "neither",
#     "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone",
#     "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on",
#     "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our",
#     "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps",
#     "please", "put", "rather", "re", "same", "see", "seem", "seemed",
#     "seeming", "seems", "serious", "several", "she", "should", "show", "side",
#     "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone",
#     "something", "sometime", "sometimes", "somewhere", "still", "such",
#     "system", "take", "ten", "than", "that", "the", "their", "them",
#     "themselves", "then", "thence", "there", "thereafter", "thereby",
#     "therefore", "therein", "thereupon", "these", "they", "thick", "thin",
#     "third", "this", "those", "though", "three", "through", "throughout",
#     "thru", "thus", "to", "together", "too", "top", "toward", "towards",
#     "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us",
#     "very", "via", "was", "we", "well", "were", "what", "whatever", "when",
#     "whence", "whenever", "where", "whereafter", "whereas", "whereby",
#     "wherein", "whereupon", "wherever", "whether", "which", "while", "whither",
#     "who", "whoever", "whole", "whom", "whose", "why", "will", "with",
#     "within", "without", "would", "yet", "you", "your", "yours", "yourself",
#     "yourselves"]
    vectorizer = CountVectorizer()

    train_reviews = []
    cv_reviews = []

    path = "./aclImdb/train/pos/"
    reviews = os.listdir(path)
    get_text(reviews, path, train_reviews)
#     get_text(reviews[10000:], path, cv_reviews)

    path = "./aclImdb/train/neg/"
    reviews = os.listdir(path)
    get_text(reviews, path, train_reviews)
#     get_text(reviews[10000:], path, cv_reviews)

    train_data = vectorizer.fit_transform(train_reviews)
    train_names = vectorizer.get_feature_names()

    print(train_data.shape)
    np.save('./train.npy', train_data.toarray())
    np.save('./train_words.npy', np.array(train_names))
#     train_data = None
#     train_names = None

#     vectorizer = CountVectorizer()
#     cv_data = vectorizer.fit_transform(cv_reviews)
#     cv_names = vectorizer.get_feature_names()

#     print(cv_data.shape)
#     np.save('./cv.npy', cv_data.toarray())
#     np.save('./cv_words.npy', np.array(cv_names))
    
    # CAN TOKENIZE NEW VALUES BY PASSING IN PARAMATER preprocessor=names
    # INTO CONSTRUCTOR CountVectorizer(preprocessor=names)

def get_text(reviews, path, all_reviews):
    for i in range(len(reviews)):
        print(i)
        text = open(path + reviews[i], encoding='utf-8').readlines()[0]
        all_reviews.append(text)

def CreateLMRTestData():
    words = np.load('train_words_reduced2.npy')
    vectorizer = CountVectorizer(vocabulary=list(words))

    test_reviews = []

    path = "./aclImdb/test/pos/"
    reviews = os.listdir(path)
    get_text(reviews, path, test_reviews)

    path = "./aclImdb/test/neg/"
    reviews = os.listdir(path)
    get_text(reviews, path, test_reviews)
    #print(np.array(test_reviews))
    test_data = vectorizer.fit_transform(test_reviews)
    np.save('./test_15236.npy', test_data.toarray())
    print(test_data.shape)


def CreateOWBCTrainTestData():
    path = "./breast-cancer-wisconsin.data"
    
    data = []
    y = []
    with open(path) as f:
        
        for line in f:
            if '?' not in line:
                d = line.strip().split(',')
                y_i = d.pop()
                if y_i == '2':
                    y.append(1)
                elif y_i == '4':
                    y.append(0)
                d = d[1:]
                for i in range(len(d)):
                    d[i] = int(d[i])
                print(d)
                data.append(d)
    data = np.array(data)
    y = np.array(y)
    print(data.shape)
    print(y.shape)

    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2)
    np.save('./obcw_train.npy', X_train)
    np.save('./obcw_train_y.npy', y_train)
    np.save('./obcw_test.npy', X_test)
    np.save('./obcw_test_y.npy', y_test)


if __name__ == '__main__':
    CreateLMRTrainData()
    CreateLMRTestData()
    CreateOWBCTrainTestData()
