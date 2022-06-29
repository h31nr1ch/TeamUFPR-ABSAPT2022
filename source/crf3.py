from itertools import chain
import nltk
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import RandomizedSearchCV
import math, string, re
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
import pycrfsuite
import spacy
nlp = spacy.load('pt_core_news_md')
from sklearn.ensemble import RandomForestClassifier

# First word2features
def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]

## Second word2features
def word2features_2(sent, i):
    word = sent[i][0]

    features = {
        'bias': 1.0,
        'word': word,
        'len(word)': len(word),
        'word[:4]': word[:4],
        'word[:3]': word[:3],
        'word[:2]': word[:2],
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word[-4:]': word[-4:],
        'word.lower()': word.lower(),
        'word.stemmed': re.sub(r'(.{2,}?)([aeiougyn]+$)',r'\1', word.lower()),
        'word.ispunctuation': (word in string.punctuation),
        'word.isdigit()': word.isdigit(),
    }
    if i > 0:
        word1 = sent[i-1][0]
        features.update({
            '-1:word': word1,
            '-1:len(word)': len(word1),
            '-1:word.lower()': word1.lower(),
            '-1:word.stemmed': re.sub(r'(.{2,}?)([aeiougyn]+$)',r'\1', word1.lower()),
            '-1:word[:3]': word1[:3],
            '-1:word[:2]': word1[:2],
            '-1:word[-3:]': word1[-3:],
            '-1:word[-2:]': word1[-2:],
            '-1:word.isdigit()': word1.isdigit(),
            '-1:word.ispunctuation': (word1 in string.punctuation),
        })
    else:
        features['BOS'] = True

    if i > 1:
        word2 = sent[i-2][0]
        features.update({
            '-2:word': word2,
            '-2:len(word)': len(word2),
            '-2:word.lower()': word2.lower(),
            '-2:word[:3]': word2[:3],
            '-2:word[:2]': word2[:2],
            '-2:word[-3:]': word2[-3:],
            '-2:word[-2:]': word2[-2:],
            '-2:word.isdigit()': word2.isdigit(),
            '-2:word.ispunctuation': (word2 in string.punctuation),
        })

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        features.update({
            '+1:word': word1,
            '+1:len(word)': len(word1),
            '+1:word.lower()': word1.lower(),
            '+1:word[:3]': word1[:3],
            '+1:word[:2]': word1[:2],
            '+1:word[-3:]': word1[-3:],
            '+1:word[-2:]': word1[-2:],
            '+1:word.isdigit()': word1.isdigit(),
            '+1:word.ispunctuation': (word1 in string.punctuation),
        })

    else:
        features['EOS'] = True
        if i < len(sent) - 2:
            word2 = sent[i+2][0]
            features.update({
            '+2:word': word2,
            '+2:len(word)': len(word2),
            '+2:word.lower()': word2.lower(),
            '+2:word.stemmed': re.sub(r'(.{2,}?)([aeiougyn]+$)',r'\1', word2.lower()),
            '+2:word[:3]': word2[:3],
            '+2:word[:2]': word2[:2],
            '+2:word[-3:]': word2[-3:],
            '+2:word[-2:]': word2[-2:],
            '+2:word.isdigit()': word2.isdigit(),
            '+2:word.ispunctuation': (word2 in string.punctuation),
        })

    return features


def sent2features_2(sent):
    return [word2features_2(sent, i) for i in range(len(sent))]

def if_O(value):
    if value == 'O':
        return 1
    else:
        return 0

def sent2labels_2(sent):
    a = [word[1] for word in sent]
    # print('word', [word[1] for word in sent])
    # print('label', [label for token, postag, label in sent])
    b = [label for token, postag, label in sent]
    # print([v1 + " " + v2 for v1, v2 in zip(a, b)])
    c = [v1 + "|" + v2 for v1, v2 in zip(a, b)]
    # print(*zip(a,b))
    ll = []
    for Va, Vb in zip(a,b):
        if(if_O(Vb) == 0):
            ll.append(Vb)
        else:
            ll.append(Va)

    # print(ll)
    return ll

def sent2tokens_2(sent):
    # return [word[0] for word in sent]
    return [label for token, postag, label in sent]

##
print('Read Dataset.')
# train_sents = list(nltk.corpus.conll2002.iob_sents('esp.train'))
# train_example = list(nltk.corpus.conll2002.iob_sents('esp.train'))
# test_sents = list(nltk.corpus.conll2002.iob_sents('esp.testb'))

# print(test_sents)

###
import pandas as pd

def tag_word(tagged_words, word, polarity, word_start_idx, aspect_start_pos, aspect_end_pos):
    tag = 'O'
    if word_start_idx >= aspect_start_pos and word_start_idx <= aspect_end_pos:
        # Check if previous word is part of the aspect
        if len(tagged_words) > 0 and tagged_words[-1][1] in ['B-ASP', 'I-ASP']:
            # If previous is part of aspect, tag current as I
            tag = 'I-ASP'
        else:
            # Else, this word starts the aspect, so tag it as B
            tag = 'B-ASP'
    converted_polarity = 'O' # -999
    if tag != 'O':
        if polarity == -1:
            converted_polarity = 'Negative'
        elif polarity == 1:
            converted_polarity = 'Positive'
        else:
            converted_polarity = 'Neutral'
    tagged_words.append((word, tag, converted_polarity))

def magic_iob(tokens, review, aspect_start_idx, aspect_end_idx):
    iob_tokens = []
    token_idx = 0
    review_idx = 0
    current_token = tokens[token_idx][0]
    while token_idx < len(tokens):
        current_token = tokens[token_idx][0]

        # if our current token starts at this position in the review
        if review[review_idx] == current_token[0]:
            # tag the token with IOB label
            label = 'O'
            if review_idx >= aspect_start_idx and review_idx <= aspect_end_idx:
                if len(iob_tokens) > 0 and iob_tokens[-1][2] != 'O':
                    label = 'I-ASP'
                else:
                    label = 'B-ASP'
            iob_tokens.append((*tokens[token_idx], label))

            # update the position using the token
            review_idx += len(current_token)

            # get next token
            token_idx += 1
        else:
            review_idx += 1

    return iob_tokens

###
# print(train_example[0])
# print(train_sents[0])

from nltk.test.portuguese_en_fixt import setup_module
# palavras_tokenize = tokenize.word_tokenize(text, language='portuguese')

data_train = pd.read_csv('../dataset/train.csv', delimiter=';', index_col='id')
data_test = pd.read_csv('../dataset/test/test_task1.csv', delimiter=';', index_col='id')

# print(data_train)
# raw_data = list(data_train['review'])
raw_data = data_train

grammar = r"""
  NP: {<DT|JJ|NN.*>+}          # Chunk sequences of DT, JJ, NN
  PP: {<IN><NP>}               # Chunk prepositions followed by NP
  VP: {<VB.*><NP|PP|CLAUSE>+$} # Chunk verbs and their arguments
  CLAUSE: {<NP><VP>}           # Chunk NP, VP
  """
cp = nltk.RegexpParser(grammar)

import joblib
from nltk import word_tokenize

# https://github.com/inoueMashuu/POS-tagger-portuguese-nltk
folder = 'trained_POS_taggers/'
teste_tagger = joblib.load('POS-tagger-portuguese-nltk/trained_POS_taggers/'+'POS_tagger_brill.pkl')
# teste_tagger = joblib.load('POS-tagger-portuguese-nltk/trained_POS_taggers/'+'POS_tagger_naive.pkl')

def convert_ie_preprocess(each_string):
    # print(each_string)
    newT = ''.join(c for c in ''.join(each_string) if c not in string.punctuation)
    # pat = r'[^a-zA-z0-9.,!?/:;\"\'\s]'
    # newT = re.sub(pat, ' ', newT)
    # print(newT)

    # sentences = nltk.sent_tokenize(each_string)
    # sentences = [nltk.word_tokenize(sent, language='portuguese') for sent in sentences]
    # sentences = [nltk.pos_tag(sent) for sent in sentences]
    # print('EN', sentences)

    sentences = [teste_tagger.tag(word_tokenize(newT))]
    # print('PT', sentences)
    # sentences = cp.parse(sentences)
    return sentences

raw_data_test = []
for each in list(data_test['review']):
    raw_data_test.append(list(chain(*convert_ie_preprocess(each))))
# print(raw_data_test)

preprocess_train_raw = []

for _, row in raw_data.iterrows():
    review, _, _, start_pos, end_pos = row
    tokens = list(chain(*convert_ie_preprocess(review)))
    iob_tokens = magic_iob(tokens, review, start_pos, end_pos)

    preprocess_train_raw.append(iob_tokens)

# print(preprocess_train_raw)
# train_sents = preprocess_train_raw[0:2500] # here
# test_sents = preprocess_train_raw[2501:] # here
#
train_sents = preprocess_train_raw # here
test_sents = list(raw_data_test) # here

# print(raw_data_test)
X_test = raw_data_test

###
print('Pre-processing..')

X_train = [sent2features_2(s) for s in train_sents]
# X_train = [sent2tokens(s) for s in train_sents]
y_train = [sent2labels_2(s) for s in train_sents]

X_test = [sent2features_2(s) for s in test_sents]
# X_test = [sent2tokens(s) for s in test_sents]
# y_test = [sent2labels_2(s) for s in test_sents] # here

crf = sklearn_crfsuite.CRF(
    algorithm='arow', # 'lbfgs', 'l2sgd', 'ap', 'pa', 'arow'
    # c1=0.25,
    # c2=0.3,
    max_iterations=300,
    all_possible_transitions=True
)

crf.fit(X_train, y_train)

labels = list(crf.classes_)
# labels.remove('O')
print(labels)

print('Predict....')
y_pred = crf.predict(X_test)

# print(y_pred) # here
print("\"input id number\";\"list of aspects\"")
for i in range(len(raw_data_test)):
    word = []
    for idx, token in enumerate(raw_data_test[i]):
        if 'B-ASP' in y_pred[i][idx] or 'I-ASP' in y_pred[i][idx]:
            word.append(token[0])
    # print(i, " ".join(word))
    print(i, ";", "\"", word, "\"", sep="")

