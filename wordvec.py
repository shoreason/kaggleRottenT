import pandas as pd

# modules for data cleaning
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import nltk.data
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)
from gensim.models import word2vec

# word2vec parameters

# Word vector dimensionality
num_features = 300
# Minimum word count
min_word_count = 40
# Number of threas to run in parallel
num_workers = 4
# context window size
context = 10
# Downsample setting for frequent words
downsampling = 1e-3 

# Read data from files
train = pd.read_csv("data/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)

test = pd.read_csv("data/testData.tsv", header=0, delimiter="\t", quoting=3)

unlabeled_train = pd.read_csv( "data/unlabeledTrainData.tsv", header=0, 
 delimiter="\t", quoting=3 )

# load the punkt tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# verify the number of reviews that were read (100,000 in total)
print("Read %d labeled train reviews, %d labeled test reviews, and %d unlabeled reviews\n" % (train["review"].size, test["review"].size, unlabeled_train["review"].size))

def review_to_wordlist(review, remove_stopwords=False):
    # remove html
    review_text = BeautifulSoup(review, "html.parser").get_text()

    #remove non-letters
    review_text = re.sub("[^a-zA-Z]", " ", review_text)

    # convert words to lower case and split them
    words = review_text.lower().split()

    # optionally remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]

    return(words)
    
def review_to_sentences(review, tokenizer, remove_stopwords=False):

    # use nltk tokenizer to split paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())

    sentences = []
    for raw_sentence in raw_sentences:
        # if sentence is empty, skip it
        if len(raw_sentence) > 0:
            # otherwise, call review_to_wordlist tp hey a list of words
            sentences.append(review_to_wordlist(raw_sentence, remove_stopwords))

    return sentences

# initialize an empty list of sentences
sentences = []

print("Parsing sentences from training set")
for review in train["review"]:
    sentences += review_to_sentences(review, tokenizer)

print("Parsing sentences from unlabeled set")
for review in unlabeled_train["review"]:
    sentences += review_to_sentences(review, tokenizer)

print(" ")
print(len(sentences))

print("first sentence " + str(sentences[0]))

print("second sentence " + str(sentences[1]))

# train and save model
print("Training model ...")
model = word2vec.Word2Vec(sentences,
                          workers = num_workers,
                          size = num_features,
                          min_count = min_word_count,
                          window = context,
                          sample = downsampling)

# init_sims makes model much more memory-efficient

model.init_sims(replace=True)

# save model for later use
model_name = "300features_40minwords_10context"
model.save(model_name)
