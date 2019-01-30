import nltk
import csv

def get_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features

def get_tweets(tweets):
    all_words = []
    for (words, sentiment) in tweets:
      all_words.extend(words)
    return all_words
    
def take_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

file = 'Training_Set.txt'
with open(file) as f:
    reader = csv.reader(f, delimiter="\t")
    tweet_set = list(reader)

tweets = []

for(words,sentiment) in tweet_set:
	filtered = [e.lower() for e in words.split() if len(e) >= 3]
	tweets.append((filtered, sentiment))
	
word_features = get_features(get_tweets(tweets))

training_set = nltk.classify.apply_features(take_features, tweets)

classifier = nltk.NaiveBayesClassifier.train(training_set)

variable = 'HowlinRays'

t_file = variable + '_input.txt'
f = open(t_file,'r')
tweets = f.read().splitlines()

t_output = variable + '_output.txt'
file = open(t_output, "w")
for tweet in tweets:
	file.write(classifier.classify(take_features(tweet.split())) + '\n')
file.close()
