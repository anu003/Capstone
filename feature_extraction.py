from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
from autocorrect import spell
from nltk.util import ngrams
from nltk.corpus import stopwords
import itertools, nltk, re, pprint, string, os, pickle
import pandas as pd
import numpy as np
import apriori

class FeatureExtraction(object):

    def __init__(self, df):
        self.df = df
        self._tokenize_sentences()

    def _tokenize_sentences(self):
        self.df['sentences'] = self.df['text'].apply(lambda x: nltk.sent_tokenize(x))

    def preprocess(self):

        # Used when tokenizing words
        sentence_re = r'(?:(?:[A-Z])(?:.[A-Z])+.?)|(?:\w+(?:-\w+)*)|(?:\$?\d+(?:.\d+)?%?)|(?:...|)(?:[][.,;"\'?():-_`])'
        # stemmer = nltk.stem.porter.PorterStemmer()

        #Taken from Su Nam Kim Paper...
        grammar = r"""
            NBAR:
                {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns

            NP:
                {<NBAR>}
                {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
        """
        chunker = nltk.RegexpParser(grammar)
        super_list = []
        for sent in self.df['sentences'].values:
            term_list = []
            sent = sent.translate(string.maketrans("",""), string.punctuation)
            toks = nltk.regexp_tokenize(sent, sentence_re)
            postoks = nltk.tag.pos_tag(toks) #nltk.tag.pos_tag(toks)
            # print postoks
            self.tree = chunker.parse(postoks)
            terms = self._get_terms()
            for term in terms:
                for word in term:
                    term_list.append(word.encode('utf-8')),
            super_list.append(term_list)
        self.feature_terms = super_list

    def _leaves(self):
        """Finds NP (nounphrase) leaf nodes of a chunk tree."""
        for subtree in self.tree.subtrees(filter = lambda t: t.label()=='NP'):
            yield subtree.leaves()

    def _normalise(self, word):
            """Normalises words to lowercase and stems and lemmatizes it."""
            lemmatizer = WordNetLemmatizer()
            word = word.lower()
        #     word = stemmer.stem_word(word)
            word = lemmatizer.lemmatize(word)
            return word

    def _acceptable_word(self, word):
            """Checks conditions for acceptable word: length, stopword."""
            stop = stopwords.words('english')
            stop += ['mr'] + ['mrs'] + ['miss']
            accepted = bool(2 <= len(word) <= 40) and (word.lower() not in stop)
            return accepted

    def _get_terms(self):
        for leaf in _leaves():
            term = [self._normalise(w) for w,t in leaf if self._acceptable_word(w)]
            yield term

    def _get_frequent_terms(self):
        C1 = apriori.createC1(self.feature_terms)
        D = map(set, self.feature_terms)
        L1, support_data = apriori.scanD(D,C1,0.01) # minimum support 0.01
        self.frequent_features = L1

    def _is_compact(self, feature_phrase):
        if 1 < len(feature_phrase.split()) <= 3:
            temp_fp = self.df[frozenset(feature_phrase.split()).issubset(frozenset(self.df['text'].split()))]
            for review in temp_fp['sentences'].values:
                for sent in review:
                    for word in feature_phrase.split():
                        if not sent.find(word):
                            break
                        break


        else:
            return False

    def _compactness_pruning(self):
        """Checks if there are more than two words between the words of a feature in a review"""
        feature_phrases = [phrase for feature in feature_phrases if _is_compact(phrase)]
        # feature_phrases = [feature for feature in self.frequent_features if 3 <= len(feature.split()) > 1 and]
        # feature_words = [feature for feature in self.frequent_features if len(feature.split()) == 1]
        # for fp in feature_phrases:
        #     word_pos = []
        #     temp_fp = self.df[frozenset(fp.split()).issubset(frozenset(self.df['text'].split()))]
        #     for word in fp.split():
        #         for review in temp_fp['sentences'].values:
        #             for sent in review:
        #                 word_pos.append(sent.find(word))
        #
        #     if temp_fp.count() >= 2:
        #



if __name__ == "__main__":
    reviews = pd.read_pickle('../data/review_data.pkl')
    terms = preprocess(reviews.text.values[:10])
    C1 = apriori.createC1(terms)
    D = map(set, terms)
    L1, support_data = apriori.scanD(D,C1,0.01) # minimum support 0.01
