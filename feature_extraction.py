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
        self.df['noun_and_np'] = self.df['sentences'].apply(self._preprocess)
        self.frequent_features = []
        self.feature_phrases = []
        self.feature_words = []
        self.features = []

    def _tokenize_sentences(self):
        """Tokenizes the reviews into list of sentences."""
        self.df['sentences'] = self.df['text'].apply(lambda x: nltk.sent_tokenize(x))

    def _leaves(self):
        """Finds NP (nounphrase) leaf nodes of a chunk tree."""
        for subtree in self.tree.subtrees(filter = lambda t: t.label()=='NP'):
            yield subtree.leaves()

    def _preprocess(self, review):

        # Used when tokenizing words
        sentence_re = r'(?:(?:[A-Z])(?:.[A-Z])+.?)|(?:\w+(?:-\w+)*)|(?:\$?\d+(?:.\d+)?%?)|(?:...|)(?:[][.,;"\'?():-_`])'
        # stemmer = nltk.stem.porter.PorterStemmer()

        #Taken from Su Nam Kim Paper...
        # grammar = r"""
        #     NBAR:
        #         {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns
        #
        #     NP:
        #         {<NBAR>}
        #         {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
        # """
        grammar = "NP: {<DT>?<JJ.*>*<NN.*>+}"
        chunker = nltk.RegexpParser(grammar)
        super_list = []
        for sent in review:
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
        return super_list

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
        """Yeilds frequent items"""
        for leaf in self._leaves():
            term = [self._normalise(w) for w,t in leaf if self._acceptable_word(w)]
            yield term

    def _get_frequent_features(self):
        """Frequent Features are found using apriori algorithm"""
        feature_terms = [sub_items for items in self.df['noun_and_np'].values for sub_items in items]
        C1 = apriori.createC1(feature_terms)
        D = map(set, feature_terms)
        L1, support_data = apriori.scanD(D,C1,0.01) # minimum support 0.01
        self.frequent_features = map(lambda x: "".join(list(x)), L1)

    def _distance(self, sentence, feature_phrase):
        """Returns True if distance between words is less than or equals to 3 else False"""
        words = feature_phrase.split()
        if len(words) == 2:
            if sentence.find(word[0]) != -1 and sentence.find(word[1]) != -1:
                if len(sentence[sentence.find(word[0]) + len(word[0]):sentence.find(word[1])].split()) <= 3:
                    return True
                return False
            return False
        else:
            if len(sentence[sentence.find(word[0]) + len(word[0]):sentence.find(word[1])].split()) <=3 and \
                    len(sentence[sentence.find(word[1]) + len(word[1]):sentence.find(word[2])].split()) <= 3:
                return True
            return False

    def _is_compact(self, feature_phrase):
        """
        input : string
        output : bool
        Returns whether the input feature phrase is compact or not
        """
        count = 0
        if 1 < len(feature_phrase.split()) <= 3:
            temp_fp = self.df[self.df['text'].str.contains(feature_phrase)]
            for review in temp_fp['sentences'].values:
                for sent in review:
                    if self._distance(sent, feature_phrase):
                        count += 1
                    if count == 2:
                        return True
            return False
        else:
            return False

    def _compactness_pruning(self):
        """Checks if there are more than two words between the words of a feature in a review"""
        feature_phrases = [phrase for phrase in self.frequent_features if self._is_compact(phrase)]
        self.features_phrases = feature_phrases

    def _is_redundant(self, ftr, phrase_list):
        """input: string, list"""
        """output: bool"""
        """Returns whether the input feature is redundant or not"""
        temp_fw = self.df[self.df['text'].str.contains(ftr)]
        if phrase_list:
            for n in temp_fw['noun_and_np'].values:
                count = 0
                for phrase in phrase_list:
                    if frozenset(phrase).issubset(frozenset(n)):
                        break
                count += 1
                if count == 3:
                    return True
            return False
        else:
            if temp_fw.count()['text'] >= 3:
                return True
            return False


    def _redundancy_pruning(self):
        """Prunes redundant single word features"""
        feature_words = [feature for feature in self.frequent_features if len(feature.split()) == 1]
        for ftr in feature_words:
            phrase_list = []
            if self.feature_phrases:
                for phrase in self.feature_phrases:
                    if ftr in phrase:
                        phrase_list.append(phrase)
            if self._is_redundant(ftr, phrase_list):
                self.feature_words.append(ftr)



    def _get_features(self):
        self.features = self.feature_words + self.feature_phrases
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

#
#
# if __name__ == "__main__":
#     # reviews = pd.read_pickle('../data/review_data.pkl')
    #   self._get_frequent_features()
#     # terms = preprocess(reviews.text.values[:10])
#     # C1 = apriori.createC1(terms)
#     # D = map(set, terms)
#     # L1, support_data = apriori.scanD(D,C1,0.01) # minimum support 0.01
