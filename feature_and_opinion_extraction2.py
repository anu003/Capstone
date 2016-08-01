from spacy.en import English
from spacy.parts_of_speech import NOUN, VERB, ADV, ADJ
from nltk.corpus import stopwords
import apriori
import pandas as pd
import numpy as np

class FeatureAndOpinionExtractor(object):

    def __init__(self, df, lang):
        self.df = df
        self.nlp = lang()
        self.frequent_features = []
        self.feature_phrases = []
        self.feature_words = []
        self.features = []
        self._preprocess()

    def _preprocess(self):
        self.df['sentences'] = self.df['text'].apply(self._tokenize_sent)
        self.df['noun_and_np'] = self.df['sentences'].apply(self._get_nouns_np)
        self._get_frequent_features()
        self._compactness_pruning()
        self._redundancy_pruning()
        self._get_features()
        self._extract_opinions()

    def _tokenize_sent(self, review):

        doc = self.nlp(review.decode('utf-8'), parse = True)
        sents = []
        # the "sents" property returns spans
        # spans have indices into the original string
        # where each index value represents a token
        for span in doc.sents:
            # go from the start to the end of each span, returning each token in the sentence
            # combine each token using join()
            sent = ''.join(doc[i].string for i in range(span.start, span.end)).strip()
            sents.append(sent.decode('utf-8'))
        return sents

    def _get_nouns_np(self, review):
        review_features = []
        for sent in review:
            doc = self.nlp(sent.decode('utf-8'))
            noun_phrase = [np.text for np in doc.noun_chunks]
            nouns = [unicode(word) for word in doc if word.pos == NOUN]
            review_features.append(nouns + noun_phrase)
        return review_features

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
            if sentence.find(words[0]) != -1 and sentence.find(words[1]) != -1:
                if len(sentence[sentence.find(words[0]) + len(words[0]):sentence.find(words[1])].split()) <= 3:
                    return True
                return False
            return False
        else:
            if len(sentence[sentence.find(words[0]) + len(words[0]):sentence.find(words[1])].split()) <=3 and \
                    len(sentence[sentence.find(words[1]) + len(words[1]):sentence.find(words[2])].split()) <= 3:
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
        stop = set(stopwords.words('english'))
        features = self.feature_words + self.feature_phrases
        self.features = [feature for feature in features if feature not in stop]

    def _remove_stop_words(self, review):
        review_list = []
        stop = stopwords.words('english')
        for sent in review:
            sent_list = []
            for item in sent:
                if item.lower() not in stop:
                    sent_list.append(item)
            review_list.append(sent_list)
        return review_list

    def _extract_pos(self, review, pos):
        pos_list = []
        stop = stopwords.words('english')
        for sent in review:
            doc = self.nlp(sent.decode('utf-8'))
            pos_ext = [unicode(word) for word in doc if word.pos == pos and str(word).lower().encode('utf-8') not in stop]
            pos_list.append(pos_ext)
        return pos_list

    def _extract_opinions(self):
        concerns = []
        for review in self.df['sentences'].values:
            user_concerns = []
            for sent in review:
                concern = frozenset(sent.split()).intersection(frozenset(self.features))
                user_concerns.append(concern)
            concerns.append(user_concerns)
        self.df['concerns'] = np.array(concerns)
        self.df['adjectives'] = self.df['sentences'].apply(lambda x: self._extract_pos(x, ADJ))
        self.df['adverbs'] = self.df['sentences'].apply(lambda x: self._extract_pos(x, ADV))
        self.df['verbs'] = self.df['sentences'].apply(lambda x: self._extract_pos(x, VERB))

# if __name__ == "__main__":
#     reviews = pd.read_pickle("reviews_15.pkl")
#     nnp = FeatureAndOpinionExtractor(reviews, English)
