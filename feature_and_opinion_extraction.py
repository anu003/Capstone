from spacy.en import English
from pattern.en import lemma, sentiment
from spacy.parts_of_speech import NOUN, VERB, ADV, ADJ
from nltk.corpus import stopwords
import apriori
import pandas as pd
import numpy as np

class FeatureAndOpinionExtractor(object):
    """
    Extracts Features and Opinions from reviews.
    """
    def __init__(self, data, lang):
        self.data = data
        self.nlp = lang()
        self.frequent_features = []
        self.feature_phrases = []
        self.feature_words = []
        self.features = []
        self._preprocess()
        self._get_sentiment()

    def _preprocess(self):
        """
        Preprocesses the data and calls the functions to extract features and its opinions.
        """
        self.data['sentences'] = self.data['text'].apply(self._tokenize_sent)
        self.data['nouns'] = self.data['sentences'].apply(self._get_nouns)
        # self._get_frequent_features()
        # self._compactness_pruning()
        # self._redundancy_pruning()
        # self._get_features()
        self._extract_opinions()

    def _tokenize_sent(self, review):
        """
        input : string
        output : list
        Returns list of sentences of a review.
        """
        return review.decode('ascii','ignore').split('.')

    def _get_nouns(self, review):
        """
        Returns features(nouns) from each sentence of a review.
        """
        review_features = []
        for sent in review:
            doc = self.nlp(sent)
            # noun_phrase = [np.text for np in doc.noun_chunks]
            nouns = [unicode(lemma(str(word).lower())) for word in doc if word.pos == NOUN]
            review_features.append(nouns)
        return review_features

    def _get_frequent_features(self):
        """Frequent Features are found using apriori algorithm"""
        feature_terms = [sub_items for items in self.data['noun_and_np'].values for sub_items in items]
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
            temp_fp = self.data[self.data['text'].str.contains(feature_phrase)]
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
        temp_fw = self.data[self.data['text'].str.contains(ftr)]
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
        """
        input : string, string
        output : list of strings
        Returns the list of words that has parts of speech as given in the input.
        """
        pos_list = []
        stop = stopwords.words('english')
        for sent in review:
            doc = self.nlp(unicode(sent))
            pos_ext = [unicode(word) for word in doc if word.pos == pos and str(word).lower().encode('utf-8') not in stop]
            pos_list.append(pos_ext)
        return pos_list

    def _extract_opinions(self):
        """
        Extracts adjectives, adverbs, verbs for each sentence of a review.
        """
        self.data['adjectives'] = self.data['sentences'].apply(lambda x: self._extract_pos(x, ADJ))
        self.data['adverbs'] = self.data['sentences'].apply(lambda x: self._extract_pos(x, ADV))
        self.data['verbs'] = self.data['sentences'].apply(lambda x: self._extract_pos(x, VERB))

    def _get_polarity(self):
        self.data['polarity'] = self.data['sentences'].apply(lambda x: [sentiment(i) for i in x])
        polarities = [polarity for sent_polarities in self.data['polarity'].values for polarity in sent_polarities]
        self._get_normalized_score(polarities)

    def _get_normalized_score(self, polarities):
        ind = 0
        scores = []
        normalized_scores = pd.cut(polarities, bins=9, right=True, labels = [1,1.5,2,2.5,3,3.5,4,4.5,5],
                    retbins=False, precision=2, include_lowest=True)
        for review in self.data['sentences']:
            l = len(review)
            scores.append(normalized_scores[ind:ind+l])
            ind += l
        self.data['scores'] = np.array(scores)

# if __name__ == "__main__":
#     reviews = pd.read_pickle("../data/cleaned_review_data.pkl")
#     foe = FeatureAndOpinionExtractor(reviews, English)
# reviews = pd.read_pickle("../data/italian_cleaned_review_data.pkl")
