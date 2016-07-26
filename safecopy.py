def preprocess(reviews):
    # Used when tokenizing words
    for review in reviews:
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
        for sent in nltk.sent_tokenize(review):
            term_list = []
            sent = sent.translate(string.maketrans("",""), string.punctuation)
            toks = nltk.regexp_tokenize(sent, sentence_re)
            postoks = nltk.tag.pos_tag(toks) #nltk.tag.pos_tag(toks)
            # print postoks
            tree = chunker.parse(postoks)
            terms = get_terms(tree)
            for term in terms:
                for word in term:
                    term_list.append(word.encode('utf-8')),
            super_list.append(term_list)
        return super_list

def leaves(tree):
        """Finds NP (nounphrase) leaf nodes of a chunk tree."""
        for subtree in tree.subtrees(filter = lambda t: t.label()=='NP'):
            yield subtree.leaves()

def normalise(word):
        """Normalises words to lowercase and stems and lemmatizes it."""
        lemmatizer = WordNetLemmatizer()
        word = word.lower()
    #     word = stemmer.stem_word(word)
        word = lemmatizer.lemmatize(word)
        return word

def acceptable_word(word):
        """Checks conditions for acceptable word: length, stopword."""
        stop = stopwords.words('english')
        stop += ['mr'] + ['mrs'] + ['miss']
        accepted = bool(2 <= len(word) <= 40) and (word.lower() not in stop)
        return accepted

def get_terms(tree):
    for leaf in leaves(tree):
        term = [normalise(w) for w,t in leaf if acceptable_word(w)]
        yield term

def compactness_pruning(features):
    """Checks if there are more than two words between the words of a feature in a review"""
    feature_phrases = [feature for feature in features if len(feature.split()) > 1]
    feature_words = [feature for feature in features if len(feature.split()) == 1]
    for review in reviews:
        for sent in nltk.sent_tokenize(review):



if __name__ == "__main__":
    reviews = pd.read_pickle('../data/review_data.pkl')
    terms = preprocess(reviews.text.values[:10])
    C1 = apriori.createC1(terms)
    D = map(set, terms)
    L1, support_data = apriori.scanD(D,C1,0.01) # minimum support 0.01
