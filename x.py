Extract features from reviews

rk = (s1k, s2k, .. , sLk) where L is tot number of features and 1 <= k <= K

Sjk = (1/njk) * sum(sijk)
where njk is the number of users who rated restaurant rk on feature
fj belongs to F

Calculate concern and requirement for each user which results in weights of preferences

concern(ui, fj) = ((count_r(ui, fi) + 1)/count_r(ui)) * Nr/(count_r(fi)+1)

count_r(ui, fj) represents the number of restaurants for which
the user ui rated feature fj, count_r(ui) is the number of restaurants
that user ui reviewed, Nr is the number of restaurants with customer
reviews, and count_r(fj) is the number of restaurants with the feature
fj being commented on

kt5MZ368FrySa2S0PB1yqg  Vhc7JJd5S347VW1s2x0F6Q 7

def combine_columns(row):
    combo = []
    scores = row['scores']
    topics = row['topics']
    for score, topic in izip(scores, topics):
        combo.append((score, topic))
    return combo

combo = []
for topic, score in izip(test_topics, test_scores):
    combo.append((topic, score))


s = data_new.apply(lambda x: pd.Series(x['topics_scores']),axis=1).stack().reset_index(level=1, drop=True)
