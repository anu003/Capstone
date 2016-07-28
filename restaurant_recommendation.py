from __future__ import division
import pandas as pd
import numpy as np

class RestaurantRecommendation(object):
    """
    Recommends top 5 restaurants to the user based on their preference on various aspects like taste, price, ambience, service"""

    def __init__(self, df):
        self.df = df
        self.user_id = self.df['user_id'].unique()
        self.restaurant_id = self.df['restaurant_id'].unique()
        self.features = self.df['features']
        self.feature_values = self.df['feature_values']
        self.satisfaction_matrix = self._satisfaction_matrix()

    def _restaurant_feature_score(self, feature, restaurant):
        has_feature = "has_" + feature
        restaurant_ratings = self.df[self.df['business_id'] == restaurant and self.df[has_feature] == True and self.df['text'] != '']
        total_ratings = restaurant_ratings['user_id'].unique().count()
        sum_of_ratings = sum(restaurant_ratings.groupby('user_id').mean()['stars'].values)
        return sum_of_ratings/total_ratings

    def _transform_stars(self, user, feature, restaurant):
        user_restaurant_feature_rating = 0
        rated_feature = []
        MAX = self.df[self.df['user_id'] == user]['stars'].values.max()
        user_rating = self.df[self.df['business_id'] == restaurant and self.df['user_id'] == user and self.df[has_feature] == True].values.mean()
        for feature in self.features:
            has_feature = "has_" + feature
            user_restaurant_feature_rating += self.df[self.df['business_id'] == restaurant and self.df['user_id'] == user and self.df[has_feature] == True]['stars'].values.mean()
            restaurant_feature_ratings = self.df[self.df['business_id'] == restaurant and self.df[has_feature] == True and self.df['text'] != '']
            rated_feature.append(reduce(or, user_restaurant[has_feature].values))
        return user_rating - (user_restaurant_feature_rating/sum(rated_feature)) + MAX

    def _user_feature_concern(self, user, feature):
        has_feature = "has_" + feature
        count_user_feature_rated = self.df[self.df['user_id'] == user and self.df[has_feature] and self.df['text'] != ""]['business_id'].unique().count()
        count_user_rated = self.df[self.df['user_id'] == user and self.df['text'] != ""]['business_id'].unique().count()
        total_restaurants_reviewed = self.df[self.df['text'] != ""]['business_id'].unique().count()
        count_restaurants_feature_reviewed = self.df[self.df[has_feature] == True and self.df['text'] != ""]['business_id'].unique().count()
        return ((count_user_feature_rated + 1 ) * total_restaurants_reviewed) / (count_user_rated * (count_restaurants_feature_reviewed + 1))

    def _avg(self, feature, restaurant):
        has_feature = "has_" + feature
        average_rating = self.df[self.df['business_id'] == restaurant and self.df[has_feature] == True]['stars'].mean()
        return average_rating

    def _delta(self, user, feature, restaurant):
        average_rating = self._avg(feature, restaurant)
        rating = self._transform_stars(user, feature, restaurant)
        if average_rating > rating > 0:
            return (average_rating - rating + 1)/average_rating
        return 1/(average_rating)

    def _requirement(self, user, feature):
        count_user_feature_rated = self.df[self.df['user_id'] == user and self.df[has_feature] and self.df['text'] != ""]['business_id'].unique().count()
        count_feature_rated = self.df[self.df[has_feature] and self.df['text'] != ""]['business_id'].unique().count()
        if count_user_feature_rated:
            return sum([self._delta(user, feature, restaurant) for restaurant in self.restaurant_id])/count_user_feature_rated
        return sum([0.1/self._avg(feature, restaurant) for restaurant in self.restaurant_id])/count_feature_rated

    def _weights(self, user, feature):
        concern = self._user_feature_concern(user, feature)
        requirement = self._user_feature_requirement(user, feature)
        return concern * requirement

    def _satisfaction(self, user, restaurant):
        user_feature_weights = [self._weights(user, feature) for feature in self.features]

        restaurant_feature_score = []
        for restaurant in self.restaurant_id:
            feature_score = []
            for feature in self.features:
                feature_score.append(self.restaurant_feature_score(restaurant, feature))
            restaurant_feature_score.append(feature_score)

        user_feature_concern = []
        for user in self.user_id:
            feature_score = []
            for feature in self.features:
                feature_score.append(self._user_feature_concern(user, feature))
            user_feature_score.append(feature_score)

        return np.dot(user_feature_weights, user_feature_score)

    def _satisfaction_matrix(self):
        rating_matrix = []
        for user in self.user_id:
            user_rating = []
            for restaurant in self.restaurant_id:
                user_rating.append(self._satisfaction(user, restaurant))
            rating_matrix.append(user_rating)
        return _satisfaction_matrix
