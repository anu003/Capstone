from __future__ import division
import pandas as pd
import numpy as np

class RestaurantRecommender(object):
    """
    Recommends top 5 restaurants to the user based on their preference on various
    aspects like taste, price, ambiance, service.
    """

    def __init__(self, data):
        self.data = data
        self.user_id = self.data['user_id'].unique()
        self.restaurant_id = self.data['business_id'].unique()
        self.features = self.data['topics']
        self.feature_values = self.data['scores']
        self.satisfaction_matrix = self._satisfaction_matrix()

    def _restaurant_feature_score(self, feature, restaurant):
        restaurant_ratings = self.data[(self.data['business_id'] == restaurant) & (self.data['topics'] == feature)]
        total_ratings = len(restaurant_ratings['user_id'].unique())
        sum_of_ratings = restaurant_ratings.groupby('user_id').mean()['scores'].sum()
        return sum_of_ratings/total_ratings

    def _transform_scores(self, user, feature, restaurant):
        user_restaurant_feature_rating = 0
        rated_feature = []
        MAX = self.data[self.data['user_id'] == user]['scores'].max()
        user_rating = self.data[(self.data['business_id'] == restaurant) & (self.data['user_id'] == user) & (self.data['topics'] == feature)].mean()
        for feature in self.features:
            user_rated = self.data[(self.data['business_id'] == restaurant) & (self.data['user_id'] == user) & (self.data['topics'] == feature)]
            user_restaurant_feature_rating += user_rated['scores'].mean()
            restaurant_feature_ratings = self.data[(self.data['business_id'] == restaurant) & (self.data['topics'] == feature)]
            rated_feature.append(user_rated['scores'].count())
        return user_rating - (user_restaurant_feature_rating/sum(rated_feature)) + MAX

    def _user_feature_concern(self, user, feature):
        user_rated_feature = self.data[(self.data['user_id'] == user) & (self.data['topics'] == feature)]
        count_user_feature_rated = len(user_rated_feature['business_id'].unique())
        count_user_rated = len(self.data[self.data['user_id'] == user]['business_id'].unique())
        total_restaurants_reviewed = len(self.data['business_id'].unique())
        count_restaurants_feature_reviewed = len(self.data[(self.data['topics'] == feature)]['business_id'].unique())
        return ((count_user_feature_rated + 1 ) * total_restaurants_reviewed) / (count_user_rated * (count_restaurants_feature_reviewed + 1))

    def _avg(self, feature, restaurant):
        average_rating = self.data[(self.data['business_id'] == restaurant) & (self.data['topics'] == feature)]['scores'].mean()
        return average_rating

    def _delta(self, user, feature, restaurant):
        average_rating = self._avg(feature, restaurant)
        rating = self._transform_scores(user, feature, restaurant)
        if average_rating > rating > 0:
            return (average_rating - rating + 1)/average_rating
        return 1/(average_rating)

    def _requirement(self, user, feature):
        count_user_feature_rated = len(self.data[(self.data['user_id'] == user) & (self.data['topics'] == feature)]['business_id'].unique())
        count_feature_rated = len(self.data[(self.data['topics'] == feature)]['business_id'].unique())
        if count_user_feature_rated:
            return sum([self._delta(user, feature, restaurant) for restaurant in self.restaurant_id])/count_user_feature_rated
        return sum([0.1/self._avg(feature, restaurant) for restaurant in self.restaurant_id])/count_feature_rated

    def _weights(self, user, feature):
        concern = self._user_feature_concern(user, feature)
        requirement = self._requirement(user, feature)
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

# if __name__ == "__main__":
#     rr = RestaurantRecommender()
