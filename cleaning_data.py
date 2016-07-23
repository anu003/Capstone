from __future__ import division
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from geopy.geocoders import Nominatim
from collections import Counter
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import requests, os, json
import pandas as pd
import numpy as np
import threading, time
import nltk
pd.set_option('display.width', 100)

def extract_locations(df):
    db = DBSCAN(eps = 1, min_samples = 100) # eps 1 = 111.2KM, since the cities provided is atleast 100 miles away from each other
    labels = db.fit_predict(df[['latitude','longitude']])
    labels = np.expand_dims(labels, axis = 1)
    df['location'] = labels
    # geocoder = Nominatim()
    # states = {}
    # centroids = {}
    # exceptions = []
    # for label in np.unique(labels):
    #     X = df[df.state == label][['latitude','longitude']]
    #     lat, lon = X.mean()

    #     location = geocoder.reverse((lat,lon))
    #     time.sleep(2)
    #     try:    # Works for most countries
    #         states[label] = location.raw['address']['state']
    #         centroids[label] = (lat,lon)
    #     except:
    #         exceptions.append((lat,lon))
    #         states[label] = (lat,lon)
    #         centroids[label] = (lat,lon)
    # df['city_state'] = df.state.map(centroids)
    # df.state = df.state.map(states)
    # return exceptions

if __name__=="__main__":
    df_business = pd.read_csv("yelp_dataset_business.csv")
    extract_locations(df_business)
