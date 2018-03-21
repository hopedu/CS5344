# -*- coding: utf-8 -*-
# file: Collaborative_Filtering.py
# author: joddiyzhang@gmail.com
# time: 21/03/2018 3:20 PM
# Copyright (C) <2017>  <Joddiy Zhang>
# ------------------------------------------------------------------------

import numpy as np
import scipy
import pandas as pd
import math
import random
import sklearn
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt

origin_data = pd.read_csv('../asset/round1_ijcai_18_train_20180301.csv', sep=" ", error_bad_lines=False)
# origin_data = pd.read_csv('../asset/sub_sample.csv', sep=" ", error_bad_lines=False)

print("Number of Columns:\n", origin_data.shape[1], "\n\n")
print("List of Columns:\n", ", ".join(origin_data.columns), "\n\n")
print("Data:\n", origin_data.head(), "\n\n")
print("Size of train data(m):\n", origin_data.shape[0])

# user info
user_info = origin_data.as_matrix(
    ["user_id", "user_gender_id", "user_age_level", "user_occupation_id", "user_star_level"])
# item info
item_info = origin_data.as_matrix(
    ["item_id", "item_category_list", "item_property_list", "item_brand_id", "item_city_id", "item_price_level",
     "item_sales_level", "item_collected_level", "item_pv_level"])
# context info
context_info = origin_data.as_matrix(
    ["context_id", "context_timestamp", "context_page_id", "predict_category_property"])
# shop info
shop_info = origin_data.as_matrix(
    ["shop_id", "shop_review_num_level", "shop_review_positive_rate", "shop_star_level", "shop_score_service",
     "shop_score_delivery", "shop_score_description"])
# transaction info
transaction_info = origin_data.as_matrix(
    ["instance_id", "user_id", "item_id", "context_id", "shop_id"])

# no-buy as 1, and buy as 5
event_type_strength = {
    0: 1.0,
    1: 5.0,
}
origin_data['info_weights'] = origin_data['is_trade'].apply(lambda x: event_type_strength[x])

users_interactions_count_df = origin_data.groupby(['user_id', 'item_id']).size().groupby('user_id').size()
print('# users: %d' % len(users_interactions_count_df))
users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= 5].reset_index()[
    ['user_id']]
print('# users with at least 5 interactions: %d' % len(users_with_enough_interactions_df))

print('# of interactions: %d' % len(origin_data))
interactions_from_selected_users_df = origin_data.merge(users_with_enough_interactions_df,
                                                        how='right',
                                                        left_on='user_id',
                                                        right_on='user_id')
print('# of interactions from users with at least 5 interactions: %d' % len(interactions_from_selected_users_df))


def smooth_user_preference(x):
    return math.log(1 + x, 2)


interactions_full_df = interactions_from_selected_users_df \
    .groupby(['user_id', 'item_id'])['info_weights'].sum() \
    .apply(smooth_user_preference).reset_index()
print('# of unique user/item interactions: %d' % len(interactions_full_df))
interactions_full_df.head(10)

interactions_train_df, interactions_test_df = train_test_split(interactions_full_df,
                                                               stratify=interactions_full_df['user_id'],
                                                               test_size=0.20,
                                                               random_state=42)

print('# interactions on Train set: %d' % len(interactions_train_df))
print('# interactions on Test set: %d' % len(interactions_test_df))

# Indexing by personId to speed up the searches during evaluation
interactions_full_indexed_df = interactions_full_df.set_index('user_id')
interactions_train_indexed_df = interactions_train_df.set_index('user_id')
interactions_test_indexed_df = interactions_test_df.set_index('user_id')


def get_items_interacted(person_id, interactions_df):
    # Get the user's data and merge in the movie information.
    interacted_items = interactions_df.loc[person_id]['item_id']
    return set(interacted_items if type(interacted_items) == pd.Series else [interacted_items])


# Top-N accuracy metrics consts
EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS = 100


class ModelEvaluator:
    def get_not_interacted_items_sample(self, person_id, sample_size, seed=42):
        interacted_items = get_items_interacted(person_id, interactions_full_indexed_df)
        all_items = set(origin_data['item_id'])
        non_interacted_items = all_items - interacted_items

        random.seed(seed)
        non_interacted_items_sample = random.sample(non_interacted_items, sample_size)
        return set(non_interacted_items_sample)

    def _verify_hit_top_n(self, item_id, recommended_items, topn):
        try:
            index = next(i for i, c in enumerate(recommended_items) if c == item_id)
        except:
            index = -1
        hit = int(index in range(0, topn))
        return hit, index

    def evaluate_model_for_user(self, model, person_id):
        # Getting the items in test set
        interacted_values_testset = interactions_test_indexed_df.loc[person_id]
        if type(interacted_values_testset['item_id']) == pd.Series:
            person_interacted_items_testset = set(interacted_values_testset['item_id'])
        else:
            person_interacted_items_testset = set([int(interacted_values_testset['item_id'])])
        interacted_items_count_testset = len(person_interacted_items_testset)

        # Getting a ranked recommendation list from a model for a given user
        person_recs_df = model.recommend_items(person_id,
                                               items_to_ignore=get_items_interacted(person_id,
                                                                                    interactions_train_indexed_df),
                                               topn=10000000000)

        hits_at_5_count = 0
        hits_at_10_count = 0
        # For each item the user has interacted in test set
        for item_id in person_interacted_items_testset:
            # Getting a random sample (100) items the user has not interacted
            # (to represent items that are assumed to be no relevant to the user)
            non_interacted_items_sample = self.get_not_interacted_items_sample(person_id,
                                                                               sample_size=EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS,
                                                                               seed=item_id % (2 ** 32))

            # Combining the current interacted item with the 100 random items
            items_to_filter_recs = non_interacted_items_sample.union(set([item_id]))

            # Filtering only recommendations that are either the interacted item or from a random sample of 100 non-interacted items
            valid_recs_df = person_recs_df[person_recs_df['item_id'].isin(items_to_filter_recs)]
            valid_recs = valid_recs_df['item_id'].values
            # Verifying if the current interacted item is among the Top-N recommended items
            hit_at_5, index_at_5 = self._verify_hit_top_n(item_id, valid_recs, 5)
            hits_at_5_count += hit_at_5
            hit_at_10, index_at_10 = self._verify_hit_top_n(item_id, valid_recs, 10)
            hits_at_10_count += hit_at_10

        # Recall is the rate of the interacted items that are ranked among the Top-N recommended items,
        # when mixed with a set of non-relevant items
        recall_at_5 = hits_at_5_count / float(interacted_items_count_testset)
        recall_at_10 = hits_at_10_count / float(interacted_items_count_testset)

        person_metrics = {'hits@5_count': hits_at_5_count,
                          'hits@10_count': hits_at_10_count,
                          'interacted_count': interacted_items_count_testset,
                          'recall@5': recall_at_5,
                          'recall@10': recall_at_10}
        return person_metrics

    def evaluate_model(self, model):
        # print('Running evaluation for users')
        people_metrics = []
        for idx, person_id in enumerate(list(interactions_test_indexed_df.index.unique().values)):
            # if idx % 100 == 0 and idx > 0:
            #    print('%d users processed' % idx)
            person_metrics = self.evaluate_model_for_user(model, person_id)
            person_metrics['_person_id'] = person_id
            people_metrics.append(person_metrics)
        print('%d users processed' % idx)

        detailed_results_df = pd.DataFrame(people_metrics) \
            .sort_values('interacted_count', ascending=False)

        global_recall_at_5 = detailed_results_df['hits@5_count'].sum() / float(
            detailed_results_df['interacted_count'].sum())
        global_recall_at_10 = detailed_results_df['hits@10_count'].sum() / float(
            detailed_results_df['interacted_count'].sum())

        global_metrics = {'modelName': model.get_model_name(),
                          'recall@5': global_recall_at_5,
                          'recall@10': global_recall_at_10}
        return global_metrics, detailed_results_df


model_evaluator = ModelEvaluator()

################################################################################################
# Popularity model
################################################################################################

# Computes the most popular items
item_popularity_df = interactions_full_df.groupby('item_id')['info_weights'].sum().sort_values(
    ascending=False).reset_index()
print(item_popularity_df.head(10))


class PopularityRecommender:
    MODEL_NAME = 'Popularity'

    def __init__(self, popularity_df, items_df=None):
        self.popularity_df = popularity_df
        self.items_df = items_df

    def get_model_name(self):
        return self.MODEL_NAME

    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        # Recommend the more popular items that the user hasn't seen yet.
        recommendations_df = self.popularity_df[~self.popularity_df['item_id'].isin(items_to_ignore)] \
            .sort_values('info_weights', ascending=False) \
            .head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how='left',
                                                          left_on='item_id',
                                                          right_on='item_id')[
                ['info_weights', 'item_id']]

        return recommendations_df


popularity_model = PopularityRecommender(item_popularity_df, origin_data)

print('Evaluating Popularity recommendation model...')
pop_global_metrics, pop_detailed_results_df = model_evaluator.evaluate_model(popularity_model)
print('\nGlobal metrics:\n%s' % pop_global_metrics)
print(pop_detailed_results_df.head(10))
exit()

################################################################################################
# Collaborative Filtering model
################################################################################################
# todo

################################################################################################
# Collaborative Filtering model
################################################################################################

# Creating a sparse pivot table with users in rows and items in columns
users_items_pivot_matrix_df = interactions_train_df.pivot(index='user_id',
                                                          columns='item_id',
                                                          values='info_weights').fillna(0)

print(users_items_pivot_matrix_df.head(10))
users_items_pivot_matrix = users_items_pivot_matrix_df.as_matrix()
users_ids = list(users_items_pivot_matrix_df.index)
# The number of factors to factor the user-item matrix.
NUMBER_OF_FACTORS_MF = 15
# Performs matrix factorization of the original user item matrix
U, sigma, Vt = svds(users_items_pivot_matrix, k=NUMBER_OF_FACTORS_MF)
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)
# Converting the reconstructed matrix back to a Pandas dataframe
cf_preds_df = pd.DataFrame(all_user_predicted_ratings, columns=users_items_pivot_matrix_df.columns,
                           index=users_ids).transpose()


class CFRecommender:
    MODEL_NAME = 'Collaborative Filtering'

    def __init__(self, cf_predictions_df, items_df=None):
        self.cf_predictions_df = cf_predictions_df
        self.items_df = items_df

    def get_model_name(self):
        return self.MODEL_NAME

    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        # Get and sort the user's predictions
        sorted_user_predictions = self.cf_predictions_df[user_id].sort_values(ascending=False) \
            .reset_index().rename(columns={user_id: 'recStrength'})

        # Recommend the highest predicted rating movies that the user hasn't seen yet.
        recommendations_df = sorted_user_predictions[~sorted_user_predictions['contentId'].isin(items_to_ignore)] \
            .sort_values('recStrength', ascending=False) \
            .head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how='left',
                                                          left_on='contentId',
                                                          right_on='contentId')[
                ['recStrength', 'contentId']]

        return recommendations_df


cf_recommender_model = CFRecommender(cf_preds_df, origin_data)
