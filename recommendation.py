'''
Method Description:

Hybrid Recommendation System

The method described is a machine learning model that predicts a numerical value (rating) based on model based recommendation system on a set of input features. Here are the steps in more detail:

Define the input features: The model uses a set of user and business features, including user_id, review_count, average_stars, useful, fans, yelping_since, cool, and funny for the user, and business_id, stars, latitude, longitude, review_count, is_open, attributes.BusinessAcceptsCreditCards, attributes.BikeParking, attributes.GoodForKids, attributes.HasTV, attributes.OutdoorSeating, attributes.RestaurantsReservations, attributes.RestaurantsTakeOut, attributes.RestaurantsGoodForGroups, attributes.RestaurantsDelivery, and attributes.OutdoorSeating for the business. Additionally, the number of photos per business, number of tips per business, and number of tips per user are also used as input features.

Train an XGBoost regressor: XGBoost is a machine learning algorithm that is commonly used for regression problems. The model is trained on a dataset of examples where each example contains the input features and the corresponding numerical rating.

Tune hyperparameters: The hyperparameters of the XGBoost algorithm are tuned to optimize the performance of the model.

Error distribution analysis: The model's performance is evaluated by analyzing the distribution of prediction errors. In this case, the errors are binned into ranges of ratings (e.g., >=0 and <1, >=1 and <2, etc.) and the number of examples in each bin is counted.

Calculate root mean squared error (RMSE): The RMSE is a commonly used metric to evaluate the performance of regression models. It measures the average distance between the predicted ratings and the true ratings in the dataset.

Overall, the described method is a machine learning model that predicts a numerical rating based on a set of input features which uses model based recommendation sysytem. The XGBoost algorithm is used with hyperparameters tuned to optimize performance, and the model's performance is evaluated using the error distribution and RMSE.

Error Distribution:
>=0 and <1: 102319
>=1 and <2: 32818
>=2 and <3: 6093
>=3 and <4: 812
>=4: 2

RMSE:
0.9775680229066512

Execution Time:
434.044s

'''

import xgboost as xgb
from xgboost import XGBRegressor
import collections
import itertools
import sys, os
import random
import json
import csv
import time
from itertools import combinations
from pyspark import SparkContext, SparkConf
from random import randint
import math
from datetime import datetime
import pickle as pkl

if __name__ == '__main__':
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

    sc = SparkContext()
    sc.setLogLevel('WARN')

    VALID_BUSINESS_COUNT = 23
    NUM_NEIGHBORS = 65536
    ALPHA = 1

    user_keys = ['user_id', 'review_count', 'average_stars', 'useful', 'fans', 'yelping_since', 'cool', 'funny']
    business_keys = ['business_id', 'stars', 'latitude', 'longitude', 'review_count', 'is_open',
                     'attributes.BusinessAcceptsCreditCards', 'attributes.BikeParking', 'attributes.GoodForKids',
                     'attributes.HasTV', 'attributes.OutdoorSeating', 'attributes.RestaurantsReservations',
                     'attributes.RestaurantsTakeOut', 'attributes.RestaurantsGoodForGroups',
                     'attributes.RestaurantsDelivery', 'attributes.OutdoorSeating', ]


    def predict_rating(row, business_map, user_map, user_avg_rdd, business_avg_rdd):
        business, user = row
        if (business not in business_map) and (user not in user_map):
            return user, business, 2.5, ALPHA
        if (business not in business_map):
            return user, business, user_avg_rdd[user], ALPHA
        if (user not in user_map):
            return user, business, business_avg_rdd[business], ALPHA
        coeffs = []
        row1 = business_map[business]
        avg1 = business_avg_rdd[business]
        curr_user_map = user_map[user]
        for business2 in curr_user_map:
            if business2 not in business_map:
                continue
            avg2 = business_avg_rdd[business2]
            row2 = business_map[business2]
            set_row1 = set(row1.keys())
            set_row2 = set(row2.keys())
            common_users = set_row1 & set_row2
            if (len(common_users) < VALID_BUSINESS_COUNT):
                h = row2[user] - avg2
                s = business_avg_rdd[business] / business_avg_rdd[business2]
                coeffs.append((h, s))
            else:
                num = 0
                norm1 = 0
                norm2 = 0
                for key in common_users:
                    rating1 = row1[key]
                    rating2 = row2[key]
                    norm1 += (rating1 - avg1) ** 2
                    norm2 += (rating2 - avg2) ** 2
                    num += (rating1 - avg1) * (rating2 - avg2)
                denom = math.sqrt(norm1 * norm2)
                if denom == 0:
                    coeff = 0
                    c = row2[user] - avg2

                    coeffs.append((c, coeff))
                else:
                    f = num / denom
                    coeffs.append((row2[user] - avg2, f))
        coeffs.sort(key=lambda x: math.fabs(x[1]), reverse=True)
        coeffs = coeffs[:NUM_NEIGHBORS]
        numer = 0
        denom = 0
        alpha_num = ALPHA * len(coeffs)
        alpha_denom = len(business_avg_rdd)
        curr_alpha = alpha_num / alpha_denom
        for curr_rating, coeff in coeffs:
            numer = numer + (curr_rating * coeff)
            denom = denom + math.fabs(coeff)
        if denom == 0:
            rating_num = (user_avg_rdd[user] + business_avg_rdd[business])
            rating = rating_num / 2
        else:
            rating = avg1 + (numer / denom)
        if rating <= 0 or rating > 5:
            e = (user_avg_rdd[user] + business_avg_rdd[business])
            rating = e / 2

        return user, business, rating, curr_alpha


    def item_based(train_path, test_path):
        input_rdd = sc.textFile(train_path)
        header = 'user_id,business_id,stars'
        input_rdd = input_rdd.filter(lambda x: x != header).map(lambda x: x.split(","))
        input_rdd = input_rdd.map(
            lambda x: (x[1], (x[0], float(x[2]))))

        business_rdd = input_rdd.groupByKey()
        business_rdd = business_rdd.mapValues(lambda x: {k: v for k, v in x})
        business_rdd = business_rdd.filter(lambda x: len(x[1]) >= VALID_BUSINESS_COUNT)
        business_map = business_rdd.collectAsMap()
        user_rdd = input_rdd.map(lambda x: (x[1][0], (x[0], x[1][1])))
        user_rdd = user_rdd.groupByKey()
        user_rdd = user_rdd.mapValues(lambda x: {k: v for k, v in x})
        user_map = user_rdd.collectAsMap()
        user_avg_rdd = user_rdd.mapValues(lambda x: sum(x.values()) / len(x))
        user_avg_rdd = user_avg_rdd.collectAsMap()
        business_avg_rdd = business_rdd.mapValues(lambda x: sum(x.values()) / len(x))
        business_avg_rdd = business_avg_rdd.collectAsMap()

        test_rdd = sc.textFile(test_path)
        header = 'user_id,business_id,stars'
        test_rdd = test_rdd.filter(lambda x: x != header)
        test_rdd = test_rdd.map(lambda x: x.split(","))
        test_rdd = test_rdd.map(lambda x: (x[1], x[0]))
        result = test_rdd.map(lambda x: predict_rating(x, business_map, user_map, user_avg_rdd, business_avg_rdd))
        r = result.toLocalIterator()
        return r


    def const_feat_set(feature_dict, feature_keys):
        result = []
        for key in feature_keys:
            levels = key.split(".")
            curr_dict = feature_dict
            for level in levels[:-1]:
                curr_dict = curr_dict[level] if level in curr_dict else {}
            if curr_dict is not None:
                curr_val = curr_dict.get(levels[-1], None)
                if levels[-1] == 'yelping_since':
                    date_time = datetime.strptime(curr_val, "%Y-%M-%d")
                    a_timedelta = datetime(2023, 1, 1) - date_time
                    curr_val = a_timedelta.total_seconds()
                elif isinstance(curr_val, str):
                    if curr_val == "True":
                        curr_val = 1
                    elif curr_val == "False":
                        curr_val = 0
            else:
                curr_val = None
            result.append(curr_val)
        return (result[0], result[1:])


    def combiner_(curr_set, curr_record):
        for i in range(len(curr_record[1])):
            if curr_record[i] is not None:
                curr_first = curr_set[i][0] + curr_record[1][i]
                curr_set[i] = (curr_first, curr_set[i][1] + 1)
            return curr_set


    def reducer_(record1, record2):
        return [(record1[i][0] + record2[i][0], record1[i][1] + record2[i][1]) for i in range(len(record1))]


    def replace_none_(record, means):
        result = [r if r is not None else means[i] for i, r in enumerate(record[1])]
        return record[0], result


    def rating_rdd_func(rating_path):
        r1_rdd = sc.textFile(rating_path)
        header = 'user_id,business_id,stars'
        r1_rdd = r1_rdd.filter(lambda x: x != header).map(lambda x: x.split(","))
        return r1_rdd


    def model_based(folder_path, test_path):

        rating_rdd = sc.textFile(f"{folder_path}/yelp_train.csv")
        header = 'user_id,business_id,stars'
        rating_rdd = rating_rdd.filter(lambda x: x != header)
        rating_rdd = rating_rdd.map(lambda x: x.split(","))

        user_rdd = sc.textFile(f"{folder_path}/user.json")
        user_rdd = user_rdd.map(lambda x: json.loads(x))
        user_rdd = user_rdd.map(lambda x: const_feat_set(x, user_keys))
        user_means = user_rdd.aggregate([(0, 0)] * (len(user_keys) - 1), combiner_, reducer_)

        user_means = [mean[0] / mean[1] if mean[1] != 0 else 0 for mean in user_means]

        user_rdd = user_rdd.map(lambda x: replace_none_(x, user_means))
        user_map = user_rdd.collectAsMap()

        for i, key in enumerate(user_map):
            user_map[key].append(i)
        user_means.append(0)
        user_rdd = user_rdd.map(lambda x: (x[0], user_map[x[0]]))

        business_rdd = sc.textFile(business_path)
        business_rdd = business_rdd.map(lambda x: json.loads(x))
        business_rdd = business_rdd.map(lambda x: const_feat_set(x, business_keys))
        business_means = business_rdd.aggregate([(0, 0)] * (len(business_keys) - 1), combiner_, reducer_)

        business_means_ = [mean[0] / mean[1] if mean[1] != 0 else 0 for mean in business_means]
        business_rdd = business_rdd.map(lambda x: replace_none_(x, business_means_))
        business_map = business_rdd.collectAsMap()

        tip_rdd = sc.textFile(f"{folder_path}/tip.json")
        tip_rdd = tip_rdd.map(lambda x: json.loads(x))
        user_tip_map = tip_rdd.map(lambda x: (x['user_id'], 1))
        user_tip_map = user_tip_map.reduceByKey(lambda x, y: x + y)
        user_tip_map = user_tip_map.collectAsMap()
        business_tip_map = tip_rdd.map(lambda x: (x['business_id'], 1))
        business_tip_map = business_tip_map.reduceByKey(lambda x, y: x + y)
        business_tip_map = business_tip_map.collectAsMap()

        mean_utip = 0
        for s in user_tip_map.values():
            mean_utip = mean_utip + s
        l_utip_values = len(user_tip_map.values())
        mean_utip = mean_utip / l_utip_values
        mean_btip = 0
        for s in business_tip_map.values():
            mean_btip = mean_btip + s
        l_btip_values = len(business_tip_map.values())
        mean_btip = mean_btip / l_btip_values

        photo_rdd = sc.textFile(f"{folder_path}/photo.json").map(lambda x: json.loads(x))

        photo_rdd = photo_rdd.map(lambda x: (x['business_id'], 1))
        photo_rdd = photo_rdd.reduceByKey(lambda x, y: x + y)
        photo_tip_map = photo_rdd.collectAsMap()

        mean_photo = 0
        for s in photo_tip_map.values():
            mean_photo = mean_photo + s
        l_photo_map = len(photo_tip_map)
        mean_photo = mean_photo / l_photo_map

        user_rating_rdd = rating_rdd.map(lambda x: (x[0], x[1:]))
        user_rating_rdd = user_rating_rdd.join(user_rdd)
        user_rating_rdd = user_rating_rdd.map(lambda x: (x[1][0][0], (x[0], float(x[1][0][1]), x[1][1])))

        feature_rating_rdd = user_rating_rdd.join(business_rdd)
        feature_rating_rdd = feature_rating_rdd.map(
            lambda x: ((x[1][0][0], x[0]), (x[1][0][2], x[1][1]), x[1][0][1]))
        feature_rating_rdd = feature_rating_rdd.sortBy(lambda x: x[0])

        train_features = feature_rating_rdd.map(lambda x: x[1][0] + x[1][1] + [user_tip_map.get(x[0][0], mean_utip), business_tip_map.get(x[0][1], mean_btip), photo_tip_map.get(x[0][1], mean_photo)])
        train_features = train_features.collect()
        train_ratings = feature_rating_rdd.map(lambda x: x[2])
        train_ratings = train_ratings.collect()

        best_params = {'max_depth': 5, 'learning_rate': 0.06999999999999999, 'subsample': 0.9000000000000001,
                       'colsample_bytree': 0.7999999999999999, 'colsample_bylevel': 0.7, 'n_estimators': 1000,
                       'min_child_weight': 5, 'tree_method': 'hist', 'gamma': 0.19000000000000003, 'random_state': 51}
        xgb_model = XGBRegressor(**best_params, verbosity=0)
        xgb_model.fit(train_features, train_ratings)

        test_rdd = sc.textFile(test_path)
        header = test_rdd.first()
        test_rdd = test_rdd.filter(lambda x: x != header)
        test_rdd = test_rdd.map(lambda x: x.split(",")[:2])
        test_features_rdd = test_rdd.map(lambda x: (x, (user_map.get(x[0], user_means), business_map.get(x[1], business_means), [user_tip_map.get(x[0], mean_utip), business_tip_map.get(x[1], mean_btip), photo_tip_map.get(x[1], mean_photo)])))
        test_features_rdd = test_features_rdd.sortBy(lambda x: x[0])
        test_features = test_features_rdd.mapValues(lambda x: x[0] + x[1] + x[2])
        test_features = test_features.values().collect()
        test_user_business = test_features_rdd.keys().collect()
        preds = xgb_model.predict(test_features)

        result = [(ub[0], ub[1], 1.0 if pred < 1.0 else 5.0 if pred > 5.0 else pred)
                  for ub, pred in zip(test_user_business, preds)]

        return result


    train_path = sys.argv[1]
    rating_path = f"{train_path}/yelp_train.csv"
    test_path = sys.argv[2]
    business_path = f"{train_path}/business.json"
    user_path = f"{train_path}/user.json"
    output_path = sys.argv[3]

    st = time.time()

    model_ratings = model_based(train_path, test_path)

    with open(output_path, "w") as out_file:
        out_file.write("user_id, business_id, prediction\n")
        out_file.writelines([f"{result[0]},{result[1]},{result[2]}\n" for result in model_ratings])

    execution_time = time.time() - st
    print("Execution Time: ", execution_time)

    y_pred = {}
    with open(output_path) as f:
        lines = f.readlines()[1:]

        for l in lines:
            x = l.strip().split(',')
            y_pred[tuple([x[0], x[1]])] = float(x[2])
    f.close()

    y_true = {}

    with open(train_path + '/yelp_val.csv') as f:
        for l in f.readlines()[1:]:
            x = l.strip().split(',')
            key = tuple([x[0], x[1]])
            y_true[key] = float(x[2])

    sum_val = 0
    cnt = 0
    error_distribution = [0, 0, 0, 0, 0]
    for key in y_true.keys():
        val1 = y_true[key]
        val2 = y_pred.get(key, 0)
        error = abs(val1 - val2)

        if 0 <= error < 1:
            error_distribution[0] += 1
        elif 1 <= error < 2:
            error_distribution[1] += 1
        elif 2 <= error < 3:
            error_distribution[2] += 1
        elif 3 <= error < 4:
            error_distribution[3] += 1
        else:
            error_distribution[4] += 1

        sum_val += (val1 - val2) ** 2
        cnt += 1

    val_rmse = math.sqrt(sum_val / cnt)

    print('Error Distribution: ')
    print('>=0 and <1:', error_distribution[0])
    print('>=1 and <2:', error_distribution[1])
    print('>=2 and <3:', error_distribution[2])
    print('>=3 and <4:', error_distribution[3])
    print('>=4:', error_distribution[4])

    print('RMSE: ')
    print(val_rmse)






