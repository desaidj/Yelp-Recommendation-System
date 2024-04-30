# Hybrid Recommendation System

This project implements a hybrid recommendation system that combines a machine learning model with a collaborative filtering approach to provide personalized recommendations for businesses on the Yelp platform.

## Overview

The machine learning component of the system is an XGBoost regressor that predicts numerical ratings based on a set of input features. These features include user-specific attributes (e.g., review count, average stars, and user engagement metrics) and business-specific attributes (e.g., location, business category, and amenities).

The collaborative filtering component leverages the user-business rating matrix to identify similar users and businesses, and then uses these similarities to provide recommendations.

## Implementation Details

### Machine Learning Model

The machine learning model follows these steps:

1. **Define Input Features**: The model uses a comprehensive set of user and business features as input, including user attributes (user_id, review_count, average_stars, useful, fans, yelping_since, cool, and funny) and business attributes (business_id, stars, latitude, longitude, review_count, is_open, various amenities, and other metadata).

2. **Train XGBoost Regressor**: The XGBoost algorithm, a powerful gradient boosting framework, is used to train a regressor on the dataset, predicting the numerical rating based on the input features.

3. **Hyperparameter Tuning**: The hyperparameters of the XGBoost algorithm are tuned to optimize the model's performance.

4. **Error Distribution Analysis**: The model's performance is evaluated by analyzing the distribution of prediction errors, binned into ranges of ratings (e.g., >=0 and <1, >=1 and <2, etc.).

5. **Calculate Root Mean Squared Error (RMSE)**: The RMSE metric is used to measure the average distance between the predicted ratings and the true ratings in the dataset.

### Collaborative Filtering

The collaborative filtering component leverages the user-business rating matrix to identify similar users and businesses. It then uses these similarities to provide recommendations based on the ratings of similar users or businesses.

## Performance

The current implementation of the hybrid recommendation system achieves the following performance:

**Error Distribution**:

- >=0 and <1: 102319
- >=1 and <2: 32818
- >=2 and <3: 6093
- >=3 and <4: 812
- >=4: 2

**RMSE**: 0.9775680229066512

**Execution Time**: 434.044s

## Data

The project uses the Yelp Dataset, which can be accessed via the following links:

- [Yelp Dataset Documentation](https://www.yelp.com/dataset/documentation/main)

## Usage

To use this hybrid recommendation system, follow these steps:

1. Clone the repository
2. Install the required dependencies
3. Preprocess the data if necessary
4. Train the machine learning model
5. Implement the collaborative filtering component
6. Combine the recommendations from both components
7. Evaluate the performance of the system





