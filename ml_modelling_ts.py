import pandas as pd
import numpy as np

'''
This py file contains the time extractor methods, the distance creation using loudness values, and other common functions.
'''


def extract_time_features(time_data_set):
    # Create some features
    # Lets simply extract year month date hour min sec from the time series as a separate feature.
    ############### Working with Time Series Data ############################
    type(time_data_set["created_at"][0])
    # Converting to timestamp format for observation
    time_data_set["created_at"] = pd.to_datetime(time_data_set["created_at"])

    # Exploring the datetime

    daysofweek = {
        0: "Monday",
        1: "Tuesday",
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday"
    }

    time_data_set["year"] = time_data_set["created_at"].apply(lambda x: x.year)
    time_data_set["month"] = time_data_set["created_at"].apply(lambda x: x.month)
    time_data_set["date"] = time_data_set["created_at"].apply(lambda x: x.date())
    time_data_set["hour"] = time_data_set["created_at"].apply(lambda x: x.hour)
    time_data_set["minute"] = time_data_set["created_at"].apply(lambda x: x.minute)
    time_data_set["day"] = time_data_set["created_at"].apply(lambda x: x.day)
    time_data_set["day_of_record"] = time_data_set["created_at"].apply(lambda x: x.dayofweek)
    # time_data_set["day_of_record"] = time_data_set["created_at"].apply(lambda x: daysofweek[x.dayofweek])

    return time_data_set


def compute_dist(x, y=None):
    '''
    For each day group the observations of the loudness and distress for px and py
    compute a similarity using euclidean distance for each s02 - loudness. When one of the values are not present
    for one of the patient only one of them is considered.
    When patient has no observations for a particular day then the previous day observation is carried over to calculate similarity.
    Doing so enables to calculate for all the available days between px and py and a final average is returned as the distance.

    :param x: attribute set of instance1, here it is day, s02
    :param y: attribute set of instance2, here it is day, s02
    :return: distance
    '''
    dist = 0
    count = 0
    prev_s02 = 0.0
    for i in range(len(x)):
        for j in range(len(y)):
            if x[i][0] == y[j][0]:
                count += 1
                s02_dist = np.linalg.norm(x[i][1] - y[j][1])
                avg_sum = s02_dist
                dist += avg_sum
                prev_s02 = y[j][1]
        # Does not hold the metric symmetry hence discarding the moving of previous day
        #if not flag:
        #    count += 1
        #    s02_dist = np.linalg.norm(x[i][1] - prev_s02)
        #    avg_sum = s02_dist
        #    dist += avg_sum
        #    print("Carrying distance " + str(dist))
    print("total_dist -- " + str(dist))
    print("n_days -- " + str(count))

    return dist/count


def preprocess_data(time_data_set, grouping="day_of_week", obs_day="mean"):

    # Extract time features
    time_data_set = extract_time_features(time_data_set)

    # Process the s02-s07
    time_data_set["s02"] = time_data_set["s02"].apply(lambda x: x / 100)
    time_data_set["s03"] = time_data_set["s03"].apply(lambda x: x / 100)
    time_data_set["s04"] = time_data_set["s04"].apply(lambda x: x / 100)
    time_data_set["s05"] = time_data_set["s05"].apply(lambda x: x / 100)
    time_data_set["s06"] = time_data_set["s06"].apply(lambda x: x / 100)
    time_data_set["s07"] = time_data_set["s07"].apply(lambda x: x / 100)
    '''
    daysofweek = {
        0: "Monday",
        1: "Tuesday",
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday"
    }
    '''
    # day_of_week level, day level
    if grouping == "day_of_week":
            user_date_ts = time_data_set.groupby(["user_id", "day_of_record"]).mean().reset_index()
    elif grouping == "day":
        user_d_ts = time_data_set.groupby(["user_id", "day"]).mean().reset_index()


    drop_user_ids = [54, 60, 140, 170, 4, 6, 7, 9,
                     12, 19, 25, 53, 59, 130, 144, 145, 148, 156, 167]

    user_ids = time_data_set["user_id"].unique()

    user_ts_days = []
    for user_id in user_ids:
        if user_id not in drop_user_ids:
            if grouping == "day_of_week":
                user_ts_days.append((user_id, user_d_ts[user_d_ts["user_id"] == user_id][["day_of_record", "s02", "s03"]].to_numpy()))
            elif grouping == "day":
                user_ts_days.append((user_id, user_d_ts[user_d_ts["user_id"] == user_id][["day", "s02", "s03"]].to_numpy()))
    return user_ts_days


import properties


def process_data(grouping="day_of_week"):
    tyt_data = pd.read_pickle(properties.ts_file_location)
    # We will initially drop all na
    tyt_data.dropna(inplace=True)
    time_data_set = tyt_data.copy()

    #Extract time info and preprocess data
    time_data_set = extract_time_features(time_data_set)
    user_ts_processed = preprocess_data(time_data_set, grouping=grouping)
    return np.asarray(user_ts_processed)


usr_ts = process_data(grouping="day")
