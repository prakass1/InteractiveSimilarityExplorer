import numpy as np
import pandas as pd
import pandas as pd
import numpy as np
import seaborn as sns
import tslearn as tsl
#import ts_preprocessing as tsp

# Preprocessing of the time series and grouping of the observations
'''
Methods that do same as time_series_grp.py
'''
def preprocess(norm):
    tyt_data = pd.read_pickle("data/input_pckl/" + "1_q.pckl")
    # We will initially drop all na
    tyt_data.dropna(inplace=True)
    time_data_set = tyt_data.copy()
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
    if norm:
        time_data_set["s02"] = time_data_set["s02"].apply(lambda x: x / 100)
        time_data_set["s03"] = time_data_set["s03"].apply(lambda x: x / 100)
        time_data_set["s04"] = time_data_set["s04"].apply(lambda x: x / 100)
        time_data_set["s05"] = time_data_set["s05"].apply(lambda x: x / 100)
        time_data_set["s06"] = time_data_set["s06"].apply(lambda x: x / 100)
        time_data_set["s07"] = time_data_set["s07"].apply(lambda x: x / 100)

    # We drop those user's who have observations less than 3 and have taken sessions less than 5day period
    #drop_user_id = [4, 6, 7, 9, 12, 25, 39, 40, 53, 33, 59, 128, 130,
    #                144, 145, 148, 156, 157, 166, 167, 168]

    drop_user_ids = [4, 6, 7, 9, 12, 25, 39, 53, 59, 128, 130, 144, 145, 148, 156, 166, 167]
    time_data_set_filtered = time_data_set[~time_data_set.user_id.isin(drop_user_ids)]

    return time_data_set_filtered

# This method aggregates values over entire days and returns value.
# Note: We realign the day to start from 0 to number of observations for the user
def get_user_day_ts(user_id, method="mean"):
    ts_day = preprocess(True)
    print("Taking by day " + method)
    if method == "mean":
        ts_day_avg = ts_day.groupby(["user_id", "day"]).mean().reset_index()
    elif method == "max":
        ts_day_avg = ts_day.groupby(["user_id", "day"]).max().reset_index()
    elif method == "min":
        ts_day_avg = ts_day.groupby(["user_id", "day"]).min().reset_index()
    elif method == "median":
        ts_day_avg = ts_day.groupby(["user_id", "day"]).median().reset_index()

    ts_user = ts_day_avg[ts_day_avg["user_id"] == int(user_id)][["s02", "s03",
                                                                 "s04", "s05", "s06", "s07"]]
    ts_user["day_session_id_count"] = [i for i in range(len(ts_user))]
    ts_user = ts_user[["day_session_id_count", "s02", "s03", "s04", "s05", "s06", "s07"]]
    ts_user_np = ts_user.to_numpy()
    return ts_user_np


# A method which first groups each of the user's time series sequences of the month and within the month,
# a data structure is created to create per day time series.
# Note: A day here means (1-31) and this day for each month is converted to start from time (t0 to tn) n is the length
# of the user time series sequence.

ts_day = preprocess(True)


def get_m_day_ts(method="median"):
    from collections import OrderedDict
    usr_grp_dict = dict()
    user_month_grp = ts_day.groupby(["user_id", "month"])
    print("Taking by day " + method)
    for grp in user_month_grp:
        if grp[0][0] not in usr_grp_dict:
            arr = []
            group_df = grp[1]
            if method == "mean":
                group_usr_df = group_df.groupby("day").mean().reset_index()
            elif method == "max":
                group_usr_df = group_df.groupby("day").max().reset_index()
            elif method == "min":
                group_usr_df = group_df.groupby("day").min().reset_index()
            elif method == "median":
                group_usr_df = group_df.groupby("day").median().reset_index()

            day_index_count = [i for i in range(0, len(group_usr_df["day"]))]
            idx = len(day_index_count)
            group_usr_df["day_index_count"] = day_index_count
            usr_grp_dict[grp[0][0]] = group_usr_df[["day_index_count", "day", "s02", "s03",
                                     "s04", "s05", "s06", "s07"]].to_numpy()
        else:
            group_user_arr = usr_grp_dict[grp[0][0]]
            group_df = grp[1]
            #print("Taking by day " + method + "at next iterations")
            if method == "mean":
                group_usr_df = group_df.groupby("day").mean().reset_index()
            elif method == "max":
                group_usr_df = group_df.groupby("day").max().reset_index()
            elif method == "min":
                group_usr_df = group_df.groupby("day").min().reset_index()
            elif method == "median":
                group_usr_df = group_df.groupby("day").median().reset_index()

            print("length of df -- for " + str(grp[0][0]) + "and len " + str(len(group_usr_df)))
            #print(len(group_user_arr))
            #print(len(group_usr_df))
            temp_arr = group_user_arr[len(group_user_arr) - 1]

            #print("start", int(temp_arr[len(temp_arr) - 1][0]) + 1)
            #print("end", len(group_user_arr) + len(group_usr_df))
            day_index_count = [i for i in range(int(temp_arr[0]) + 1,
                                                int(temp_arr[0]) + 1 + len(group_usr_df))]
            group_usr_df["day_session_id_count"] = day_index_count
            usr_grp_dict[grp[0][0]] = np.append(group_user_arr, group_usr_df[["day_session_id_count", "day", "s02",
                                                "s03", "s04", "s05", "s06", "s07"]].to_numpy(), axis=0)
    return usr_grp_dict


def get_m_day_ts_index_corrected(method="median"):
    from collections import OrderedDict
    usr_grp_dict = dict()
    count = 0
    user_month_grp = ts_day.groupby(["user_id", "month"])
    print("Taking by day " + method)
    for grp in user_month_grp:
        if grp[0][0] not in usr_grp_dict:
            print("User- " + str(grp[0][0]))
            count += 1
            arr = []
            group_df = grp[1]
            if method == "mean":
                group_usr_df = group_df.groupby("day").mean().reset_index()
            elif method == "max":
                group_usr_df = group_df.groupby("day").max().reset_index()
            elif method == "min":
                group_usr_df = group_df.groupby("day").min().reset_index()
            elif method == "median":
                group_usr_df = group_df.groupby("day").median().reset_index()

            #Compute to get the fraction of the day from 1-30.

            days = np.sort(group_usr_df["day"].to_numpy())
            prev_diff = round((1/30), 2)
            norm_day_index = list()
            norm_day_index.append(prev_diff)
            for i in range(len(days) - 1):
                if i >= (len(days) - 1):
                    break
                diff = abs(days[i] - days[i + 1])
                if diff > 1:
                    print(diff)
                    new_diff = diff / 30
                    print(new_diff)
                    print(prev_diff)
                    norm_day_index.append(round((new_diff + prev_diff), 2))
                    prev_diff = new_diff + prev_diff
                else:
                    new_diff = 1 / 30
                    norm_day_index.append(round((new_diff + prev_diff), 2))
                    prev_diff = new_diff + prev_diff

            #day_index_count = [i for i in range(0, len(group_usr_df["day"]))]
            day_index_count = norm_day_index
            idx = len(day_index_count)
            group_usr_df["day_index_count"] = day_index_count
            usr_grp_dict[grp[0][0]] = group_usr_df[["day", "day_index_count", "s02", "s03",
                                     "s04", "s05", "s06", "s07"]].to_numpy()
        else:
            group_user_arr = usr_grp_dict[grp[0][0]]
            group_df = grp[1]
            #print("Taking by day " + method + "at next iterations")
            if method == "mean":
                group_usr_df = group_df.groupby("day").mean().reset_index()
            elif method == "max":
                group_usr_df = group_df.groupby("day").max().reset_index()
            elif method == "min":
                group_usr_df = group_df.groupby("day").min().reset_index()
            elif method == "median":
                group_usr_df = group_df.groupby("day").median().reset_index()

            print("length of df -- for " + str(grp[0][0]) + "and len " + str(len(group_usr_df)))
            #print(len(group_user_arr))
            #print(len(group_usr_df))


            #Computation of the day_index continue from where ever last month was left.
            '''
            Math:
            It is evident that month-1 last reading of a user could be on a day1 and the next month-2 would be on a day2.
            Let us look at this with an example:
            if month-1 and day-25 has one of the reading and month-2 and day-10 has a reading, then we do the following:
            1. We make a general number of days as 30 and then we start taking the difference in day of the month readings
            Here:
            A conditions is handled to check if the curr_day is greater than prev_day, if yes then,
             -- Difference would be: mdiff = (30 + |curr_day - prev|) and it's fractional value would be mdiff/30.
            else it would be mdiff = (30 - |prev_day - curr_day|) and its fractional value as mdiff/30
            In the case of the example, mdiff = 15 and n_mdiff = 15/30 ==> 0.5.
            2. This value obtained is then added to the enumeration day_index into the first 
            date of the next month which can keep the day intact.
            '''

            temp_arr = group_user_arr[len(group_user_arr) - 1]

            days = np.sort(group_usr_df["day"].to_numpy())

            # Computation for gap filling. Index is not a true index, we account for the gap by a logic
            prev_mday = temp_arr[0]
            curr_mday = group_usr_df["day"].iloc[0]

            # We commonly take 30 as the number of days. This should be noted
            if curr_mday > prev_mday:
                day_diff = 30 + abs(curr_mday - prev_mday)
            else:
                day_diff = 30 - abs(prev_mday - curr_mday)

            # Exceptional case of 31 days
            if day_diff == 0:
                day_diff = 1

            norm_day_index = list()
            prev_diff = temp_arr[1]
            new_diff = (day_diff / 30)


            norm_day_index = list()
            norm_day_index.append(round((prev_diff + new_diff), 2))
            prev_diff = prev_diff + new_diff
            for i in range(len(days) - 1):
                if i >= (len(days) - 1):
                    break
                diff = abs(days[i] - days[i + 1])
                if diff > 1:
                    print(diff)
                    new_diff = diff / 30
                    print(new_diff)
                    print(prev_diff)
                    norm_day_index.append(round((new_diff + prev_diff), 2))
                    prev_diff = new_diff + prev_diff
                else:
                    new_diff = 1 / 30
                    norm_day_index.append(round((new_diff + prev_diff), 2))
                    prev_diff = new_diff + prev_diff


            #print("start", int(temp_arr[len(temp_arr) - 1][0]) + 1)
            #print("end", len(group_user_arr) + len(group_usr_df))
            day_index_count = norm_day_index
            group_usr_df["day_session_id_count"] = day_index_count
            usr_grp_dict[grp[0][0]] = np.append(group_user_arr, group_usr_df[["day", "day_session_id_count","s02",
                                                "s03", "s04", "s05", "s06", "s07"]].to_numpy(), axis=0)
    return usr_grp_dict



user_grp_dict1 = get_m_day_ts(method="median")


def get_usr_mday_ts(user_id):
    return user_grp_dict1[user_id]


user_grp_dict2 = get_m_day_ts_index_corrected(method="median")


def get_usr_mday_ts_index_corrected(user_id):
    return user_grp_dict2[user_id]

