# Instantiation via class initialization
import numpy as np
import pandas as pd
import properties
from pathlib import Path
'''
Description:
This is core of the time series data class. This class utilizes the raw data and converts into data utilized for predictions.
Encoding of time to fill gap is incorporated in two ways:
1. A simple date handling logic is first built.
2. Another data structure with logic handling with date is built. The idea is explained below. 
But, there might be occurances of peaks impacting the algorithm output so used for visualized. 
It is a good start to extend it for complicated techniques
3. A new simulate dataset addition is created to add to the existing time series dataset. This is done to evaluate the
obtained nearest neighbors without looking into the actual dataset. The simulation is created not for the entire dataset 
but for set of handpicked users such as a digital twin, outliers and normal test instances.
'''


class TimeSeriesGroupProcessing:
    norm = True
    file_path = properties.ts_file_location
    simulate_file_path = properties.simulate_ts_file_location

    def __init__(self, method):
        self.ts_day = self.pre_process()
        self.user_grp_dict_predict = self.get_m_day_ts_enumerate(method=method)
        self.user_grp_dict_vis = self.get_m_day_ts_encode(method=method)

    def pre_process(self):
        tyt_data = pd.read_pickle(self.file_path)

        # Do the simulation concatenation if the file exists
        p = Path(self.simulate_file_path)
        if p.exists() and p.is_file():
            simulate_tyt_data = pd.read_pickle(self.simulate_file_path)
            # We will initially drop all na - s03
            simulate_tyt_data["user_id"] = simulate_tyt_data["user_id"].apply(int)
            overall_data = tyt_data.append(simulate_tyt_data)
            overall_data.dropna(subset=["s03"], inplace=True)
        else:
            overall_data = tyt_data.copy()
            overall_data.dropna(subset=["s02","s03","s04","s05","s06","s07"], inplace=True)
        # Create some features
        time_data_set = overall_data.copy()
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
        if self.norm:
            time_data_set["s02"] = time_data_set["s02"].apply(lambda x: x / 100)
            time_data_set["s03"] = time_data_set["s03"].apply(lambda x: x / 100)
            time_data_set["s04"] = time_data_set["s04"].apply(lambda x: x / 100)
            time_data_set["s05"] = time_data_set["s05"].apply(lambda x: x / 100)
            time_data_set["s06"] = time_data_set["s06"].apply(lambda x: x / 100)
            time_data_set["s07"] = time_data_set["s07"].apply(lambda x: x / 100)

        drop_user_ids = [4, 6, 7, 9, 12, 19, 25, 53, 59, 130, 144, 145, 148, 156, 167]
        time_data_set_filtered = time_data_set[~time_data_set.user_id.isin(drop_user_ids)]
        time_data_set_filtered = time_data_set_filtered[
            ["user_id", "month", "day", "date", "s02", "s03", "s04", "s05", "s06", "s07"]]

        return time_data_set_filtered

    '''
    Proposed second method to ts_Encode
    '''
    def get_m_day_ts_encode(self, method="mean"):
        from collections import OrderedDict
        usr_grp_dict = dict()
        count = 0
        user_month_grp = self.ts_day.groupby(["user_id", "month"])
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

                # Compute to get the fraction of the day from 1-30.


                days = np.sort(group_usr_df["day"].to_numpy())
                prev_diff = (1 / 30)
                norm_day_index = list()
                norm_day_index.append(prev_diff)
                for i in range(len(days) - 1):
                    if i >= (len(days) - 1):
                        break
                    diff = abs(days[i] - days[i + 1])
                    if diff > 1:
                        print(diff)
                        new_diff = (diff / 30)
                        print(new_diff)
                        print(prev_diff)
                        norm_day_index.append(new_diff + prev_diff)
                        prev_diff = new_diff + prev_diff
                    else:
                        new_diff = (1 / 30)
                        norm_day_index.append((new_diff + prev_diff))
                        prev_diff = new_diff + prev_diff

                # day_index_count = [i for i in range(0, len(group_usr_df["day"]))]
                # Rounding for cleaner view and manipulations of day_index
                day_index_count = [round(val, 2) for val in norm_day_index]
                idx = len(day_index_count)
                group_usr_df["day_session_id"] = day_index_count
                usr_grp_dict[grp[0][0]] = group_usr_df[["day", "day_session_id", "s02", "s03",
                                                        "s04", "s05", "s06", "s07"]].to_numpy()
            else:
                group_user_arr = usr_grp_dict[grp[0][0]]
                group_df = grp[1]
                # print("Taking by day " + method + "at next iterations")
                if method == "mean":
                    group_usr_df = group_df.groupby("day").mean().reset_index()
                elif method == "max":
                    group_usr_df = group_df.groupby("day").max().reset_index()
                elif method == "min":
                    group_usr_df = group_df.groupby("day").min().reset_index()
                elif method == "median":
                    group_usr_df = group_df.groupby("day").median().reset_index()

                print("length of df -- for " + str(grp[0][0]) + "and len " + str(len(group_usr_df)))
                # print(len(group_user_arr))
                # print(len(group_usr_df))

                # Computation of the day_index continue from where ever last month was left.

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
                        new_diff = (diff / 30)
                        print(new_diff)
                        print(prev_diff)
                        norm_day_index.append((new_diff + prev_diff))
                        prev_diff = new_diff + prev_diff
                    else:
                        new_diff = (1 / 30)
                        norm_day_index.append(new_diff + prev_diff)
                        prev_diff = new_diff + prev_diff

                # print("start", int(temp_arr[len(temp_arr) - 1][0]) + 1)
                # print("end", len(group_user_arr) + len(group_usr_df))
                # day_index_count = [round(val, 2) for val in norm_day_index]
                day_index_count = [round(val, 2) for val in norm_day_index]
                group_usr_df["day_session_id"] = day_index_count
                usr_grp_dict[grp[0][0]] = np.append(group_user_arr, group_usr_df[["day", "day_session_id", "s02",
                                                                                  "s03", "s04", "s05", "s06",
                                                                                  "s07"]].to_numpy(), axis=0)
        return usr_grp_dict

    '''
    1st method proposed for ts_encode
    '''
    def get_m_day_ts_enumerate(self, method="mean"):
        from collections import OrderedDict
        usr_grp_dict = dict()
        user_month_grp = self.ts_day.groupby(["user_id", "month"])
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
                group_usr_df["day_session_id"] = day_index_count
                usr_grp_dict[grp[0][0]] = group_usr_df[["day", "day_session_id", "s02", "s03",
                                                        "s04", "s05", "s06", "s07"]].to_numpy()
            else:
                group_user_arr = usr_grp_dict[grp[0][0]]
                group_df = grp[1]
                # print("Taking by day " + method + "at next iterations")
                if method == "mean":
                    group_usr_df = group_df.groupby("day").mean().reset_index()
                elif method == "max":
                    group_usr_df = group_df.groupby("day").max().reset_index()
                elif method == "min":
                    group_usr_df = group_df.groupby("day").min().reset_index()
                elif method == "median":
                    group_usr_df = group_df.groupby("day").median().reset_index()

                print("length of df -- for " + str(grp[0][0]) + "and len " + str(len(group_usr_df)))
                # print(len(group_user_arr))
                # print(len(group_usr_df))
                temp_arr = group_user_arr[len(group_user_arr) - 1]

                # print("start", int(temp_arr[len(temp_arr) - 1][0]) + 1)
                # print("end", len(group_user_arr) + len(group_usr_df))
                day_index_count = [i for i in range(int(temp_arr[1]) + 1,
                                                    int(temp_arr[1]) + 1 + len(group_usr_df))]
                group_usr_df["day_session_id"] = day_index_count
                usr_grp_dict[grp[0][0]] = np.append(group_user_arr, group_usr_df[["day", "day_session_id", "s02",
                                                                                  "s03", "s04", "s05", "s06",
                                                                                  "s07"]].to_numpy(), axis=0)
        return usr_grp_dict

    def get_user_mday_ts_visualize(self, user_id):
        return self.user_grp_dict_vis[user_id]

    def get_usr_mday_ts_predict(self, user_id):
        return self.user_grp_dict_predict[user_id]

