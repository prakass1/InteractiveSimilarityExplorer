import utility
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import static_sim_functions as smf
# import ts_preprocessing as ts_data
import numpy as np
import os
from pathlib import Path
import ast
'''
This class is created to simulate models for UI for especially similarity comparison.
Uses the same functionalities as per other ML modelling classes. Is maintained as a separate file.
The simulated data itself is created from the jupyter notebook file Synthethic Test Data.ipynb.
'''

def common_processing(df):
    # Getting percentage between 0 to 1 rather than score values
    df["tschq12"] = df["tschq12"].apply(lambda x: x / 100)
    df["tschq16"] = df["tschq16"].apply(lambda x: x / 100)
    df["tschq17"] = df["tschq17"].apply(lambda x: x / 100)

    df["tschq04"] = df.apply(create_cols_family_hist, axis=1)

    return df


# Common elements
# Feature engineering family history
def create_cols_family_hist(x):
    if x["tschq04-1"] == "YES":
        if isinstance(x["tschq04-2"], str):
            res = ast.literal_eval(x["tschq04-2"])
        else:
            res = x["tschq04-2"]

        lst_sorted = sorted(res)
        list_to_str = "_".join([val for val in lst_sorted])
        return list_to_str
    else:
        return x["tschq04-1"]


def get_common_cols(col1, col2):
    common_elements = set(col1).intersection(col2)
    return common_elements


import properties
import pandas as pd


def check_access(location):
    if location.exists() and location.is_file():
        return True
    else:
        return False


def initial_processing():
    # Read the csv of the tschq data and make the necessary things
    tschq = pd.read_pickle(properties.registration_file_location)

    hq = pd.read_pickle(properties.hearing_file_location)

    # If simulation file for tchq dataset exists add it.
    path_access = Path(properties.simulate_registration_file_location)
    hearing_path_access = Path(properties.simulate_hearing_file_location)

    if check_access(path_access):
        simulation_reg_file = pd.read_pickle(properties.simulate_registration_file_location)
        # Append the simulation file alongside when True
        tschq = tschq.append(simulation_reg_file)
    else:
        print("Simulated registration file is not created !!!")

    if check_access(hearing_path_access):
        simulation_hearing_file = pd.read_pickle(properties.simulate_hearing_file_location)
        hq = hq.append(simulation_hearing_file)
    else:
        print("Simulated hearing file is not created !!!")


    # Dropping users who do not have their time series
    drop_indexs = []

    drop_user_ids = [54, 60, 140, 170, 4, 6, 7, 9, 12, 19, 25, 53, 59, 130, 144, 145, 148, 156, 167]
    # indexes to be obtained
    for val in drop_user_ids:
        drop_indexs.append(tschq[tschq["user_id"] == val].index[0])

    # Drop those indexes of the users who do not have their time recordings
    tschq.drop(drop_indexs, inplace=True)
    tschq.reset_index(inplace=True, drop=True)

    # Cleaning tschq05 question. There is an abstraction for a row we add common value

    def filter_age(x):
        if isinstance(x, int):
            # Append the most common value obtained
            return tschq["tschq05"].value_counts().head(1).index[0]
        else:
            return x

    tschq["tschq05"] = tschq["tschq05"].apply(filter_age)

    # Drop the questionnaire_id and created_at
    tschq.drop(["questionnaire_id", "created_at"], axis=1, inplace=True)

    hq.isna().sum(axis=0)
    # By looking at the output we are sure that h5 and h6 do not contribute much and can be dropped
    hq.drop(["hq05", "hq06"], axis=1, inplace=True)
    hq_df = hq.set_index("user_id")
    df = tschq.join(hq_df.iloc[:, 2:], on="user_id")

    # Repeated code but it should be okay
    # Looking at the output, we can drop tschq25, tschq07-02, tschq04-2
    drop_cols = ["tschq01", "tschq25", "tschq07-2",
                 "tschq13", "tschq04-1", "tschq04-2"]

    # Getting percentage between 0 to 1 rather than score values
    df["tschq12"] = df["tschq12"].apply(lambda x: x / 100)
    df["tschq16"] = df["tschq16"].apply(lambda x: x / 100)
    df["tschq17"] = df["tschq17"].apply(lambda x: x / 100)

    df["tschq04"] = df.apply(create_cols_family_hist, axis=1)

    df.drop(drop_cols, axis=1, inplace=True)

    # Set the heom object, while using the required similarity
    # Alternative
    # Categorical boolean mask
    categorical_feature_mask = df.iloc[:, 1:].infer_objects().dtypes == object
    other_feature_mask = df.iloc[:, 1:].infer_objects().dtypes != object
    # filter categorical columns using mask and turn it into a list
    categorical_cols = df.iloc[:, 1:].columns[categorical_feature_mask].tolist()
    num_cols = df.iloc[:, 1:].columns[other_feature_mask].tolist()
    cat_idx = [df.iloc[:, 1:].columns.get_loc(val) for val in categorical_cols]
    num_idx = [df.iloc[:, 1:].columns.get_loc(val) for val in num_cols]

    return cat_idx, num_idx, df


import os
import traceback


def save_data_objs(df, quest_cmbs="all"):
    try:
        if not os.path.isdir(properties.model_location + "/simulate/" + quest_cmbs):
            os.makedirs(properties.model_location + "/simulate/" + quest_cmbs)
        utility.save_model("".join("/simulate/" + quest_cmbs + "/" + quest_cmbs + "_stat_q_data"), df)

        encoded_combined_df = smf.preprocess(df, quest_cmbs, age_bin=False,
                                             process_model_name="".join("/simulate/" + quest_cmbs + "/" +
                                                                        quest_cmbs + "_stat_q_data_oe_model"),
                                             prediction=False)

        # Save this encoded_data
        utility.save_model("".join("/simulate/" + quest_cmbs + "/" +
                                   quest_cmbs + "_stat_q_data_encoded"), encoded_combined_df)

        return encoded_combined_df

        # Use this data to build the data model over static data.
    except Exception:
        print(traceback.print_exc())


def weighted_average(distress_list):
    average = np.asarray(distress_list, dtype=float).mean()
    return average


# Function computes the weighted average as predictions for given prediction time point
def compute_weighted_avg(n_idx, encoded_data, pred_at_list, method="mean", dist_nn=None, wt_flag=False):

    preds = list()
    # Prediction for four time points
    for pval in pred_at_list:
        distress_list = list()
        for vals in n_idx:
            # print(user_id)
            # For this user now get the time series.
            # might have to iterate over as well.
            u_id = encoded_data["user_id"].iloc[vals]
            user_ts = tsg_data.get_usr_mday_ts_predict(int(u_id))
            # 3rd val of the series is s03 of the neighbor
            print("{}, {} Values ".format(int(pval), int(u_id)))
            if len(user_ts) > int(pval):
                value = user_ts[int(pval), :][3]
            elif len(user_ts) <= int(pval):
                value = user_ts[len(user_ts) - 1, :][3]

            distress_list.append(value)

        if wt_flag:
            print("Calling by weighted distance prediction for distress")
            preds.append(weighted_distance_prediction(distress_list, dist_nn))
        else:
            print("Calling weighted average to predict distress")
            preds.append(weighted_average(distress_list))
    return preds


def weighted_distance_prediction(p_preds, distance):
    # Inverse distance so that highest weight is given to the nearest one and least to the farther
    inv_dist = np.divide(1, distance)

    # s03 - tinnitus distress weighted by distance is given as
    s03_pred = (np.sum(np.multiply(p_preds, inv_dist)) / (np.sum(inv_dist)))

    return s03_pred


def compute(test_nn, encoded_data,
            pred_list, method="mean", dist_nn=None, wt_dist=False):
    # test_nn = [0, 3, 4]
    # pred_list = [0.1,0.23,0.27]
    from sklearn.linear_model import LinearRegression

    preds = list()
    for point in pred_list:
        nn_preds = list()
        intercepts_list = list()
        coeff_list = list()
        for nn in test_nn:
            u_id = encoded_data["user_id"].iloc[nn]
            user_ts = tsg_data.get_usr_mday_ts_predict(int(u_id))
            # Obtain the time series until time point and fit the data for linear regression
            diff_arr = np.abs(np.subtract(point, user_ts[:, 1]))
            diff_near_idx = np.where(diff_arr == diff_arr.min())
            print("minimum to the time point is at -- ", diff_near_idx)
            # difference near index. Handling for the length of users
            usr_idx = diff_near_idx[0][0]

            user_ts_p = user_ts[:usr_idx]
            user_ts_df = pd.DataFrame(user_ts_p, columns=["day", "day_sess_index",
                                                          "s02", "s03", "s04",
                                                          "s05", "s06", "s07"])
            X = user_ts_df[["day_sess_index"]]
            # We show for tinnitus distress. This can be extended to other physiological variables as well.
            y = user_ts_df[["s03"]]

            # Fit on X axis as time and Y as the s03 predictive value.
            reg_fit = LinearRegression(normalize=True)
            reg_fit.fit(X, y)

            # If weighted_distance is true, then predict by each of the nn_user and add to list. This will be used for
            # calculating weighted_distance_predictions.
            if wt_dist:
                nn_pred = reg_fit.predict(np.asarray(point).reshape(1, -1))
                nn_preds.append(nn_pred[0][0])
            else:
                intercepts_list.append(reg_fit.intercept_)
                coeff_list.append(reg_fit.coef_)

        if wt_dist:
            print("Predicting the value of s03 for the user by a weighted average weighted by distance")
            preds.append(weighted_distance_prediction(nn_preds, dist_nn))
        else:
            print("Predicting the value of s3 over the averaged slope and intercepts of "
                  "observations of the neighbors")

            # y = mx + c, where m is the average slope of the neighbors and c is the average intercept obtained.
            print("The equation to estimate s03 for the user is {}".format("".join(str(np.asarray(coeff_list).mean())) +
                                                                           "* time_index + " +
                                                                           str(np.asarray(intercepts_list).mean())))
            y = np.multiply(np.asarray(coeff_list).mean(), point) + np.asarray(intercepts_list).mean()
            preds.append(y)

    return preds


# Create test label as ground truth at prediction point.
def create_y_labels(test_data, prediction_at, method="mean"):
    y_test = list()
    for i in range(0, len(test_data)):
        test_ts_test1 = tsg_data.get_usr_mday_ts_predict(int(test_data.iloc[i]["user_id"]))
        # print(len(test_ts_test1))
        if len(test_ts_test1) >= prediction_at:
            y_test.append(test_ts_test1[prediction_at - 1][2])
        elif len(test_ts_test1) < prediction_at:
            y_test.append(test_ts_test1[len(test_ts_test1) - 1][2])
    return y_test


# Create reference points for multiple reference predictions
def get_pred_ref_points(user_id, ndays, method="mean"):
    # Using the default tsg which is mean observations of the user
    test_user_ts = tsg_data.get_usr_mday_ts_predict(user_id)

    # user_ts = tsg.get_usr_mday_ts_index_corrected(int(user_id))
    user_ts_idx = test_user_ts[:, 1]
    # ["date", "time_idx", "s02", "s03", "s04", "s05", "s06", "s07]
    user_distress = test_user_ts[:, 3]

    # Near evaluation. Change this for farther evaluations
    # Near -> 0.25
    # Far -> 1 - (Near)

    # Near points are of the sequence of observation because we are sure all stay until here.
    prediction_at = 10

    # Far prediction point is the last N% of the test user time series
    # prediction_at = round(len(user_ts_idx) * 0.80)
    y_labels = user_distress[prediction_at:prediction_at + ndays].tolist()
    prediction_at_list = user_ts_idx[prediction_at:prediction_at + ndays].tolist()

    return y_labels, prediction_at_list


def do_test(test_data, out_writer, csv_out_writer,
            ndays, near_idxs, encoded_data, fold_count="final",
            method="mean", dist_nn=None, wt_dist_flag=False):
    for i in range(0, len(test_data)):
        user_id = int(test_data.iloc[i]["user_id"])
        print("User- Id ", user_id)
        y_labels, prediction_at_list = get_pred_ref_points(user_id, ndays, method=method)

        # y_labels = create_y_labels(X_test, preds, method="mean")
        if wt_dist_flag:
            test_user_nn = near_idxs[i]
            test_user_dist = dist_nn[i]
            pred_weighted_average = compute_weighted_avg(test_user_nn, encoded_data, prediction_at_list,
                                                         method=method, dist_nn=test_user_dist, wt_flag=wt_dist_flag)

            pred_lr = compute(test_user_nn, encoded_data, prediction_at_list,
                              method=method, dist_nn=test_user_dist, wt_dist=wt_dist_flag)
        else:
            test_user_nn = near_idxs[i]
            pred_weighted_average = compute_weighted_avg(test_user_nn, encoded_data, prediction_at_list,
                                                         method=method, dist_nn=None, wt_flag=False)
            pred_lr = compute(test_user_nn, encoded_data, prediction_at_list,
                              method=method, dist_nn=None, wt_dist=False)

        # calculate MAE, MSE, RMSE
        if not fold_count == "final":
            print("Evaluating for the fold-" + str(count) + " for the forecast reference points - " +
                  str(prediction_at_list))
            out_writer.write("Evaluating for the fold-" + str(count) + " for the forecast reference points -- " +
                             str(prediction_at_list) + "for the method evaluation -- " + str(method) + "\n")
        else:
            print("Evaluating for the final model over the " + " forecast reference points - " +
                  str(prediction_at_list))
            out_writer.write("Evaluating for the final model over the" + " forecast reference points -- " +
                             str(prediction_at_list) + "for the method evaluation -- " + str(method) + "\n")

        print("Computing MAE, MSE, RMSE for weighted average based predictions on the User -- " + str(user_id))
        out_writer.write("Computing MAE, MSE, RMSE for weighted average based predictions"
                         " plain and on N days on the User -- " + str(user_id) + "\n")
        print("---------------------------------------------------------------")
        out_writer.write("---------------------------------------------------------------\n")
        print("MAE -- ", mean_absolute_error(y_labels, pred_weighted_average))
        out_writer.write("MAE -- " + str(mean_absolute_error(y_labels, pred_weighted_average)) + "\n")

        # MAE for N days
        print("MAE for N days -- ",
              str(mean_absolute_error(y_labels, pred_weighted_average) / ndays))
        out_writer.write("MAE for N days -- "
                         + str(mean_absolute_error(y_labels, pred_weighted_average) / ndays) + "\n")

        print("MSE -- ", mean_squared_error(y_labels, pred_weighted_average))
        out_writer.write("MSE -- " + str(mean_squared_error(y_labels, pred_weighted_average)) + "\n")

        # MSE for N days
        print("MSE for N days-- ", str(mean_squared_error(y_labels, pred_weighted_average) / ndays))
        out_writer.write(
            "MSE for N days -- " + str(mean_squared_error(y_labels, pred_weighted_average) / ndays) + "\n")

        print("RMSE -- ", np.sqrt(mean_squared_error(y_labels, pred_weighted_average)))
        out_writer.write("RMSE -- " + str(np.sqrt(mean_squared_error(y_labels, pred_weighted_average))) + "\n")

        # RMSE for N days
        print("RMSE for N days -- ", str(np.sqrt(mean_squared_error(y_labels, pred_weighted_average)) / ndays))
        out_writer.write("RMSE for N days -- " + str(
            np.sqrt(mean_squared_error(y_labels, pred_weighted_average)) / ndays) + "\n")

        # pred_lr = compute_linear_regression(test_user_nn, encoded_data, prediction_at_list , method="mean")
        m_count = 0
        # Writing to csv file
        if not fold_count == "final":
            csv_out_writer.write("".join(str(user_id) + "," +
                                         str(count) + "," +
                                         str(mean_absolute_error(y_labels, pred_weighted_average)) + "," +
                                         str(mean_squared_error(y_labels, pred_weighted_average)) + "," +
                                         str(np.sqrt(mean_squared_error(y_labels, pred_weighted_average))) + "," +
                                         "weighted_average" + ","
                                         # str(y_labels) + "," +
                                         # str(pred_weighted_average)
                                         + str(y_labels[0]) + "," + str(y_labels[1]) + "," + str(y_labels[2])
                                         + "," + str(pred_weighted_average[0]) + "," + str(pred_weighted_average[1])
                                         + "," + str(pred_weighted_average[2]) + "\n"))
        else:
            csv_out_writer.write("".join(str(user_id) + "," +
                                         str("test") + "," +
                                         str(mean_absolute_error(y_labels, pred_weighted_average)) + "," +
                                         str(mean_squared_error(y_labels, pred_weighted_average)) + "," +
                                         str(np.sqrt(mean_squared_error(y_labels, pred_weighted_average))) + "," +
                                         "weighted_average" + ","
                                         + str(y_labels[0]) + "," + str(y_labels[1]) + "," + str(y_labels[2])
                                         + "," + str(pred_weighted_average[0]) + "," + str(pred_weighted_average[1])
                                         + "," + str(pred_weighted_average[2]) + "\n"))
            # + str(y_labels) + str(pred_weighted_average)

        print("---------------------------------------------------------------")
        out_writer.write("---------------------------------------------------------------\n")
        print("Computing MAE, MSE, RMSE for {} {} based predictions for the user -- {}"
              .format(str("weighted_distance" + str(wt_dist_flag)), str("linear_regression"), str(user_id)))
        out_writer.write("Computing MAE, MSE, RMSE for {} {} based predictions for the user -- {} \n"
                         .format(str("weighted_distance" + str(wt_dist_flag)), str("linear_regression"), str(user_id)))
        print("MAE -- ", mean_absolute_error(y_labels, pred_lr))
        out_writer.write("MAE -- " + str(mean_absolute_error(y_labels, pred_lr)) + "\n")
        print("MSE -- ", mean_squared_error(y_labels, pred_lr))
        out_writer.write("MSE -- " + str(mean_squared_error(y_labels, pred_lr)) + "\n")
        print("RMSE -- ", np.sqrt(mean_squared_error(y_labels, pred_lr)))
        out_writer.write("RMSE -- " + str(np.sqrt(mean_squared_error(y_labels, pred_lr))) + "\n")
        print("---------------------------------------------------------------")
        out_writer.write("---------------------------------------------------------------\n")

        # Write to csv file
        if not fold_count == "final":
            csv_out_writer.write("".join(str(user_id) + "," +
                                         str(count) + "," +
                                         str(mean_absolute_error(y_labels, pred_lr)) + "," +
                                         str(mean_squared_error(y_labels, pred_lr)) + "," +
                                         str(np.sqrt(mean_squared_error(y_labels, pred_lr))) + "," +
                                         str("lr") + ","
                                         + str(y_labels[0]) + "," + str(y_labels[1]) + "," + str(y_labels[2])
                                         + "," + str(pred_lr[0]) + "," + str(pred_lr[1]) + "," + str(
                pred_lr[2]) + "\n"))
        else:
            csv_out_writer.write("".join(str(user_id) + "," +
                                         str("test") + "," +
                                         str(mean_absolute_error(y_labels, pred_lr)) + "," +
                                         str(mean_squared_error(y_labels, pred_lr)) + "," +
                                         str(np.sqrt(mean_squared_error(y_labels, pred_lr))) + "," +
                                         str("lr") + ","
                                         + str(y_labels[0]) + "," + str(y_labels[1]) + "," + str(y_labels[2])
                                         + "," + str(pred_lr[0]) + "," + str(pred_lr[1]) + "," + str(
                pred_lr[2]) + "\n"))


import properties
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
# Create prediction reference points
from sklearn.metrics import *
# Here, to change to different evaluations
from time_series_grp import TimeSeriesGroupProcessing
from HEOM import HEOM
from scipy.spatial.distance import pdist, squareform

# Change method and execute to get the predictions appropriately, these are configurations
eval_method = "mean"
wt_distance = False
# Random Neighbors
rand_neighbors = False
# Default day readings for all test users must be at mean and prediction are between min - mean - max
tsg_data = TimeSeriesGroupProcessing(method=eval_method)
# For all combinations evaluation it must be set to True
quest_cmb_all = True
# Same random state needs to be maintained to get consistent test data over all combinations and repeatable results
random_state = 1220
# It is the setting to get the ahead prediction for tinnitus distress, 3 here means for 3 days
# min it is a day and max of about 60days between points which is not an usual scenario
ndays = 3
# Load user build models over the time series observations


# user_best_models = utility.load_ts_model("best_params_usr_models")
# user_best_estimators = utility.load_ts_model("cross_val_estimators")
# KFOLDS - Evaluation over K=5 folds are done.
# Build the default model with all the combination.
if not quest_cmb_all:
    for key, val in properties.quest_comb.items():
        # Build model for each category
        print("Building model for the question combination -- " + str(key))
        out_writer = open("".join("output/output_simulate_" + str(key) + "_" + str(eval_method) + "_heom_norm.txt"), "w+")
        csv_out_writer = open("".join("output/output__simulate_" + str(key) + "_" + str(eval_method) + "_heom_norm.csv"), "w+")

        # key = "bg_tinnitus_history"
        # val = properties.quest_comb[key]

        cat_idx, num_idx, combined_df = smf.initial_processing(key, val, append_synthethic=True)
        # Build and get the knn model for prediction over test instances.
        # Save the data objs

        encoded_data = save_data_objs(combined_df, key)

        csv_out_writer.write("".join("user_id,fold,mae,mse,rmse,algorithm,"
                                     "ref_p1,ref_p2,ref_p3,pred_p1,pred_p2,pred_p3\n"))


        # Create a specific test set as per requirements to contain digital twin, outlier and normal instances

        random_user_ids = encoded_data["user_id"].sample(n=3, random_state=42).to_list()

        """
        10 test users in following format:
        1. Outliers -- [8,20,27,149]
        2. DT - [44428, 444154, 444133]
        3. Random Users - random_user_ids with random state same so always same test set is retrieved.
        """
        test_simulation_ids = [8, 20, 27, 149, 44428, 444154, 444133] + random_user_ids

        test = encoded_data[encoded_data["user_id"].isin(test_simulation_ids)]
        X = encoded_data[~encoded_data["user_id"].isin(test_simulation_ids)]

        def filter_train_ids(x):
            # print(x)
            if x["user_id"] in train_user_ids:
                return x


        def filter_test_ids(x):
            # print(x)
            if x["user_id"] in test_user_ids:
                return x


        train_user_ids = X["user_id"].to_list()

        X_train_data_ui = combined_df.apply(filter_train_ids, axis=1, result_type="broadcast").dropna()
        X_train_data_ui["user_id"] = X_train_data_ui["user_id"].apply(int)
        # Save the non encoded train data for visualization purposes
        utility.save_model("".join("/simulate/" + key + "/" + key + "_train_stat_q_data"), X_train_data_ui)

        # filter and get the data to show to the UI for the test data.
        test_user_ids = test["user_id"].to_list()

        X_test_data_ui = combined_df.apply(filter_test_ids, axis=1, result_type="broadcast").dropna()

        X_test_data_ui["user_id"] = X_test_data_ui["user_id"].apply(int)

        # Save the data_ui object as json
        test_data = {}
        test_data["users"] = X_test_data_ui.to_dict("r")
        utility.save_data("".join("simulate/test_data_ui_" + key), test_data)

        heom = HEOM(X.to_numpy(), cat_idx, num_idx)
        sim_matrix = pdist(X.to_numpy()[:, 1:], heom.heom_distance)
        mean_heom_distance = sim_matrix.mean()

        knn = NearestNeighbors(n_neighbors=5, metric=heom.heom_distance, radius=mean_heom_distance)
        knn.fit(X.iloc[:, 1:])
        dist, test_idx = knn.kneighbors(test.to_numpy()[:, 1:], n_neighbors=5)

        do_test(test, out_writer, csv_out_writer, ndays, test_idx, X,
                fold_count="final", method=eval_method, dist_nn=None, wt_dist_flag=wt_distance)

        utility.save_model("".join("simulate/" + key + "/" + "knn_static"), knn)
        utility.save_model("".join("simulate/" + key + "/" + "train_sim_data.pckl"), X)

        out_writer.close()
        csv_out_writer.close()

else:
    cat_idx, num_idx, combined_df = initial_processing()

    # Build model for each category
    print("Building model for the question combination -- " + str("overall"))

    # Take this combined_df and split into train and test.

    # Split some data out of test as part unseen from the UI

    # data_ui_val, data = combined_df.iloc[:5, :], combined_df.iloc[5:, :]


    # Save the data objs
    encoded_data = save_data_objs(combined_df, "overall")

    random_user_ids = encoded_data["user_id"].sample(n=3, random_state=42).to_list()
    test_simulation_ids = [8, 20, 27, 149, 44428, 444154, 444133] + random_user_ids

    test = encoded_data[encoded_data["user_id"].isin(test_simulation_ids)]
    X = encoded_data[~encoded_data["user_id"].isin(test_simulation_ids)]


    def filter_train_ids(x):
        # print(x)
        if x["user_id"] in train_user_ids:
            return x


    def filter_test_ids(x):
        # print(x)
        if x["user_id"] in test_user_ids:
            return x


    train_user_ids = X["user_id"].to_list()

    X_train_data_ui = combined_df.apply(filter_train_ids, axis=1, result_type="broadcast").dropna()
    X_train_data_ui["user_id"] = X_train_data_ui["user_id"].apply(int)

    utility.save_model("".join("/simulate/" + "overall" + "/" + "overall" + "_train_stat_q_data"), X_train_data_ui)

    # filter and get the data to show to the UI for the test data.
    test_user_ids = test["user_id"].to_list()

    X_test_data_ui = combined_df.apply(filter_test_ids, axis=1, result_type="broadcast").dropna()

    X_test_data_ui["user_id"] = X_test_data_ui["user_id"].apply(int)

    # Save the data_ui object as json
    test_data = {}
    test_data["users"] = X_test_data_ui.to_dict("r")
    utility.save_data("simulate/test_data_ui_x_test", test_data)

    count = 0

    out_writer = open("output/simulate_overall_output_folds_" + str(eval_method) + ".txt", "w+")
    csv_out_writer = open("output/simulate_overall_output_folds_" + str(eval_method) + ".csv", "w+")

    # First get the time series for a given test patient and the reference point and iterate to evaluate
    csv_out_writer.write("user_id,fold,mae,mse,rmse,algorithm,"
                         "ref_p1,ref_p2,ref_p3,pred_p1,pred_p2,pred_p3\n")

    # Split the data into train and test and evaluate as a final model
    from sklearn.model_selection import train_test_split
    import utility
    from HEOM import HEOM
    # Can be done at prediction too.
    from sklearn.metrics.pairwise import cosine_distances
    from sklearn.linear_model import LinearRegression
    from scipy.spatial.distance import pdist, squareform

    heom = HEOM(X.to_numpy()[:, 1:], cat_idx, num_idx)
    sim_matrix = pdist(X.to_numpy()[:, 1:], heom.heom_distance)
    mean_heom_distance = sim_matrix.mean()

    knn = NearestNeighbors(n_neighbors=5, metric=heom.heom_distance, radius=mean_heom_distance)
    knn.fit(X.to_numpy()[:, 1:])
    dist, idx_test = knn.kneighbors(test.to_numpy()[:, 1:], n_neighbors=5)

    do_test(test, out_writer, csv_out_writer, ndays, idx_test, X,
            fold_count="final", method=eval_method, dist_nn=None, wt_dist_flag=wt_distance)

    out_writer.close()
    csv_out_writer.close()

    # Save the simulated neighborhood results
    utility.save_model("".join("/simulate/overall/" + "knn_static"), knn)
    utility.save_model("".join("/simulate/overall" + "/" + "train_sim_data.pckl"), X)

