import utility
import static_sim_functions as smf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from time_series_grp import TimeSeriesGroupProcessing
from RandomNeighbors import RandomNeighbors
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import KFold

import ml_modelling_ts as ml_ts

'''
This is just a run of the approaches using the methodologies, save the neighborhood for UI.
'''


def common_processing(df):
    # Getting percentage between 0 to 1 rather than score values
    df["tschq12"] = df["tschq12"].apply(lambda x: x / 100)
    df["tschq16"] = df["tschq16"].apply(lambda x: x / 100)
    df["tschq17"] = df["tschq17"].apply(lambda x: x / 100)

    # Feature engineering family history
    df["tschq04"] = df.apply(create_cols_family_hist, axis=1)

    return df


# Common elements
# Feature engineering family history
def create_cols_family_hist(x):
        if x["tschq04-1"] == "YES":
            lst_sorted = sorted(x["tschq04-2"])
            list_to_str = "_".join([val for val in lst_sorted])
            return list_to_str
        else:
            return x["tschq04-1"]


def get_common_cols(col1, col2):
    common_elements = set(col1).intersection(col2)
    return common_elements

import properties
import pandas as pd
def initial_processing():
    # Read the csv of the tschq data and make the necessary things
    tschq = pd.read_pickle(properties.data_location + "/input_pckl/" + "3_q.pckl")

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

    # Lets read and join two questionnaires tschq and hq
    hq = pd.read_pickle("data/input_pckl/4_q.pckl")
    hq.isna().sum(axis=0)
    # By looking at the output we are sure that h5 and h6 do not contribute much and can be dropped
    hq.drop(["hq05", "hq06"], axis=1, inplace=True)
    hq_df = hq.set_index("user_id")
    df = tschq.join(hq_df.iloc[:, 2:], on="user_id")

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
        if not os.path.isdir(properties.model_location + quest_cmbs):
            os.makedirs(properties.model_location + quest_cmbs)
        utility.save_model("".join(quest_cmbs + "/" + quest_cmbs + "_stat_q_data"), df)

        encoded_combined_df = smf.preprocess(df, quest_cmbs, age_bin=False,
                                                 process_model_name="".join(quest_cmbs + "/" +
                                                                            quest_cmbs + "_stat_q_data_oe_model"),
                                                 prediction=False)

        # Save this encoded_data
        utility.save_model("".join(quest_cmbs + "/" +
                                       quest_cmbs + "_stat_q_data_encoded"), encoded_combined_df)

        return encoded_combined_df

        # Use this data to build the data over static data.
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
            u_id = encoded_data["user_id"].iloc[vals]
            user_ts = tsg_data.get_usr_mday_ts_predict(int(u_id))
            # 3rd val of the series is s03 of the neighbor
            print("{}, {} Values ".format(int(pval), int(u_id)))
            if len(user_ts) > int(pval):
                value = user_ts[int(pval), :][3]
            elif len(user_ts) <= int(pval):
                value = user_ts[len(user_ts)-1, :][3]

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

    #s03 - tinnitus distress weighted by distance is given as
    s03_pred = (np.sum(np.multiply(p_preds, inv_dist)) / (np.sum(inv_dist)))

    return s03_pred


def compute(test_nn, encoded_data,
            pred_list, method="mean", dist_nn=None, wt_dist=False):
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


def compute_linear_regression(test_nn, encoded_data, pred_list, method="mean"):
    #test_nn = test_user_nn
    #pred_list = prediction_at_list
    from sklearn.linear_model import LinearRegression
    preds = list()
    for point in pred_list:
        attr_list = list()
        intercepts_list = list()
        coeff_list = list()
        for nn in test_nn:
            u_id = encoded_data["user_id"].iloc[nn]
            user_ts = tsg_data.get_m_day_ts_enumerate(int(11))
            diff_arr = np.abs(np.subtract(point, user_ts[:, 1]))
            diff_near_idx = np.where(diff_arr == diff_arr.min())
            print(diff_near_idx)
            # difference near index
            usr_vals = np.array([user_ts[n_id] for n_id in diff_near_idx[0]])
            if len(usr_vals) > 1:
                value = usr_vals.mean(axis=0)
                print("vavg" + str(value))
            else:
                value = usr_vals[0]
                print("v" + str(value))

            attr_list.append(value)


            df = pd.DataFrame(user_ts)
            df.columns = ["day", "day_session_id",
                          "s02", "s03",
                          "s04", "s05",
                          "s06", "s07"]
            reg_model = LinearRegression(normalize=True)
            user_x = df[["day_session_id", "s04", "s05", "s06"]].to_numpy()
            user_s03 = df[["s03"]].to_numpy().ravel()
            reg_model.fit(user_x, user_s03)
            intercepts_list.append(reg_model.intercept_)
            coeff_list.append(reg_model.coef_)
        # y = mx + c, where m is the average slope of the neighbors and c is the average intercept obtained.

        # convert coeff's to numpy for manipulations
        numpy_attr_list = np.array(attr_list)
        print(numpy_attr_list)
        avg_np_attr_list = numpy_attr_list[:, 4:].mean(axis=0)

        print(avg_np_attr_list)

        numpy_coeff_list = np.array(coeff_list)

        print(numpy_coeff_list)
        print(numpy_coeff_list.mean(axis=0))

        # Day_index, s02, s04, s05, s06 ,s07 - Use only the fit independent features to estimate the dependent
        y = np.multiply(numpy_coeff_list[:, 0].mean(), point) + \
            np.multiply(numpy_coeff_list[:, 1].mean(), avg_np_attr_list[0]) + \
            np.multiply(numpy_coeff_list[:, 2].mean(), avg_np_attr_list[1]) + \
            np.multiply(numpy_coeff_list[:, 3].mean(), avg_np_attr_list[2]) + \
            np.asarray(intercepts_list).mean()
        preds.append(y)
    print(preds)
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

    user_ts_idx = test_user_ts[:, 1]
    # ["date", "time_idx", "s02", "s03", "s04", "s05", "s06", "s07]
    user_distress = test_user_ts[:, 3]

    # Near evaluation. Change this for farther evaluations
    # Near -> 0.20, 0.10
    # Far -> 1 - (Near)

    # Near points are of the sequence of observation because we are sure all stay until here.
    #prediction_at = 10

    # Far prediction point is the last N% of the test user time series
    # It is tested for 0.75, 0.8, 0.9
    prediction_at = round(len(user_ts_idx) * 0.80)
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
        # Weighting by inverse of neighbor
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


        # calculate
        if not fold_count == "final":
            print("Evaluating for the fold-" + str(fold_count) + " for the forecast reference points - " +
                  str(prediction_at_list))
            out_writer.write("Evaluating for the forecast reference points -- " +
                             str(prediction_at_list) + "for the method evaluation -- " + str(method) + "\n")
        else:
            print("Evaluating for forecast reference points - " +
                  str(prediction_at_list))
            out_writer.write("Evaluating over the forecast reference points -- " +
                             str(prediction_at_list) + "for the method evaluation -- " + str(method) + "\n")

        print("Computing RMSE for weighted average based predictions on the User -- " + str(user_id))
        print("---------------------------------------------------------------")
        out_writer.write("---------------------------------------------------------------\n")

        print("RMSE -- ", np.sqrt(mean_squared_error(y_labels, pred_weighted_average)))
        out_writer.write("RMSE -- " + str(np.sqrt(mean_squared_error(y_labels, pred_weighted_average))) + "\n")


        # Writing to csv file
        if not fold_count == "final":
            csv_out_writer.write("".join(str(user_id) + "," +
                                          str(np.sqrt(mean_squared_error(y_labels, pred_weighted_average))) + "," +
                                          "weighted_average" + ","
                                         + str(y_labels[0]) + "," + str(y_labels[1]) + "," + str(y_labels[2])
                                         + "," + str(pred_weighted_average[0]) + "," + str(pred_weighted_average[1])
                                         + "," + str(pred_weighted_average[2]) + "\n"))
        else:
            csv_out_writer.write("".join(str(user_id) + "," +
                                          str(np.sqrt(mean_squared_error(y_labels, pred_weighted_average))) + "," +
                                          "weighted_average" + ","
                                         + str(y_labels[0]) + "," + str(y_labels[1]) + "," + str(y_labels[2])
                                         + "," + str(pred_weighted_average[0]) + "," + str(pred_weighted_average[1])
                                         + "," + str(pred_weighted_average[2]) + "\n"))

        print("-----------------------------------------------------------------------------")
        out_writer.write("---------------------------------------------------------------\n")
        print("Computing RMSE for {} {} based predictions for the user -- {}"
              .format(str("weighted_distance" + str(wt_dist_flag)), str("linear_regression"), str(user_id)))
        out_writer.write("Computing RMSE for {} {} based predictions for the user -- {} \n"
                         .format(str("weighted_distance" + str(wt_dist_flag)), str("linear_regression"), str(user_id)))
        print("RMSE -- ", np.sqrt(mean_squared_error(y_labels, pred_lr)))
        out_writer.write("RMSE -- " + str(np.sqrt(mean_squared_error(y_labels, pred_lr))) + "\n")
        print("---------------------------------------------------------------")
        out_writer.write("---------------------------------------------------------------\n")

        # Write to csv file
        if not fold_count == "final":
            csv_out_writer.write("".join(str(user_id) + "," +
                                         str(np.sqrt(mean_squared_error(y_labels, pred_lr))) + "," +
                                         str("lr") + ","
                                         + str(y_labels[0]) + "," + str(y_labels[1]) + "," + str(y_labels[2])
                                         + "," + str(pred_lr[0]) + "," + str(pred_lr[1]) + "," + str(
                pred_lr[2]) + "\n"))
        else:
            csv_out_writer.write("".join(str(user_id) + "," +
                                         str(np.sqrt(mean_squared_error(y_labels, pred_lr))) + "," +
                                         str("lr") + ","
                                         + str(y_labels[0]) + "," + str(y_labels[1]) + "," + str(y_labels[2])
                                         + "," + str(pred_lr[0]) + "," + str(pred_lr[1]) + "," + str(
                pred_lr[2]) + "\n"))


# Change method and execute to get the predictions appropriately, these are configurations
eval_method = "mean"
# Default day readings for all test users must be at mean and prediction are between min - mean - max

tsg_data = TimeSeriesGroupProcessing(method=eval_method)
# For all combinations evaluation it must be set to True
quest_cmb_all = False
# Same random state needs to be maintained to get consistent test data over all combinations and repeatable results
random_state = 1220
# It is the setting to get the ahead prediction for tinnitus distress, 3 here means for 3 days
# min it is a day and max of about 60days between points which is not an usual scenario
ndays = 3

# Build the default NN with all the combination.
if not quest_cmb_all:
    for key, val in properties.quest_comb.items():
        # Build NN for each category
        print("Building NN for the question combination -- " + str(key))

        cat_idx, num_idx, combined_df = smf.initial_processing(key, val, append_synthethic=False)
        # Build and get the knn NN for prediction over test instances.
        # Save the data objs

        encoded_data = save_data_objs(combined_df, key)

        out_writer = open("".join("output/output_" + str(key) + "_" + str(eval_method) + "_heom_norm.txt"), "w+")
        csv_out_writer = open("".join("output/output_" + str(key) + "_" + str(eval_method) + "_heom_norm.csv"), "w+")

        csv_out_writer.write("".join("user_id,rmse,algorithm,"
                                     "ref_p1,ref_p2,ref_p3,pred_p1,pred_p2,pred_p3\n"))

        #Create a test set
        X, test = train_test_split(encoded_data,
                                   test_size=0.20,
                                   random_state=random_state)

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
        utility.save_model("".join(key + "/" + key + "_train_stat_q_data"), X_train_data_ui)

        # filter and get the data to show to the UI for the test data.
        test_user_ids = test["user_id"].to_list()

        X_test_data_ui = combined_df.apply(filter_test_ids, axis=1, result_type="broadcast").dropna()

        X_test_data_ui["user_id"] = X_test_data_ui["user_id"].apply(int)

        # Save the data_ui object as json
        #test_data = {}
        #test_data["users"] = X_test_data_ui.to_dict("r")
        #utility.save_data("".join("test_data_ui_" + key), test_data)

        from HEOM import HEOM
        # Can be done at prediction too.
        from sklearn.metrics.pairwise import cosine_distances
        from sklearn.linear_model import LinearRegression
        from scipy.spatial.distance import pdist, squareform
        from scipy.stats import zscore

        heom = HEOM(X.to_numpy(), cat_idx, num_idx)
        sim_matrix = pdist(X.to_numpy()[:, 1:], heom.heom_distance)
        mean_heom_distance = sim_matrix.mean()

        knn = NearestNeighbors(n_neighbors=5, metric=heom.heom_distance, radius=mean_heom_distance)
        knn.fit(X.iloc[:, 1:])
        dist, test_idx = knn.kneighbors(test.to_numpy()[:, 1:], n_neighbors=5)

        # Execute without any varying for saving the KNN as pickle to be used by UI
        do_test(test, out_writer, csv_out_writer, ndays, test_idx, X,
                fold_count="final", method=eval_method, dist_nn=None, wt_dist_flag=False)

        utility.save_model("".join(key + "/" + "knn_static"), knn)
        utility.save_model("".join(key + "/" + "train_sim_data.pckl"), X)

        out_writer.close()
        csv_out_writer.close()


# All feature combinations

cat_idx, num_idx, combined_df = initial_processing()


# Build KNN for each category
print("Building KNN for the question combination -- " + str("overall"))

# Save the data objs
encoded_data = save_data_objs(combined_df, "overall")


X, test = train_test_split(encoded_data,
                                   test_size=0.20,
                                   random_state=random_state)


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

# Save in overall.
utility.save_model("".join("overall" + "/" + "overall" + "_train_stat_q_data"), X_train_data_ui)

# filter and get the data to show to the UI for the test data.
test_user_ids = test["user_id"].to_list()

X_test_data_ui = combined_df.apply(filter_test_ids, axis=1, result_type="broadcast").dropna()

X_test_data_ui["user_id"] = X_test_data_ui["user_id"].apply(int)

# Save the data_ui object as json
test_data = {}
test_data["users"] = X_test_data_ui.to_dict("r")
utility.save_data("test_data_ui_x_test", test_data)

# Save the results to out_writer
out_writer = open("output/overall_output_folds_" + str(eval_method) + ".txt", "w+")
csv_out_writer = open("output/overall_output_folds_" + str(eval_method) + ".csv", "w+")

# First get the time series for a given test patient and the reference point and iterate to evaluate
csv_out_writer.write("user_id,rmse,algorithm,"
                     "ref_p1,ref_p2,ref_p3,pred_p1,pred_p2,pred_p3\n")


# Split the data into train and test
from sklearn.model_selection import train_test_split
import utility
from HEOM import HEOM
#Can be done at prediction too.
from sklearn.metrics.pairwise import cosine_distances
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import pdist, squareform
from scipy.stats import zscore


heom = HEOM(X.to_numpy()[:, 1:], cat_idx, num_idx)
sim_matrix = pdist(X.to_numpy()[:, 1:], heom.heom_distance)
mean_heom_distance = sim_matrix.mean()

knn = NearestNeighbors(n_neighbors=5, metric=heom.heom_distance, radius=mean_heom_distance)
knn.fit(X.to_numpy()[:, 1:])
dist, idx_test = knn.kneighbors(test.to_numpy()[:, 1:], n_neighbors=5)

# First get the time series for a given test patient and the reference point and iterate to evaluate

do_test(test, out_writer, csv_out_writer, ndays, idx_test, X,
        fold_count="final", method=eval_method, dist_nn=None, wt_dist_flag=False)

out_writer.close()
csv_out_writer.close()

# End save the nearest neighbor as data objects, so that can be used from the UI
utility.save_model("".join("overall/" + "knn_static"), knn)
utility.save_model("".join("overall" + "/" + "train_sim_data.pckl"), X)


'''
 ML Modelling based on s02 - loudness.
 Note: This has to be run once the all feature execution is completed since we build upon a custom similarity matrix,
 it is essential that the same split of train test happen so that it can be verified from the application.
'''

# Create train and test containing same users in train and test as per static data. (Note: Run above code and then this
# because same set of train test users are used)

def splitData(dataset, test_user_ids):
    train_data = dataset[~dataset["user_id"].isin(test_user_ids)]
    test_data = dataset[dataset["user_id"].isin(test_user_ids)]
    return train_data, test_data


# Save both train and test matrix
def save_ts_objs(train, test, location_name):
    try:
        if not os.path.isdir(properties.model_location + location_name):
            os.makedirs(properties.model_location + location_name)
        utility.save_model("".join(location_name + "/" + location_name + "_train_data"), train)
        utility.save_model("".join(location_name + "/" + location_name + "_test_data"), test)

    except Exception:
        print(traceback.print_exc())


X = ml_ts.process_data(grouping="day")

# Calculate pairwise distance and create a dataframe for the same
from scipy.spatial.distance import pdist, squareform
# Cross validate here based on the same split of static data here.
# Note: Only one combination will be present
C = np.zeros((X.shape[0], X.shape[0]))
for i in range(0, len(X)):
    for j in range(0, len(X)):
        dist = ml_ts.compute_dist(X[:, 1][i], X[:, 1][j])
        C[i][j] = dist

C_df = pd.DataFrame(C)


#C_df.to_csv("sim_ema.csv")

# Threshold overall distance for making within radius
threshold_distance = sum(C_df.mean())/len(C_df)


user_ids = []
for val in X:
    user_ids.append(val[0])

C_df["user_id"] = user_ids


train_data, test_data = splitData(C_df, test_user_ids)
# Save the time series data objects as dynamic_ts into model folder
save_ts_objs(train_data, test_data, "dynamic_ts")

out_writer = open("".join("output/output_ema_" + str(eval_method) + "_.txt"), "w+")
csv_out_writer = open("".join("output/output_ema_" + str(eval_method) + "_.csv"), "w+")

csv_out_writer.write("user_id,rmse,algorithm,"
                     "ref_p1,ref_p2,ref_p3,pred_p1,pred_p2,pred_p3\n")

# Test on the final test set. Note there is no varying K just to save the NN here.
# It should be noted we use NearesetNeighbors and not KNearestNeighbors classifier.
knn_ema = NearestNeighbors(n_neighbors=5, metric="precomputed", radius=threshold_distance)
knn_ema.fit(train_data[train_data.index])
ema_dist, ema_idx = knn_ema.kneighbors(test_data[train_data.index], n_neighbors=5)
# First get the time series for a given test patient and the reference point and iterate to evaluate
do_test(test_data, out_writer, csv_out_writer, ndays, ema_idx, encoded_data,
        fold_count="final", method=eval_method, dist_nn=None, wt_dist_flag=False)

# Close the writers
out_writer.close()
csv_out_writer.close()

# Save the similarity search index KNN
utility.save_model("".join("dynamic_ts" + "/" + "dynamic_ts" + "_knn"), knn_ema)
