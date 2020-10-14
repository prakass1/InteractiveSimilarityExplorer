import utility
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import static_sim_functions as smf
# import ts_preprocessing as ts_data
import numpy as np
import pickle
# import ts_group_processing as tsg_data
# import machine_learning as ml
import pandas as pd
import properties
from sklearn.preprocessing import PolynomialFeatures

'''
Model building and entity prediction is done here. We evaluate it with 5-fold cv and then against each sub set of features.
As we are creating a visualization tool we do not do automatic sub space identification.
'''
test_rmses = []

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

    # Dropping users who do not have their time series
    drop_indexs = []
    # User having less than 10 days of observations when grouped by their day at each of the months
    # are not included in the analysis.
    drop_user_ids = [54, 60, 140, 170, 4, 6, 7, 9,
                     12, 19, 25, 53, 59, 130, 144, 145, 148, 156, 167]
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

    # Lets read and join two questionnaires tschq and hq
    hq = pd.read_pickle("data/input_pckl/4_q.pckl")
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
        #if not os.path.isdir(properties.model_location + quest_cmbs):
        #    os.makedirs(properties.model_location + quest_cmbs)
        #utility.save_model("".join(quest_cmbs + "/" + quest_cmbs + "_stat_q_data"), df)

        # Preprocess and save the encoded. This is much needed while testing the users from the app.
        # Note on the file name here passed as a parameter to the function.
        encoded_combined_df = smf.preprocess(df, quest_cmbs, age_bin=False,
                                             process_model_name="".join(quest_cmbs + "/" +
                                                                        quest_cmbs + "_stat_q_data_oe_model"),
                                             prediction=False, save_model=False)


        return encoded_combined_df

        # Use this data to build the data NN over static data.
    except Exception:
        print(traceback.print_exc())


def weighted_average(distress_list):
    average = np.asarray(distress_list, dtype=float).mean()
    return average


# Function computes the weighted average as predictions for given prediction time point
def compute_weighted_avg(n_idx, encoded_d, pred_at_list, method="mean", random_idx=False,
                         ema_s02=False, dist_nn=None, wt_flag=False):

    train_uids = encoded_d["user_id"].to_numpy()

    preds = list()
    # Prediction for four time points
    for pval in pred_at_list:
        distress_list = list()
        for vals in n_idx:

            if random_idx:
                u_id = encoded_d["user_id"].loc[vals]

            elif ema_s02:
                u_id = train_uids[vals]
            else:
                u_id = encoded_d["user_id"].iloc[vals]

            user_ts = tsg_data.get_usr_mday_ts_predict(int(u_id))

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


# inverse of distance based.
def weighted_distance_prediction(p_preds, distance):
    # Inverse distance so that highest weight is given to the nearest one and least to the farther
    inv_dist = np.divide(1, distance)

    # s03 - tinnitus distress weighted by distance is given as
    s03_pred = (np.sum(np.multiply(p_preds, inv_dist)) / (np.sum(inv_dist)))

    return s03_pred


def compute(test_nn, encoded_d,
            pred_list, method="mean", dist_nn=None, wt_dist=False, random_idx=False, ema_s02=False):

    from sklearn.linear_model import LinearRegression
    train_uids = encoded_d["user_id"].to_numpy()

    preds = list()
    for point in pred_list:
        nn_preds = list()
        intercepts_list = list()
        coeff_list = list()
        for nn in test_nn:
            if random_idx:
                u_id = encoded_d["user_id"].loc[nn]
            elif ema_s02:
                u_id = train_uids[nn]
            else:
                u_id = encoded_d["user_id"].iloc[nn]
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

    from sklearn.linear_model import LinearRegression
    preds = list()
    # predictions for n ahead days
    for point in pred_list:
        attr_list = list()
        intercepts_list = list()
        coeff_list = list()
        for nn in test_nn:
            u_id = encoded_data["user_id"].iloc[nn]
            user_ts = tsg_data.get_m_day_ts_enumerate(int(u_id))
            diff_arr = np.abs(np.subtract(point, user_ts[:, 1]))
            diff_near_idx = np.where(diff_arr == diff_arr.min())
            print(diff_near_idx)
            # difference near index
            usr_vals = np.array([user_ts[n_id] for n_id in diff_near_idx[0]])
            if len(usr_vals) > 1:
                value = usr_vals.mean(axis=0)
            else:
                value = usr_vals[0]

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
    # Near -> 0.25 or points such as a randomly choosen instance.
    # Far -> 1 - (Near)

    # A time point is fixed for all test users and from here for 3 days prediction is made.
    #prediction_at = 10  # It is to check, how well for an early timepoint a suitable k can be seen.

    # Far prediction point is the last N% of the test user time series
    percentage_range = 0.80
    prediction_at = round(len(user_ts_idx) * percentage_range)

    y_labels = user_distress[prediction_at:prediction_at + ndays].tolist()
    prediction_at_list = user_ts_idx[prediction_at:prediction_at + ndays].tolist()

    return y_labels, prediction_at_list

    # Second approach not in use.
    # prediction_at = user_ts_idx[round(len(user_ts_idx) * 0.25)]

    # pred_idx = int(np.where(user_ts_idx == prediction_at)[0])

    # if abs(pred_idx - (len(user_ts_idx) - 1)) == 0:
    #     # Last point no ground truth needs only forecast
    #     ref_pred_at = prediction_at
    #     prediction_at_list = list()
    #     for i in range(0, ndays):
    #         ref_pred_at += (1 / 30)
    #         prediction_at_list.append(round(ref_pred_at, 2))
    #
    # else:
    #     # Other reference points only to the points available. Note: This is our assumption can be changed here.
    #     prediction_at_list = user_ts_idx[pred_idx:pred_idx + ndays].tolist()
    #     y_labels = user_distress[pred_idx:pred_idx + ndays].tolist()
    #     if len(prediction_at_list) < ndays:
    #         len_p_list = len(prediction_at_list)
    #         day_prop = round((1 / 30), 2)
    #         prev_day_idx_val = prediction_at_list[len(prediction_at_list) - 1]
    #         for _ in range(len_p_list, ndays):
    #             prev_day_idx_val = prediction_at_list[len(prediction_at_list) - 1]
    #             prediction_at_list.append(prev_day_idx_val + day_prop)
    # return y_labels, prediction_at_list


def do_test(test_d, ndays, near_idxs, encoded_d, fold_count="final",
            method="mean", dist_nn=None, wt_dist_flag=False, random_idx=False, ema_s02=False):
    rmse_wa_list = []
    rmse_lr_list = []
    for i in range(0, len(test_d)):
        user_id = int(test_d.iloc[i]["user_id"])
        print("User- Id ", user_id)
        y_labels, prediction_at_list = get_pred_ref_points(user_id, ndays, method=method)

        # y_labels = create_y_labels(X_test, preds, method="mean")
        if wt_dist_flag:
            test_user_nn = near_idxs[i]
            test_user_dist = dist_nn[i]
            pred_weighted_average = compute_weighted_avg(test_user_nn, encoded_d, prediction_at_list,
                                                         method=method, random_idx=random_idx,
                                                         ema_s02=ema_s02, dist_nn=test_user_dist,
                                                         wt_flag=wt_dist_flag)

            pred_lr = compute(test_user_nn, encoded_d, prediction_at_list,
                              method=method, dist_nn=test_user_dist,
                              wt_dist=wt_dist_flag, random_idx=False, ema_s02=ema_s02)
        elif random_idx:
            test_user_nn = near_idxs[i]
            pred_weighted_average = compute_weighted_avg(test_user_nn, encoded_d, prediction_at_list,
                                                         method=method, random_idx=random_idx,
                                                         ema_s02=ema_s02, dist_nn=None,
                                                         wt_flag=False)
            pred_lr = compute(test_user_nn, encoded_d, prediction_at_list,
                              method=method, dist_nn=None, wt_dist=False, random_idx=random_idx, ema_s02=ema_s02)

        else:
            test_user_nn = near_idxs[i]
            pred_weighted_average = compute_weighted_avg(test_user_nn, encoded_d, prediction_at_list,
                                                         method=method, random_idx=random_idx,
                                                         ema_s02=ema_s02, dist_nn=None,
                                                         wt_flag=False)
            pred_lr = compute(test_user_nn, encoded_d, prediction_at_list,
                              method=method, dist_nn=None,
                              wt_dist=False, random_idx=False, ema_s02=ema_s02)

        # calculate MAE, MSE, RMSE

        if not fold_count == "final":
            print("Evaluating for the fold-" + str(count) + " for the forecast reference points - " +
                  str(prediction_at_list))

        else:
            print("Evaluating for the final NN over the " + " forecast reference points - " +
                  str(prediction_at_list))


        print("Computing RMSE for weighted average based predictions on the User -- " + str(user_id))

        print("---------------------------------------------------------------")

        print("====== Weighted Average ==========================")

        print("RMSE -- ", np.sqrt(mean_squared_error(y_labels, pred_weighted_average)))

        print("Computing RMSE for lr based predictions on the User -- " + str(user_id))

        print("---------------------------------------------------------------")

        print("====== Linear Regression ==========================")
        print("RMSE -- ", np.sqrt(mean_squared_error(y_labels, pred_lr)))

        rmse_wa_list.append(np.sqrt(mean_squared_error(y_labels, pred_weighted_average)))
        rmse_lr_list.append(np.sqrt(mean_squared_error(y_labels, pred_lr)))

    return np.mean(rmse_wa_list), np.mean(rmse_lr_list)



#Call the method to do things like weighteimport properties
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
# Create prediction reference points
### Evaluate library metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import *
# Here, to change to different evaluations
from time_series_grp import TimeSeriesGroupProcessing
from RandomNeighbors import RandomNeighbors


# Change method and execute to get the predictions appropriately, these are configurations
# This is the settings for each of the scenarios. May be this can go as a main() in future.
eval_method = "mean"
wt_distance = False
# Random Neighbors
rand_neighbors = False
# Default day readings for all test users must be at mean and prediction are between min - mean - max
tsg_data = TimeSeriesGroupProcessing(method=eval_method)
# For all combinations evaluation it must be set to True
quest_cmb_all = False
# Same random state needs to be maintained to get consistent test data over all combinations and repeatable results
random_state = 1220
# It is the setting to get the ahead prediction for tinnitus distress and ahead prediction can be achieved by providing
# the value for ndays. Ideally, predictions are considered which is 3 ahead days predictions
ndays = 3


if not quest_cmb_all:
    eval_k_rmse_dict = {}
    final_k_rmse_dict = {}
    for key, val in properties.quest_comb.items():
        # Build NN for each category
        print("Building NN for the question combination -- " + str(key))

        cat_idx, num_idx, combined_df = smf.initial_processing(key, val, append_synthethic=False)
        # Build and get the knn for prediction over test instances.
        # Save the data objs

        encoded_data = save_data_objs(combined_df, key)

        #kf = KFold(n_splits=5)
        count = 0

        # Create a test set
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
        #utility.save_model("".join(key + "/" + key + "_train_stat_q_data"), X_train_data_ui)

        # filter and get the data to show to the UI for the test data.
        test_user_ids = test["user_id"].to_list()

        X_test_data_ui = combined_df.apply(filter_test_ids, axis=1, result_type="broadcast").dropna()

        X_test_data_ui["user_id"] = X_test_data_ui["user_id"].apply(int)

        # Save the data_ui object as json
        #test_data = {}
        #test_data["users"] = X_test_data_ui.to_dict("r")
        #utility.save_data("".join("test_data_ui_" + key), test_data)

        # creating odd list of K for KNN
        neighbors = list(range(1, 30, 1))


        avg_kwafold_rmse = []
        avg_lrfold_rmse = []
        final_rmselr_score = []
        final_rmsewa_score = []

        for k in neighbors:
            folds_rmsewa_score = []
            folds_rmselr_score = []

            from sklearn.model_selection import train_test_split
            import utility
            from HEOM import HEOM
            from sklearn.metrics.pairwise import cosine_distances
            from sklearn.linear_model import LinearRegression
            from scipy.spatial.distance import pdist, squareform

            if rand_neighbors:
                rknn = RandomNeighbors(X, kneighbors=k)
                rand_test_idx = rknn.get_random_neighbors(test)

            else:
                heom = HEOM(X.to_numpy()[:, 1:], cat_idx, num_idx)
                sim_matrix = pdist(X.to_numpy()[:, 1:], heom.heom_distance)
                mean_heom_distance = sim_matrix.mean()

                knn = NearestNeighbors(n_neighbors=k, metric=heom.heom_distance, radius=mean_heom_distance)
                knn.fit(X.iloc[:, 1:])
                dist, test_idx = knn.kneighbors(test.to_numpy()[:, 1:], n_neighbors=k)

            if rand_neighbors:
                frmsewa_score, frmselr_score = do_test(test, ndays, rand_test_idx, X,
                        fold_count="final", method=eval_method, dist_nn=None,
                                                       wt_dist_flag=wt_distance, random_idx=rand_neighbors)
            elif wt_distance:
                frmsewa_score, frmselr_score = do_test(test, ndays, test_idx, X,
                        fold_count="final", method=eval_method, dist_nn=dist,
                                                       wt_dist_flag=wt_distance, random_idx=rand_neighbors)
            else:
                frmsewa_score, frmselr_score = do_test(test, ndays, test_idx, X,
                        fold_count="final", method=eval_method, dist_nn=None,
                                                       wt_dist_flag=wt_distance, random_idx=rand_neighbors)

            final_rmsewa_score.append(frmsewa_score)
            final_rmselr_score.append(frmselr_score)


        final_k_rmse_dict[key] = {"wa_rmse": final_rmsewa_score, "lr_rmse": final_rmselr_score}


    if rand_neighbors:
        f_test_eval = open("".join("evals_k_rmse/" + str(eval_method) + "_far_random_test" + "vary_k_folds_test.pckl"), "wb")
        pickle.dump(final_k_rmse_dict, f_test_eval)

    elif wt_distance:
        f_test_eval = open("".join("evals_k_rmse/" + str(eval_method) + "_wt_" + "_fartest_vary_k_folds_test.pckl"),  "wb")
        pickle.dump(final_k_rmse_dict, f_test_eval)
    else:
        f_test_eval = open("".join("evals_k_rmse/" + str(eval_method) + "_fartestmock_vary_k_folds_test.pckl"),  "wb")
        pickle.dump(final_k_rmse_dict, f_test_eval)

    f_test_eval.close()
else:
    overall_eval_k_rmse_dict = {}
    overall_final_k_rmse_dict = {}
    cat_idx, num_idx, combined_df = initial_processing()

    # Build NN for each category
    print("Building NN for the question combination -- " + str("overall"))

    # Save the data objs
    encoded_data = save_data_objs(combined_df, "overall")

    # from sklearn.model_selection import train_test_split (80 and 20 throughout)
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

    # Save the train data for UI
    utility.save_model("".join("overall" + "/" + "overall" + "_train_stat_q_data"), X_train_data_ui)

    # filter and get the data to show to the UI for the test data.
    test_user_ids = test["user_id"].to_list()

    X_test_data_ui = combined_df.apply(filter_test_ids, axis=1, result_type="broadcast").dropna()

    X_test_data_ui["user_id"] = X_test_data_ui["user_id"].apply(int)

    # Save the data_ui object as json, enable this usually when you want to save a new data into the UI.
    test_data = {}
    test_data["users"] = X_test_data_ui.to_dict("r")
    utility.save_data("test_data_ui_x_test", test_data)

    count = 0

    # creating odd list of K for KNN
    neighbors = list(range(1, 30, 1))

    overall_avg_kwafold_rmse = []
    overall_avg_lrfold_rmse = []
    overall_final_rmselr_score = []
    overall_final_rmsewa_score = []
    for k in neighbors:
        folds_rmsewa_score = []
        folds_rmselr_score = []

        # Split the data into train and test.
        from sklearn.model_selection import train_test_split
        import utility
        from HEOM import HEOM
        from sklearn.metrics.pairwise import cosine_distances
        from sklearn.linear_model import LinearRegression
        from scipy.spatial.distance import pdist, squareform

        if rand_neighbors:
            rknn = RandomNeighbors(X, kneighbors=k)
            rand_test_idx = rknn.get_random_neighbors(test)
        else:
            heom = HEOM(X.to_numpy()[:, 1:], cat_idx, num_idx)
            sim_matrix = pdist(X.to_numpy()[:, 1:], heom.heom_distance)
            mean_heom_distance = sim_matrix.mean()

            knn = NearestNeighbors(n_neighbors=k, metric=heom.heom_distance, radius=mean_heom_distance)
            knn.fit(X.to_numpy()[:, 1:])
            dist, test_idx = knn.kneighbors(test.to_numpy()[:, 1:], n_neighbors=k)

        if rand_neighbors:
            frmsewa_score, frmselr_score = do_test(test, ndays, rand_test_idx, X,
                    fold_count="final", method=eval_method, dist_nn=None,
                                                   wt_dist_flag=wt_distance, random_idx=rand_neighbors)
        elif wt_distance:
            frmsewa_score, frmselr_score = do_test(test, ndays, test_idx, X,
                    fold_count="final", method=eval_method, dist_nn=dist,
                                                   wt_dist_flag=wt_distance, random_idx=rand_neighbors)
        else:
            frmsewa_score, frmselr_score = do_test(test, ndays, test_idx, X,
                    fold_count="final", method=eval_method, dist_nn=None,
                                                   wt_dist_flag=wt_distance, random_idx=rand_neighbors)

        overall_final_rmsewa_score.append(frmsewa_score)
        overall_final_rmselr_score.append(frmselr_score)

        overall_final_k_rmse_dict["overall"] = {"wa_rmse": overall_final_rmsewa_score, "lr_rmse": overall_final_rmselr_score}

    # Set the file name of your choice, while evaluating the via KNN and regression.
    if rand_neighbors:
        f_test_eval = open("".join("evals_k_rmse/" + str(eval_method) + "_neartest_overall_random_" + "vary_k_folds_test.pckl"), "wb")
        pickle.dump(overall_final_k_rmse_dict, f_test_eval)

    elif wt_distance:
        f_test_eval = open("".join("evals_k_rmse/" + str(eval_method) + "_wt_overall_" + "_neartest_vary_k_folds_test.pckl"),  "wb")
        #pickle.dump(overall_eval_k_rmse_dict, f_eval)
        pickle.dump(overall_final_k_rmse_dict, f_test_eval)
    else:
        f_test_eval = open("".join("evals_k_rmse/" + str(eval_method) + "_overall_neartest_vary_k_folds_test.pckl"),  "wb")
        pickle.dump(overall_final_k_rmse_dict, f_test_eval)

    #f_eval.close()
    f_test_eval.close()


'''
 ML Modelling based on s02 - loudness. The concept is simple and similar to moving average.
 First the time series observations are grouped by day so that we get observations between 1-31 across users.
 For each of the day a similarity is computed and when there is a match counter is incremented.
 When there is no match, basically if the user has no observations for a given day then we move his previous day value and
 compute the similarity. Finally, sum of all days by the counter is the similarity value.
'''

import ml_modelling_ts as ml_ts
import numpy as np
import pandas as pd

# Create train and test containing same users in train and test as per static data.
# This is for UI, otherwise split and perform kfolds


def splitData(dataset, test_user_ids):
    train_data = dataset[~dataset["user_id"].isin(test_user_ids)]
    test_data = dataset[dataset["user_id"].isin(test_user_ids)]
    return train_data, test_data


X = ml_ts.process_data(grouping="day")

# Calculate pairwise distance and create a dataframe for the same
from scipy.spatial.distance import pdist, squareform

# Cross validate here based on the same split of static data here.
# Note: Only one combination will be present
C = np.zeros((X.shape[0], X.shape[0]))
for i in range(0, len(X)):
    #print("User is -- {}", X[i][0])
    #print("User is -- {}", len(X[i][1]))
    for j in range(0, len(X)):
        dist = ml_ts.compute_dist(X[:, 1][i], X[:, 1][j])
        C[i][j] = dist
C_df = pd.DataFrame(C)


# Threshold overall distance for making within radius
threshold_distance = sum(C_df.mean()) / len(C_df)


user_ids = []
for val in X:
    user_ids.append(val[0])

C_df["user_id"] = user_ids

train_data, test_data = splitData(C_df, test_user_ids)

#### A KNN over the obtained similarity matrix for searching
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# creating odd list of K for KNN
neighbors = list(range(1, 30, 1))

count = 0

overall_ema_eval_k_rmse_dict = {}
overall_ema_final_k_rmse_dict = {}

overall_avg_kwafold_rmse = []
overall_avg_lrfold_rmse = []
overall_final_rmselr_score = []
overall_final_rmsewa_score = []
for k in neighbors:
    # Test on the final test set to see the performance of the NN over the subspaces

    knn_ema = NearestNeighbors(n_neighbors=k, metric="precomputed", radius=threshold_distance)
    knn_ema.fit(train_data[train_data.index])
    ema_dist, ema_idx = knn_ema.kneighbors(test_data[train_data.index], n_neighbors=k)
    # First get the time series for a given test patient and the reference point and iterate to evaluate
    if wt_distance:
        frmsewa_score, frmselr_score = do_test(test_data, ndays, ema_idx, train_data,
                                               fold_count="final", method=eval_method, dist_nn=ema_dist,
                                               wt_dist_flag=wt_distance, random_idx=False, ema_s02=True)
    else:
        frmsewa_score, frmselr_score = do_test(test_data, ndays, ema_idx, train_data,
                                               fold_count="final", method=eval_method, dist_nn=None,
                                               wt_dist_flag=wt_distance, random_idx=False, ema_s02=True)

    overall_final_rmsewa_score.append(frmsewa_score)
    overall_final_rmselr_score.append(frmselr_score)

overall_ema_final_k_rmse_dict["overall"] = {"wa_rmse": overall_final_rmsewa_score,
                                            "lr_rmse": overall_final_rmselr_score}

if wt_distance:
    f_test_eval = open("".join("evals_k_rmse/" + str(eval_method) + "ema_wt_overall_neartest" + "vary_k_folds_test.pckl"), "wb")
    pickle.dump(overall_ema_final_k_rmse_dict, f_test_eval)
    f_test_eval.close()
else:
    f_test_eval = open("".join("evals_k_rmse/" + str(eval_method) + "ema_overall_neartest2_vary_k_folds_test.pckl"), "wb")
    pickle.dump(overall_ema_final_k_rmse_dict, f_test_eval)
    f_test_eval.close()

