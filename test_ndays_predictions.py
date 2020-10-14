######## Perform Test run fixing k, combination, aggregations

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import static_sim_functions as smf
from scipy.spatial.distance import pdist, squareform
import properties
import pandas as pd
import ml_modelling_ts as ml_ts
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from HEOM import HEOM

# Global declaration. Set this once, and run each of the functions with different bounds.
# Should be moved later to a class level based executions. So that entire py file can be executed.
# Note: Select Line based execution is required since the store for each of percentage is held in globals.
lr_rmse_ = []
wa_rmse_ = []
# Set true for Loudness based calculation.
ema = True
# Percentage value in 0 to 1. (20%, 30%, 50% bounds are looked at)
bound = 0.50
###################
lr_usr_bounds_dict = {}
wa_usr_bounds_dict = {}
lr_usr_bounds_dict_ema = {}
lr_usr_bounds_dict_ema = {}

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


def initial_processing():
    # Read the csv of the tschq data and make the necessary things
    tschq = pd.read_pickle(properties.data_location + "/input_pckl/" + "3_q.pckl")

    # Dropping users
    drop_indexs = []

    # Users with very few observations and user do not containing the time series are filtered.
    drop_user_ids = [54, 60, 140, 170, 4, 6, 7, 9, 12, 19, 25, 53, 59, 130, 144, 145, 148, 156, 167]

    # indexes to be obtained
    for val in drop_user_ids:
        drop_indexs.append(tschq[tschq["user_id"] == val].index[0])

    # Drop those indexes of the users who have very less observations (less than 10 days)
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
    categorical_feature_mask = df.iloc[:, 1:].dtypes == object
    other_feature_mask = df.iloc[:, 1:].dtypes != object
    # filter categorical columns using mask and turn it into a list
    categorical_cols = df.iloc[:, 1:].columns[categorical_feature_mask].tolist()
    num_cols = df.iloc[:, 1:].columns[other_feature_mask].tolist()
    cat_idx = [df.iloc[:, 1:].columns.get_loc(val) for val in categorical_cols]
    num_idx = [df.iloc[:, 1:].columns.get_loc(val) for val in num_cols]

    return cat_idx, num_idx, df


# Create reference points for multiple reference predictions
def get_pred_ref_points(user_id, ndays, method="mean"):
    # Using the default tsg which is mean observations of the user
    test_user_ts = tsg_data.get_usr_mday_ts_predict(user_id)

    # user_ts = tsg.get_usr_mday_ts_index_corrected(int(user_id))
    user_ts_idx = test_user_ts[:, 1]
    # ["date", "time_idx", "s02", "s03", "s04", "s05", "s06", "s07]
    user_distress = test_user_ts[:, 3]

    # bound -> 0.20, 0.30, 0.5 percentage of the time series recordings
    # from where the start day is chosen.
    # Should be made this as a function parameter rather.
    percentage_range = bound

    # percentage because of unequal length in the time series
    prediction_at = round(len(user_ts_idx) * percentage_range)

    y_labels = user_distress[prediction_at:prediction_at + ndays].tolist()
    prediction_at_list = user_ts_idx[prediction_at:prediction_at + ndays].tolist()

    return y_labels, prediction_at_list


def weighted_average(distress_list):
    average = np.asarray(distress_list, dtype=float).mean()
    return average


def splitData(dataset, test_user_ids):
    train_d = dataset[~dataset["user_id"].isin(test_user_ids)]
    test_d = dataset[dataset["user_id"].isin(test_user_ids)]
    return train_d, test_d


# Function computes the weighted average as predictions for given prediction time point
def compute_weighted_avg(n_idx, encoded_data, pred_at_list, is_ema=False, method="mean", dist_nn=None, wt_flag=False):
    preds = list()
    train_users = encoded_data["user_id"].to_numpy()
    # Prediction for ahead time points
    for pval in pred_at_list:
        distress_list = list()
        for vals in n_idx:
            if is_ema:
                u_id = train_users[vals]
            else:
                u_id = encoded_data["user_id"].iloc[vals]

            user_ts = tsg_data.get_usr_mday_ts_predict(int(u_id))
            # 3rd val of the series is s03 of the neighbor
            print("{}, {} Values ".format(int(pval), int(u_id)))
            if len(user_ts) > int(pval):
                value = user_ts[int(pval), :][3]
            elif len(user_ts) <= int(pval):
                value = user_ts[len(user_ts)-1, :][3]

            distress_list.append(value)

        print("Calling weighted average to predict distress")
        preds.append(weighted_average(distress_list))
    return preds


def compute(test_nn, encoded_data,
            pred_list, is_ema=False, method="mean", dist_nn=None, wt_dist=False):

    train_users = encoded_data["user_id"].to_numpy()
    from sklearn.linear_model import LinearRegression
    preds = list()
    for point in pred_list:
        nn_preds = list()
        intercepts_list = list()
        coeff_list = list()
        for nn in test_nn:

            if is_ema:
                u_id = train_users[nn]
            else:
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

            intercepts_list.append(reg_fit.intercept_)
            coeff_list.append(reg_fit.coef_)

        print("Predicting the value of s3 over the averaged slope and intercepts of observations of the neighbors")

        # y = mx + c, where m is the average slope of the neighbors and c is the average intercept obtained.
        print("The equation to estimate s03 for the user is {}".format("".join(str(np.asarray(coeff_list).mean())) +
                                                                   "* time_index + " +
                                                                   str(np.asarray(intercepts_list).mean())))
        y = np.multiply(np.asarray(coeff_list).mean(), point) + np.asarray(intercepts_list).mean()
        preds.append(y)

    return preds


def plot_bar(x, y, plot_props=None):
    fig = sns.barplot(x=x, y=y, order=x, color="steelblue")
    if plot_props:
        fig.set(xlabel=plot_props["xlabel"],
                ylabel=plot_props["ylabel"],
                title=plot_props["title"],
                ylim=plot_props["ylim"])
    return fig


def setup_dict_usr_vals(rmse_scores, test_users):
    temp_dict = {}
    for u_id, rmse_val in zip(test_users, rmse_scores):
        if u_id not in temp_dict:
            temp_dict[u_id] = rmse_val
    return temp_dict

import operator
def sort_dict_vals(dictionary):
    return {key: values for key, values in sorted(dictionary.items(), key=lambda item: item[1])}


def calculate_mse_users(y_labels, wa_user_preds, lr_usr_preds):
    mse_wa_list = []
    mse_lr_list = []

    from sklearn.metrics import mean_squared_error
    for y_label, wa_pred, lr_pred in zip(y_labels, wa_user_preds, lr_usr_preds):

        mse_val_wa = np.square(np.subtract(y_label, wa_pred))
        mse_val_lr = np.square(np.subtract(y_label, lr_pred))

        #print(np.sqrt(mean_squared_error(y_label, wa_pred)))
        #print(np.sqrt(mean_squared_error(y_label, lr_pred)))

        # Unequal lengths append zeroes for the unavailable predictions, so that numpy computation is possible.
        if mse_val_wa.shape[0] < ndays:
            mse_val_wa = np.append(mse_val_wa, np.zeros(ndays - mse_val_wa.shape[0]))

        if mse_val_lr.shape[0] < ndays:
            mse_val_lr = np.append(mse_val_lr, np.zeros(ndays - mse_val_lr.shape[0]))

        # Append
        mse_wa_list.append(mse_val_wa)
        mse_lr_list.append(mse_val_lr)
    return mse_wa_list, mse_lr_list


def compute_predictions(test_info, train_info, nn_idx, ndays=3, eval_cond="mean", is_ema=False):
    y_labels_list = list()  # truth label list
    wa_usr_list = list()  # wa list of user predictions
    lr_usr_list = list()  # lr list of user predictions
    prediction_tp_list = list()
    for t_user in range(0, len(test_info)):
        user_id = int(test_info.iloc[t_user]["user_id"])
        print("User- Id ", user_id)
        y_labels, prediction_at_list = get_pred_ref_points(user_id, ndays, method=eval_cond)
        test_user_nn = nn_idx[t_user]
        #test_user_ema_nn = ema_idx[i]
        pred_weighted_average = compute_weighted_avg(test_user_nn, train_info, prediction_at_list, is_ema=is_ema,
                                                     method=eval_cond)
        #pred_weighted_average_ema = compute_weighted_avg(test_user_ema_nn, train_data, prediction_at_list,
        #                                                 method=eval_cond)
        pred_lr = compute(test_user_nn, train_info, prediction_at_list, is_ema=is_ema,
                          method=eval_cond)
        #pred_lr_ema = compute(test_user_ema_nn, train_data, prediction_at_list,
        #                      method=eval_cond)
        # Append all
        if user_id == 51:
            print("User 51 ---- ", np.sqrt(mean_squared_error(pred_weighted_average, y_labels)))
            print("User 51 ----- ", np.sqrt(mean_squared_error(pred_lr, y_labels)))
        y_labels_list.append(y_labels)
        prediction_tp_list.append(prediction_at_list)
        wa_usr_list.append(pred_weighted_average)
        lr_usr_list.append(pred_lr)
    return y_labels_list, wa_usr_list, lr_usr_list


##### Start of Main #################
# For ema set k= 11 and for static reg set k=9
ndays = 3
k = 11
quest_cmb = "related_conditions"
eval_cond = "mean"
random_state = 1220

from time_series_grp import TimeSeriesGroupProcessing

tsg_data = TimeSeriesGroupProcessing(method=eval_cond)

user_obs_cond = tsg_data.user_grp_dict_predict



# Initial cleaning of the data.
cat_idx, num_idx, df = smf.initial_processing(quest_cmb, properties.quest_comb[quest_cmb],
                                              append_synthethic=False)

# unnecessary column not required for the computations
drop_cols = ["tschq01", "tschq04-1", "tschq04-2", "tschq07-2", "tschq13", "tschq25"]

if quest_cmb not in ["all", "overall"]:
    filtered_cols = [x for x in properties.quest_comb[quest_cmb] if x not in drop_cols]
    if quest_cmb == "bg_tinnitus_history":
        filtered_query_data = df[filtered_cols + ["tschq04"]]
    else:
        filtered_query_data = df[filtered_cols]
else:
    filtered_query_data = df

# Label and ordinal encoding scheme
encoded_combined_df = smf.preprocess(filtered_query_data, quest_cmb, age_bin=False,
                                     process_model_name="",
                                     prediction=False, save_model=False)

#Split into train test with same random state as per eval
X, test = train_test_split(encoded_combined_df,
                           test_size=0.20,
                           random_state=random_state)


# This is required for obtaining the same train and test sets from EMA data.
train_user_ids = X["user_id"].to_list()
test_user_ids = test["user_id"].to_list()

#train_len = 0
#test_len = 0
#for k, v in user_obs_cond.items():
#    if k in train_user_ids:
#        train_len += len(v)
#    else:
#        test_len += len(v)

EMA_data = ml_ts.process_data(grouping="day")

# Calculate pairwise distance and create a dataframe for the same
from scipy.spatial.distance import pdist, squareform

# Note: Only one combination will be present
C = np.zeros((EMA_data.shape[0], EMA_data.shape[0]))
for i in range(0, len(EMA_data)):
    #print("User is -- {}", X[i][0])
    #print("User is -- {}", len(X[i][1]))
    for j in range(0, len(EMA_data)):
        dist = ml_ts.compute_dist(EMA_data[:, 1][i], EMA_data[:, 1][j])
        C[i][j] = dist

C_df = pd.DataFrame(C)


# Threshold overall distance for making within radius
threshold_distance = sum(C_df.mean()) / len(C_df)


user_ids = []
for val in EMA_data:
    user_ids.append(val[0])

C_df["user_id"] = user_ids

train_data, test_data = splitData(C_df, test_user_ids)

# Fit the train into nearest neighbors and predict over test by choosing the specified neighborhood
heom = HEOM(X.to_numpy()[:, 1:], cat_idx, num_idx)
sim_matrix = pdist(X.to_numpy()[:, 1:], heom.heom_distance)
mean_heom_distance = sim_matrix.mean()
knn = NearestNeighbors(n_neighbors=k, radius=mean_heom_distance, metric=heom.heom_distance)
knn.fit(X.to_numpy()[:, 1:])
# Fit with static data
dist, idx = knn.kneighbors(test.to_numpy()[:, 1:], n_neighbors=k)

# Fit with EMA data based on Loudness
knn_ema = NearestNeighbors(n_neighbors=k, metric="precomputed", radius=threshold_distance)
knn_ema.fit(train_data[train_data.index])
ema_dist, ema_idx = knn_ema.kneighbors(test_data[train_data.index], n_neighbors=k)

if ema:
    y_labels_list, wa_usr_list, lr_usr_list = compute_predictions(test_data, train_data, ema_idx,
                                                                  ndays=ndays, eval_cond=eval_cond, is_ema=ema)
    rmse_wa_list, rmse_lr_list = calculate_mse_users(y_labels_list, wa_usr_list, lr_usr_list)
else:
    y_labels_list, wa_usr_list, lr_usr_list = compute_predictions(test, X, idx,
                                                                  ndays=ndays, eval_cond=eval_cond)
    rmse_wa_list, rmse_lr_list = calculate_mse_users(y_labels_list, wa_usr_list, lr_usr_list)

# At each timepoints
mean_rmse_wa_list = np.sqrt(np.mean(rmse_wa_list, axis=0))
mean_rmse_lr_list = np.sqrt(np.mean(rmse_lr_list, axis=0))


### Visualize a bar chart of test users average rmse values for ndays

# At each user level
user_mean_rmse_wa_list = np.sqrt(np.mean(rmse_wa_list, axis=1))
user_mean_rmse_lr_list = np.sqrt(np.mean(rmse_lr_list, axis=1))

lr_rmse_.append(user_mean_rmse_lr_list)
wa_rmse_.append(user_mean_rmse_wa_list)


## Save the user-rmse in dictionary and sort based on rmse.

plot_props = {
    "ylim": (0, 0.6),
    "xlabel": "user_ids",
    "ylabel": "RMSE",
    "title": "Sorted RMSE values of the test users"
}

import utility

# uncomment while processing ema based similarity
if ema:
    lr_usr_bounds_dict_ema = sort_dict_vals(setup_dict_usr_vals(np.mean(np.asarray(lr_rmse_), axis=0), test_data["user_id"].to_list()))
    wa_usr_bounds_dict_ema = sort_dict_vals(setup_dict_usr_vals(np.mean(np.asarray(wa_rmse_), axis=0), test_data["user_id"].to_list()))
    utility.save_model("lr_usr_bounds_dict_ema.pckl", lr_usr_bounds_dict_ema)
    utility.save_model("wa_usr_bounds_dict_ema.pckl", wa_usr_bounds_dict_ema)
    fig_set1 = plot_bar(x=list(lr_usr_bounds_dict_ema.keys()), y=list(lr_usr_bounds_dict_ema.values()), plot_props=plot_props)
    plt.savefig("eval_images/" + "barplot_lr_ema-{}_k-{}_x_0.2".format(eval_cond, k) + "_.png", dpi=300, bbox_inches='tight')
    plt.show()
    fig_set2 = plot_bar(x=list(wa_usr_bounds_dict_ema.keys()), y=list(wa_usr_bounds_dict_ema.values()), plot_props=plot_props)
    plt.savefig("eval_images/" + "barplot_wa_ema-{}_k-{}_x_0.2".format(eval_cond, k) + "_.png", dpi=300, bbox_inches='tight')
    plt.show()
else:
    lr_usr_bounds_dict = sort_dict_vals(setup_dict_usr_vals(np.mean(np.asarray(lr_rmse_), axis=0), test_user_ids))
    wa_usr_bounds_dict = sort_dict_vals(setup_dict_usr_vals(np.mean(np.asarray(wa_rmse_), axis=0), test_user_ids))
    utility.save_model("lr_usr_bounds_dict.pckl", lr_usr_bounds_dict)
    utility.save_model("wa_usr_bounds_dict.pckl", wa_usr_bounds_dict)
    fig_set3 = plot_bar(x=list(lr_usr_bounds_dict.keys()), y=list(lr_usr_bounds_dict.values()), plot_props=plot_props)
    plt.savefig("eval_images/" + "barplot_lr_c3-{}_k-{}_x_0.2".format(eval_cond, k) + "_.png", dpi=300, bbox_inches='tight')
    plt.show()
    fig_set4 = plot_bar(x=list(wa_usr_bounds_dict.keys()), y=list(wa_usr_bounds_dict.values()), plot_props=plot_props)
    plt.savefig("eval_images/" + "barplot_wa_C3-{}_k-{}_x_0.2".format(eval_cond, k) + "_.png", dpi=300, bbox_inches='tight')
    plt.show()
