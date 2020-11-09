import numpy as np
from sklearn.neighbors import NearestNeighbors
import static_sim_functions as smf
from scipy.spatial.distance import pdist, squareform
from pyod.utils import generate_data

from pyod.utils.data import generate_data
from pyod.utils.data import generate_data_clusters
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.cof import COF
from pyod.utils.data import get_outliers_inliers
from pyod.utils.utility import standardizer
from sklearn.metrics import roc_auc_score
import pyod
from pyod.utils.example import data_visualize
from pyod.utils.example import visualize
import properties
import pandas as pd
import ml_modelling_ts as ml_ts

"""
Rank based outlier detection.
Author: Subash Prakash
"""


class RBOD:
    """
    The notion of reverse neighborhood concept has been widely useful to detect varying densities point. There has been
    recent attention to this reverse neighborhood in many fields related to medical, social network etc..
    The reverse neighborhood means to ask if for a query q there are set of nearest neighbors, then how close are these
    neighors?. Closeness is defined by obtain the rank of the query in the neighbors. Ideally, for a neighbor p of q, q
    must be one of the neighbors. This is always not true and could lead to outliers. This class implements such a detection
    as per "Rank-based outlier detection" mostly a good start. Since, the paper do not have their implementation this is
    contributed in the thesis work.
    """

    def __init__(self, sim_df, kneighbors=5, metric=None, radius=1.0, z_val=2.5, dyn=False):

        """
        :param sim_df: Precompute a similarity matrix by a defined metric and pass to the class.
        :param kneighbors: Number of K neighbors same as KNN
        :param metric: If None, default euclidean is perform, else the passed metric is used for computation same as sim_df
        :param radius: Defaults to the radius around query to 1.0, if there is a threshold in radius it can be passed.
        :param z_val: The cut-off for outlier as per the original paper is 2.5, however based on the domain this can change.
        :param dyn: Obtain the outlier detection over dynamic data for the constructed similarity.
        """

        self.kneighbors = kneighbors
        self.metric = metric
        self.radius = radius
        self.sim_df = sim_df
        self.z_val = z_val
        self.dyn = dyn

    def detect(self, X):
        """
        :param X: The processed Train data. This contains all the necessary information. This implementation also needs
        an id, this can be user_id, or just an index_id for the identifier between the sim_df and X_train.
        :param X: In case of already computed data, it would be a similarity matrix numpy processed array.
        :return: outlier_list contains a tuple (<<Id>>,<<z_val>>,<<True/False indicating the outlierness>>)
        """

        outlier_list = []
        if None is not self.metric:
            knn = NearestNeighbors(n_neighbors=self.kneighbors, radius=self.radius, metric=self.metric)
            #knn = NearestNeighbors(n_neighbors=20, radius=1.0, metric="precomputed")
        else:
            # This is the default Euclidean distance. Mostly useful while doing over public dataset
            knn = NearestNeighbors(n_neighbors=self.kneighbors)

        # Fit to ball_tree entire data. We will obtain the neighbors indexed from this matrix and use it find and label outliers if any.
        if self.dyn:
            knn.fit(self.sim_df.iloc[:, 1:])
           #knn.fit(C_df.iloc[:, 1:])
        else:
            knn.fit(X[:, 1:])

        user_ids_np = self.sim_df["user_id"].to_numpy()
        #user_ids_np = C_df["user_id"].to_numpy()
        for user in X:
            dist, idx = knn.kneighbors(user[1:].reshape(1, -1), n_neighbors=self.kneighbors)

            # Not concerned with the distance but their ranks across the neighbors
            dist = dist.flatten()
            idx = idx.flatten()
            ranks_list = []
            for u_id in idx[1:]:
                user_rank_dict = {}
                u_id_dists = self.sim_df[self.sim_df["user_id"] == int(user_ids_np[u_id])].to_numpy()
                #u_id_dists = C_df[C_df["user_id"] == int(user_ids_np[26])].to_numpy()
                for i, dists in enumerate(u_id_dists.flatten()[1:]):
                    if not float(dists) == 0.0:
                        user_rank_dict[user_ids_np[i]] = dists

                # Sort in ascending order and find the rank of the respective user_id
                sorted_usr_rank_tuple = sorted(user_rank_dict.items(), key=lambda x: x[1])
                rank = 1
                for val in sorted_usr_rank_tuple:
                    # print(val)
                    # print(int(user_ids_np[u_id]))
                    if int(user[0]) == val[0]:
                        break
                    rank += 1
                ranks_list.append(rank)
                #print(ranks_list)
            outlier_list.append((int(user[0]), np.sum(ranks_list) / self.kneighbors))
            #outlier_list.append((int(user[0]), np.sum(ranks_list) / 20))

        np_outlier_list = np.asarray([val[1] for val in outlier_list])
        mean_rank_ratio = np_outlier_list.mean()
        std_rank_ratio = np_outlier_list.std()
        z_list = []
        for ele in outlier_list:
            z_score = (ele[1] - mean_rank_ratio) / std_rank_ratio
            if z_score >= self.z_val:
            #if z_score >= 2.5:
                z_list.append((ele[0], z_score, True))
            else:
                z_list.append((ele[0], z_score, False))

        return z_list


def common_processing(df):
    # Getting percentage between 0 to 1 rather than score values
    df["tschq12"] = df["tschq12"].apply(lambda x: x / 100)
    df["tschq16"] = df["tschq16"].apply(lambda x: x / 100)
    df["tschq17"] = df["tschq17"].apply(lambda x: x / 100)

    # Feature engineering family history

    # Create bins over tschq12-loudness, tschq16-stress, tschq17-annoyance
    # bins = [0, 25, 50, 75, 100]
    # labels = ["LOW","MEDIUM","HIGH", "VERYHIGH"]
    # df['tschq12'] = pd.cut(df['tschq12'], bins=bins, labels=labels, include_lowest=True)
    # df.drop(["age"], axis=1, inplace=True)

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


def initial_processing():
    # Read the csv of the tschq data and make the necessary things
    #tschq = pd.read_pickle(properties.data_location + "/input_pckl/" + "3_q.pckl")

    # Dropping users who do not have their time series
    #drop_indexs = []

    # Users with very few observations and user do not containing the time series are filtered.
    #drop_user_ids = [54, 60, 140, 170, 4, 6, 7, 9, 12, 19, 25, 53, 59, 130, 144, 145, 148, 156, 167]

    # indexes to be obtained
    #for val in drop_user_ids:
    #    drop_indexs.append(tschq[tschq["user_id"] == val].index[0])

    # Drop those indexes of the users who do not have their time recordings
    #tschq.drop(drop_indexs, inplace=True)
    #tschq.reset_index(inplace=True, drop=True)

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


def main(top_n=5, k_range=[3, 5, 7, 11, 15, 17, 19], alpha=1.0):
    k_outliers_dict = {}
    outliers_data = {}
    combinations_dict = {}
    user_outliers = {}
    user_lof_outliers = {}
    k_cmb_z_scores = list()
    k_z_scores = list()
    for quest_cmb in list(
            properties.quest_comb.keys()):  # Only take the subset of feature and not consider entire -- + ["overall"]:

        for k in k_range:
            # Can be changed to random set of features for making it feature bagging + voting strategy with ensembles.
            if quest_cmb == "overall":
                cat_idx, num_idx, df = initial_processing()
            else:
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

            cmb_data_np = encoded_combined_df.to_numpy()
            from HEOM import HEOM
            heom = HEOM(cmb_data_np[:, 1:], cat_idx, num_idx)

            # Create the similarity matrix, for references the user_id is tagged along
            C = np.zeros((cmb_data_np.shape[0], cmb_data_np.shape[0]))
            for i in range(0, len(cmb_data_np)):
                for j in range(0, len(cmb_data_np)):
                    dist = heom.heom_distance(cmb_data_np[i][1:], cmb_data_np[j][1:])
                    C[i][j] = dist

            C_df = pd.DataFrame(C)
            #C_df["user_id"] = encoded_combined_df["user_id"]
            C_df.insert(0, "user_id", encoded_combined_df["user_id"].to_numpy())
            # Threshold radius to pass. If not required can be kept to default 1.0
            sim_matrix = pdist(encoded_combined_df.iloc[:, 1:], heom.heom_distance)
            threshold_distance = sim_matrix.mean()

            # Build the Rank Based Outlier Detection. Z_val is kept as per original paper. Can be modified according to the dataset.
            rbod = RBOD(C_df, kneighbors=k, metric=heom.heom_distance, radius=threshold_distance, z_val=2.5)
            lof = LOF(n_neighbors=k, metric=heom.heom_distance)
            lof.fit(cmb_data_np[:, 1:])
            labels = lof.labels_

            outlier_list = rbod.detect(cmb_data_np)
            user_lof_list = np.where(labels == 1)
            users_lof_outlier = [encoded_combined_df["user_id"].to_numpy()[idx] for idx in user_lof_list[0]]

            for val in outlier_list:
                if val[2]:
                    if val[0] not in user_outliers:
                        user_outliers[val[0]] = 1
                    else:
                        user_outliers[val[0]] += 1

            for val in users_lof_outlier:
                if val not in user_lof_outliers:
                    user_lof_outliers[val] = 1
                else:
                    user_lof_outliers[val] += 1

            k_z_scores.append(np.asarray([item[1] for item in outlier_list]))
        k_cmb_z_scores.append(np.mean(k_z_scores, axis=0))

    # Sorted in descending order of the outlier occurance and return the top-n outliers
    user_desc_rbod_outliers = sorted(user_outliers.items(), key=lambda x: x[1], reverse=True)
    user_desc_lof_outliers = sorted(user_lof_outliers.items(), key=lambda x: x[1], reverse=True)

    # Median as not to impact the zscore with each k and alpha parameter for number of std to go. Ideal for 1std.
    from scipy import stats
    mean_z_scores = np.mean(k_cmb_z_scores, axis=0)
    median_z_scores = np.median(k_cmb_z_scores, axis=0)
    user_z_scores = [(idx, values) for idx, values in zip(encoded_combined_df["user_id"].to_numpy(), mean_z_scores)]
    # Extreme Outliers = Q3 + alpha * IQR, where alpha = 3.0
    # Mild Outliers = Q3 + alpha * IQR, where alpha=1.0
    # Q3 - Q1 -> Mid point which is 75% - 25% of the data point

    iqr = stats.iqr(median_z_scores, interpolation="midpoint")
    #k_z_scores = median_z_scores + alpha * iqr

    grad_outliers = np.where(median_z_scores > alpha * iqr)

    #There is a provision to extend to printout based on the iqr values too which returns the gradual outliers.

    # Return the (id,occurance) tuple in highest outlier occurance by voting process.
    return user_desc_rbod_outliers[:top_n], user_desc_lof_outliers[:top_n], user_z_scores


def main_dyn(top_n=5, k_range=[3, 5, 7, 11, 15, 17, 19], alpha=1.0):
    k_outliers_dict = {}
    outliers_data = {}
    combinations_dict = {}
    user_outliers = {}
    user_lof_outliers = {}
    k_cmb_z_scores = list()
    k_z_scores = list()
    X = ml_ts.process_data(grouping="day")

    C = np.zeros((X.shape[0], X.shape[0]))
    for i in range(0, len(X)):
        for j in range(0, len(X)):
            dist = ml_ts.compute_dist(X[:, 1][i], X[:, 1][j])
            C[i][j] = dist

    C_df = pd.DataFrame(C)
    user_ids = [val[0] for val in X]


    # Insert at the begining the Id
    C_df.insert(0, "user_id", user_ids)

   # C_df.to_csv("sim_ema.csv")

    #C_df["user_id"] = user_ids
    for k in k_range:
        # Build the Rank Based Outlier Detection. Z_val is kept to 1.5 and seems to obtain some of outliers. Can be modified according to the dataset.
        rbod = RBOD(C_df, kneighbors=k, metric="precomputed", z_val=1.5, dyn=True)

        lof = LOF(n_neighbors=k, metric="precomputed")
        lof.fit(C_df.iloc[:, 1:])
        labels = lof.labels_

        outlier_list = rbod.detect(C_df.to_numpy())
        user_lof_list = np.where(labels == 1)
        users_lof_outlier = [user_ids[idx] for idx in user_lof_list[0]]

        for val in outlier_list:
            if val[2]:
                if val[0] not in user_outliers:
                    user_outliers[val[0]] = 1
                else:
                    user_outliers[val[0]] += 1

        for val in users_lof_outlier:
            if val not in user_lof_outliers:
                user_lof_outliers[val] = 1
            else:
                user_lof_outliers[val] += 1

        k_z_scores.append(np.asarray([item[1] for item in outlier_list]))

    # Sorted in descending order of the outlier occurance and return the top-n outliers
    user_desc_rbod_outliers = sorted(user_outliers.items(), key=lambda x: x[1], reverse=True)
    user_desc_lof_outliers = sorted(user_lof_outliers.items(), key=lambda x: x[1], reverse=True)

    # Median as not to impact the zscore with each k and alpha parameter for number of std to go. Ideal for 1std.
    from scipy import stats
    mean_z_scores = np.mean(k_z_scores, axis=0)
    median_z_scores = np.median(k_z_scores, axis=0)
    user_z_scores = [(idx, values) for idx, values in zip(user_ids, mean_z_scores)]
    # Extreme Outliers = Q3 + alpha * IQR, where alpha = 3.0
    # Mild Outliers = Q3 + alpha * IQR, where alpha=1.0
    # Q3 - Q1 -> Mid point which is 75% - 25% of the data point

    iqr = stats.iqr(median_z_scores, interpolation="midpoint")
    #k_z_scores = median_z_scores + alpha * iqr

    grad_outliers = np.where(median_z_scores > alpha * iqr)

    #There is a provision to extend to printout based on the iqr values too which returns the gradual outliers.

    # Return the (id,occurance) tuple in highest outlier occurance by voting process.
    return user_desc_rbod_outliers[:top_n], user_desc_lof_outliers[:top_n], user_z_scores


if __name__ == '__main__':
    save = True  # Set to true normally when one want to save the outlier scores to be utilized to anything.
    k_range = [_ for _ in range(3, 19, 2)]
    # Need to go more higher neighborhood to find extreme outliers for dynamic data
    dyn_k_range = [_ for _ in range(3, 19, 2)]
    # Alpha is set for the outlier value threshold as per the original paper for RBDA, can be changed accordingly.
    n = 10
    a = 2.5
    print("Starting Outlier Detection --- with top_n -- {}, k_range -- {} and alpha -- {}".format(n, k_range, a))
    rbod_outliers, lof_outliers, user_zscores = main(top_n=n, k_range=k_range, alpha=a)

    dyn_rbod_outliers, dyn_lof_outliers, dyn_user_zscores = main_dyn(top_n=n, k_range=dyn_k_range, alpha=a)
    print("Starting Outlier Detection over Loudness value (s02) --- with top_n -- {}, k_range -- {} and alpha -- {}"
          .format(n, dyn_k_range, a))
    print("---------------------------------------------------------------")
    print("RBOD Top-n Outliers Static Similarity-- {}".format(rbod_outliers))
    print("LOF Top-n Outliers Static Similarity-- {}".format(lof_outliers))
    print("Common Outliers Static Similarity-- {}".format(list(set([idx[0] for idx in rbod_outliers])
                                              .intersection([idx[0] for idx in lof_outliers]))))
    print("Obtained z-scores for all k_range are -- {}".format(user_zscores))
    print('Writing the Outlier-Scores for each of the users to {/}')
    print("---------------------------------------------------------------")
    print("RBOD Top-n Outliers obtained from the time series recordings -- {}".format(dyn_rbod_outliers))
    print("LOF Top-n Outliers  from the time series recordings -- {}".format(dyn_lof_outliers))
    print("Common Outliers from the "
          "time series recordings -- {}"
          .format(list(set([idx[0] for idx in dyn_rbod_outliers])
                       .intersection([idx[0] for idx in dyn_lof_outliers]))))
    print("Obtained z-scores for all k_range are -- {}".format(dyn_user_zscores))
    print('Writing the Outlier-Scores for each of the users to {/}')

    if save:
        import pickle
        model_file = open("".join(properties.user_os_name + ".pckl"), "wb")
        dyn_model_file = open("".join(properties.user_os_dynname + ".pckl"), "wb")
        # Save the model
        pickle.dump(user_zscores, model_file)
        pickle.dump(dyn_user_zscores, dyn_model_file)
        model_file.close()
        dyn_model_file.close()
    print("Finished!!")


##########################################################################################
#
# Synthetic simulation, to test the constructed algorithm approach against various scenarios
# Uncomment to work with synthethic data analysis using - https://pyod.readthedocs.io/en/latest/
#
##########################################################################################

# from sklearn.metrics.pairwise import euclidean_distances
#
# #X, y_train = generate_data(100, train_only=True, contamination=0.10, n_features=2)
# contamination = 0.1
# X_train, X_test, y_train, y_test = generate_data_clusters(n_train=100, n_test=50,
#                             n_clusters=4, n_features=15,
#                             contamination=contamination, size="different", density="different",
#                             random_state=11, return_in_clusters=False)
#
# ##
# # 4 clusters
# # 30 f, 15 outliers, clusters have same size and different density.
# # random_state=11 --> Constant
#
# from sklearn.metrics import roc_auc_score
# #X_train = standardizer(X)
# outlier_output_file = open("output/od_eval/outlier_detection_cluster(4,15,10)_100.csv", "w+")
# outlier_output_file.write("k,precision_n,roc_auc,algorithm\n")
# # Create the similarity matrix
# C = np.zeros((X_train.shape[0], X_train.shape[0]))
# # A simple euclidean distance over the synthethic dataset. Not against our similarity
# for i in range(0, len(X_train)):
#     for j in range(0, len(X_train)):
#         dist = np.linalg.norm(X_train[i].reshape(1, -1) - X_train[j].reshape(1, -1))
#         C[i][j] = dist
#
# C_df = pd.DataFrame(C)
# u_id = np.asarray([i for i in range(len(X_train))]).reshape(len(X_train), 1)
# C_df.insert(0, "user_id", u_id)
# X_train = np.hstack((u_id, X_train))
# #clf = KNN(n_neighbors=k)
# for k in range(10, 61, 1):
#     #clf = COF(n_neighbors=k)
#     clf_lof = LOF(n_neighbors=k)
#     #clf.fit(X_train[:, 1:])
#     clf_lof.fit(X_train[:, 1:])
#
#     combination_dict = {}
#     rbod = RBOD(C_df, kneighbors=k)
#     combination_dict["outliers"] = rbod.detect(X_train)
#
#     #print("Classifer Outlier Labels (COF) - {}".format(clf.labels_))
#     print("Classifer Outlier Labels (LOF) - {}".format(clf_lof.labels_))
#     #To show labels for RBDA
#     # This code based on numpy executions of precision_scoring
#     rbod_decision_scores = np.asarray([val[1] for val in combination_dict["outliers"]])
#     threshold = np.percentile(rbod_decision_scores, 100 * (1 - contamination))
#     rbod_labels = (rbod_decision_scores > threshold).astype('int')
#     print("Classifier RBDA Outlier labels are - {}".format(rbod_labels))
#
#     from pyod.utils import evaluate_print
#
#     #evaluate_print('COF with k=' + str(k), y=y_train, y_pred=clf.decision_scores_)
#     evaluate_print('LOF with k=' + str(k), y=y_train, y_pred=clf_lof.decision_scores_)
#     # The precision at rank is invalid for RBDA since. It is not possible to exactly create a threshold.
#     evaluate_print('RBOD with k=' + str(k), y=y_train, y_pred=np.asarray([val[1] for val in combination_dict["outliers"]]))
#
#     #roc_cof = np.round(roc_auc_score(y_train, clf.decision_scores_), decimals=4)
#     #prn_cof = np.round(pyod.utils.data.precision_n_scores(y_train, clf.decision_scores_), decimals=4)
#     #outlier_output_file.write("".join(str(k) + "," + str(prn_cof) + "," + str(roc_cof) + "," + "COF" + "\n"))
#
#     roc_lof = np.round(roc_auc_score(y_train, clf_lof.decision_scores_), decimals=4)
#     prn_lof = np.round(pyod.utils.data.precision_n_scores(y_train, clf_lof.decision_scores_), decimals=4)
#     outlier_output_file.write("".join(str(k) + "," + str(prn_lof) + "," + str(roc_lof) + "," + "LOF" + "\n"))
#
#     roc_rbod = np.round(roc_auc_score(y_train,
#                                       [val[1] for val in combination_dict["outliers"]]), decimals=4)
#     prn_rbod = np.round(pyod.utils.data.precision_n_scores(y_train,
#                                                            np.asarray([val[1] for val in combination_dict["outliers"]])), decimals=4)
#     outlier_output_file.write("".join(str(k) + "," + str(prn_rbod) + "," + str(roc_rbod) + "," + "RBOD" + "\n"))
#
# outlier_output_file.close()
