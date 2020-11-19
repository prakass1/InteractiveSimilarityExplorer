import properties
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import json
import pandas as pd
import numpy as np
import utility
import ast


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


def common_processing(df, item_list):
    # Getting percentage between 0 to 1 rather than score values
    if item_list in ["bg_tinnitus_history", "all"]:
        df["tschq12"] = df["tschq12"].apply(lambda x: x / 100)
        df["tschq16"] = df["tschq16"].apply(lambda x: x / 100)
        df["tschq17"] = df["tschq17"].apply(lambda x: x / 100)

    if item_list in ["bg_tinnitus_history", "all"]:
        df["tschq04"] = df.apply(create_cols_family_hist, axis=1)

    if item_list in ["modifying_influences", "related_conditions"]:
        df["tschq12"] = df["tschq12"].apply(lambda x: x / 100)

    return df

#Common elements


def get_common_cols(col1, col2):
    common_elements = set(col1).intersection(col2)
    return common_elements

from pathlib import Path


def check_access(location):
    if location.exists() and location.is_file():
        return True
    else:
        return False


def initial_processing(item_list, quest_cmbs=None, append_synthethic=False):
    # Read the csv of the tschq data and make the necessary things
    # tschq = pd.read_csv("data/input_csv/3_q.csv", index_col=0, na_filter=False)
    tschq = pd.read_pickle(properties.registration_file_location)
    hq = pd.read_pickle(properties.hearing_file_location)


    # If append synthethic is true then add the synthethic data.
    if append_synthethic:
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
            print("Simulated hearing file is not created !!!")

    tschq.reset_index(inplace=True, drop=True)

    if item_list in ["bg_tinnitus_history"]:
        tschq = tschq[quest_cmbs]

    # Cleaning tschq05 question. There is an abstraction for a row we add common value
    if item_list == "bg_tinnitus_history" or item_list == "all":
        def filter_age(x):
            if isinstance(x, int):
                # Append the most common value obtained
                return tschq["tschq05"].value_counts().head(1).index[0]
            else:
                return x

        tschq["tschq05"] = tschq["tschq05"].apply(filter_age)

    # Drop the questionnaire_id and created_at
    if item_list == "all":
        tschq.drop(["questionnaire_id", "created_at"], axis=1, inplace=True)

    # Lets read and join two questionnaires tschq and hq
    if item_list in ["modifying_influences", "related_conditions", "all"]:
        hq.isna().sum(axis=0)
        # By looking at the output we are sure that h5 and h6 do not contribute much and can be dropped
        hq.drop(["hq05", "hq06"], axis=1, inplace=True)
        hq_df = hq.set_index("user_id", inplace=False)
        df = tschq.join(hq_df.iloc[:, 2:], on="user_id")

        # Repeated code but it should be okay
        # Looking at the output, we can drop tschq25, tschq07-02, tschq04-2
        drop_cols = ["tschq01", "tschq25", "tschq07-2",
                     "tschq13", "tschq04-1", "tschq04-2"]

        # Feature engineering tschq04
        if item_list == "all":
            df = common_processing(df, item_list)

        if item_list != "all":

            df = df[quest_cmbs]
            # Normalize loudness- This is added so that we do not get same HEOM distance values
            df["tschq12"] = df["tschq12"].apply(lambda x: x / 100)
            common_cols = get_common_cols(drop_cols, quest_cmbs)
            df.drop(list(common_cols), axis=1, inplace=True)
        else:
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
    else:
        # Drop the other sub questions with respect to tschq04
        processed_df = common_processing(tschq, item_list)
        # processed_df.isna().sum()
        # Looking at the output, we can drop tschq25, tschq07-02, tschq04-2
        drop_cols = ["tschq01", "tschq25", "tschq07-2",
                 "tschq13", "tschq04-1", "tschq04-2"]

        if item_list != "all":
            common_cols = get_common_cols(drop_cols, quest_cmbs)
            processed_df.drop(list(common_cols), axis=1, inplace=True)

        # Set the heom object, while using the required similarity
        # Alternative
        # Categorical boolean mask
        categorical_feature_mask = processed_df.iloc[:, 1:].infer_objects().dtypes == object
        other_feature_mask = processed_df.iloc[:, 1:].infer_objects().dtypes != object
        # filter categorical columns using mask and turn it into a list
        categorical_cols = processed_df.iloc[:, 1:].columns[categorical_feature_mask].tolist()
        num_cols = processed_df.iloc[:, 1:].columns[other_feature_mask].tolist()
        cat_idx = [processed_df.iloc[:, 1:].columns.get_loc(val) for val in categorical_cols]
        num_idx = [processed_df.iloc[:, 1:].columns.get_loc(val) for val in num_cols]

        return cat_idx, num_idx, processed_df


def get_query_data(query_id):
    with open("".join(properties.data_location + query_id + ".json")) as json_file:
        query_json = json.load(json_file)
        cols = query_json.keys()
        # Transpose since taken vertical.
        query_df = pd.DataFrame.from_dict({1: query_json}, columns=cols, orient="index")

    return query_df


def get_user_data(query_id, simulate=False):
    if simulate:
        file_name = properties.data_location + "simulate/test_data_ui_x_test" + ".json"
    else:
        file_name = properties.data_location + "test_data_ui_x_test" + ".json"

    with open("".join(file_name)) as json_file:
        data_json = json.load(json_file)

        for i in range(0, len(data_json["users"])):
            if int(query_id) == data_json["users"][i]["user_id"]:
                query_json = data_json["users"][i]
                cols = query_json.keys()
                # Transpose since taken vertical.
                query_df = pd.DataFrame.from_dict({1: query_json}, columns=cols, orient="index")

                return query_df

'''
This is the fed info to HEOM to basically make it understand we have an order for some questions
'''
def convert_ord_5(x):
    if x == "3DAYS":
        return 0
    elif x == "6-12MONTHS":
        return 1
    elif x == "1-2YEARS":
        return 2
    elif x == "3-5YEARS":
        return 3
    elif x == "5-10YEARS":
        return 4
    elif x == "MORETHAN10YEARS":
        return 5


def convert_ord_15(x):
    if x == "LOW":
        return 0
    elif x == "MEDIUM":
        return 1
    elif x == "HIGH":
        return 2
    elif x == "VERYHIGH":
        return 3


def convert_ord_18(x):

    if x == "NONE":
        return 0
    elif x == "ONE":
        return 1
    elif x == "2TO4":
        return 2
    elif x == "5ANDMORE":
        return 3


def convert_ord_28(x):
    if x == "RARELY":
        return 0
    elif x == "SOMETIMES":
        return 1
    elif x == "USUALLY":
        return 2
    elif x == "ALWAYS":
        return 3
    elif x == "NEVER":
        return 4

#Preprocessing of the dataframe for modelling

def preprocess(df, key, age_bin=False,
               process_model_name = "data_model_encoder", prediction=False, save_model=True):
    # Identify numeric and non numeric cols

    numeric_cols = df.infer_objects().select_dtypes('number').columns
    non_numeric_cols = df.infer_objects().select_dtypes('object').columns

    # Alternative
    # Categorical boolean mask
    categorical_feature_mask = df.infer_objects().dtypes == object

    # filter categorical columns using mask and turn it into a list

    categorical_cols = df.columns[categorical_feature_mask].tolist()
    if key in ["bg_tinnitus_history"]:
        filtered_categorical_cols = [x for x in categorical_cols
                                     if x not in ["user_id", "tschq05",
                                                  "tschq15", "tschq18"]]
    elif key in ["related_conditions"]:
        filtered_categorical_cols = [x for x in categorical_cols
                                     if x not in ["user_id", "tschq28"]]

    elif key in ["all", "overall"]:
        filtered_categorical_cols = [x for x in categorical_cols
                                     if x not in ["user_id", "tschq05",
                                                  "tschq15", "tschq18", "tschq28"]]
    else:
        filtered_categorical_cols = categorical_cols

    df_copy = df.copy()

    #Keep order conversion
    if key in ["bg_tinnitus_history"]:
        df_copy["tschq05"] = df_copy["tschq05"].apply(convert_ord_5)
        df_copy["tschq15"] = df_copy["tschq15"].apply(convert_ord_15)
        df_copy["tschq18"] = df_copy["tschq18"].apply(convert_ord_18)
    elif key in ["related_conditions"]:
        df_copy["tschq28"] = df_copy["tschq28"].apply(convert_ord_28)
    elif key in ["all", "overall"]:
        df_copy["tschq05"] = df_copy["tschq05"].apply(convert_ord_5)
        df_copy["tschq15"] = df_copy["tschq15"].apply(convert_ord_15)
        df_copy["tschq18"] = df_copy["tschq18"].apply(convert_ord_18)
        df_copy["tschq28"] = df_copy["tschq28"].apply(convert_ord_28)


    # Ordinal Encode Instead of dummies
    if not prediction:
        from sklearn.preprocessing import OrdinalEncoder
        oe = OrdinalEncoder()
        #df_le = df.copy()
        #df_le[categorical_cols] = df_le[categorical_cols].apply(lambda col: le.fit_transform(col))
        oe.fit(df_copy[filtered_categorical_cols])
        if save_model:
            import utility
            utility.save_model(process_model_name, oe)
        df_copy[filtered_categorical_cols] = oe.transform(df_copy[filtered_categorical_cols])
        # Save the labelEncoder and use it while making a predictions
    else:
        import utility
        oe = utility.load_model(process_model_name)
        #pred_df = df.copy()
        df_copy[filtered_categorical_cols] = oe.transform(df_copy[filtered_categorical_cols])

    # Age binarizations. (Not used)
    if age_bin:
        bins = [0, 30, 40, 45, 50, 60, 70, 85]
        labels = [0, 1, 2, 3, 4, 5, 6]
        df_copy['age_bins'] = pd.cut(df_copy['age'], bins=bins, labels=labels, include_lowest=True)
        df_copy.drop(["age"], axis=1, inplace=True)

    return df_copy


# Cosine distance definition but not used.
def cosine_distances(X, Y=None):
    """Compute cosine distance between samples in X and Y.

    Cosine distance is defined as 1.0 minus the cosine similarity.

    Read more in the :ref:`User Guide <metrics>`.

    Parameters
    ----------
    X : array_like, sparse matrix
        with shape (n_samples_X, n_features).

    Y : array_like, sparse matrix (optional)
        with shape (n_samples_Y, n_features).

    Returns
    -------
    distance matrix : array
        An array with shape (n_samples_X, n_samples_Y).

    See also
    --------
    sklearn.metrics.pairwise.cosine_similarity
    scipy.spatial.distance.cosine (dense matrices only)
    """
    # 1.0 - cosine_similarity(X, Y) without copy
    S = cosine_similarity(X, Y)
    S *= -1
    S += 1
    np.clip(S, 0, 2, out=S)
    if X is Y or Y is None:
        # Ensure that distances between vectors and themselves are set to 0.0.
        # This may not be the case due to floating point rounding errors.
        S[np.diag_indices_from(S)] = 0.0
    return S
