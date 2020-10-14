import properties
import numpy as np
import pandas as pd

# Basically convert all data objects to string format for the sake of processing.
def convert_to_string(dict):
    temp_dict = {}
    for key,val in dict.items():
        temp_dict[key] = str(val)
    return temp_dict

# This is method to obtain the scores from RBDA algorithm. Note we do not run the algorithm via UI. It has done offline
# and scores are saved.
def get_outlier_scores(user_id, flag=False):
    # Load the outlier scores for adding them to be visible from the UI
    import pickle
    from pathlib import Path
    if flag:
        # Load the file
        os_location = "".join(properties.user_os_dynname + ".pckl")
    else:
        # Load the file
        os_location = "".join(properties.user_os_name + ".pckl")
    p = Path(os_location)
    if p.exists() and p.is_file():
        outlier_scores = pickle.load(open(os_location, 'rb'))
        score = [users[1] for users in outlier_scores if users[0] == user_id]

        if score:
            return score[0]
        else:
            return None
    else:
        return None


# determine dissimilarity. Basically this is the threshold (tau) which is the distance difference from the nearest neighbor
def dissimilarity_output(scores, alpha=1.0):
    close_score = scores[0]
    difference_scores = np.asarray([(scores[i] - close_score)
                for i in range(1, len(scores))])

    difference_scores_mean = np.mean(difference_scores)
    difference_scores_std = np.std(difference_scores)
    # 75 percentile cut-off.
    # If anything greater than that is dissimilar in the neighboring group (Not used)
    #threshold = np.percentile(difference_scores, 75, interpolation="midpoint")
    threshold = difference_scores_mean + alpha * difference_scores_std
    score_indexes = np.where(difference_scores > threshold)
    return [score_index + 1 for score_index in score_indexes[0]], round(threshold, 7)


# Create the distance table to fill to UI. Table option to show distance is populated here
def create_distance_table(query_id, links_list, scores_index, threshold):
    table_row = '<table class="table table-sm table-bordered "><tbody>'
    temp_row = '<tr><td>--</td>'
    dist_row = '<tr><td>' + str(query_id) + '</td>'
    for index, element in enumerate(links_list):

        if element["close"] == 1:
            temp_row += '<td><b>' + element["target"] + '</b></td>'
            dist_row += '<td><b>' + element["score"] + '</b></td>'
        else:
                if index in scores_index:
                    temp_row += '<td bgcolor="pink">' + element["target"] + '</td>'
                    dist_row += '<td bgcolor="pink">' + element["score"] + '</td>'
                else:
                    temp_row += '<td>' + element["target"] + '</td>'
                    dist_row += '<td>' + element["score"] + '</td>'

        if index == len(links_list) - 1:
            temp_row += "</tr>"
            dist_row += "</tr>"

    table_row += temp_row + dist_row + '</tbody></table>'

    p_row = '<p>Distance difference from the closest neighbor ' \
            'is <span style="background-color:pink;"> >' + str(threshold) + '</span> threshold value</p>'

    table_row += p_row

    return table_row


# Generates the node and link required for force directed graph
def present_json(query_id, quest_cmb="all", k=5, simulate=False):
    import properties
    import utility
    import static_sim_functions as smf
    from sklearn.metrics.pairwise import cosine_distances
    from HEOM import HEOM
    pres_dict = {}
    p_node_list = []
    p_link_list = []

    # Read the necessary model definitions. (Note: The names will have to be changed here.
    # Should be moved to json of props ideally)

    if simulate:
        exp_df = utility.load_model("".join("/simulate/" + quest_cmb + "/" + quest_cmb + "_train_stat_q_data"))
        train_sim_df = utility.load_model("".join("/simulate/" + quest_cmb + "/" + "train_sim_data.pckl"))
        encoding_dm_model = "/simulate/" + quest_cmb + "/" + quest_cmb + "_stat_q_data_oe_model"
        knn_model = "/simulate/" + quest_cmb + "/" + "knn_static"

    else:
        exp_df = utility.load_model("".join(quest_cmb + "/" + quest_cmb + "_train_stat_q_data"))
        train_sim_df = utility.load_model("".join(quest_cmb + "/" + "train_sim_data.pckl"))
        encoding_dm_model = quest_cmb + "/" + quest_cmb + "_stat_q_data_oe_model"
        knn_model = quest_cmb + "/" + "knn_static"


    #exp_df.reset_index(inplace=True, drop=True)
    query_data = smf.get_user_data(query_id, simulate=simulate)

    # drop based on combinations:
    drop_cols = ["tschq01", "tschq04-1", "tschq04-2", "tschq07-2", "tschq13", "tschq25"]

    if quest_cmb not in ["all", "overall"]:
        filtered_cols = [x for x in properties.quest_comb[quest_cmb] if x not in drop_cols]
        if quest_cmb == "bg_tinnitus_history":
            filtered_query_data = query_data[filtered_cols + ["tschq04"]]
        else:
            filtered_query_data = query_data[filtered_cols]
    else:
        filtered_query_data = query_data


    X = smf.preprocess(filtered_query_data, quest_cmb, age_bin=False,
                       process_model_name="".join(encoding_dm_model),
                       prediction=True, save_model=False)


    #Note this is heom based model
    model = utility.load_model(knn_model)
    dist, idx = model.kneighbors(X.to_numpy()[:, 1:], n_neighbors=k)
    dist = dist.flatten()
    idx = idx.flatten()

    #An index of the score to highlighted
    scores_index, threshold = dissimilarity_output(dist, alpha=1.0)


    # Construct the graph data structure.
    q_data_dict = query_data.to_dict("r")[0]
    p_init_dict = convert_to_string(q_data_dict)
    p_init_dict["query"] = 0
    p_init_dict['zscore'] = get_outlier_scores(int(query_id))


    p_node_list.append(p_init_dict)
    min_dist = min(dist)
    for i, val in enumerate(idx):
        user_id = train_sim_df.iloc[val]["user_id"]
        p_dict = exp_df[exp_df["user_id"] == int(user_id)].to_dict("r")[0]
        #Iterate to convert all to string
        p_dict = convert_to_string(p_dict)
        p_dict["query"] = 1
        p_dict['zscore'] = get_outlier_scores(int(user_id))

        p_dict["score"] = str(dist[i])
        '''
        scales_arr = []
        for qval in ["tschq12", "tschq16", "tschq17"]:
            scale_dict = {}
            scale_dict["name"] = qval
            scale_dict["val"] = p_dict[qval]
            scales_arr.append(scale_dict)

        p_dict["pie_scales"] = scales_arr
        '''

        p_node_list.append(p_dict)
        p_link_dict = {}
        #it should simple be the query_id
        p_link_dict["source"] = str(query_id)
        #All the targets to the query
        p_link_dict["target"] = str(int(train_sim_df.iloc[val]["user_id"]))
        #Their distances
        p_link_dict["score"] = str(round(dist[i], 7))
        p_link_dict["zscore"] = get_outlier_scores(int(user_id))

        if dist[i] > min_dist:
            p_link_dict["close"] = 0
        else:
            p_link_dict["close"] = 1

        p_link_list.append(p_link_dict)
    pres_dict["nodes"] = p_node_list
    pres_dict["links"] = p_link_list

    # Create the distance table
    pres_dict["distance_table"] = create_distance_table(query_id, p_link_list, scores_index, threshold)
    return pres_dict

# Get the heatmap for visualization
def get_patient_information(quest_cmb, query_id, nearest_pid, simulate=False):
    heatmap_dict = {}
    #Get row label encode and return it assuming there are lot categorical attributes.
    # Load the encoded data.
    import utility
    import static_sim_functions as smf

    if simulate:
        encoded_data = utility.load_model("".join("/simulate/" + quest_cmb + "/" + quest_cmb + "_stat_q_data_encoded"))
        exp_df = utility.load_model("".join("/simulate/" + quest_cmb + "/" + quest_cmb + "_stat_q_data"))
        encoded_data_model = "/simulate/" + quest_cmb + "/" + quest_cmb + "_stat_q_data_oe_model"
    else:
        encoded_data = utility.load_model("".join(quest_cmb + "/" + quest_cmb + "_stat_q_data_encoded"))
        exp_df = utility.load_model("".join(quest_cmb + "/" + quest_cmb + "_stat_q_data"))
        encoded_data_model = quest_cmb + "/" + quest_cmb + "_stat_q_data_oe_model"

    #encoded_data = utility.load_model("".join(quest_cmb + "/" + quest_cmb + "_stat_q_data_encoded"))
    # load models according to the combinations. Move this to a method.
    #exp_df = utility.load_model("".join(quest_cmb + "/" + quest_cmb + "_stat_q_data"))
    query_data = smf.get_user_data(query_id, simulate=simulate)
    # Before preprocessing removing the cols from the actual dataset by their combinations.
    drop_cols = ["tschq01", "tschq04-1", "tschq04-2", "tschq07-2", "tschq13", "tschq25"]

    if quest_cmb not in ["all", "overall"]:
        filtered_cols = [x for x in properties.quest_comb[quest_cmb] if x not in drop_cols]
        if quest_cmb == "bg_tinnitus_history":
            filtered_query_data = query_data[filtered_cols + ["tschq04"]]
        else:
            filtered_query_data = query_data[filtered_cols]
    else:
        filtered_query_data = query_data

    query_id_row_encoded = smf.preprocess(filtered_query_data, quest_cmb, age_bin=False,
                                          process_model_name="".join(encoded_data_model),
                                          prediction=True)
    nearest_pid_row = encoded_data[encoded_data["user_id"] == int(nearest_pid)]

    #Get only numeric data and visualize. Or sometimes the oridinal which can be all normalized.

    nearest_pid_row_selected = nearest_pid_row
    query_id_row_selected = query_id_row_encoded
    #columns = nearest_pid_row.columns

    combined_p_df =  query_id_row_selected.append(nearest_pid_row_selected)

    #combined_p_df_1 = nearest_pid_row_1.append(processed_query_data)
    #combined_p_df_1.set_index("user_id", inplace=True)

    combined_p_df.set_index('user_id', inplace=True)
    #combined_p_df.drop(["id"], axis=1, inplace=True)


    import patient
    import patient_view
    import json
    parent_data_objs = []
    pres_dict = {}
    for index in combined_p_df.index:
        dict = convert_to_string(pd.Series.to_dict(combined_p_df.loc[index]))
        #To get actual attributes for heatmap. Look at this later
        #dict1 = convert_to_string(pd.Series.to_dict(combined_p_df_1.loc[index]))
        for key, val in dict.items():
            obj = patient.Patient(index, key, dict[key])
            parent_data_objs.append(obj)
    import jsonpickle
    data = patient_view.PatientView(parent_data_objs)
    json_data = jsonpickle.encode(data, unpicklable=False)
    return json_data


def create_patient_objs(key, series, var_type="tinnitus_distress"):
    import patient
    import patient_view
    patient_data_list = []
    for row in series:
        #pObj = patient.TemporalPatient(str(row[0]), str(row[1]), str(row[2]))
        if var_type == "tinnitus_distress":
            pObj = patient.TemporalPatient(str(round(row[1], 3)), str(round(row[3], 3)))
        else:
            pObj = patient.TemporalPatient(str(round(row[1], 3)), str(round(row[2], 3)))
        patient_data_list.append(pObj)
    p_list_obj = patient.TemporalPatientList(key, patient_data_list)
    return p_list_obj


# Visualize time series by realign them.
def get_patient_ts(query_id, nearest_id, tsuser_avg_grp, var_type):
    patients_temporal_list = []
    query_id = int(query_id)
    nearest_id = int(nearest_id)
    id_list = [query_id, nearest_id]
    import patient_view
    for ids in id_list:
        val = tsuser_avg_grp.get_user_mday_ts_visualize(ids)
        if val is not None:
            patients_temporal_list.append(create_patient_objs(ids, val, var_type))
    #for key, series in copy_data_series_tuple:
    #    if key == query_id or key == nearest_id:
    import jsonpickle
    temporal_patient_list = patient_view.TemporalPatientView(patients_temporal_list)
    json_output = jsonpickle.encode(temporal_patient_list, unpicklable=False)
    return json_output


# Get the time series of the query patient for making point predictions.

def get_query_ts(query_id, tsg_usr_data):
    #import ts_preprocessing as tsp
    from time_series_grp import TimeSeriesGroupProcessing
    import patient_view
    patients_temporal_list = []
    #query_id = "8"
    #tsg_data = TimeSeriesGroupProcessing(method="mean")
    val = tsg_usr_data.get_usr_mday_ts_predict(int(query_id))
    if val is not None:
        patients_temporal_list.append(create_patient_objs(int(query_id), val, "tinnitus_distress"))
    import jsonpickle
    temporal_patient_list = patient_view.TemporalPatientView(patients_temporal_list)
    json_output = jsonpickle.encode(temporal_patient_list, unpicklable=False)
    return json_output


def weighted_average(distress_list):
    average = np.asarray(distress_list, dtype=float).mean()
    return average


def compute_weighted_avg(user_id, stress, nearest_n, prediction_at, ts_method_obj):
    import patient
    import patient_view
    #import ts_group_processing as tsg
    import jsonpickle

    #tsg = TimeSeriesGroupProcessing(method=method)

    user_ts = ts_method_obj.get_usr_mday_ts_predict(int(user_id))
    user_ts_idx = user_ts[:, 1]
    pred_idx = int(np.where(user_ts_idx == prediction_at)[0])

    # Only forecast this point is ndays
    if pred_idx - (len(user_ts_idx) - 1) == 0:
        # Last point no ground truth needs only forecast
        ref_pred_at = prediction_at
        prediction_at_list = list()
        for i in range(0, 3):
            ref_pred_at += 1
            prediction_at_list.append(ref_pred_at)

    else:
        #Other reference points only to the points available. Note: This is our assumption can be changed here.
        prediction_at_list = user_ts_idx[pred_idx + 1:pred_idx + 4].tolist()
        if len(prediction_at_list) < 3:
            len_p_list = len(prediction_at_list)
            day_prop = 1
            #prev_day_idx_val = prediction_at_list[len(prediction_at_list) - 1]
            for _ in range(len_p_list, 3):
                prev_day_idx_val = prediction_at_list[len(prediction_at_list) - 1]
                prediction_at_list.append(prev_day_idx_val + day_prop)

    preds = list()

    #Prediction for four time points
    for val in prediction_at_list:
        distress_list = list()
        for user_id in nearest_n:
            # print(user_id)
            # For this user now get the time series.
            user_ts = ts_method_obj.get_usr_mday_ts_predict(int(user_id))
            diff_arr = np.abs(np.subtract(val, user_ts[:, 1]))
            diff_near_idx = np.where(diff_arr == diff_arr.min())

            #difference near index
            usr_vals = np.array([user_ts[:, 3][n_idx] for n_idx in diff_near_idx])
            if len(usr_vals) > 0:
                value = np.average(usr_vals)
            else:
                value = usr_vals[0]

            distress_list.append(value)
        preds.append((val, weighted_average(distress_list)))

    patient_data_list = []
    pObj = patient.TemporalPatient(str(prediction_at), str(stress))
    patient_data_list.append(pObj)
    for time, p_stress in preds:
        pObj = patient.TemporalPatient(str(time), str(round(p_stress, 3)))
        patient_data_list.append(pObj)

    import jsonpickle
    temporal_patient_list = patient_view.TemporalPatientView(patient_data_list)
    json_output = jsonpickle.encode(temporal_patient_list, unpicklable=False)

    return json_output

# A weighted average approach using Linear regression logic. This can be modified to contain weighted by distance logic


# The linear regression technique
def compute_linear_regression(user_id, stress, nearest_n, prediction_at, ts_method_obj):
    import patient
    import patient_view
    from sklearn.linear_model import LinearRegression
    preds = list()
    #Prediction for four time points
    for i in range(int(prediction_at) + 1, int(prediction_at) + 4):
        intercepts_list = list()
        coeff_list = list()
        for n_u_id in nearest_n:
            # print(user_id)
            # For this user now get the time series.
            user_ts = ts_method_obj.get_usr_mday_ts_predict(int(n_u_id))
            # Obtain the time series until time point and fit the data for linear regression
            diff_arr = np.abs(np.subtract(i, user_ts[:, 1]))
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

            # Add the line's coefficient and intercept.
            intercepts_list.append(reg_fit.intercept_)
            coeff_list.append(reg_fit.coef_)

        print("Predicting the value of s3 over the averaged slope and intercepts of "
              "observations of the neighbors")

        # y = mx + c, where m is the average slope of the neighbors and c is the average intercept obtained.
        print("The equation to estimate s03 for the user is {}".format("".join(str(np.asarray(coeff_list).mean())) +
                                                                       "* time_index + " +
                                                                       str(np.asarray(intercepts_list).mean())))
        y = np.multiply(np.asarray(coeff_list).mean(), i) + np.asarray(intercepts_list).mean()
        preds.append((i, y))

    patient_data_list = []
    pObj = patient.TemporalPatient(str(prediction_at), str(stress))
    patient_data_list.append(pObj)
    for time, p_stress in preds:
        pObj = patient.TemporalPatient(str(time), str(round(p_stress, 3)))
        patient_data_list.append(pObj)

    import jsonpickle
    temporal_patient_list = patient_view.TemporalPatientView(patient_data_list)
    json_output = jsonpickle.encode(temporal_patient_list, unpicklable=False)

    return json_output



### Dynamic functions to present data.

def present_json_ts(query_id, k=5):
    # Load the train and test data.
    import utility
    import properties
    p_link_list = []
    p_node_list = []

    train_data = utility.load_model("".join("dynamic_ts/" + "dynamic_ts_train_data"))
    test_data = utility.load_model("".join("dynamic_ts/" + "dynamic_ts_test_data"))
    knn_ts = utility.load_model("".join("dynamic_ts/" + "dynamic_ts_knn"))
    query_data = test_data[test_data["user_id"] == int(query_id)]
    exp_df = utility.load_model("".join("overall" + "/" + "overall" + "_stat_q_data"))
    dist, idx = knn_ts.kneighbors(query_data[train_data.index], n_neighbors=k)
    dist = dist.flatten()
    idx = idx.flatten()

    # An index of the score to highlighted
    scores_index, threshold = dissimilarity_output(dist, alpha=1.0)

    query_data = exp_df[exp_df["user_id"] == int(query_id)].to_dict("r")[0]
    p_init_dict = convert_to_string(query_data)
    p_init_dict["query"] = 0
    p_init_dict["zscore"] = get_outlier_scores(int(query_id))
    p_node_list.append(p_init_dict)
    min_dist = min(dist)
    train_data_users = train_data["user_id"].to_numpy()
    for i, val in enumerate(idx):
        user_id = train_data_users[val]
        nn_dict = {}
        node = exp_df[exp_df["user_id"] == int(user_id)].to_dict("r")[0]
        node_dict = convert_to_string(node)
        node_dict["query"] = 1
        node_dict["score"] = str(dist[i])
        node_dict["zscore"] = get_outlier_scores(int(train_data.iloc[val]["user_id"]))
        p_node_list.append(node_dict)
        link = {}
        link["source"] = str(query_id)
        link["target"] = str(int(train_data.iloc[val]["user_id"]))
        link["score"] = str(round(dist[i], 7))
        link["zscore"] = get_outlier_scores(int(train_data.iloc[val]["user_id"]))
        if dist[i] > min_dist:
            link["close"] = 0
        else:
            link["close"] = 1
        p_link_list.append(link)
    nn_dict["nodes"] = p_node_list
    nn_dict["links"] = p_link_list

    # Create the distance table
    nn_dict["distance_table"] = create_distance_table(query_id, p_link_list, scores_index, threshold)

    return nn_dict


'''
This is a plot to compare the time series recording of the patients distribution in hour of day.
Mostly, binned into Morning to Night through feature creation (See: Explore.py) and utilize this
for showing a distribution plot for comparing.
Hypothesis: Nearest neighbor and query time series recordings may be are similar on parts of the day (This can change)
'''


def call_box_plot(user_data, var_type, col_dict, query_id):
    def custom_sort(x):
        pos_session = {
            "Early Morning": 0,
            "Morning": 1,
            "Afternoon": 2,
            "Evening": 3,
            "Night": 4,
            "Late Night": 5
        }
        return str(pos_session[x])

    import plotly
    import plotly.graph_objs as go
    import numpy as np
    sorted_hour_bins = []

    for sess in user_data["hour_bins"].unique():
        sorted_hour_bins.append(custom_sort(sess) + "-" + sess)

    sorted_hour_bins = np.sort(sorted_hour_bins)

    col_list = [col_dict["box"] for i in range(len(sorted_hour_bins))]
    creation_date_list = np.sort(user_data["created_at"].unique())

    var_dict = {
        "s02": "Tinnitus Loudness",
        "s03": "Tinnitus Distress",
        "s04": "Wellness of hearing",
        "s05": "Limited by hearing ability",
        "s06": "Level of stress",
        "s07": "Level of Exhaustion"
    }

    fig = go.Figure()
    for creation_date, hour_bin, col in zip(creation_date_list, sorted_hour_bins, col_list):
        hour_sess = hour_bin.split("-")[1]
        fig.add_trace(go.Box(
            y=user_data[user_data["hour_bins"] == hour_sess][str(var_type)].to_numpy(),
            name="".join(str(hour_sess)),
            hovertext=["".join("(Date:" + str(d).split("T")[0] + "), (Time:" + str(d).split("T")[1].split(".")[0] + ")")
                       for d in
                       user_data[user_data["hour_bins"] == hour_sess]["created_at"].to_numpy()],
            jitter=0.3,
            pointpos=0,
            boxmean=True,
            boxpoints='all',  # all points are shown
            marker=dict(
                color=col_dict["marker"],
                size=1.5,
                line=dict(
                    color=col_dict["line"],
                    width=1
                )),
            fillcolor=str(col),
            width=0.6
        ))
    fig.update_layout(
        width=505,
        height=400,
        boxgap=0.05,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title_text="(User - {}, Feature - {})"
            .format(str(query_id), str(var_dict[str(var_type)])))
    fig.update_xaxes(title_text="Session")
    fig.update_yaxes(title_text=str(var_dict[str(var_type)]))
    fig.update_layout(showlegend=False)
    return fig.to_dict()


def create_boxplot_compare(query_list, var_type, simulate=False):
    import json
    import plotly
    #query_list = [8, 18]
    #var_type = "s03"
    # query_list 1st element is the query and 2nd is the clicked nearest neighbor
    # var_type - The time series variables utilized for the visualization
    query_color_dict = {"box": "rgb(255,159,0)", "marker": "rgb(200, 90, 50)", "line": "rgb(21,38,53)"}
    nn_col_dict = {"box": "rgb(0, 91, 131)", "marker": "rgb(50, 0, 200)", "line": "rgb(255, 102, 44)"}

    from exploration import Explore
    explore_ts = Explore(simulate)
    ts_feature_df = explore_ts.features_df
    # Create the required plot and add it.
    graph_data = list()
    for idx, user_id in enumerate(query_list):
        user_id_data = ts_feature_df[ts_feature_df["user_id"] == int(user_id)]
        if idx == 0:
            figure_dict = call_box_plot(user_id_data, var_type, query_color_dict, user_id)
        else:
            figure_dict = call_box_plot(user_id_data, var_type, nn_col_dict, user_id)

        graph_data.append(figure_dict)

    graph_data = json.dumps(graph_data, cls=plotly.utils.PlotlyJSONEncoder)
    return graph_data

