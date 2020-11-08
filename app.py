from flask import Flask
from flask import request, render_template, current_app, flash, redirect, url_for
import json
from flask import jsonify
import similarity_functions
import traceback
from time_series_grp import TimeSeriesGroupProcessing
import properties
import sys

'''
This is the base controller file for the app.
Author: Subash Prakash
Python Version: 3.6.X or greater
'''
app = Flask(__name__)

# Create the time series grouping each of : min, max, mean.
# This is an initialize to cache and reduce the time of prediction from UI. Can change if required to store as pickle and load
user_tsg_min = TimeSeriesGroupProcessing(method="min")
user_tsg_max = TimeSeriesGroupProcessing(method="max")
user_tsg_mean = TimeSeriesGroupProcessing(method="mean")

# Loads the default index, Commented to show directly the similarity overview instead.
#@app.route("/")
#def index():
#    return render_template("index.html")

# Load the static dashboard
@app.route("/app/static_vis")
def static_vis_template():
    return render_template("static_vis_dash.html")

# Load the dynamic dashboard
@app.route("/app/dynamic_vis")
def dynamic_vis_template():
    return render_template("dynamic_vis_dash.html")


#Run the POST request to recommend similar users and return json to D3
@app.route("/app/recommend")
def recommend():
    user_id = request.args.get("user_id")
    #simulate = request.args.get("simulate")
    return render_template("recommendations.html", user_id=user_id)

    # When simulate is True rather return random simulated users
    #if simulate == "true" or simulate == "True":
    #    return render_template("recommendations.html", user_id=user_id, simulate=simulate)
    #elif simulate == "false" or simulate == "False":
    #    return render_template("recommendations.html", user_id=user_id, simulate=simulate)
    #else:
    #    # Redirect to page home with an error message.
    #    return render_template("index.html", error="Not a correct simulate option. It should be true or false only!!")

# The Post request
@app.route("/api/recommendations", methods=["POST", "GET"])
def get_recommendations():
    # Use the form data and build a similarity for the queried patient.
    '''
    Form data contains:
    1. Patient Id - Use this to get the one patient information.
    2. Dimensions to be handled, nulls to be handled etc..
    3. To this patient, compute a pairwise similarity against all patients and
    store the similarity value.
    4. Use a threshold, to get only top N similar patients and get average similarity(a threshold).

    Choose nearest neighbors to this patients greater than threshold and return the output.
    Output: A json output must be returned which is acceptable into Directed graph.
    Use this output to build the required visualization.
    '''

    # Mocking response can also be sent for testing any improvements to the visualization.
    # with open("static/js/example.json", "r") as json_reader:
    #    data_output = json.load(json_reader)

    # return jsonify(data_output)

    # Get the information from the front-end.
    parameters = request.form.to_dict()
    static_sim = ""
    dynamic_sim = ""
    data = {}
    #simulate = request.args.get("simulate")

    # Get the simulated flag. If flag is True, load the simulated model for the UI.

    print(parameters)

    try:
        if "static-checkbox" in parameters.keys():
            static_sim = parameters["static-checkbox"]
            #simulate = parameters["simulate"]
        if "dynamic-checkbox" in parameters.keys():
            dynamic_sim = parameters["dynamic-checkbox"]
            #simulate = parameters["simulate"]

        #if simulate == "true" or simulate == "True":
        #    simulate = True
        #elif simulate == "false" or simulate == "False":
        #    simulate = False

        query_id = parameters["patient-id"]

        print(static_sim + " " + dynamic_sim + " " + query_id)

        # groups = parameters["sim-sel-grp"]
        # print(static_sim + " " + dynamic_sim + " " + query_id)
        if (static_sim == "") and (dynamic_sim == ""):
            print("Came here")
            data["message"] = "Please check one of the box for similarity criteria"

        elif static_sim == "1" and dynamic_sim == "":
            # Static recommendations
            print("Calling static similarity and building predictive visualization!!")
            quest_cmb = parameters["sim-sel-grp"]
            json_data = similarity_functions.present_json(query_id, quest_cmb, simulate=False)
            query_ts = similarity_functions.get_query_ts(query_id, user_tsg_mean)
            print(json_data)
            data["static"] = json_data
            data["query_ts"] = query_ts
            data["combination"] = quest_cmb

        elif (dynamic_sim == "1") and (static_sim == ""):
            # Dynamic recommendations
            print("Calling Dynamic similarity computations and building predictive visualizations!!")
            json_data = similarity_functions.present_json_ts(query_id)
            query_ts = similarity_functions.get_query_ts(query_id, user_tsg_mean)
            print(json_data)
            data["dynamic"] = json_data
            data["query_ts"] = query_ts

        elif (dynamic_sim == "1") and (static_sim == "1"):
            # Both together
            quest_cmb = parameters["sim-sel-grp"]
            json_data_static = similarity_functions.present_json(query_id, quest_cmb, simulate=False)
            query_ts_static = similarity_functions.get_query_ts(query_id,  user_tsg_mean)
            print(json_data_static)
            data["static"] = json_data_static
            data["query_ts"] = query_ts_static
            data["combination"] = quest_cmb
            # Dynamic recommendations
            print("Calling Dynamic similarity computations and building predictive visualizations!!")
            json_data_dynamic = similarity_functions.present_json_ts(query_id)
            print(json_data_dynamic)
            data["dynamic"] = json_data_dynamic

        return jsonify(data), "Test Message"
    except Exception:
        print("Something went wrong. Check the application", traceback.print_exc())
        data["message"] = "Something went wrong. Check the application"
        return jsonify(data)

# This request will do a POST for filling questionnaires
@app.route("/app/questionnaire", methods=["GET", "POST"])
def questionnaire():
    # Parse the data and load the json to the questions
    if request.method == "GET":
        import utility
        import os
        import properties
        from pathlib import Path
        import json
        try:
            total_q = utility.load_data("total_questions")
            # Create dummy id file in questions dir of data.
            if not os.path.isdir(properties.questions_location):
                os.makedirs(properties.questions_location)
            # Create an id file as a json. Note, not to delete it.
            if not Path(properties.questions_location + "user_id.json").is_file():
                id = {}
                id["user_id"] = 11001
                print("Saving data to the location ", properties.data_location)
                with open("".join(properties.data_location + "user_questions/user_id" + ".json"), 'w') as f:
                    json.dump(id, f)

            return render_template("questionnaire.html", questionnaire=total_q)
        except Exception:
            print("There has been some problem while loading the questionnaire")
            print(traceback.print_exc())

    elif request.method == "POST":
        # Get the information from the front-end.
        question_kv = request.form.to_dict(flat=True)
        checkbox_list = request.form.getlist("tschq04-2")
        question_kv["tschq04-2"] = checkbox_list
        # Save the questions in the user_questions dir
        # First reading the existing id.
        try:
            import properties
            import utility
            import os
            import json
            from pathlib import Path
            file_reader = open(properties.questions_location + "user_id.json")
            user_id_json = json.load(file_reader)
            # Save the question submission utilizing the id
            print("Saving user questionnaire to the location ", properties.questions_location)
            with open("".join(properties.questions_location + str(user_id_json["user_id"]) + ".json"), 'w') as f:
                json.dump(question_kv, f)
            # Increment the user_id file by rewriting.
            temp_id = user_id_json["user_id"] + 1
            user_id_json["user_id"] = temp_id
            print("Incrementing the user_id and saving")
            with open("".join(properties.questions_location + "user_id.json"), 'w') as f:
                json.dump(user_id_json, f)
            return render_template("success.html")
        except Exception:
            print(traceback.print_exc())
            error = traceback.print_exc()
            return render_template("error.html", error=error)

@app.route("/api/plot/switch", methods = ["GET"])
def switch_plot():
    # Get the request params from the get request
    nearest_pid = request.args.get("nearest_pid")
    query_id = request.args.get("query_id")
    plot_type = request.args.get("plot_type")
    var_type = request.args.get("var_type")
    simulate = request.args.get("simulate")

    if simulate == "true" or simulate == "True":
        simulate = True
    elif simulate == "false" or simulate == "False":
        simulate = False

    # Plot types acts as the action parameter to load the respective plot change
    # Options can be boxplot, heatmap, lineplot, etc...

    # Implementation of boxplot is shown
    if plot_type == "boxplot":
        # Obtain box plot for both nearest neighbor and query
        # Very important to be in this order else will change the plot. (Especially coloring factors)
        output = {}
        query_list = [query_id, nearest_pid]
        graph_data = similarity_functions.create_boxplot_compare(query_list, var_type, simulate=simulate)
        output["graph_data"] = graph_data
        return jsonify(output)

# Click level visualizations
@app.route("/api/plot", methods=["GET"])
def get_information():
    # request_data = request.get_json()
    nearest_pid = request.args.get("nearest_pid")
    query_id = request.args.get("query_id")
    combination = request.args.get("combination")
    plot_type = request.args.get("plot_type")
    var_type = request.args.get("var_type")
    simulate = request.args.get("simulate")
    print(var_type)
    # print(nearest_pid + " " + query_id)
    # Construct heatmap view information as json and return back for d3 to perform necessary visualization
    try:

        if simulate == "true" or simulate == "True":
            simulate = True
        elif simulate == "false" or simulate == "False":
            simulate = False

        if plot_type == "heatmap":
            data_output = similarity_functions.get_patient_information(combination, query_id, nearest_pid,
                                                                       simulate=simulate)
        elif plot_type == "timeseries":
            output = {}
            query_list = [query_id, nearest_pid]
            graph_data = similarity_functions.create_boxplot_compare(query_list, var_type, simulate=simulate)
            output["graph_data"] = graph_data
            return jsonify(output)
        elif plot_type == "heatmap_ts":
            output = {}
            data_hm_output = similarity_functions.get_patient_information(combination, query_id, nearest_pid,
                                                                          simulate=simulate)
            output["hm"] = data_hm_output
            # Very important to be in this order else will change the plot. (Especially coloring factors)
            query_list = [query_id, nearest_pid]
            graph_data = similarity_functions.create_boxplot_compare(query_list, var_type, simulate=simulate)
            output["graph_data"] = graph_data
            return jsonify(output)

    except Exception:
        traceback.print_exc()
        return "Something has gone wrong with the plots. Check the application"

# Define the application level itself. More like login in real sense.
@app.route("/app", methods=["GET"])
def application():
    if request.args.get("identifiers") is None:
        return "Error in request"
    form_data = request.args.get("identifiers")
    if int(form_data) == 1:
        identifier = "physician"
        return render_template("application.html", identifier=identifier)
    elif int(form_data) == 2:
        identifier = "patient"
        return render_template("application.html", identifier=identifier)
    else:
        return "Error identifier is being accessed"

# Perform predictions via a GET request
@app.route("/api/predict", methods=["GET"])
def predict():
    user_id = request.args.get("user_id")
    time_point = request.args.get("time_point")
    nearest_n = request.args.getlist("nearest_neighbors[]")
    stress = request.args.get("ref_stress")
    print("---------------------------------------------------")
    print("user_id -- " + str(user_id))
    print("time_point -- " + str(time_point))
    print("reference stress point" + str(stress))
    print(" nearest_neighbors -- " + str(nearest_n))
    print("Calling for making predictions -- weighted average at the moment")
    data = {}
    for method in ["mean", "min", "max"]:
        if method == "mean":
            #Mean of nearest neighbors
            predict_json = similarity_functions.compute_linear_regression(user_id,
                                                                          stress,
                                                                          nearest_n,
                                                                          float(time_point), user_tsg_mean)

        elif method == "min":
            # Min of nearest neighbors
            predict_json = similarity_functions.compute_linear_regression(user_id, stress,
                                                                          nearest_n, float(time_point), user_tsg_min)
        else:
            # Max of nearest neighbors
            predict_json = similarity_functions.compute_linear_regression(user_id, stress,
                                                                          nearest_n, float(time_point), user_tsg_max)

        # Encode it as JSON object to the visualizations
        data["".join(method + "_pred")] = predict_json
    return jsonify(data)

# The dashboard for data explorations
@app.route("/dash_explore", methods=["GET", "POST"])
def explore_dash():
    # Send user ids here so that select has them. Make a ajax calls after plotting to
    # dynamically change

    # Create initial plot to a default first user and change according to the selection.
    from exploration import Explore
    import exploration as plot_methods
    import numpy as np
    exp = Explore()
    # Graph json for box plot for s02 and s03
    user_ids = np.sort(exp.features_df["user_id"].unique())
    feature_df = exp.features_df
    user_data = feature_df[feature_df["user_id"] == user_ids[0]]
    usr_years = np.sort(user_data["year"].unique())

    # Get the user month's based on year.
    # Note: Year is not dynamically generated as it is known this is a 2 year data.
    data_years = np.sort(exp.features_df["year"].unique())

    user_y_data = user_data[user_data["year"] == usr_years[0]]
    months_sel = np.sort(user_y_data["month"].unique())
    user_ym_data = user_y_data[user_y_data["month"] == months_sel[0]]

    # Default Colname to s03
    graph_json = plot_methods.create_box_plot(user_ym_data, user_ids[0], col_name="s02")

    # Graph json for bar plot of number of observations per day of the user
    nobs_users = plot_methods.create_number_of_observations_user(user_ids[0], usr_years[0], months_sel[0], "bar")
    #nobs_user_df = nobs_users[nobs_users["user_id"] == user_ids[0]].groupby(["days"]).sum().reset_index()
    graph_bar_plot = plot_methods.create_bar_plot(nobs_users, user_ids[0], "bar")

    #Plot an overview accross months
    user = user_y_data[["s03", "month"]]
    user_overview_summary = user.groupby(by="month").count().reset_index()
    graph_monthly_overview = plot_methods.plot_monthly_overview(user_overview_summary, user_ids[0])

    # Plot is kept but hidden from UI
    #cmb_box_graph_json = plot_methods.visualize_all_users_box("day_hour", "Tinnitus_Loudness")

    # Construct all data and render them dynamically in the exploration dashboard. Dictionary for faster processing.
    construct_data = {"user_ids":user_ids,
                      "months":months_sel,
                      "box_graph_json": graph_json,
                      "bar_graph_json":graph_bar_plot,
                      "box_graph_monthly_overview": graph_monthly_overview,
                      "data_years": data_years}

    # reload the variable to visualize dynamically...
    return render_template("explore_dash.html", construct_data=construct_data)
                           #user_ids=user_ids,
                           #months=months_sel,
                           #box_graph_json=graph_json,
                           #bar_graph_json=graph_bar_plot,
                           #box_graph_monthly_overview = graph_monthly_overview)


# Dynamic changes GET query on demand
@app.route("/change_boxplot", methods=["GET", "POST"])
def change_feature_boxplot():
    user_id_name = request.args['user_id']
    user_id = int(user_id_name.split("-")[1])
    col_name = str(request.args["col_name"])
    year = int(request.args["year"])
    month = int(request.args["month"])
    # Create initial plot to a default first user and change according to the selection.
    from exploration import Explore
    import exploration as plot_methods
    import numpy as np
    exp = Explore()
    #user_ids = np.sort(exp.features_df["user_id"].unique())
    feature_df = exp.features_df
    user_data = feature_df[feature_df["user_id"] == user_id]

    # User year data
    user_data_y = user_data[user_data["year"] == year]

    # User monthly data
    user_data_ym = user_data_y[user_data_y["month"] == month]

    graph_json = plot_methods.create_box_plot(user_data_ym, user_id, col_name=col_name)
    return graph_json

# On demand query
@app.route("/update_obs_plot", methods=["GET", "POST"])
def change_feature_barplot():
    user_id_name = request.args['user_id']
    plot_type = request.args["plot_type"]
    user_id = int(user_id_name.split("-")[1])
    year = int(request.args["year"])
    month = int(request.args["month"])


    # Create initial plot to a default first user and change according to the selection.
    from exploration import Explore
    import exploration as plot_methods
    import numpy as np
    if plot_type == "bar":
        nobs_users = plot_methods.create_number_of_observations_user(user_id, year, month, plot_type)
        # Takes the sum of [values per days, which is an added observations] values
        #nobs_user_df = nobs_users[nobs_users["user_id"] == user_id].groupby(["days"]).sum().reset_index()
        plots = {}
        graph_bar_plot = plot_methods.create_bar_plot(nobs_users, user_id, plot_type)
        exp = Explore()
        # Graph json for box plot for s02 and s03
        feature_df = exp.features_df
        user_data = feature_df[feature_df["user_id"] == user_id]
        user_y_data = user_data[user_data["year"] == year]
        user = user_y_data[["s03", "month"]]
        user_overview_summary = user.groupby(by="month").count().reset_index()
        graph_monthly_overview = plot_methods.plot_monthly_overview(user_overview_summary, user_id)
        plots["bar_plot_graph"] = graph_bar_plot
        plots["graph_monthly_overview"] = graph_monthly_overview

        return plots
    elif plot_type == "box":
        nobs_users = plot_methods.create_number_of_observations_user(user_id, year, month, plot_type)
        #nobs_user_df = nobs_users[nobs_users["user_id"] == user_id]
        graph_bar_plot = plot_methods.create_bar_plot(nobs_users, user_id, plot_type)
        return graph_bar_plot

# On demand request for populating months from the dataset.
@app.route("/populate_months", methods=["GET"])
def populate_months():
    data = {}
    user_id_name = request.args['user_id']
    year = int(request.args["year"])
    user_id = int(user_id_name.split("-")[1])

    from exploration import Explore
    import exploration as plot_methods
    import numpy as np

    exp = Explore()
    # Graph json for box plot for s02 and s03
    feature_df = exp.features_df
    user_data = feature_df[feature_df["user_id"] == user_id]

    # Get the user month's based on year.
    # Note: Year is not dynamically generated as we know this is a 2 year data.

    user_y_data = user_data[user_data["year"] == year]
    months_sel = np.sort(user_y_data["month"].unique())

    data["months"] = months_sel.tolist()

    return data

@app.route("/", methods=["GET"])
def index():
    return redirect(url_for('similarity_dash'))

# Similarity overview page to be loaded. It should be noted the created test users are loaded as JSON to the UI.
# In reality this can come from the Database.
@app.route("/similarity_dash", methods=["GET"])
def similarity_dash():
    import utility
    print("Loading normal test instance!!!")
    #Ideally load from DB
    data = utility.load_data("test_data_ui_x_test")
    return render_template("similarity_dashboard.html", test_data=data)

    # Simulation is disabled for the public views.
    #import utility
    #simulate = request.args.get("simulate")

    # if simulate == "true" or simulate == "True":
    #     data = utility.load_data("simulate/test_data_ui_x_test")
    #     print("Loading simulated data instances")
    #     return render_template("similarity_dashboard.html", test_data=data, simulate=simulate)
    # elif simulate == "false" or simulate == "False":
    #     print("Loading normal test instance!!!")
    #     #Ideally load from DB
    #     data = utility.load_data("test_data_ui_x_test")
    #     return render_template("similarity_dashboard.html", test_data=data, simulate=simulate)
    # else:
    #     error = "The simulate option must be true or false only!!!"
    #     return render_template("index.html", error=error)


@app.route("/get_details_nobs", methods=["GET"])
def get_details_nobs():
    user_id_name = request.args['user_id']
    user_id = int(user_id_name.split("-")[1])
    year = int(request.args["year"])
    month = int(request.args["month"])
    day = int(request.args["day"])

    from exploration import Explore
    exp = Explore()
    time_data_set_f = exp.features_df
    #Obtain the subset of the data, pretty like sql select but via pandas
    user_results = time_data_set_f[(time_data_set_f["user_id"] == int(user_id)) &
                                   (time_data_set_f["year"] == int(year)) &
                                   (time_data_set_f["month"] == int(month)) &
                                   (time_data_set_f["day"] == int(day))]

    user_results = user_results.dropna()
    data_output = {"user_data": user_results.to_dict("r")}
    return data_output


# Change K and replot. Dynamic update the Nearest neighbor visualization
@app.route("/api/replot", methods = ["GET"])
def change_k_plot():
    user_id = request.args['user_id']
    static_sim = request.args['static_sim']
    dynamic_sim = request.args['dyn_sim']
    quest_cmb = request.args['combination']
    k_val = int(request.args['k_val'])
    simulate = request.args.get("simulate")

    print("Static_sim - {}",static_sim)
    print("Dyn_sim - {}", dynamic_sim)

    if static_sim == "true" and dynamic_sim == "true":
        static_sim = True
        dynamic_sim = True
    elif static_sim == "true" and dynamic_sim != "true":
        static_sim = True
        dynamic_sim = False
    elif static_sim == "false" and dynamic_sim == "true":
        static_sim = False
        dynamic_sim = True

    if simulate == "true" or simulate == "True":
        simulate = True
    elif simulate == "false" or simulate == "False":
        simulate = False


    try:
        data = {}
        if static_sim and (not dynamic_sim):
            # Static recommendations
            print("Calling static similarity and building predictive visualization!!")
            json_data = similarity_functions.present_json(user_id, quest_cmb, k=k_val, simulate=simulate)
            #query_ts = similarity_functions.get_query_ts(user_id, user_tsg_mean)
            #print(json_data)
            data["static"] = json_data
            #data["query_ts"] = query_ts
            data["combination"] = quest_cmb

        elif dynamic_sim and (not static_sim):
            # Dynamic recommendations
            print("Calling Dynamic similarity computations and building predictive visualizations!!")
            json_data = similarity_functions.present_json_ts(user_id, k=k_val)
            #query_ts = similarity_functions.get_query_ts(quest_cmb, user_tsg_mean)
            print(json_data)
            data["dynamic"] = json_data
            #data["query_ts"] = query_ts

        elif dynamic_sim and static_sim:
            # Both together
            #quest_cmb = parameters["sim-sel-grp"]
            json_data_static = similarity_functions.present_json(user_id, quest_cmb, k=k_val, simulate = simulate)
            #query_ts_static = similarity_functions.get_query_ts(user_id,  user_tsg_mean)
            print(json_data_static)
            data["static"] = json_data_static
            #data["query_ts"] = query_ts_static
            data["combination"] = quest_cmb
            # Dynamic recommendations
            print("Calling Dynamic similarity computations and building predictive visualizations!!")
            json_data_dynamic = similarity_functions.present_json_ts(user_id, k=k_val)
            print(json_data_dynamic)
            data["dynamic"] = json_data_dynamic

            # data = {}
            # data_static_output = similarity_functions.present_json()
            # data["static"] = data_static_output
            # data_ts_output = similarity_functions.present_ts_nn_json()
            # data["dynamic"] = data_ts_output
            # data

        return jsonify(data), "Test"

    except Exception:
        print("Something went wrong. Check the application", traceback.print_exc())
        data["message"] = "Something went wrong. Check the application"
        return jsonify(data)

# Host and port to run the app

app.run(host="0.0.0.0", port=5000, debug=False)
