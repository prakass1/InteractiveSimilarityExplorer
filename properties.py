###########################################################################################################
# Description: This is the core definition of the properties used throughout the application.
# A change in any of it will result in errors. Please see through this file as a start point for the configs
#
############################################################################################################

model_location = "models/"
data_location = "data/"
ts_model_location = "ts_models/"
questions_location = "data/user_questions/"

# Time stamp questionnaire - This is the (s01 - s07) questions

ts_file_location = "data/input_pckl/1_q.pckl"
registration_file_location = "data/input_pckl/3_q.pckl"
hearing_file_location = "data/input_pckl/4_q.pckl"

# Simulation Files
simulate_ts_file_location = "data/simulate/1_q_sim.pckl"
simulate_registration_file_location = "data/simulate/3_q_sim.pckl"
simulate_hearing_file_location = "data/simulate/4_q_sim.pckl"

# User outlier Score Property
user_os_name = "user_outlier_scores"
user_os_dynname = "user_outlier_dyn_scores"

# questionnaire Subspace combination. (This is adaptive so that other combination or groups can be included)
quest_comb = {
    # Is the combination of background and history. This is done because age was masked as per data.
    "bg_tinnitus_history": ["user_id"] + ["".join("tschq0"+str(i)) for i in range(1, 4)] +
                           ["tschq04-1", "tschq04-2"] +
                           ["".join("tschq0"+str(i)) for i in range(5, 7)] +
                           ["tschq07-1", "tschq07-2"] +
                           ["".join("tschq0"+str(i)) if i <= 9 else "".join("tschq"+str(i))
                            for i in range(8, 19)],
    "modifying_influences": ["user_id"] + ["tschq12"] + ["".join("tschq"+str(i)) for i in range(19, 26)] + ["hq01", "hq03"],

                            #["".join("hq0"+str(i)) for i in range(1, 5)],
    "related_conditions": ["user_id"] + ["tschq12"] + ["".join("tschq"+str(i)) for i in range(28, 36)] + ["hq02", "hq04"]
                          #["".join("hq0"+str(i)) for i in range(1, 5)]
}

