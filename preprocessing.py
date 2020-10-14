import pandas as pd
import os
import ast
from collections import OrderedDict
import numpy as np
import utility
import properties

'''
Read the answersheets as raw json output out by data conversion as csv
'''

def read_answersheets(file_path, separator=","):
    """
    Read the actual answers of the subjects, from the table answersheets
    :param file_path:
    :return:
    """

    column_names = ['id', 'user_id', 'questionnaire_id', 'locale', 'answers', 'sensordata', 'client', 'flags',
                    'collected_at', 'deleted_at', 'created_at', 'updated_at']
    # read the answersheets csv and keep only the relevant columns
    answer_df = pd.read_csv(file_path,  sep=separator)#,names=column_names, header=None)
    answer_df = answer_df[['user_id', 'questionnaire_id', 'answers','collected_at', 'deleted_at', 'created_at','updated_at']]

    questions_list = np.sort(answer_df.questionnaire_id.unique())

    questionnaires_dict = {}

    # For each question build a dictionary and later we can write to separate files
    for idx in questions_list:
        li = []
        for row in answer_df.itertuples():
            if row.questionnaire_id == idx:
                #print(row.questionnaire_id)
                # the answers are json objects, which are evaluated by ast.literal_eval
                a = manipulate_df(pd.DataFrame(ast.literal_eval(row.answers)),
                                  row.user_id, row.questionnaire_id, row.collected_at, row.created_at, row.updated_at)
                # append the list till all the answersheet rows are processed
                li.append(a)
        questionnaires_dict[idx] = li

    return questionnaires_dict

    #final_df = pd.DataFrame(li)
    #return final_df


def manipulate_df(df, user_id, questionnaire_id, collected_at, created_at, updated_at, keep_answer_timestamps=False):
    """
    Reads the elements of the answers and creates keys and values from the answers in a format fit for appending to the
    final dataframe
    :param df:
    :param user_id:
    :param questionnaire_id:
    :return: A dictionary that becomes the row of the dataframe
    """

    a = OrderedDict()
    a['user_id'] = user_id
    a['questionnaire_id'] = questionnaire_id
    a['created_at'] = created_at
    if keep_answer_timestamps:
        a['collected_at'] = collected_at
        a['updated_at'] = updated_at
        for row in df.itertuples():
            time_label = row.label + '_collected_at'
            a[time_label] = row.collected_at
            a[row.label] = row.value

    else:
        for row in df.itertuples():
            #time_label = row.label + '_collected_at'
            #a[time_label] = row.collected_at
            a[row.label] = row.value

    return a


# returns the final dataframe
def build_data(file_path):
    """
    :param file_path:
    :return:
    """
    print('Preparing data. Please wait...')
    return read_answersheets(file_path,";")

# Write to csv all 6 questionnaires
def write_from_dict(total_dict,csv=False):
    #Note that, key is a question_id => (1 - 6) and val is the data which will be made to csv.
    for key, val in total_dict.items():
        temp_df = pd.DataFrame(val)
        if csv:
            temp_df.to_csv("".join(properties.data_location + str(key) + "_q.csv"))
        else:
            temp_df.to_pickle("".join(properties.data_location + str(key) + "_q.pckl"))


# Uncomment when this has to be reprocessed
#import properties
#file_path = properties.data_location + "answersheets.csv"
#total_questions_dict = build_data(file_path)
#write_from_dict(total_questions_dict, csv=True)

def preprocess_questions():
    question_with_translation = pd.read_csv("data/questions_with_translations.csv", sep=";")
    question_filtered = question_with_translation[['id', 'questiontype', 'label', 'values', 'required',
                                                   'locale', 'question', 'answers']]
    question_tschq_en = question_filtered[(question_filtered["locale"] == "en") &
                                          question_filtered["label"].str.match("tschq")]
    question_hq_en = question_filtered[(question_filtered["locale"] == "en") &
                                          question_filtered["label"].str.match("hq")]
    total_questions = question_tschq_en.append(question_hq_en)
    return total_questions.to_dict("r")


def remove_quotes(data):
    format_data = []
    formatted_dict = {}
    for dicts in data:
        for key, val in dicts.items():
            if isinstance(val, str):
                if val.startswith("[") and val.endswith("]"):
                    dicts[key] = ast.literal_eval(val)
                elif val.startswith("{") and val.endswith("}"):
                    dicts[key] = ast.literal_eval(val)
        format_data.append(dicts)

    return format_data


#remove_quotes(preprocess_questions())

# Uncomment to recreate questions again
#total_questions = {}
#total_questions["questions"] = remove_quotes(preprocess_questions())

#utility.save_data("total_questions", total_questions)

#utility.save_data("filter_data", remove_quotes(utility.load_data("total_questions")))