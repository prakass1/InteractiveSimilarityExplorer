import pandas as pd
import numpy as np
import seaborn as sns
import tslearn as tsl
import matplotlib.pyplot as plt
#import ts_preprocessing as tsp
import properties
import plotly

import numpy as np
import json
from pathlib import Path


class Explore:
    def __init__(self, simulate=False, method="mean"):
        self.tyt_file_path = properties.ts_file_location
        self.simulate_file_path = properties.simulate_ts_file_location
        self.features_df = self.preprocess_df(simulate)

    @staticmethod
    def categorize_days(val):
        if (val > 4) and (val <= 8):
            return 'Early Morning'
        elif (val > 8) and (val <= 12):
            return 'Morning'
        elif (val > 12) and (val <= 16):
            return 'Afternoon'
        elif (val > 16) and (val <= 20):
            return 'Evening'
        elif (val > 20) and (val <= 24):
            return 'Night'
        elif val <= 4:
            return 'Late Night'

    @staticmethod
    def create_hour_num(x):
        if x[1] > 9:
            return float(str(x[0]) + "." + str(x[1]))
        else:
            return float(str(x[0]) + "." + "".join("0" + str(x[1])))

    def preprocess_df(self, simulate):
        tyt_data = pd.read_pickle(self.tyt_file_path)

        if simulate:
            # Do the simulation concatenation if the file exists
            p = Path(self.simulate_file_path)
            if p.exists() and p.is_file():
                simulate_tyt_data = pd.read_pickle(self.simulate_file_path)
                simulate_tyt_data["user_id"] = simulate_tyt_data["user_id"].apply(int)
                overall_data = tyt_data.append(simulate_tyt_data)
                time_data_set = overall_data.copy()
            else:
                # Original data copy, avoid the error and go ahead to visualization.
                print("No synthetic data present, however extracting over original data only.")
                time_data_set = tyt_data.copy()
        else:
            # Original data copy
            time_data_set = tyt_data.copy()

        # Create some features
        # Lets simply extract year month date hour min sec from the time series as a separate feature.

        # Converting to timestamp format for observation
        time_data_set["created_at"] = pd.to_datetime(time_data_set["created_at"])

        # Exploring the datetime

        daysofweek = {
            0: "Monday",
            1: "Tuesday",
            2: "Wednesday",
            3: "Thursday",
            4: "Friday",
            5: "Saturday",
            6: "Sunday"
        }

        time_data_set["year"] = time_data_set["created_at"].apply(lambda x: x.year)
        time_data_set["month"] = time_data_set["created_at"].apply(lambda x: x.month)
        time_data_set["date"] = time_data_set["created_at"].apply(lambda x: x.date())
        time_data_set["hour"] = time_data_set["created_at"].apply(lambda x: x.hour)
        time_data_set["minute"] = time_data_set["created_at"].apply(lambda x: x.minute)
        time_data_set["day"] = time_data_set["created_at"].apply(lambda x: x.day)
        time_data_set["day_of_record"] = time_data_set["created_at"].apply(lambda x: x.dayofweek)
        # time_data_set["day_of_record"] = time_data_set["created_at"].apply(lambda x: daysofweek[x.dayofweek])

        #Preprocess from 0 to 1
        time_data_set["s02"] = time_data_set["s02"].apply(lambda x: x / 100)
        time_data_set["s03"] = time_data_set["s03"].apply(lambda x: x / 100)
        time_data_set["s04"] = time_data_set["s04"].apply(lambda x: x / 100)
        time_data_set["s05"] = time_data_set["s05"].apply(lambda x: x / 100)
        time_data_set["s06"] = time_data_set["s06"].apply(lambda x: x / 100)
        time_data_set["s07"] = time_data_set["s07"].apply(lambda x: x / 100)

        #Create feature by binning categories for boxplot such as morning,
        # evening, late evening, night
        time_data_set["hour_num"] = time_data_set[["hour", "minute"]].apply(self.create_hour_num, axis=1)
        time_data_set["hour_bins"] = time_data_set["hour_num"].apply(self.categorize_days)

        return time_data_set


def create_number_of_observations_user(user_id, year, month, plot_type):
    explore = Explore()
    time_data_set = explore.features_df
    user_data_ym = time_data_set[(time_data_set["user_id"] == int(user_id)) &
                  (time_data_set["year"] == int(year)) &
                  (time_data_set["month"] == int(month))]

    #user_data = time_data_set[time_data_set["user_id"] == int(user_id)]
    #user_data_y = user_data[user_data["year"] == int(year)]
    #user_data_ym = user_data_y[user_data_y["month"] == int(month) ]

    if plot_type == "bar":
        # Hour based grouping
        print("Grouping by user, day")
        user_day_grp = user_data_ym.groupby(by=["user_id", "day"])
        number_of_obs_s02 = []
        number_of_obs_s03 = []
        day_user_id = []
        grp_day_usr = []
        for grp in user_day_grp:
            s03_list = [val for val in grp[1]["s03"].to_numpy() if not np.isnan(val)]
            s02_list = [val for val in grp[1]["s02"].to_numpy() if not np.isnan(val)]
            n_count_s03 = len(s03_list)
            n_count_s02 = len(s02_list)
            prev_day = grp[0][1]
            # prev_hr = grp[0][2]
            prev_user = grp[0][0]
            number_of_obs_s02.append(n_count_s02)
            number_of_obs_s03.append(n_count_s03)
            day_user_id.append(prev_user)
            grp_day_usr.append(prev_day)
            # grp_hr_usr.append(prev_hr)

        print("Length of n_observations s02 {}, s03 {} length of user_id {} and length of day {} for grouping by hour".format(
            len(number_of_obs_s02), len(number_of_obs_s03),
            len(day_user_id),
            len(grp_day_usr)
        ))

        dict_df_nobs2 = {
            "user_id": day_user_id,
            "days": grp_day_usr,
            "n_obs_s02": number_of_obs_s02,
            "n_obs_s03": number_of_obs_s03
        }

        df_user_nobs_day = pd.DataFrame(dict_df_nobs2)

        return df_user_nobs_day

    else:
        # Day based grouping
        print("Grouping by user, day and hour")
        user_day_grp = user_data_ym.groupby(by=["user_id", "day", "hour"])
        number_of_obs_s03_dh = []
        number_of_obs_s02_dh = []
        day_hruser_id = []
        grp_hrday_usr_day = []
        grp_hrday_usr_hr = []

        for grp in user_day_grp:
            s03_list = [val for val in grp[1]["s03"].to_numpy() if not np.isnan(val)]
            s02_list = [val for val in grp[1]["s02"].to_numpy() if not np.isnan(val)]
            n_count_s03 = len(s03_list)
            n_count_s02 = len(s02_list)
            prev_day = grp[0][1]
            prev_hr = grp[0][2]
            prev_user = grp[0][0]
            number_of_obs_s02_dh.append(n_count_s02)
            number_of_obs_s03_dh.append(n_count_s03)
            day_hruser_id.append(prev_user)
            grp_hrday_usr_day.append(prev_day)
            grp_hrday_usr_hr.append(prev_hr)

        dict_df_nobs1 = {
            "user_id": day_hruser_id,
            "days": grp_hrday_usr_day,
            "hour": grp_hrday_usr_hr,
            "n_obs_s02": number_of_obs_s02_dh,
            "n_obs_s03": number_of_obs_s03_dh
        }

        df_user_nobs_day_hr = pd.DataFrame(dict_df_nobs1)
        return df_user_nobs_day_hr


def plot_monthly_overview(user_summ_data, user_id):
    monthly_dict = {1: "January", 2: "February", 3: "March", 4: "April", 5:"May", 6:"June",
                    7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"}
    graph_overview_dict = dict(
        data=[
            dict(
                x=[monthly_dict[int(month)] for month in np.unique(user_summ_data["month"].to_numpy())],
                y=user_summ_data["s03"],
                width = [0.8 for _ in range(0, len(user_summ_data))],
                text=[str(val) for val in user_summ_data["s03"].to_numpy()],
                textposition='auto',
                hoverinfo='none',
                fillcolor='rgba(0,128,128,0.4)',
                marker=dict(
                    color='rgba(0,128,128,0.4)',
                    line=dict(
                        color='rgba(255, 63, 20, 0.7)',
                        width=1
                    )),
                type="bar"
            ),
        ],
        layout=dict(
            title="".join('User - ' + str(user_id) + " Overview of s03 Observations"),
            width="600",
            height="400",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(
                showgrid=False,
                title='Number of Observations',
                zeroline=False,
                gridwidth=2
            ),
            xaxis=dict(
                showgrid=False,
                title='Months',
                #tickvals=[int(month) for month in np.unique(user_summ_data["month"].to_numpy())],
                #ticktext=[monthly_dict[int(month)] for month in np.unique(user_summ_data["month"].to_numpy())]
            )
        )
    )

    graph_json = json.dumps(graph_overview_dict, cls=plotly.utils.PlotlyJSONEncoder)

    return graph_json




'''
Creates a box plot for the dataframe provided.
'''


def call_hour_boxplot(user_data, var_dict, user_id, col_name="s03"):
    import plotly
    import plotly.graph_objs as go
    days_list = np.sort(user_data["date"].unique())
    hours_list = np.sort(user_data["hour"].unique())
    col_list = ["rgba(0,128,128, 0.4)" for i in range(len(hours_list))]

    fig = go.Figure()
    for day, colors, hours in zip(days_list, col_list, hours_list):
        fig.add_trace(go.Box(
            y=user_data[user_data["hour"] == hours][str(col_name)].to_numpy(),
            name="".join("Hour-" + str(hours)),
            hovertext=["".join("(Date:" + str(d) + ")") for d in
                       user_data[user_data["hour"] == hours]["date"].to_numpy()],
            jitter=0.3,
            pointpos=0,
            boxmean=True,
            boxpoints='all',  # all points are shown
            marker=dict(
                color='rgba(24, 38, 114, 1)',
                size=2.0,
                line=dict(
                    color='rgba(255, 63, 20, 1)',
                    width=1
                )),
            fillcolor=str(colors),
            width=0.45
        ))
    fig.update_layout(
        boxgap=0.05,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title_text="(User - {},{}-{}) at hour level"
            .format(str(user_id), str(col_name), str(var_dict[str(col_name)])))
    fig.update_xaxes(title_text="Hours")
    fig.update_yaxes(title_text=str(var_dict[str(col_name)]))
    fig.update_layout(showlegend=False)
    return fig.to_dict()
    #plotly.offline.plot(fig)
    # plotly.offline.plot(fig, filename="users_distress_each_days_s03_notch_msd.html")
    # plotly.offline.plot(fig, filename="users_distress_each_day_notch_msd_total_day_s03.html")


def create_box_plot(user_data, user_id, col_name="s03"):
    # Create the plot using plotly to be used from UI and ajax
    var_dict = {
        "s02": "Tinnitus Loudness",
        "s03": "Tinnitus Distress",
        "s04": "Wellness of hearing",
        "s05": "Limited by hearing ability",
        "s06": "Level of stress",
        "s07": "Level of Exhaustion"
    }

    graphs = [
        dict(
            data=[
                dict(
                    x=user_data["day"],
                    y=user_data[col_name],
                    #x=["".join("Day-" + str(day)) for day in np.unique(user_data["day"].to_numpy())],
                    boxpoints='all',
                    jitter=0.3,
                    pointpos=0,
                    boxmean=True,
                    fillcolor='rgba(0,128,128,0.4)',
                    marker=dict(
                        color='rgba(24, 38, 114, 1)',
                        size=1.5,
                        line=dict(
                            color='rgba(255, 63, 20, 1)',
                            width=1
                        )),
                    type="box"
                ),
            ],
            layout=dict(
                title="".join('(User ' + str(user_id) + "," + str(col_name) + "-" + var_dict[str(col_name)] + ") at day level"),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                #width="950",
                #height="400",
                yaxis=dict(
                    showgrid = False,
                    title = "".join(var_dict[str(col_name)] + " - " + str(col_name)),
                    rangemode='tozero'
                ),
                xaxis=dict(
                    showgrid=False,
                    title='Days',
                    tickvals=[int(day) for day in np.unique(user_data["day"].to_numpy())],
                    ticktext=["".join("Day-" + str(day)) for day in np.unique(user_data["day"].to_numpy())]
                )
            )
        ),

        call_hour_boxplot(user_data, var_dict, user_id, col_name)
    ]

    graph_json = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    return graph_json


'''
Creates a bar plot for the dataframe provided for the users
'''


def call_nobs_hr_boxplot(nobs_data, var, user_id, col_name):
    import plotly
    import plotly.graph_objs as go
    days_list = np.sort(nobs_data["days"].unique())
    hours_list = np.sort(nobs_data["hour"].unique())
    col_list = ["rgba(41, 168, 214, 1)" for _ in range(len(hours_list))]

    fig = go.Figure()
    for day, colors, hours in zip(days_list, col_list, hours_list):
        fig.add_trace(go.Box(
            y=nobs_data[nobs_data["days"] == day][str(col_name)].to_numpy(),
            name="".join("Day-" + str(day)),
            hovertext=["".join("(Hour of Day:" + str(h) + ")") for h in
                       nobs_data[nobs_data["days"] == day]["hour"].to_numpy()],
            jitter=0.3,
            pointpos=0,
            boxmean=True,
            boxpoints='all',  # all points are shown
            marker=dict(
                color='rgba(24, 38, 114, 1)',
                size=2.0,
                line=dict(
                    color='rgba(255, 63, 20, 1)',
                    width=1
                )),
            fillcolor=str(colors),
            width=0.45
        ))
    fig.update_layout(
        boxgap=0.05,
        title_text="Number of observations for User - {} over days for {}"
            .format(str(user_id), str(var)))
    fig.update_xaxes(title_text="Hours")
    fig.update_yaxes(title_text="Number of observations")
    return fig.to_dict()


def create_bar_plot(nobs_data, user_id, plot_type):
    # Create the plot and mock plotly json to be used from UI and ajax requests

    if plot_type == "bar":
        graphs = [
            dict(
                data=[
                    dict(
                        x=nobs_data["days"],
                        y=nobs_data["n_obs_s03"],
                        text=[str(val) for val in nobs_data["n_obs_s03"].to_numpy()],
                        width=[0.8 for _ in range(0, len(nobs_data))],
                        textposition= 'auto',
                        hoverinfo= 'none',
                        fillcolor='rgba(0,128,128,0.4)',
                        marker=dict(
                            color='rgba(0,128,128,0.4)',
                            line=dict(
                                color='rgba(255, 63, 20, 0.7)',
                                width=1
                            )),
                        type="bar"
                    ),
                ],
                layout=dict(
                    title="".join('User - ' + str(user_id) + " "),
                    #width="950",
                    #height="400",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    yaxis=dict(
                        showgrid=False,
                        title = 'Total number of observations (Tinnitus Distress - S03)',
                        zeroline= False,
                        gridwidth= 2
                    ),
                    xaxis=dict(
                        showgrid=False,
                        title='Days',
                        tickvals=[int(day) for day in np.unique(nobs_data["days"].to_numpy())],
                        ticktext=["".join("Day-" + str(day)) for day in np.unique(nobs_data["days"].to_numpy())]
                    )
                )
            ),

            dict(
                data=[
                    dict(
                        x=nobs_data["days"],
                        y=nobs_data["n_obs_s02"],
                        text=[str(val) for val in nobs_data["n_obs_s02"].to_numpy()],
                        textposition='auto',
                        hoverinfo='none',
                        fillcolor='#29a8d6',
                        marker=dict(
                            color='#29a8d6',
                            line=dict(
                                color='rgba(255, 63, 20, 0.7)',
                                width=1
                            )),
                        type="bar"
                    ),
                ],
                layout=dict(
                    title="".join('User - ' + str(user_id) + "bar plot for tinnitus loudness"),
                    #width="950",
                    #height="400",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    yaxis=dict(
                        title='Total number of observations (Tinnitus Loudness - (S02))',
                        zeroline=False,
                        gridwidth=2
                    ),
                    xaxis=dict(
                        title='Days',
                        tickvals=[int(day) for day in np.unique(nobs_data["days"].to_numpy())],
                        ticktext=["".join("Day-" + str(day)) for day in np.unique(nobs_data["days"].to_numpy())]
                    )
                )
            )

        ]

        graph_json = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    elif plot_type == "box":
        '''
        graphs = [
            dict(
                data=[
                    dict(
                        x=nobs_data["days"],
                        y=nobs_data["n_obs_s03"],
                        #hovertext = ["".join("Hour of Day - " + str(hr)) for hr in nobs_data["hour"].to_numpy()],
                        boxpoints='all',
                        jitter=0.3,
                        pointpos=0,
                        boxmean=True,
                        fillcolor='rgba(41, 168, 214, 1)',
                        marker=dict(
                            color='rgba(24, 38, 114, 1)',
                            size=1.5,
                            line=dict(
                                color='rgba(255, 63, 20, 1)',
                                width=1
                            )),
                        type="box"
                    ),
                ],
                layout=dict(
                    title="".join('Tinnitus Distress Box plot for the User - ' + str(user_id)),
                    # width="950",
                    # height="400",
                    yaxis=dict(
                        title='Number of observations (Tinnitus Distress - S03) per day',
                        rangemode='tozero'
                    ),
                    xaxis=dict(
                        title='Days',
                        tickvals=[int(day) for day in np.unique(nobs_data["days"].to_numpy())],
                        ticktext=["".join("Day-" + str(day)) for day in np.unique(nobs_data["days"].to_numpy())]
                    )
                )
            ),

            dict(
                data=[
                    dict(
                        x=nobs_data["days"],
                        y=nobs_data["n_obs_s02"],
                        boxpoints='all',
                        jitter=0.3,
                        pointpos=0,
                        boxmean=True,
                        fillcolor='rgba(41, 168, 214, 1)',
                        marker=dict(
                            color='rgba(24, 38, 114, 1)',
                            size=1.5,
                            line=dict(
                                color='rgba(255, 63, 20, 1)',
                                width=1
                            )),
                        type="box"
                    ),
                ],
                layout=dict(
                    title="".join('Tinnitus Loudness Box plot for the User - ' + str(user_id)),
                    # width="950",
                    # height="400",
                    yaxis=dict(
                        title='Number of observations (Tinnitus Loudness - (S02)) per day',
                        rangemode='tozero'
                    ),
                    xaxis=dict(
                        title='Days',
                        tickvals=[int(day) for day in np.unique(nobs_data["days"].to_numpy())],
                        ticktext=["".join("Day-" + str(day)) for day in np.unique(nobs_data["days"].to_numpy())]
                    )
                )
            )

        ]
        '''

        graphs = [call_nobs_hr_boxplot(nobs_data, "tinnitus distress", user_id, "n_obs_s03"),
                  call_nobs_hr_boxplot(nobs_data, "tinnitus loudness", user_id, "n_obs_s02")]

        graph_json = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    return graph_json


def visualize_all_users_box(g_type, variable):
    import plotly.graph_objs as go
    if g_type == "day_hour":
        nobs_data = create_number_of_observations_user(grouping=["user_id", "day", "hour"])
        # Do it by day and hour
        df_grp_day_nobs_dh = nobs_data.groupby(["user_id", "days"]).sum().reset_index()
        total_day_users_dayhr = pd.DataFrame({"day": df_grp_day_nobs_dh["days"].value_counts().index,
                                              "total_users": df_grp_day_nobs_dh["days"].value_counts().values})
        total_users_dh = total_day_users_dayhr.sort_values(["day"])["total_users"].to_numpy()
        if variable == "Tinnitus_Loudness":
            col_list = ["#778899" for _ in range(31)]

            days_list = np.sort(nobs_data["days"].unique())
            user_ids = nobs_data["user_id"].unique()
            fig = go.Figure()
            for day, colors, t_users in zip(days_list, col_list, total_users_dh):
                fig.add_trace(go.Box(
                    y=nobs_data[nobs_data["days"] == day]["n_obs_s02"].to_numpy(),
                    name="".join("(Day-" + str(day) + " \n total_users-" + str(t_users) + ")"),
                    hovertext=["".join("(User_id: " + str(id) + "\n Hour: " + str(hr) + ")") for id, hr in
                               zip(nobs_data[nobs_data["days"] == day]["user_id"].to_numpy(),
                                   nobs_data[nobs_data["days"] == day]["hour"].to_numpy())],
                    jitter=0.3,
                    pointpos=0,
                    boxmean=True,
                    boxpoints='all',  # all points are shown
                    fillcolor=str(colors),
                    marker_color='rgb(7,40,89)',
                    line_color='rgb(7,40,89)',
                    width=.55
                ))
            fig.update_layout(
                boxgap=0.05,
                title_text="Users (S02 - Tinnitus Loudness) sequence of observations at day level")
            fig.update_xaxes(title_text="Days")
            fig.update_yaxes(title_text="number_of_observations")
            # plotly.offline.plot(fig, filename="users_distress_each_days_s03_notch_msd.html")
            # plotly.offline.plot(fig, filename="users_distress_each_day_notch_msd1.html")
            return fig.to_json()

        else:
            col_list = ["#778899" for i in range(31)]

            days_list = np.sort(nobs_data["days"].unique())
            user_ids = nobs_data["user_id"].unique()
            fig = go.Figure()
            for day, colors, t_users in zip(days_list, col_list, total_users_dh):
                fig.add_trace(go.Box(
                    y=nobs_data[nobs_data["days"] == day]["n_obs_s03"].to_numpy(),
                    name="".join("(Day-" + str(day) + " \n total_users-" + str(t_users) + ")"),
                    hovertext=["".join("(User_id: " + str(id) + "\n Hour: " + str(hr) + ")") for id, hr in
                               zip(nobs_data[nobs_data["days"] == day]["user_id"].to_numpy(),
                                   nobs_data[nobs_data["days"] == day]["hour"].to_numpy())],
                    jitter=0.3,
                    pointpos=0,
                    boxmean=True,
                    boxpoints='all',  # all points are shown
                    fillcolor=str(colors),
                    marker_color='rgb(7,40,89)',
                    line_color='rgb(7,40,89)',
                    width=.55
                ))
            fig.update_layout(
                boxgap=0.05,
                title_text="Users (S03 - Tinnitus Distress) sequence of observations at day level ")
            fig.update_xaxes(title_text="Days")
            fig.update_yaxes(title_text="number_of_observations")
            # plotly.offline.plot(fig, filename="users_distress_each_days_s03_notch_msd.html")
            #plotly.offline.plot(fig, filename="users_distress_each_day_notch_msd1.html")
            return fig.to_json

    else:
        nobs_data = create_number_of_observations_user(grouping=["user_id", "day"])
        total_day_users = pd.DataFrame({"day": nobs_data["days"].value_counts().index,
                                        "total_users": nobs_data["days"].value_counts().values})
        total_users = total_day_users.sort_values(["day"])["total_users"].to_numpy()

        # This is by day
        if variable == "Tinnitus_Loudness":
            col_list = ["#778899" for i in range(31)]

            import plotly.graph_objects as go
            days_list = np.sort(nobs_data["days"].unique())
            user_ids = nobs_data["user_id"].unique()
            fig = go.Figure()
            for day, colors, t_users in zip(days_list, col_list, total_users):
                fig.add_trace(go.Box(
                    y=nobs_data[nobs_data["days"] == day]["n_obs_s02"].to_numpy(),
                    name="".join("(Day-" + str(day) + " \n total_users-" + str(t_users) + ")"),
                    hovertext=["".join("(User_id: " + str(id) + ")") for id in
                               nobs_data[nobs_data["days"] == day]["user_id"].to_numpy()],
                    jitter=0.3,
                    pointpos=0,
                    boxmean="sd",
                    boxpoints='all',  # all points are shown
                    fillcolor=str(colors),
                    marker_color='rgb(7,40,89)',
                    line_color='rgb(7,40,89)',
                    width=.55
                ))
            fig.update_layout(
                boxgap=0.05,
                title_text="Users (S02 - Tinnitus Loudness) sequence of observations at day level ")
            fig.update_xaxes(title_text="Days")
            fig.update_yaxes(title_text="Total number_of_observations (Tinnitus Loudness - (S02))")
            # plotly.offline.plot(fig, filename="users_distress_each_days_s03_notch_msd.html")
            #plotly.offline.plot(fig, filename="users_distress_each_day_notch_msd_total_day_s03.html")
            return fig.to_json

        else:
            col_list = ["#778899" for i in range(31)]

            import plotly.graph_objects as go
            days_list = np.sort(nobs_data["days"].unique())
            user_ids = nobs_data["user_id"].unique()
            fig = go.Figure()
            for day, colors, t_users in zip(days_list, col_list, total_users):
                fig.add_trace(go.Box(
                    y=nobs_data[nobs_data["days"] == day]["n_obs_s03"].to_numpy(),
                    name="".join("(Day-" + str(day) + " \n total_users-" + str(t_users) + ")"),
                    hovertext=["".join("(User_id: " + str(id) + ")") for id in
                               nobs_data[nobs_data["days"] == day]["user_id"].to_numpy()],
                    jitter=0.3,
                    pointpos=0,
                    boxmean="sd",
                    boxpoints='all',  # all points are shown
                    fillcolor=str(colors),
                    marker_color='rgb(7,40,89)',
                    line_color='rgb(7,40,89)',
                    width=.55
                ))
            fig.update_layout(
                boxgap=0.05,
                title_text="Users (S03 - Tinnitus Distress) sequence of observations at day level ")
            fig.update_xaxes(title_text="Days")
            fig.update_yaxes(title_text="Total number_of_observations (Tinnitus Distress - (S03))")
            #plotly.offline.plot(fig, filename="users_distress_each_days_s03_notch_msd.html")
            #plotly.offline.plot(fig, filename="users_distress_each_day_notch_msd_total_day_s03.html")
            return fig.to_json
