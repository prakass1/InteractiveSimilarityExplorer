import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import utility
import properties

'''
Test experiments with the obtained parameters, bar plot by each of the user sorted.
'''

# load the save results
lr_rmse_ema = utility.load_model("lr_usr_bounds_dict_ema.pckl")
wa_rmse_ema = utility.load_model("wa_usr_bounds_dict_ema.pckl")
lr_rmse_static = utility.load_model("lr_usr_bounds_dict.pckl")
wa_rmse_static = utility.load_model("wa_usr_bounds_dict.pckl")

# Prepare and plot side by side
plot_props = {
    "ylim": (0, 0.65),
    "xlabel": "user_ids",
    "ylabel": "RMSE",
    "title": "Sorted RMSE values"
}
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
plt1 = sns.barplot(x=list(wa_rmse_ema.keys()),
                   y=list(wa_rmse_ema.values()),
                   color="steelblue", order= list(wa_rmse_ema.keys()),
                   ax=ax[0])
plt1.set(xlabel=plot_props["xlabel"],
        ylabel=plot_props["ylabel"],
        title="Weighted Average " + plot_props["title"],
        ylim=plot_props["ylim"])

plt2 = sns.barplot(x=list(lr_rmse_ema.keys()),
                   y=list(lr_rmse_ema.values()),
                   color="steelblue", order= list(lr_rmse_ema.keys()),
                   ax=ax[1])
plt2.set(xlabel=plot_props["xlabel"],
        ylabel=plot_props["ylabel"],
        title="Linear Regression " + plot_props["title"],
        ylim=plot_props["ylim"])

# for p in plt1.patches:
#     height = p.get_height()
#     plt1.text(p.get_x()+p.get_width()/2.,
#             height + 0.009,
#             '{:1.3f}'.format(height),
#             ha="center")
#
# for p in plt2.patches:
#     height = p.get_height()
#     plt2.text(p.get_x()+p.get_width()/2.,
#             height + 0.009,
#             '{:1.3f}'.format(height),
#             ha="center")
fig.tight_layout()
plt.savefig("evals_k_rmse/images/" + "barplot_ema-{}_k-{}_x_(20,30,50)".format("mean", 11) + "_.png", dpi=300, bbox_inches='tight')
plt.show()

fig1, ax1 = plt.subplots(1, 2, figsize=(12, 6))
plt3 = sns.barplot(x=list(wa_rmse_static.keys()),
                   y=list(wa_rmse_static.values()),
                   color="steelblue", order= list(wa_rmse_static.keys()),
                   ax=ax1[0])
plt3.set(xlabel=plot_props["xlabel"],
        ylabel=plot_props["ylabel"],
        title="Weighted Average " + plot_props["title"],
        ylim=plot_props["ylim"])
plt4 = sns.barplot(x=list(lr_rmse_static.keys()),
                   y=list(lr_rmse_static.values()),
                   color="steelblue", order= list(lr_rmse_static.keys()),
                   ax=ax1[1])
plt4.set(xlabel=plot_props["xlabel"],
        ylabel=plot_props["ylabel"],
        title="Linear regression " + plot_props["title"],
        ylim=plot_props["ylim"])

# for p in plt3.patches:
#    height = p.get_height()
#    plt3.text(p.get_x()+p.get_width()/2.,
#            height + 0.009,
#            '{:1.3f}'.format(height),
#           ha="center")
#
# for p in plt4.patches:
#    height = p.get_height()
#    plt4.text(p.get_x()+p.get_width()/2.,
#            height + 0.009,
#            '{:1.3f}'.format(height),
#            ha="center")
fig.tight_layout()
plt.savefig("evals_k_rmse/images/" + "barplot__static-{}_k-{}_x_(20,30,50)".format("mean", 9) + "_.png", dpi=300, bbox_inches='tight')
plt.show()
