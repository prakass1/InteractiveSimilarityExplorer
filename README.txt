The file structure of the entire app is as follows:

root_dir
|------------- data
|------------- evals_k_rmse/images
|------------- jupyter_notebooks
|------------- models
|------------- output
|------------- static
|------------- templates
|------------- app.py

###############################################################################################################################
From the above file structure,
First, evals_k_rmse/images, models, output folders are created. This is one time activity.

data --> This is the starting for the entire data, both the processed csvs, other important forms such as test.json,
simulation data are present.

eval_k_rmse -> This is just the naming file given to save all the results. The images directory also is required.
This is created manually since it is a one time process only.

jupyter_notebooks -> Contains the analysis, other infos done to understand the data, explore etc..

models -> This is an important folder which contains the data objects such as nearest neighbors, data, encoded infos.
All of these informations are required for the UI and hence this folder should be handled with care.
(We save the results and run the UI and do not do computations from UI which is expensive)

static -> static contains the required javascript, css, and the images. New visualizations can sit here.
The contribution to the introduction of two visualizations lies here. Look at vis module.
js/vis
|---- common_module.js
|---- d3-wrapper.js --> contains the node-link construction using d3.js force-direction scheme.
|---- heatmap_js.js
|---- plot_box_compare.js
|---- ts_compare.js

js/app.js --> This drives the entire javascript code where the ajax requests to python are served.
js/plots_ajax.js --> Serves ajax requests for data explorations.

templates --> This is the place where the design layout for the visualization app is created. Any new changes to the layout must go here.

######################################################################################################################################################

#### Python files ########
app.py -> This is the starting point for the flask request response. Any GET,POST request must be written here.
HEOM.py -> The implementation of HEOM similarity is present here.
ml_modelling_ts.py -> Contains the distance measure introduced based on loudness recordings over the day.
category_similarities.py -> Some of the explored similarities for category_data is written based on a survey paper. The overlap metric is derived from this and added to HEOM
exploration.py -> The file contains the infos for the plotting, basically creating the necessary information.
machine_learning_model.py -> Does not iterative run over different neighborhood. Just to create the Neighborhood data object this is used.
similarity_functions.py -> Contains functions that creates the necessary data structures for nearest neighbor visualization, comparison and the line plot visualization.
simulation_ml_modelling.py -> Runs the same executions as machine_learning_model.py, but over a created simulated data.
vary_k_ml_modelling.py -> Same as machine_learning_model.py, runs iteratively over different k, combinations etc...
utility.py -> This file helps in the utility function to load, save data objects.
outlier_detection.py -> The implementation of the voting based approach and the implementation of RBDA is present here.
RandomNeighbors.py -> The implementation of obtaining random neighbors with sklearn style is created.
data/user_questions/user_id.json -> This file contains the user_id incrementer. So that everytime a question is filled from app comes and saves here with the associated ID
########################################################################################################

#### Properties ########
** This is the most important file **
properties.py -> Well this file drives the app, it should be noted to check this file for any changes in the structure of executions
###########################

### For executions ###############
Option-1: Dockerfile:
With the execution of Dockerfile:
1. go into the root folder where the Dockerfile is present and run.
-- docker build -t <<tag_name>> .
2. Once image has e been build
-- docker run -d -p 5000:5000 <<tag_name>>
It must load the prototype app. Initially it might take sometime I would suggest after run to wait for sometime for some cache loading.
############################################################

Option-2: Manual
1. Setup the requirement.txt with the required packages etc...
2. run app.py which must open the flask web app.
############################################################


########### Cons
1. Variable naming in the app. It should be noted that the EMA variables are manually given into the templates.
See:
-- explore_dash.html: s02-s07 is specified as select options. It is diffficult to automate this factor.
-- Hover, tooltip, inner circle functions --> This is generic enough in its definition. However, the variable names could not be generic here.
So, in code one might find as follows:
function getStrokeColor(combination, node) {
    if ((combination === "all") || (combination === "overall") || (combination === "modifying_influences")) {
        return node.tschq21 === "YES" ? "#ee27e1" : "transparent"
    }
    else if ((combination === "bg_tinnitus_history")){
        return node.tschq06 === "ABRUPT" ? "#ee27e1" : "transparent"
    }
    else if (combination === "related_conditions") {
        return node.tschq26 === "YES" ? "#ee27e1" : "transparent"
       }
    else{
        return "transparent"
    }
}

From the above code, node.tschq26 --> here, tschq26 is the variable name and is not generalized.
Hence, if in case this app will need to be adapted for a completely new domain.
This should be changed or just import common_module and change the function according the structure of the new app.

2. Database integration into app. If someone wish to run it over a live environment then I would suggest the prototype application can be migrated to a db approach with storing all results into db and picking it from there.
###################################################################################################################################################################