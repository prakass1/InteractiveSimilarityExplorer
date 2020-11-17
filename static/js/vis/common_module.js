//Function to code for stroke color of the outer node.
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

/*
function plot_comp_boxplots(graph_data){
//Create the plots by utilizing plotly.js
    //$("#loader").css("display","none");
    var data_obj = JSON.parse(graph_data);
    var config = {displaylogo:false, responsive:true, editable:true, displayModeBar: false};
     //, scrollZoom:true};
    console.log("api response " + data_obj[0].data)

    Plotly.newPlot("ts-vis-statq",
                data_obj[0].data,
                data_obj[0].layout || {}, config);

    Plotly.newPlot("ts-vis-statnn",
                data_obj[1].data,
                data_obj[1].layout || {}, config);
}
*/

// Switch plot ajax workflow (simulate parameter is taken out)
function switch_plot_information(target, query, plot_type, var_type, flag){
console.log("Entering to call switch_plot_information()");
$.ajax({
        url: "/api/plot/switch",
        type: "GET",
        data: {
            "nearest_pid": target,
            "query_id": query,
            "plot_type": plot_type,
            "var_type": var_type
            //"simulate": simulate_val
        },
        cache: false,
        success: function (data, test) {
            $("#loader").hide();
            console.log("api response from the ajax switch_plot_information()")
            // plot_comp_boxplots(query, nn);

            if(flag){
               plot_comp_boxplots(data.graph_data, "ts-vis-query", "ts-vis-target");
            }
            else{
               plot_comp_boxplots(data.graph_data,"ts-vis-statq", "ts-vis-statnn" );
            }

        },
        error: function (e) {
            $("#loader").hide();
            $('#log').append("<span style='color:#EC7063;font-weight:bold;font-size:120%;'>" + e + "</span>").fadeOut(5000, function () {});
        }
    });

}


// Call for plotting ajax
function ema_graph_plot(target, query, var_type, ts_flag){
$("#loader").show();
if(ts_flag){
var var_type = document.getElementById('plot_vars_ts').value
}
else{
var var_type = document.getElementById('plot_vars').value}
$.ajax({
        url: "/api/plot",
        type: "GET",
        data: {
            "nearest_pid": target,
            "query_id": query,
            "plot_type": "timeseries",
            "var_type": var_type
        },
        cache: false,
        success: function (data, test) {
            $("#loader").hide();
            console.log("api response from the ajax ema_graph_plot()")
            d_obj = JSON.parse(data.ts);
            console.log("Length of the the time series objects " + d_obj.pTList.length);
            if (d_obj.pTList.length < 2) {
                $('#log').append("<span style='color:#EC7063;font-weight:bold;font-size:120%;'>" + "INFO: One of the user has no time series." + "</span>").fadeOut(2000, function () {});
            } else {
                //Plot the time series on click here.

                console.log("Plotting the time series function plot_ts()");
                plot_ts(data.ts, false);
            }
        },
        error: function (e) {
            $("#loader").hide();
            $('#log').append("<span style='color:#EC7063;font-weight:bold;font-size:120%;'>" + e + "</span>").fadeOut(5000, function () {});
        }
    });
}

//Binary search algorithm to obtain the common neighbors quickly
function bSearch(arr, x){
    let l = 0;
    let r = arr.length - 1;
    let found = false;

    while((l <= r) && !(found)){
        //Begin binary search
        let mid = Math.floor((l+r)/2);
        console.log("Mid", mid);
        console.log(arr[mid]);
        if(arr[mid] == x){
            console.log("Found it !!");
            found = true;
        }
        else if(arr[mid] < x){
            console.log("Came Here for lt -- ", arr[mid], x);
            l = mid + 1;
        }
        else{
            console.log("Came Here for gt -- ", arr[mid], x);
            r = mid - 1;
        }
    }
    return found;
}

function getNodeColor(node) {
    if (node.query === 0) {
        query = node.user_id
    }
    return node.query === 0 ? '#ff6310' : '#005b83'
};

//Similar function for ts
function getNodeColor_ts(node) {
    if (node.query === 0) {
        query_ts = node.user_id
    }
    return node.query === 0 ? '#ff6310' : '#005b83'
};

function colorByQuery_ts(query_flag) {

    return query_flag === true ? '#ff6310' : '#005b83'
}

function toolTipText(combination, node) {
    if ((combination === "all") || (combination === "overall")) {
        return "<span class='badge badge-secondary'>" +
        "User Id:" + node.user_id + "<br/>" +
        "Gender:" + node.tschq02 + "<br/>" +
        "Loudness:" + node.tschq12 + "<br/>" +
        "family history of tinnitus:" + node["tschq04"] + "<br/>" +
        "Pattern:" + node.tschq08 + "<br/>" + //+
        "Initial Onset:" + node.tschq06 + "<br/>" + //+
        "Noise Tolerance:" + node.tschq28 + "<br/>" + "</span>" //+
        //"<b>Outcome:</b>" + node.tinnitus + "</span>"
    } else if (combination === "bg_tinnitus_history") {
        return "<span class='badge badge-secondary'>" +
        "User Id:" + node.user_id + "<br/>" +
        "Gender:" + node.tschq02 + "<br/>" +
        "Handedness:" + node.tschq03 + "<br/>" +
        "family history of tinnitus:" + node["tschq04"] + "<br/>" +
        "Loudness:" + parseFloat(node.tschq12) * 100 + "%" + "<br/>" +
        "Pattern:" + node.tschq08 + "<br/>" + //+
        "Initial Onset:" + node.tschq06 + "<br/>" + //+
        "Awake Time:" + parseFloat(node.tschq16) * 100 + "%" + "<br/>" + "</span>"
    } else if (combination === "related_conditions") {
        return "<span class='badge badge-secondary'>" +
        "User ID:" + node.user_id + "<br/>" +
        "Hearing aids:" + node.hq04 + "<br/>" +
        "Noise Tolerance:" + node.tschq28 + "<br/>" + "</span>" //+
        //"<b>Outcome:</b>" + node.tinnitus + "</span>"
    } else if (combination === "modifying_influences") {
        return "<span class='badge badge-secondary'>" +
        "User ID:" + node.user_id + "<br/>" +
        "Diagnosed hearing loss:" + node.hq03 + "<br/>" +
        "Effects of stress:" + node.tschq24 + "<br/>" +
        "Effects of loud noise:" + node.tschq20 + "<br/>" + "</span>" //+
        //"<b>Outcome:</b>" + node.tinnitus + "</span>"
    }
    else {
    return "<span class='badge badge-secondary'>" +
        "User Id:" + node.user_id + "<br/>" +
        "Gender:" + node.tschq02 + "<br/>" +
        "Loudness:" + node.tschq12 + "<br/>" +
        "family history of tinnitus:" + node["tschq04"] + "<br/>" +
        "Pattern:" + node.tschq08 + "<br/>" + //+
        "Initial Onset:" + node.tschq06 + "<br/>" + //+
        "Noise Tolerance:" + node.tschq28 + "<br/>" + "</span>" //+
    }
}

function showToolTip(combination, node) {
    //Background and all combinations
    if (node.user_id === query) {
        //Should handle NULL
        if ((combination === "all") || (combination === "overall")) {
            return "<span class='badge badge-secondary'>" +
            "User Id:" + node.user_id + "<br/>" +
            "Gender:" + node.tschq02 + "<br/>" +
            "Loudness:" + parseFloat(node.tschq12) * 100 + "%" + "<br/>" +
            "family history of tinnitus:" + node["tschq04"] + "<br/>" +
            "Pattern:" + node.tschq08 + "<br/>" + //+
            "Initial Onset:" + node.tschq06 + "<br/>" + //+
            "Noise Tolerance:" + node.tschq28 + "<br/>" + "</span>" //+
            //"<b>Outcome:</b>" + node.tinnitus + "</span>"
        } else if (combination === "bg_tinnitus_history") {
            return "<span class='badge badge-secondary'>" +
            "User Id:" + node.user_id + "<br/>" +
            "Gender:" + node.tschq02 + "<br/>" +
            "Handedness:" + node.tschq03 + "<br/>" +
            "family history of tinnitus:" + node["tschq04"] + "<br/>" +
            "Loudness:" + parseFloat(node.tschq12) * 100 + "%" + "<br/>" +
            "Pattern:" + node.tschq08 + "<br/>" + //+
            "Initial Onset:" + node.tschq06 + "<br/>" + //+
            "Awake Time:" + parseFloat(node.tschq16) * 100 + "%" + "<br/>" + "</span>"
        } else if (combination === "related_conditions") {
            return "<span class='badge badge-secondary'>" +
            "User ID:" + node.user_id + "<br/>" +
            "Loudness:" + parseFloat(node.tschq12) * 100 + "%" + "<br/>" +
            "Hearing aids:" + node.hq04 + "<br/>" +
            "Noise Tolerance:" + node.tschq28 + "<br/>" + "</span>" //+
            //"<b>Outcome:</b>" + node.tinnitus + "</span>"
        } else if (combination === "modifying_influences") {
            return "<span class='badge badge-secondary'>" +
            "User ID:" + node.user_id + "<br/>" +
            "Loudness:" + parseFloat(node.tschq12) * 100 + "%" + "<br/>" +
            "Effects of stress:" + node.tschq24 + "<br/>" +
            "Diagnosed hearing loss:" + node.hq03 + "<br/>" +
            "Effects of loud noise:" + node.tschq20 + "<br/>" + "</span>" //+
            //"<b>Outcome:</b>" + node.tinnitus + "</span>"
        }
        else{
        return "<span class='badge badge-secondary'>" +
            "User Id:" + node.user_id + "<br/>" +
            "Gender:" + node.tschq02 + "<br/>" +
            "Loudness:" + parseFloat(node.tschq12) * 100 + "%" + "<br/>" +
            "family history of tinnitus:" + node["tschq04"] + "<br/>" +
            "Pattern:" + node.tschq08 + "<br/>" + //+
            "Initial Onset:" + node.tschq06 + "<br/>" + //+
            "Noise Tolerance:" + node.tschq28 + "<br/>" + "</span>" //+
            //"<b>Outcome:</b>" + node.tinnitus + "</span>"

        }

    } else {
        if ((combination === "all") || (combination === "overall")) {
            return "<span class='badge badge-secondary'>" +
            "User Id:" + node.user_id + "<br/>" +
            "Gender:" + node.tschq02 + "<br/>" +
            "Loudness:" + parseFloat(node.tschq12) * 100 + "%" + "<br/>" +
            "family history of tinnitus:" + node["tschq04"] + "<br/>" +
            "Pattern:" + node.tschq08 + "<br/>" + //+
            "Initial Onset:" + node.tschq06 + "<br/>" + //+
            "Noise Tolerance:" + node.tschq28 + "<br/>" + "</span>" //+
            //"<b>Outcome:</b>" + node.tinnitus + "</span>"
        } else if (combination === "bg_tinnitus_history") {
            return "<span class='badge badge-secondary'>" +
            "User Id:" + node.user_id + "<br/>" +
            "Gender:" + node.tschq02 + "<br/>" +
            "Handedness:" + node.tschq03 + "<br/>" +
            "family history of tinnitus:" + node["tschq04"] + "<br/>" +
            "Loudness:" + parseFloat(node.tschq12) * 100 + "%" + "<br/>" +
            "Pattern:" + node.tschq08 + "<br/>" + //+
            "Initial Onset:" + node.tschq06 + "<br/>" + //+
            "Awake Time:" + parseFloat(node.tschq16) * 100 + "%" + "<br/>" + "</span>"
        } else if (combination === "related_conditions") {
            return "<span class='badge badge-secondary'>" +
            "User ID:" + node.user_id + "<br/>" +
            "Loudness:" + parseFloat(node.tschq12) * 100 + "%" + "<br/>" +
            "Hearing aids:" + node.hq04 + "<br/>" +
            "Noise Tolerance:" + node.tschq28 + "<br/>" + "</span>" //+
            //"<b>Outcome:</b>" + node.tinnitus + "</span>"
        } else if (combination === "modifying_influences") {
            return "<span class='badge badge-secondary'>" +
            "User ID:" + node.user_id + "<br/>" +
            "Loudness:" + parseFloat(node.tschq12) * 100 + "%" + "<br/>" +
            "Effects of stress:" + node.tschq24 + "<br/>" +
            "Diagnosed hearing loss:" + node.hq03 + "<br/>" +
            "Effects of loud noise:" + node.tschq20 + "<br/>" + "</span>" //+
            //"<b>Outcome:</b>" + node.tinnitus + "</span>"
        }
        else{
        return "<span class='badge badge-secondary'>" +
            "User Id:" + node.user_id + "<br/>" +
            "Gender:" + node.tschq02 + "<br/>" +
            "Loudness:" + parseFloat(node.tschq12) * 100 + "%" + "<br/>" +
            "family history of tinnitus:" + node["tschq04"] + "<br/>" +
            "Pattern:" + node.tschq08 + "<br/>" + //+
            "Initial Onset:" + node.tschq06 + "<br/>" + //+
            "Noise Tolerance:" + node.tschq28 + "<br/>" + "</span>" //+
            //"<b>Outcome:</b>" + node.tinnitus + "</span>"
        }
    }
}

var max = 0
function storeByDistance(link, flag) {

    if (flag) {
        return link.close === 1 ? "#3939ff" : "#B7E9F7";
    } else {
        //(link.close)
        console.log(link.close)
        return link.close === 1 ? "#3939ff" : "#B7E9F7";
    }

}

function fillByZScore(node){

     if ((node.zscore < 0) && (node.zscore > -.5)) {
            return "#A3C0E4"
        }

     else if ((node.zscore <= -.5) && (node.zscore > -1.0)) {
            return "#88A0BE"
        }

     else if ((node.zscore <= -1.0) && (node.zscore > -1.5)) {
            return "#7694CB"
        }

     else if ((node.zscore <= -1.5) && (node.zscore > -2.0)) {
            return "#5D7AB9"
        }

/*
     else if ((node.zscore <= -2.0) && (node.zscore >= -2.4)) {
            return "#4662AC"
        }

     else if ((node.zscore <= -2.0) && (node.zscore >= -2.4)) {
            return "#364C86"
        }
*/
     else if ((node.zscore <= -2.0)){
        return "#274B9F"
     }

     else if(node.zscore == 0){
        return "#fafcff"
     }
     else if ((node.zscore > 0) && (node.zscore < .5)) {
            return "#FFECEC"
        }

     else if ((node.zscore >= 0.5) && (node.zscore < 1.0)) {
            return "#FFCACA"
        }
     else if ((node.zscore >= 1.0) && (node.zscore < 1.5)){
        return "#FF6464"
     }
     else if ((node.zscore >= 1.5) && (node.zscore < 2.0)){
        return "#FF4444"
     }
     else if ((node.zscore > 2.0)){
        return "#E50000"
     }

     else{
        return "transparent"
     }

     /*
     else if ((node.zscore > 2.4)){

        return "#870009"

     }
     */
}


function colorInnerCircle(combination, node) {
    //Color palette for the rule must be explained - Blue combination
    if ((parseFloat(node.tschq12) > 0) && (parseFloat(node.tschq12) <= .25)) {
            return "#3c82ac"
    }
    else if ((parseFloat(node.tschq12) > .25) && (parseFloat(node.tschq12) <= .50)) {
            return "#6badd8"
    }
    else if ((parseFloat(node.tschq12) > .50) && (parseFloat(node.tschq12) <= .75)) {
            return "#98daff"
    }
    else if (parseFloat(node.tschq12) > .75) {
            return "#c7ffff"
    }

    /*
    if ((combination === "bg_tinnitus_history") ||
    (combination === "all") ||
    (combination === "overall") ||
    (combination === "modifying_influences") ||
    (combination === "related_conditions")) {
        if ((parseFloat(node.tschq12) > 0) && (parseFloat(node.tschq12) <= .25)) {
            return "#3c82ac"
        } else if ((parseFloat(node.tschq12) > .25) && (parseFloat(node.tschq12) <= .50)) {
            return "#6badd8"
        } else if ((parseFloat(node.tschq12) > .50) && (parseFloat(node.tschq12) <= .75)) {
            return "#98daff"
        } else if (parseFloat(node.tschq12) > .75) {
            return "#c7ffff"
        }
    }
    */
}

function fillRadiusInnerCircle(combination, node) {
    //Radius manipulation rules for loudness
    var r = 3
        if ((combination === "bg_tinnitus_history") ||
        (combination === "all") ||
        (combination === "overall") ||
        (combination === "modifying_influences") ||
        (combination === "related_conditions")) {
            if ((parseFloat(node.tschq12) > 0) && (parseFloat(node.tschq12) <= .25)) {
                return r + parseFloat(node.tschq12)
            } else if ((parseFloat(node.tschq12) > .25) && (parseFloat(node.tschq12) <= .50)) {
                return r + parseFloat(node.tschq12)
            } else if ((parseFloat(node.tschq12) > .50) && (parseFloat(node.tschq12) <= .75)) {
                return r + parseFloat(node.tschq12)
            } else if (parseFloat(node.tschq12) > .75) {
                return r + parseFloat(node.tschq12)
            }
        }
        else{
        if ((parseFloat(node.tschq12) > 0) && (parseFloat(node.tschq12) <= .25)) {
                return r + parseFloat(node.tschq12)
            } else if ((parseFloat(node.tschq12) > .25) && (parseFloat(node.tschq12) <= .50)) {
                return r + parseFloat(node.tschq12)
            } else if ((parseFloat(node.tschq12) > .50) && (parseFloat(node.tschq12) <= .75)) {
                return r + parseFloat(node.tschq12)
            } else if (parseFloat(node.tschq12) > .75) {
                return r + parseFloat(node.tschq12)
            }

        //All others same radius is retained.
        //return r
        }
}

//Add legend as a function and then call it at the end
function addLegend(choices) {
    //alert("Came", choices);
    if(choices === "static"){
        alert("Here when static");
        var svg = d3.select("#static-legend")
                    .append("svg")
                    .attr("class","float-right")
                    .attr("width", 150)
                    .attr("height", 100)
    }
    else{
        alert("Here when it is false");
        var svg = d3.select("#dynamic-legend")
                    .append("svg")
                    .attr("class","float-right")
                    .attr("width", 150)
                    .attr("height", 100)
        //var g = d3.select("#my-legend-ts")
    }
        var keys = ["study-user", "neighbors"]

        var colors = ["#ff6310", "#005b83"]

        var color = d3.scaleOrdinal()
        .domain(keys)
        .range(colors);
    // Add one dot in the legend for each name.
    svg.selectAll("mydots")
    .data(keys)
    .enter()
    .append("circle")
    .attr("cx", 20)
    .attr("cy", function (d, i) {
        return 20 + i * 30
    }) // 100 is where the first dot appears. 25 is the distance between dots
    .attr("r", 5)
    .style("fill", function (d) {
        return color(d)
    })

    // Add one dot in the legend for each name.
    svg.selectAll("mylabels")
    .data(keys)
    .enter()
    .append("text")
    .attr("x", 30)
    .attr("y", function (d, i) {
        return 30 + i * 25
    }) // 100 is where the first dot appears. 25 is the distance between dots
    .style("fill", function (d) {
        return color(d)
    })
    .text(function (d) {
        return d
    })
    .attr("text-anchor", "left")
    .style("alignment-baseline", "auto")
}

//list keys
var keyValue1 = [];
function keyList(obj) {
    Object.keys(obj).forEach(function (key) {
        keyValue1.push(key);
        if (typeof(obj[key]) == 'object') {
            keyList(obj[key]);
        }
    });

}


//Add prediction legend
function addLegendPredictions(choices) {
       if(choices){
        var svg = d3.select("#pred_legend_static")
                    .append("svg")
                    .attr("class","float-right")
                    .attr("width", 150)
                    .attr("height", 100)
        }
        else{
                var svg = d3.select("#pred_legend_dynamic")
                    .append("svg")
                    .attr("class","float-right")
                    .attr("width", 150)
                    .attr("height", 100)

        }
        //var g = d3.select("#my-legend-ts")
        var keys = ["Min Predictions", "Mean Predictions", "Max Predictions"]

        var colors = ["#39FF14", "#C40097", "#2471A3"]

        var color = d3.scaleOrdinal()
        .domain(keys)
        .range(colors);
    // Add one dot in the legend for each name.
    svg.selectAll("mydots")
    .data(keys)
    .enter()
    .append("circle")
    .attr("cx", 20)
    .attr("cy", function (d, i) {
        return 20 + i * 30 //Magic number. Tried many times to come to this value
    }) // 100 is where the first dot appears. 25 is the distance between dots
    .attr("r", 3)
    .style("fill", function (d) {
        return color(d)
    })

    // Add one dot in the legend for each name.
    svg.selectAll("mylabels")
    .data(keys)
    .enter()
    .append("text")
    .attr("x", 30)
    .attr("y", function (d, i) {
        return 30 + i * 25
    }) // 100 is where the first dot appears. 25 is the distance between dots
    .style("fill", function (d) {
        return color(d)
    })
    .text(function (d) {
        return d
    })
    .attr("text-anchor", "left")
    .style("alignment-baseline", "auto")
}

function create_tooltip_lc() {
    // create a tooltip
    var Tooltip = d3.select("body")
        .append("div")
        .style("opacity", 0)
        .attr("class", "tooltip")
        .style("background-color", "white")
        .style("border", "solid")
        .style("border-width", "2px")
        .style("border-radius", "5px")
        .style("padding", "5px")

        return Tooltip
}

// Three function that change the tooltip when user hover / move / leave a cell
function mouseover(d, Tooltip) {
    Tooltip
    .style("opacity", 1)
}
function mousemove(d, Tooltip, freeText) {
    Tooltip
    .html(freeText + "value: " + d.xvar)
    .style("left", (d3.event.pageX) + "px")
    .style("top", (d3.event.pageY) + "px")
}

function mouseleave(d, Tooltip) {
    Tooltip
    .style("opacity", 0)
}

//Sort numbers
function sortNumber(n1, n2){
return n1 - n2;
}

//Stroke according to the prediction type
function storePrediction(key){
if(key === "min_pred"){
         return "#39ff14" //Green
    }
    else if(key === "mean_pred"){
         return "#C40097" //Purple
    }
    else{
         return "#2471A3" //Steel-Blue
    }
}


//Adds the information about the coloring for both the inner circle.
/* All the code implementation is adapted from https://d3-legend.susielu.com/#color-threshold */

function legend_coloring(data, flag, legend_inner_id, zscore_id){

if(flag){
    var thresholdScale = d3.scaleOrdinal()
    .domain(data["range_scale"])
    // Range is domain array + 1
    .range(data["color_scale"]);

    //var svg = d3.selectAll(legend_inner_id).append("svg").attr("width", 450).attr("height",200);

    var svg = d3.selectAll(legend_inner_id)
            .append("svg")
            .attr("viewBox", "0 0 " + 300 + " " + 200 )
            .attr("preserveAspectRatio", "xMinYMin meet");

    svg.append("g")
    .attr("class", "legend_inner_circle")
    .attr("transform", "translate(20,20)");

    var legend = d3.legendColor()
    .shapePadding(10)
    .title(data["title"])
    .scale(thresholdScale)

    svg.select(".legend_inner_circle")
    .call(legend);
}
else{
var linear = d3.scaleLinear()
  .domain(data["range_scale"])
  .range(data["color_scale"]);

/*
var svg = d3.selectAll(zscore_id)
            .append("svg")
            .attr("width", 450)
            .attr("height",100);

*/

var svg = d3.selectAll(zscore_id)
            .append("svg")
            .attr("viewBox", "0 0 " + 300 + " " + 100 )
            .attr("preserveAspectRatio", "xMinYMin meet");

svg.append("g")
  .attr("class", "legendzscore")
  .attr("transform", "translate(0,20)");

var legendLinear = d3.legendColor()
  .shapeWidth(26)
  .cells([-2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0])
  .orient('horizontal')
  .scale(linear)
  .title(data["title"]);

$(zscore_id).append(data["span_help"]);

svg.select(".legendzscore")
  .call(legendLinear);

}
}
