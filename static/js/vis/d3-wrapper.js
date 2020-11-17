//Setup and work with d3.js
/*
Author: Subash Prakash (Otto-von-guericke Universitat)
Description: This file contains the creation of the customized visualization functions over tinnitus data
1. get_force_direct_graph() - This builds the necessary node, edge graph and visualizes
2. plot_heatmap() - Builds a heatmap like visualization for knowing the closeness over questions
3. plot_ts() - Pretty much takes care of plotting time series and visualization
4. predict() - Prediction and update components drives the visualization to predict and update dynamically

All of the above are driven through python apis which returns us the json.
The architecture is defined in such a way that with minimal modifications it can be integrated to many other components.
*/

// keep track of if tooltip is hidden or not
var isTooltipHidden = true;
var width = 350
var height = 400
var query = ""
var query_ts = ""
var target_nn_static = ""
var target_nn_ema = ""


//Onchange trigger on select call and plot again for static data
//d3.select("#plot_vars").on("change", (function(e){
//    d3.select("#ts-vis-statq").selectAll("svg").remove();
//    d3.select("#ts-vis-statnn").selectAll("svg").remove();
//    //Remove the headers as well
//    d3.select("#ts-vis-statq").selectAll("h6").remove();
//    d3.select("#ts-vis-statnn").selectAll("h6").remove();
//    ema_graph_plot(target_nn_static, query, false);
//}));

//d3.select("#static_plot_vars").on("change", (function(e){
//    d3.select("#ts-vis-statq").selectAll("svg").remove();
//    d3.select("#ts-vis-statnn").selectAll("svg").remove();
//    //Remove the headers as well
//    d3.select("#ts-vis-statq").selectAll("h6").remove();
//    d3.select("#ts-vis-statnn").selectAll("h6").remove();
//    ema_graph_plot(target_nn_static, query, false);
//}));

//Ts_flag for ema
d3.select("#static_plot_vars").on("change", function(){
    d3.select("#ts-vis-statq").selectAll("svg").remove();
    d3.select("#ts-vis-statnn").selectAll("svg").remove();
    //Remove the headers as well
    d3.select("#ts-vis-statq").selectAll("h6").remove();
    d3.select("#ts-vis-statnn").selectAll("h6").remove();
    var target_nn = target_nn_static;
    var query_id = query;
    var var_type = document.getElementById('static_plot_vars').value;
    var plot_type = "boxplot";
    //var simulate_val = document.getElementById('simulate').value
    switch_plot_information(target_nn, query_id, plot_type, var_type, false);
});

d3.select("#dynamic_plot_vars").on("change", function(){
    d3.select("#ts-vis-query").selectAll("svg").remove();
    d3.select("#ts-vis-target").selectAll("svg").remove();
    //Remove the headers as well
    d3.select("#ts-vis-query").selectAll("h6").remove();
    d3.select("#ts-vis-target").selectAll("h6").remove();
    var target_nn = target_nn_ema;
    var query_id = query_ts;
    var var_type = document.getElementById('dynamic_plot_vars').value;
    var plot_type = "boxplot";
   // var simulate_val = document.getElementById('simulate').value
    switch_plot_information(target_nn, query_id, plot_type, var_type, true);
});


//Source Help: https://observablehq.com/@skofgar/force-directed-graph-integrated-html
//Static click functionality
function getprofile(combination, target) {
    alert("Performing visualization analysis for the user " + " " + target.user_id);
    d3.select("#heat-map").selectAll("svg").remove();
    d3.select("#ts-vis-statq").selectAll("svg").remove();
    d3.select("#ts-vis-statnn").selectAll("svg").remove();
    //Remove the headers as well
    d3.select("#ts-vis-statq").selectAll("h6").remove();
    d3.select("#ts-vis-statnn").selectAll("h6").remove();
    target_nn_static = target.user_id;
    getInformation(combination, target.user_id);
}

//Time series click functionality
function getprofile_ts(target) {

    if (query_ts === target.user_id) {
        console.log("Cannot click over target user under study" + " " + target.user_id);
        return
    }
    target_nn_ema = target.user_id;
    get_ts_information(target.user_id);
}

function get_ts_information(target_id) {
    if (target_id === query_ts) {
        console.log("Cannot call on user under study");
        return
    }

    $("#dynamic_plot_vars").show();
    $("#loader").show();
    d3.select("#ts-vis-target").selectAll("svg").remove();
    d3.select("#ts-vis-query").selectAll("svg").remove();
    //alert(target_id)
    $.ajax({
        url: "/api/plot",
        type: "GET",
        data: {
            "nearest_pid": target_id,
            "query_id": query_ts,
            "plot_type": "timeseries",
            "var_type": document.getElementById('dynamic_plot_vars').value
            //"simulate": "false"
        },
        cache: false,
        success: function (data, test) {
            console.log("api response " + data.graph_data);
            //plot timeseries for the dynamic data
            plot_comp_boxplots(data.graph_data, "ts-vis-query", "ts-vis-target");
            $("#loader").hide();
        },
        error: function (e) {
            $('#log').append("<span style='color:#EC7063;font-weight:bold;font-size:120%;'>" + e + "</span>").fadeOut(30000, function () {});
            $("#loader").hide();
        }
    });

}

function getInformation(combination, target_id) {
    if (target_id === query) {
        console.log("Cannot call on user under study");
        return
    }
    $("#static_plot_vars").show();
    $("#loader").show();
    console.log("getInformation() is called for " + target_id)
    $.ajax({
        url: "/api/plot",
        type: "GET",
        data: {
            "nearest_pid": target_id,
            "query_id": query,
            "combination": combination,
            "plot_type": "heatmap_ts",
            "var_type": document.getElementById('static_plot_vars').value
            //"simulate": document.getElementById("simulate").value
        },
        cache: false,
        success: function (data, test) {
            $("#loader").hide();
            console.log("api response from the ajax getInformation()")
            //get_force_direct_graph(data);
            //plot heatmap
            //alert(data);
            plot_heatmap(data.hm);

            // plot_comp_boxplots(query, nn);

            plot_comp_boxplots(data.graph_data, "ts-vis-statq", "ts-vis-statnn");

            /*
            d_obj = JSON.parse(data.ts);
            console.log("Length of the the time series objects " + d_obj.pTList.length);
            if (d_obj.pTList.length < 2) {
                $('#log').append("<span style='color:#EC7063;font-weight:bold;font-size:120%;'>" + "INFO: One of the patient has no time series." + "</span>").fadeOut(2000, function () {});
            } else {
                //Plot the time series on click here.

                console.log("Plotting the time series function plot_ts()");
                plot_ts(data.ts, false);
            }
            */

        },
        error: function (e) {
            $('#log').append("<span style='color:#EC7063;font-weight:bold;font-size:120%;'>" + e + "</span>").fadeOut(5000, function () {});
            $("#loader").hide();
        }
    });

}

/*
// reset nodes to not be pinned
function unPinNode(node) {
node.fx = null;
node.fy = null;
}
 */
/*
function loadTooltipContent(node){
html = "<div id='patient_info'>"
html += "<p> Patient Id: " + node.id
html += "<p> Patient Age: " + node.age
html += "<p> Patient loudness: " + node.loudness
html += "<p> Patient other syndromes " + node.other_syndromes
html += "<p> Patient awake percentage " + node.awake_perc
html += "<p> Patient hearing problem " + node.hearing_problem
html += "<p> Patient has tinnitus " + node.tinnitus
html += "<p> Patient pitch level " + node.pitch + "</div>"
tooltip.html(html)
}

// add tooltip to HTML body
var tooltip = d3.select("body")
.append("div")
.attr("class", "tooltip")
.style("position", "absolute")
.style("padding", "10px")
.style("z-index", "10")
.style("width", "300px")
.style("height", "200px")
.style("background-color", "rgba(230, 242, 255, 0.8)")
.style("border-radius", "5px")
.style("visibility", "hidden")
.text("");
 */
//d3.json("../static/js/example.json", function(error, graph) {
//  if (error) throw error;

//obtain nearest neighbors

function fillColors(n) {
  var colors = ["#3c82ac", "#6badd8", "#98daff", "#c7ffff"]
  return colors[n % colors.length];
}

var option = "";

//Setting for both profile
function set_choice(chc){
    option = chc;
}

var nn_id_static = [];
var nn_id_dynamic = [];
function set_nearest_neighbor(links, flag) {
    if(flag){
        //Static
        if((nn_id_static.length > 0)){
            nn_id_static.length = 0;
        }
        for (var i = 0; i < links.length; i++) {
            nn_id_static.push(parseInt(links[i].target));
        }
    }
    else{
        //Dynamic
        if((nn_id_dynamic.length > 0)){
            nn_id_dynamic.length = 0;
        }
        for (var i = 0; i < links.length; i++) {+
            nn_id_dynamic.push(parseInt(links[i].target));
        }
    }
    //console.log(nearest_neighbor_ids);
}


if(option === "both"){
    console.log("Yes it is both");
}


function getScaleValues(graph){
val_list = []



}

function get_force_direct_graph(graph, combination, flag) {

    //add encompassing group for the zoom
    console.log("1 parameter " + graph.links);
    //set_nearest_neighbor
    set_nearest_neighbor(graph.links, flag);
    //find the nearest neighbors and add to the list

    if (flag) {
          if ($("#pred_header_static_fd").length) {
                console.log("Element already exists");
          } else {
                $("#fd_static").prepend("<span id='pred_header_static_fd' style='color:white;'>"+"Nearest Neighbors visualization for user id -- " + graph.links[0].source + "</span>");
          }
        //var svg = d3.select("#force-direct").append("svg").attr("width", width).attr("height", height);
        var svg = d3.select("#force-direct")
        .append("svg")
        .attr("viewBox", "0 0 " + width + " " + height )
        .attr("preserveAspectRatio", "xMinYMin meet");

        //.attr("viewBox", `0 0 350 400`);
        addLegend("static");
    } else {
          if ($("#pred_header_dynamic_fd").length) {
                console.log("Element already exists");
          } else {
                $("#fd_dynamic").prepend("<span id='pred_header_dynamic_fd' style='color:white;'>"+"Nearest Neighbors visualization for user id -- " + graph.links[0].source + "</span>");
          }
        //var svg = d3.select("#force-direct-ts").append("svg").attr("width", width).attr("height", height);
        var svg = d3.select("#force-direct-ts").append("svg")
        .attr("viewBox", "0 0 " + width + " " + height )
        .attr("preserveAspectRatio", "xMinYMin meet");

        //.attr("viewBox", `0 0 350 400`);
        addLegend("dynamic");
    }
    var color = d3.scaleOrdinal(d3.schemeCategory20);

    //Get the scores and check if they are greater than 0.9, if so then simply make distance else make strength.
    var linkForce = d3.forceLink()
        .id(function (link) {
            return link.user_id 
        })

        var scores = false;
        var sum_inv_score = 0.0;
    for (var i = 0; i < graph.links.length; i++) {
           if (parseFloat(graph.links[i]["score"]) >= 0.1){
            sum_inv_score = sum_inv_score + (1/parseFloat(graph.links[i]["score"]))
           }

        //if (parseFloat(graph.links[i]["score"]) < 0.1) {
            //alert("Came Here!!");
        //    scores = true;
        //    break;
        }

    //Scores to manipulate the link force
    /*
    if (scores === true) {
        console.log("heom distance based NN, with default n=20 *( inverse distance scoring)");
        //alert("Came Here too!!");
        linkForce.distance(function (link){
                return (20 + parseFloat(20 * parseFloat(link.score)));
        });
     }
     else { */
        //console.log("For strength similarity is utilized");
        linkForce.strength(function (link) {
            //alert(1.0-parseFloat(link.score))
            //return (Math.abs(1.0-parseFloat(link.score)))
            //if(parseFloat(link.score) > 1.0){
            //    return (1.0 - parseInt(link.score))
            //}

            if(parseFloat(link.score) < 0.1){
                return (1.0)
            }
            else{
                console.log(sum_inv_score);
                var inv_dist = (1/parseFloat(link.score));
                console.log(inv_dist/sum_inv_score);
                return (inv_dist/sum_inv_score);
            }
        });

     //}

        //alert("Came Here too!!");
        //linkForce.distance(function (link) {
        //    return (parseFloat(10 * link.score))
        //})

    // simulation setup with all forces
    //var linkForce = d3
    //    .forceLink()
    //    .id(function (link) { return link.user_id })
    //    .strength(function (link) { return (1.0 - parseFloat(link.score)) })

    var simulation = d3.forceSimulation()
        .force("link", linkForce)
        .force("charge", d3.forceManyBody().strength(-120))
        .force("center", d3.forceCenter(width / 2, height / 2));

    var g = svg.append("g")
        .attr("class", "everything");

    var linkElements = g.append("g")
        .attr("class", "links")
        .selectAll("line")
        .data(graph.links).enter().append("line")
        .attr("stroke", function (link) {
            return storeByDistance(link, flag);
        })
        .attr("stroke-width", function (d) {
            return (1 - parseFloat(d.score));
        })

        //A base node.
        /*
        var nodeElements = g.append("g")
        .attr("class","nodes")
        .selectAll("circle")
        .data(graph.nodes).enter().append("circle")
        .attr("r", 10)
        .attr("fill", getNodeColor)
        .on("click", function(d){getprofile(d.id);});
         */

  var nodeElements = g.append("g").selectAll("path");

    //Outer
    if (flag) {
        //True == Static get profile
        //alert(flag);
        var outer = nodeElements.data(graph.nodes).enter().append("circle")
            .attr("r", 10)
            .attr("fill", getNodeColor).on("click", function (d) {
                getprofile(combination, d);
            })
            .attr("stroke", function (d) {
                return getStrokeColor(combination, d);
            })
            .attr("stroke-width", 0.5);
    } else {
        //False == Dynamic get profile
        var outer = nodeElements.data(graph.nodes).enter().append("circle")
            .attr("r", 10)
            .attr("fill", getNodeColor_ts).on("click", function (d) {
                getprofile_ts(d);
            })
            .attr("stroke", function (d) {
                return getStrokeColor(combination, d);
            })
            .attr("stroke-width", 0.5);
    }
    //Have a outer circle and draw a lot of inner circles.
    var inner = nodeElements.data(graph.nodes).enter().append("circle")
        .attr("r", function (d) {
            return fillRadiusInnerCircle(combination, d);
        })
        .attr("fill", function (d) {
            return colorInnerCircle(combination, d);
        });

    var innerOutlier = nodeElements.data(graph.nodes).enter().append("rect")
        .attr("width", 2.5)
        .attr("height", 2)
        .attr("fill", function(d){return fillByZScore(d)});

    /* Create the text for each block */
   var lables = nodeElements.data(graph.nodes).enter().append("text")
        .text(function(d){return d.user_id})
        .style("font-size", function (d) {
            return fillRadiusInnerCircle(combination, d);})
         .attr("fill", "black");




/*
    var arcGenerator = d3.arc()
    .innerRadius(1)
    .outerRadius(7);

    var pie = d3.pie()
    .sort(function(a, b) {
        return a.name.localeCompare(b.name);})
    .value(function(d) {
     alert("Val - " + d.val);
     return parseInt(d.val) * 100;
     });


    var inner = nodeElements.data(graph.nodes).enter().append("g");

    inner.selectAll("path")
    .data(function(d) {return pie(d.pie_scales); })
    .enter().append("path")
    .attr("d", arcGenerator)
    .style("fill", "orange")
    .style("stroke", "white")
    .style("stroke-width", 0.5);


    nodeElements.data(graph.nodes).append("title")
    .text(function (d) {
        return d.score;
    });

    var lables = nodeElements.data(graph.nodes).append("text")
        .text(function (d) {
            return d.score;
        })
        .attr('dx', 6)
        .attr('dy', 3);

*/

/*
    var arcGenerator = d3.arc()
    .innerRadius(3)
    .outerRadius(7);

    var pieGenerator = d3.pie();
    var data = [10, 40, 30];
    var arcData = pieGenerator(data);
    var pieElement = nodeElements
    .selectAll('path')
    .data(arcData)
    .enter()
    .append('path')
    .attr('d',arcGenerator);
*/
/*
    nodeElements
    .selectAll("circle")
    .data(graph.nodes)
    .append("title")
    .text(function (d) {
        return d.score;
    });

    var lables = nodeElements
        .selectAll("circle")
    .data(graph.nodes)
    .append("text")
        .text(function (d) {
            return d.score;
        })
        .attr('dx', 6)
        .attr('dy', 3);
*/
    //Zoom action and functions
    function zoom_actions() {
        g.attr("transform", d3.event.transform)
    }
    var zoom_handler = d3.zoom()
        .on("zoom", zoom_actions);

    zoom_handler(svg);

    // tooltip div:
    if(flag){
    var tooltip = d3.select("body").append("div").style("opacity", 0).attr("class", "tooltip");
    }
    else{
    var tooltip = d3.select("body").append("div").style("opacity", 0).attr("class", "tooltip");
    }
        //tooltip.class("tooltip", true)
        tooltip
        .style("opacity", 0) // start invisible
        inner
        .on("mouseover", function(d){
             tooltip
            .style("opacity", 1)
            .html(showToolTip(combination, d))
            .style("left", (d3.event.pageX + "px"))
            .style("top", (d3.event.pageY + "px"));

             // show the tooltip
        })
        .on("mousemove", function (d) {
        })
        .on("mouseleave", function (d) {
            tooltip
            .style("opacity", 0)
        });

    simulation.nodes(graph.nodes).on('tick', () => {
        outer
        .attr('cx', function (node) {
            return node.x
        })
        .attr('cy', function (node) {
            return node.y
        })

        inner
        .attr('cx', function (node) {
            return node.x
        })
        .attr('cy', function (node) {
            return node.y
        })

        innerOutlier
        .attr("ex", function(node){return node.x + 11})
        .attr("ey", function(node){return node.y + 1})
        .attr("transform", function(node){
        return "translate(" + (node.x + 11) + "," + node.y + 1 + ")"
        })

        lables
        .attr('dx', function (node) {
            return node.x + 10
        })

        .attr('dy', function (node) {
            return node.y
        })


      //  inner
       // .attr('cx', function (node) {
        //    return node.x
        //})
        //.attr('cy', function (node) {
         //   return node.y
        //})

        linkElements
        .attr('x1', function (link) {
            return link.source.x
        })
        .attr('y1', function (link) {
            return link.source.y
        })
        .attr('x2', function (link) {
            return link.target.x
        })
        .attr('y2', function (link) {
            return link.target.y
        })
    });

    simulation.force("link").links(graph.links);

    //Exit
    //nodeElements.exit().remove();
    //linkElements.exit().remove();
}
//});




function plot_ts(data, flag) {

    //Get all the variable of the plot. Knowing this as more than one dimension.
    d_obj = JSON.parse(data);

    console.log("Object -- " + d_obj.pTList.length);

    for (var i = 0; i < d_obj.pTList.length; i++) {
        visualize(d_obj.pTList[i], flag);
    }
}

function visualize(data, flag) {
    if (flag === true) {
        //Time Series based
        call_ts_plot(data, query_ts, flag);
    } else {
        //Static one
        call_ts_plot(data, query, flag);
    }
}

function call_ts_plot(data, query, flag) {
    alert("Calling comparison plot for " + data.pid);
    if (data.pid === query) {
        query_flag = true;
        //This is the patient-under-study
        //Getting his time series.
        var query_data = data.data;
        var pid = data.pid
        var xaxis_label = document.getElementById('plot_vars').value
        if (flag === true) {
            console.log("Perform time series plot for dynamic data");
            var id = "#ts-vis-query";
        } else {
            console.log("Perform time series plot for static data");
            var id = "#ts-vis-statq";
        }
        console.log(JSON.stringify(query_data));
        var attributes = d3.keys(query_data[0]);
        console.log(attributes);
        for (var i = 0; i < attributes.length; i++) {
            if (attributes[i] != "time_index") {
                //Perform the plotting for the specific multivariate attribute
                plot_timeseries(query_data, attributes[i], query_flag, id, pid, xaxis_label);
            }
        }
    } else {
        var query_flag = false;
        var query_data = data.data;
        var pid = data.pid;
        var xaxis_label = document.getElementById('plot_vars').value
        if (flag === true) {
            console.log("Perform time series plot for dynamic data for nearest neighbor");
            var id = "#ts-vis-target";
        } else {
            console.log("Perform time series plot for static data for nearest neighbor");
            var id = "#ts-vis-statnn";
        }
        console.log(JSON.stringify(query_data));
        var attributes = d3.keys(query_data[0]);
        console.log(attributes);
        for (var i = 0; i < attributes.length; i++) {
            if (attributes[i] != "time_index") {
                //Perform the plotting for the specific multivariate attribute
                plot_timeseries(query_data, attributes[i], query_flag, id, pid, xaxis_label);
            }
        }
    }
}


//Plot the time series based on the attribute
//selections enabling each attribute to go separately
// This can be a selection in html itself think about it.
//Measure the value for title.

//Definition functions to set the ids and classes for both set of plots
function get_line_class(q_f) {
    return q_f === true ? "line_q" : "line_t"
}

function get_line_circle(q_f) {
    return q_f === true ? "myCircle_q" : "myCircle_t"
}

//Creating a plot for query_ts
function call_query_ts_plot(data, flag) {
    //This is the patient-under-study
    //Getting his time series.
    d_obj = JSON.parse(data);
    console.log(d_obj.pTList.length);

    var query_data = d_obj.pTList[0].data;
    var pid = d_obj.pTList[0].pid;
    console.log(JSON.stringify(query_data));
    var attributes = d3.keys(query_data[0]);
    console.log("The attributes of the series are:" + attributes);
    for (var i = 0; i < attributes.length; i++) {
        if (attributes[i] != "time_index") {
            //Perform the plotting for the specific multivariate attribute
            plot_query_ts(query_data, attributes[i], pid, flag);
        }
    }
}

//Actual query_ts plot for making point ahead predictions.
//data -- time series, attributes -- json keys of time series, pid -- patient id
function plot_query_ts(data, attributes, pid, flag) {
    console.log("The user_id is -- " + pid);
    //plot of timeseries as a d3.js
    var margin = {
        top: 10,
        right: 30,
        bottom: 30,
        left: 50
    };
    var width = 750 - margin.left - margin.right,
    height = 250 - margin.top - margin.bottom;

    var val1 = width + margin.left + margin.right
        var val2 = height + margin.top + margin.bottom

        // append the svg object to the body of the page
        // update the margin based on the title size

    console.log("Flag-- ", flag);
    if(flag){
            if ($("#pred_header_static").length){
                console.log("Element already exists");
            } else {
               // $("#exp_more").append("<a class='btn' href='dash_explore'><i class='fas fa-external-link-alt'></i></i></a>")
                $("#card_header_static").prepend("<span id='pred_header_static' style='color:white;'>"+"User under study - " + pid + "</span>");
                $("#card_header_static").append("<a class='btn' id='usr_vis_box_plot'><i class='fas fa-external-link-alt'></i></i></a>");
            }
            var svg = d3.select("#query-ts-static")
                      .append("svg")
                      //.attr("preserveAspectRatio", "xMidMid meet")
                      .attr("viewBox", "0 0 " + (val1) + " " + (val2))
                      //.attr("width", val1)
                      //.attr("height", val2)
                      .attr("id", "p_ts_plot_static")
                      .append("g")
                        .attr("transform",
                              "translate(" + margin.left + "," + margin.top + ")");
    }
    else{
             if ($("#pred_header_dyn").length) {
                console.log("Element already exists");
            } else {
                $("#card_header_dynamic").prepend("<span id='pred_header_dyn' style='color:white;'>"+"User under study - " + pid + "</span>");
            }
            var svg = d3.select("#query-ts-dynamic")
                      .append("svg")
                      //.attr("preserveAspectRatio", "xMidYMid meet")
                      .attr("viewBox", "0 0 " + val1 + " " + val2 )
                      .attr("id", "p_ts_plot_dynamic")
                      .append("g")
                      .attr("transform",
                            "translate(" + margin.left + "," + margin.top + ")");

                      //.attr("width", val1)
                      //.attr("height", val2)

    }


    // Add X axis --> it is a date format
    //var formatxAxis = d3.format('%.0f');
    var x = d3.scaleLinear()
        .domain(d3.extent(data, function (d) {
                return parseFloat(d.time_index);
            }))
        .range([0, width]);
    var xAxis = svg.append("g")
    .attr("transform", "translate(0," + height + ")")
    .call(d3.axisBottom(x));

    //Create Title
    /*
    svg.append("text")
    .attr("x", width / 2 )
    .attr("y", height / 2)
    .attr("class", "title")
    .style("text-anchor", "top")
    .text("Title of Diagram");
     */
    // text label for the x axis
    svg.append("text")
    .attr("transform",
        "translate(" + (width / 2) + " ," +
        (height + margin.top + 15) + ")")
    .style("text-anchor", "middle")
    .text("day_session_index");

    if (attributes === "xvar") {
        // Add Y axis
        var y = d3.scaleLinear()
            //.domain([0, d3.max(data, function(d) { return +d.s03; })])
            .domain([0, 1])
            .range([height, 0]);
        yAxis = svg.append("g")
        .call(d3.axisLeft(y));

		//Add the clipping and brush
            var clip = svg.append("defs").append("svg:clipPath")
                .attr("id", "clip")
                .append("svg:rect")
                .attr("width", width + 5)
                .attr("height", height + 5)
                .attr("x", 0)
                .attr("y", 0);
            // Add brushing
            var brush = d3.brushX() // Add the brush feature using the d3.brush function
                .extent([[0, 0], [width, height]]) // initialise the brush area: start at 0,0 and finishes at width,height: it means I select the whole graph area
                .on("end", updateChart) // Each time the brush selection changes, trigger the 'updateChart function

        // Add the line
        var line = svg.append("g").attr("clip-path", "url(#clip)");

        line.append("path")
        .datum(data)
        .attr("class", "query_pred_line")
        .attr("fill", "none")
        .attr("stroke", "#ff6310")
        .attr("stroke-width", 1.0)
        .attr("d", d3.line()
            .x(function (d) {
                return x(d.time_index)
            })
            .y(function (d) {
                return y(d.xvar)
            }))

        // text label for the y axis
        line.append("text")
        .attr("transform", "rotate(-90)")
        .attr("y", 0)
        .attr("x", -10)
        .attr("dy", "0.7em")
        .style("text-anchor", "end")
        .text("tinnitus distress");

		   // Add the brushing
            line
            .append("g")
            .attr("class", "brush")
            .call(brush);

        //
        var Tooltip = create_tooltip_lc();

        var max_time_index = d3.max(data, function (d) {
                return +d.time_index;
            });
        console.log("Max time index" + max_time_index);

        // Add the points
        line
        .append("g")
        .selectAll("dot")
        .data(data)
        .enter()
        .append("circle")
        .attr("class", "myCircle_pred")
        .attr("cx", function (d) {
            return x(d.time_index)
        })
        .attr("cy", function (d) {
            return y(d.xvar)
        })
        .attr("r", 3.5)
        .attr("stroke", "#69b3a2")
        .attr("stroke-width", 1)
        .attr("fill", "white")
        .on("mouseover", function (d) {
            return mouseover(d, Tooltip)
        })
        .on("mousemove", function (d) {
            return mousemove(d, Tooltip, "Actual ")
        })
        .on("mouseleave", function (d) {
            return mouseleave(d, Tooltip)
        })
        .on("click", function (d) {
            predict_stress(d, pid, line, x, xAxis,y,yAxis, data, width, height, flag);
        });

		// A function that set idleTimeOut to null
            var idleTimeout
            function idled() {
                idleTimeout = null;
            }

            // A function that update the chart for given boundaries
            function updateChart() {

                // What are the selected boundaries?
                extent = d3.event.selection
                    console.log(extent)

                    // If no selection, back to initial coordinate. Otherwise, update X axis domain
                    if (!extent) {
                        if (!idleTimeout)
                            return idleTimeout = setTimeout(idled, 350); // This allows to wait a little bit
                        x.domain([4, 8])
                    } else {
                        x.domain([x.invert(extent[0]), x.invert(extent[1])])
                        line.select(".brush").call(brush.move, null) // This remove the grey brush area as soon as the selection has been done
                    }

                    // Update axis and line position

                    xAxis.transition().duration(1000).call(d3.axisBottom(x))

                    line
                    .selectAll('.query_pred_line')
                    .transition()
                    .duration(1000)
                    .attr("d", d3.line()
                        .x(function (d) {
                            return x(d.time_index)
                        })
                        .y(function (d) {
                            return y(d.xvar)
                        }));

                    line
                    .selectAll(".myCircle_pred")
                    .attr("cx", function (d) {
                        return x(d.time_index)
                    })
                    .attr("cy", function (d) {
                        return y(d.xvar)
                    });
                    line.selectAll(".myTriangle_pred")
                        .attr("transform", function(d) {
                        return "translate(" + x(d.time_index) +
                                      "," + y(d.xvar) + ")";
                    });
            }

            // If user double click, reinitialize the chart
            svg.on("dblclick", function () {

                x.domain(d3.extent(data, function (d) {
                        return parseFloat(d.time_index);
                    }))
                .range([0, width])
                xAxis.transition().call(d3.axisBottom(x))
                line
                .selectAll('.query_pred_line')
                .transition()
                .attr("d", d3.line()
                    .x(function (d) {
                        return x(d.time_index)
                    })
                    .y(function (d) {
                        return y(d.xvar)
                    }));
                line
                .selectAll(".myCircle_pred")
                .attr("cx", function (d) {
                    return x(d.time_index)
                })
                .attr("cy", function (d) {
                    return y(d.xvar)
                });
                line.selectAll(".myTriangle_pred")
                .attr("transform", function(d) {
                    return "translate(" + x(d.time_index) +
                                      "," + y(d.xvar) + ")";
                });
            });

    }
}

//Prediction of stress ahead by some points
function predict_stress(d, query_id, line, x, xAxis,y,yAxis, entire_data, width, height, flag) {
    alert("Predicting for the time point " + d.time_index + " and for the user - " + query_id);
    //console.log(nearest_neighbor_ids);

    //if (parseFloat(d.time_index) === max_time_index) {
    //    console.log("okay");
    //}

    if(flag){
    d3.select("#query-ts-static").selectAll("#min_pred").remove();
    d3.select("#query-ts-static").selectAll("#max_pred").remove();
    d3.select("#query-ts-static").selectAll("#mean_pred").remove();
    d3.select("#query-ts-static").selectAll("#circle_pred").remove();
    d3.select("#pred_legend_static").selectAll("svg").remove();
    if(option != "both"){
        nearest_neighbor_ids = nn_id_static;
    }
    else{
        cm_array = [];
        //Sort both arrays first
        let nn_id_static_sorted = nn_id_static.sort(sortNumber);
        let nn_id_dynamic_sorted = nn_id_dynamic.sort(sortNumber);
        console.log("Static Sorted", nn_id_static_sorted);
        console.log("Dynamic Sorted", nn_id_dynamic_sorted);
        for(let i=0;i<nn_id_static_sorted.length;i++){
            if(bSearch(nn_id_dynamic_sorted, nn_id_static_sorted[i])){
                cm_array.push(nn_id_static_sorted[i]);
            }
        }
        console.log("Common NN - ", cm_array);
        common_neighbor = cm_array;
        if(common_neighbor.length >= 3){
            //Calling by common neighbors
            nearest_neighbor_ids = common_neighbor;
        }
        else{
            //Calling by normal neighbors
            nearest_neighbor_ids = nn_id_static;
        }
    }
    }
    else{
    d3.select("#query-ts-dynamic").selectAll("#min_pred").remove();
    d3.select("#query-ts-dynamic").selectAll("#max_pred").remove();
    d3.select("#query-ts-dynamic").selectAll("#mean_pred").remove();
    d3.select("#query-ts-dynamic").selectAll("#circle_pred").remove();
    d3.select("#pred_legend_dynamic").selectAll("svg").remove();

    if(option != "both"){
          nearest_neighbor_ids = nn_id_dynamic;
    }
    else{
        cm_array = [];
        //Sort both arrays first
        let nn_id_static_sorted = nn_id_static.sort(sortNumber);
        let nn_id_dynamic_sorted = nn_id_dynamic.sort(sortNumber);
        console.log("Static Sorted", nn_id_static_sorted);
        console.log("Dynamic Sorted", nn_id_dynamic_sorted);
        for(let i=0;i<nn_id_static_sorted.length;i++){
            if(bSearch(nn_id_dynamic_sorted, nn_id_static_sorted[i])){
                cm_array.push(nn_id_static_sorted[i]);
            }
        }
        console.log("Common NN - ", cm_array);
        common_neighbor = cm_array;
        if(common_neighbor.length >= 3){
            //Calling by common neighbors
            nearest_neighbor_ids = common_neighbor;
        }
        else{
            //Calling by normal neighbors
            nearest_neighbor_ids = nn_id_dynamic;
        }
    }
    }

    $.ajax({
        url: "/api/predict",
        type: "GET",
        data: {
            "user_id": query_id,
            "time_point": d.time_index,
            "ref_stress": d.xvar,
            "nearest_neighbors": nearest_neighbor_ids
        },
        cache: false,
        success: function (data, test) {
            console.log("api response " + data);
            console.log(JSON.stringify(data));
            update_graph_predictions_wrapper(data, line, x, xAxis,y,yAxis, entire_data, width,height, flag);
            //get_force_direct_graph(data);
            //plot heatmap
            //Add path to the existing time series
        },
        error: function (e) {
            $('#log').append("<span style='color:#EC7063;font-weight:bold;font-size:120%;'>" + e + "</span>").fadeOut(30000, function () {});
        }
    });

}

// Call the graph predictions for each set
function update_graph_predictions_wrapper(data, line, x, xAxis,y,yAxis, entire_data, w, h, flag){
 for(key in data){
    //Call for each key meaning the mean min and max
    console.log("Calling to make predictions for -- ", key)
    update_graph_predictions(data[key], line, x, xAxis,y,yAxis, entire_data, w, h, key);
 }

 //Add the legend then
 addLegendPredictions(flag);
}



function get_s03_last_val(query_data){
y_arr = []
for(var td=0; td< query_data.length; td++){
    y_arr.push(query_data[td]["xvar"])
}

    console.log("The y vals of the user -- ", y_arr);
    return y_arr[y_arr.length - 1]
}

function update_graph_predictions(data, line, x, xAxis,y,yAxis, entire_data, w, h, predictionKey) {
    d_obj = JSON.parse(data);
    console.log(d_obj.pTList.length);
    console.log("X-axis domain minimum " + x.domain()[0]);
    console.log("X-axis domain max " + x.domain()[1]);

    var query_data = d_obj.pTList;
    console.log(JSON.stringify(query_data));
    var attributes = d3.keys(query_data);
    console.log("The attributes of the series are:" + attributes);
    // Add the line
    //Update the axis
      // Update X axis
    xlim_obs = d3.max(entire_data, function(d){ return +d.time_index});
    xlim_pred = d3.max(query_data, function(d){ return +d.time_index});
    var last_y_val = get_s03_last_val(query_data);
    console.log("The last val of the pred ", last_y_val);

    if(xlim_obs > xlim_pred){
      //  console.log(xlim_obs);
       // var val_range = d3.extent(entire_data, function (d) {
        //        return parseInt(d.time_index);
         //   })
        console.log(x.domain());
        //x.domain(val_range).range([0, w])
    }
    else if(xlim_obs < xlim_pred){
        console.log(xlim_pred);
        x.domain([x.domain()[0], xlim_pred]).range([0, w])
    }
    xAxis.transition().duration(1000).call(d3.axisBottom(x))
    line
    .select('.query_pred_line')
    .transition()
    .attr("d", d3.line()
    .x(function (d) {
        return x(d.time_index)
    })
    .y(function (d) {
        return y(d.xvar)
    }))

    line
    .selectAll(".myCircle_pred")
    .attr("cx", function (d) {
        return x(d.time_index)
    })
    .attr("cy", function (d) {
        return y(d.xvar)
    })

    line
    .selectAll(".myTriangle_pred")
    .attr("transform", function(d) {
                    return "translate(" + x(d.time_index) +
                                      "," + y(d.xvar) + ")";
        })

    //create tooltip
    var Tooltip = create_tooltip_lc();

     // Individual points
      line.append("g").selectAll("dot")
       .data(query_data)
       .enter().append("path")
       .attr("id", "circle_pred")
       .attr("class", "myTriangle_pred")
       .attr("fill", "white")
       .attr("stroke", "#69b3a2")
       .attr("stroke-width", 1.5)
       .attr("d", d3.symbol().size([60]).type(d3.symbolTriangle))
       .attr("transform", function(d) {
                    return "translate(" + x(d.time_index) +
                                      "," + y(d.xvar) + ")";
        })
        .on("mouseover", function (d) {
        return mouseover(d, Tooltip)
    })
    .on("mousemove", function (d) {
        return mousemove(d, Tooltip, "Predicted ")
    })
    .on("mouseleave", function (d) {
        return mouseleave(d, Tooltip)
    });


    //svg.style("width", val1 + 100 + "px")
    line.append("path")
    .datum(query_data)
    .attr("id", predictionKey)
    .attr("class", "query_pred_line")
    .attr("fill", "none")
    .attr("stroke", storePrediction(predictionKey))
    .attr("stroke-width", 1.0)
    .attr("d", d3.line()
        .x(function (d) {
            return x(d.time_index)
        })
        .y(function (d) {
            return y(d.xvar)
        }))




    //create circles
    /*
    line
    .append("g")
    .selectAll("dot")
    .data(query_data)
    .enter()
    .append("circle")
    .attr("id", "circle_pred")
    .attr("class", "myCircle_pred")
    .attr("cx", function (d) {
        return x(d.time_index)
    })
    .attr("cy", function (d) {
        return y(d.s03)
    })
    .attr("r", 3.5)
    .attr("stroke", "#69b3a2")
    .attr("stroke-width", 1)
    .attr("fill", "white")
    .on("mouseover", function (d) {
        return mouseover(d, Tooltip)
    })
    .on("mousemove", function (d) {
        return mousemove(d, Tooltip, "Predicted ")
    })
    .on("mouseleave", function (d) {
        return mouseleave(d, Tooltip)
    })
    */
}

//var ts_example_json = "../static/js/ts-example.json";
/*
function plot_ts(ts_example_json){
//plot of timeseries as a d3.js
var margin = {top: 10, right: 30, bottom: 30, left: 60},
width = 300 - margin.left - margin.right,
height = 250 - margin.top - margin.bottom;

// append the svg object to the body of the page
var svg = d3.select("#ts-plot")
.append("svg")
.attr("width", width + margin.left + margin.right)
.attr("height", height + margin.top + margin.bottom)
.append("g")
.attr("transform",
"translate(" + margin.left + "," + margin.top + ")");

//Read the data
console.log(ts_example_json);
d3.json(ts_example_json,
// Now I can use this dataset:
function(data) {
console.log(data.node);
var keys = d3.keys(data.node[0]);
alert(keys);
for(val in keys){
alert(val);

}

// Add X axis --> it is a date format
//var formatxAxis = d3.format('%.0f');
var x = d3.scaleLinear()
.domain(d3.extent(data.node, function(d) { return d.date; }))
.range([ 0, width ]);
svg.append("g")
.attr("transform", "translate(0," + height + ")")
.call(d3.axisBottom(x).ticks(5));


// Add Y axis
var y = d3.scaleLinear()
.domain([0, d3.max(data.node, function(d) { return +d.rating; })])
.range([ height, 0 ]);
svg.append("g")
.call(d3.axisLeft(y));

// Add the line
svg.append("path")
.datum(data.node)
.attr("fill", "none")
.attr("stroke", "steelblue")
.attr("stroke-width", 1.5)
.attr("d", d3.line()
.x(function(d) { return x(d.date) })
.y(function(d) { return y(d.rating) })
)

});

}
 */