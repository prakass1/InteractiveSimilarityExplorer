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
    console.log("The patient user_id -- " + pid);
    //plot of timeseries as a d3.js
    var margin = {
        top: 10,
        right: 30,
        bottom: 30,
        left: 50
    };
    var width = 700 - margin.left - margin.right,
    height = 250 - margin.top - margin.bottom;

    var val1 = width + margin.left + margin.right
        var val2 = height + margin.top + margin.bottom

        // append the svg object to the body of the page
        // update the margin based on the title size

    console.log("Flag-- ", flag);
    if(flag){
            if ($("#pred_header_static").length) {
                console.log("Element already exists");
            } else {
                $("#card_header_static").prepend("<span id='#pred_header_static' style='color:white;'>"+"Tinnitus distress prediction over Study Patient - " + pid + "</span>");
            }
            var svg = d3.select("#query-ts-static")
                      .append("svg")
                      .attr("viewBox", "0 0 " + (val1 + 5) + " " + (val2 + 50))
                      .attr("preserveAspectRatio", "xMidYMid meet")
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
                $("#card_header_dynamic").prepend("<span id='#pred_header_dyn' style='color:white;'>"+"Tinnitus distress prediction over Study Patient - " + pid + "</span>");
            }
            var svg = d3.select("#query-ts-dynamic")
                      .append("svg")
                      .attr("viewBox", "0 0 " + val1 + " " + val2 )
                      .attr("preserveAspectRatio", "xMidYMid meet")
                      //.attr("width", val1)
                      //.attr("height", val2)
                      .attr("id", "p_ts_plot_dynamic")
                      .append("g")
                      .attr("transform",
                            "translate(" + margin.left + "," + margin.top + ")");
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
        (height + margin.top + 20) + ")")
    .style("text-anchor", "middle")
    .text("day session Index");

    if (attributes === "s03") {
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
                .attr("height", height)
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
                return y(d.s03)
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
                            return y(d.s03)
                        }));

                    line
                    .selectAll(".myCircle_pred")
                    .attr("cx", function (d) {
                        return x(d.time_index)
                    })
                    .attr("cy", function (d) {
                        return y(d.s03)
                    });
                    line.selectAll(".myTriangle_pred")
                        .attr("transform", function(d) {
                        return "translate(" + x(d.time_index) +
                                      "," + y(d.s03) + ")";
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
                        return y(d.s03)
                    }));
                line
                .selectAll(".myCircle_pred")
                .attr("cx", function (d) {
                    return x(d.time_index)
                })
                .attr("cy", function (d) {
                    return y(d.s03)
                });
                line.selectAll(".myTriangle_pred")
                .attr("transform", function(d) {
                    return "translate(" + x(d.time_index) +
                                      "," + y(d.s03) + ")";
                });
            });

    }
}

//Prediction of stress ahead by some points
function predict_stress(d, query_id, line, x, xAxis,y,yAxis, entire_data, width, height, flag) {
    alert("Predicting for the time point" + d.time_index + " and for the user - " + query_id);
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
        if(common_neighbor.length > 1){
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
        if(common_neighbor.length > 1){
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
            "ref_stress": d.s03,
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
    y_arr.push(query_data[td]["s03"])
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
        return y(d.s03)
    }))

    line
    .selectAll(".myCircle_pred")
    .attr("cx", function (d) {
        return x(d.time_index)
    })
    .attr("cy", function (d) {
        return y(d.s03)
    })

    line
    .selectAll(".myTriangle_pred")
    .attr("transform", function(d) {
                    return "translate(" + x(d.time_index) +
                                      "," + y(d.s03) + ")";
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
                                      "," + y(d.s03) + ")";
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
            return y(d.s03)
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
    .html(freeText + "Stress value: " + d.s03)
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