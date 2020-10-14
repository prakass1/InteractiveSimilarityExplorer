function plot_ts(data, flag) {

    //Get all the variable of the plot. Knowing this as more than one dimension.
    d_obj = JSON.parse(data);

    console.log("Object -- " + d_obj.pTList.length);

    for (var i = 0; i < d_obj.pTList.length; i++) {
        visualize(d_obj.pTList[i], flag);
    }
}

function call_ts_plot(data, query, flag) {
    alert("Calling time series plot for " + data.pid);
    if (data.pid === query) {
        query_flag = true;
        //This is the patient-under-study
        //Getting his time series.
        var query_data = data.data;
        var pid = data.pid
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
                plot_timeseries(query_data, attributes[i], query_flag, id, pid);
            }
        }
    } else {
        var query_flag = false;
        var query_data = data.data;
        var pid = data.pid;
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
                plot_timeseries(query_data, attributes[i], query_flag, id, pid);
            }
        }
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

function plot_timeseries(data, attribute, query_flag, id, pid) {
    //plot of timeseries as a d3.js
    var margin = {
        top: 10,
        right: 30,
        bottom: 30,
        left: 30
    };
    var width = 450 - margin.left - margin.right,
    height = 250 - margin.top - margin.bottom;

    // append the svg object to the body of the page
    if (query_flag) {

        if ($(id).find("#pred_header").length) {
            console.log("Element already exists");
        } else {
            $(id).prepend("<h6 id='pred_header'>" + "Tinnitus distress plot of Study Patient - " + pid + "</h6>");
        }
        //$("#ts-vis-query").html("<h6>Patient Under Study Time Series</h6>");
        // update the margin based on the title size
        var svg = d3.select(id)
            .append("svg")
            .attr("viewBox", "0 0 " + width + " " + height )
            .attr("preserveAspectRatio", "xMidYMid meet")
            //.attr("width", width + margin.left + margin.right)
            //.attr("height", height + margin.top + margin.bottom)
            .append("g")
            .attr("transform",
                "translate(" + margin.left + "," + margin.top + ")");

        //Create interactive plot for query_ts
        // Add X axis --> it is a date format
        //var formatxAxis = d3.format('%.0f');
        var x = d3.scaleLinear()
            .domain(d3.extent(data, function (d) {
                    return parseFloat(d.time_index);
                }))
            .range([0, width]);
        xAxis = svg.append("g")
            .attr("transform", "translate(0," + height + ")")
            .call(d3.axisBottom(x));
        //Create Title
        /*
        svg.append("text")
        .attr("x", width / 2 )
        .attr("y", height / 2)
        .attr("class", "title")
        .style("text-anchor", "middle")
        .text("Title of Diagram");
         */

        // text label for the x axis
        svg.append("text")
        .attr("transform",
            "translate(" + (width / 2) + " ," +
            (height + margin.top + 20) + ")")
        .style("text-anchor", "middle")
        .text("day session Index");

        if (attribute === "s03") {
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
                .attr("width", width)
                .attr("height", height)
                .attr("x", 0)
                .attr("y", 0);
            // Add brushing
            var brush = d3.brushX() // Add the brush feature using the d3.brush function
                .extent([[0, 0], [width, height]]) // initialise the brush area: start at 0,0 and finishes at width,height: it means I select the whole graph area
                .on("end", updateChart) // Each time the brush selection changes, trigger the 'updateChart' function

                // Create the line variable: where both the line and the brush take place
                var line = svg.append('g')
                .attr("clip-path", "url(#clip)")
                // Add the line
                line.append("path")
                .datum(data)
                .attr("class", "line_q")
                .attr("fill", "none")
                .attr("stroke", colorByQuery_ts(query_flag))
                .attr("stroke-width", 1.5)
                .attr("d", d3.line()
                    .x(function (d) {
                        return x(d.time_index)
                    })
                    .y(function (d) {
                        return y(d.s03)
                    }));

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

                // Three function that change the tooltip when user hover / move / leave a cell
                var mouseover = function (d) {
                Tooltip
                .style("opacity", 1)
            }
            var mousemove = function (d) {
                Tooltip
                .html("Stress value: " + d.s03)
                .style("left", (d3.event.pageX) + "px")
                .style("top", (d3.event.pageY) + "px")
            }
            var mouseleave = function (d) {
                Tooltip
                .style("opacity", 0)
            }


            // Add the points
            line
            .append("g")
            .selectAll("dot")
            .data(data)
            .enter()
            .append("circle")
            .attr("class", "myCircle_q")
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
            .on("mouseover", mouseover)
            .on("mousemove", mousemove)
            .on("mouseleave", mouseleave)



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
                    .select('.line_q')
                    .transition()
                    .duration(1000)
                    .attr("d", d3.line()
                        .x(function (d) {
                            return x(d.time_index)
                        })
                        .y(function (d) {
                            return y(d.s03)
                        }))

                    line
                    .selectAll(".myCircle_q")
                    .attr("cx", function (d) {
                        return x(d.time_index)
                    })
                    .attr("cy", function (d) {
                        return y(d.s03)
                    })
            }

            // If user double click, reinitialize the chart
            svg.on("dblclick", function () {

                x.domain(d3.extent(data, function (d) {
                        return parseFloat(d.time_index);
                    }))
                .range([0, width])
                xAxis.transition().call(d3.axisBottom(x))
                line
                .select('.line_q')
                .transition()
                .attr("d", d3.line()
                    .x(function (d) {
                        return x(d.time_index)
                    })
                    .y(function (d) {
                        return y(d.s03)
                    }))
                line
                .selectAll(".myCircle_q")
                .attr("cx", function (d) {
                    return x(d.time_index)
                })
                .attr("cy", function (d) {
                    return y(d.s03)
                })

            });

        }

    } else {
        if ($(id).find("#pred_header").length) {
            console.log("Element already exists");
        } else {
            $(id).prepend("<h6 id='pred_header'>" + "Tinnitus distress plot of the targeted nearest neighbor - " + pid + "</h6>");
        }

        //$("#ts-vis-query").html("<h6>Patient Under Study Time Series</h6>");
        // update the margin based on the title size
        var svg = d3.select(id)
            .append("svg")
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom)
            .append("g")
            .attr("transform",
                "translate(" + margin.left + "," + margin.top + ")");

        //Create an interactive plot for nearest neighbor time series
        // Add X axis --> it is a date format
        //var formatxAxis = d3.format('%.0f');
        var x = d3.scaleLinear()
            .domain(d3.extent(data, function (d) {
                    return parseFloat(d.time_index);
                }))
            .range([0, width]);
        xAxisTarget = svg.append("g")
            .attr("transform", "translate(0," + height + ")")
            .call(d3.axisBottom(x));
        //Create Title
        /*
        svg.append("text")
        .attr("x", width / 2 )
        .attr("y", height / 2)
        .attr("class", "title")
        .style("text-anchor", "middle")
        .text("Title of Diagram");
         */

        // text label for the x axis
        svg.append("text")
        .attr("transform",
            "translate(" + (width / 2) + " ," +
            (height + margin.top + 20) + ")")
        .style("text-anchor", "middle")
        .text("day session index");

        if (attribute === "s03") {
            // Add Y axis
            var y = d3.scaleLinear()
                //.domain([0, d3.max(data, function(d) { return +d.s03; })])
                .domain([0, 1])
                .range([height, 0]);
            yAxisTarget = svg.append("g")
                .call(d3.axisLeft(y));
            //Add the clipping and brush
            var clip = svg.append("defs").append("svg:clipPath")
                .attr("id", "clip")
                .append("svg:rect")
                .attr("width", width)
                .attr("height", height)
                .attr("x", 0)
                .attr("y", 0);
            // Add brushing
            var brush = d3.brushX() // Add the brush feature using the d3.brush function
                .extent([[0, 0], [width, height]]) // initialise the brush area: start at 0,0 and finishes at width,height: it means I select the whole graph area
                .on("end", updateChart) // Each time the brush selection changes, trigger the 'updateChart' function

                // Create the line variable: where both the line and the brush take place
                var line = svg.append('g')
                .attr("clip-path", "url(#clip)")
                // Add the line
                line.append("path")
                .datum(data)
                .attr("class", "line_t")
                .attr("fill", "none")
                .attr("stroke", colorByQuery_ts(query_flag))
                .attr("stroke-width", 1.5)
                .attr("d", d3.line()
                    .x(function (d) {
                        return x(d.time_index)
                    })
                    .y(function (d) {
                        return y(d.s03)
                    }));

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

                // Three function that change the tooltip when user hover / move / leave a cell
                var mouseover = function (d) {
                Tooltip
                .style("opacity", 1)
            }

            var x_min = 0;
            var mousemove = function (d) {
                Tooltip
                .html("Stress value: " + d.s03)
                .style("left", (d3.event.pageX) + "px")
                .style("top", (d3.event.pageY) + "px")
            }
            var mouseleave = function (d) {
                Tooltip
                .style("opacity", 0)
            }

            // Add the points
            line
            .append("g")
            .selectAll("dot")
            .data(data)
            .enter()
            .append("circle")
            .attr("class", "myCircle_t")
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
            .on("mouseover", mouseover)
            .on("mousemove", mousemove)
            .on("mouseleave", mouseleave)

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
                        console.log("Extent min " + x.invert(extent[0]) );
                        extent_min = x.invert(extent[0]);
                        x.domain([x.invert(extent[0]), x.invert(extent[1])])
                        line.select(".brush").call(brush.move, null) // This remove the grey brush area as soon as the selection has been done
                    }

                    // Update axis and line position

                    xAxisTarget.transition().duration(1000).call(d3.axisBottom(x))

                    x_min = x.domain();

                    line
                    .select('.line_t')
                    .transition()
                    .duration(1000)
                    .attr("d", d3.line()
                        .x(function (d) {
                            return x(d.time_index)
                        })
                        .y(function (d) {
                            return y(d.s03)
                        }))

                    line
                    .selectAll(".myCircle_t")
                    .attr("cx", function (d) {
                        return x(d.time_index)
                    })
                    .attr("cy", function (d) {
                        return y(d.s03)
                    })

               updated_x_domain = x.domain();
            }
            // If user double click, reinitialize the chart
            svg.on("dblclick", function () {

                x.domain(d3.extent(data, function (d) {
                        return parseFloat(d.time_index);
                    }))
                .range([0, width])
                xAxisTarget.transition().call(d3.axisBottom(x))
                line
                .select('.line_t')
                .transition()
                .attr("d", d3.line()
                    .x(function (d) {
                        return x(d.time_index)
                    })
                    .y(function (d) {
                        return y(d.s03)
                    }))
                line
                .selectAll(".myCircle_t")
                .attr("cx", function (d) {
                    return x(d.time_index)
                })
                .attr("cy", function (d) {
                    return y(d.s03)
                })
            });

        }
    }
}