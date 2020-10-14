//Plotting Heatmap here
function plot_heatmap(data) {
    //Read the data first and map the required things
    //var svg = d3.select("#heat-map").append("svg").attr("width",width).attr("height",height);
    d_obj = JSON.parse(data);
    console.log(d_obj.plist);
    data_items = d_obj.plist;
    var x_axis = d3.set(data_items.map(function (item) {
                return item.id;
            })).values(),
    y_axis = d3.set(data_items.map(function (item) {
                return item.attr;
            })).values();
    console.log(x_axis);
    console.log(y_axis);

    // set the dimensions and margins of the graph
    var itemSize = 22,
    cellSize = itemSize - 1,
    margin = {
        top: 80,
        right: 20,
        bottom: 20,
        left: 60
    };
    var width = 300 - margin.right - margin.left,
    height = 450 - margin.top - margin.bottom;

    // append the svg object to the body of the page
    if ($("#pred_header_hm").length) {
                console.log("Element already exists");
    } else {
                $("#hm_static").prepend("<p id='#pred_header_hm' style='color:white;'>"+"Heatmap visualization over questions" + "</p>");
    }
    var svg = d3.select("#heat-map")
        .append("svg")
        .attr("viewBox", "0 0 " + width + " " + height )
        .attr("preserveAspectRatio", "xMidYMid meet")
        //.attr("width", width + margin.left + margin.right)
        //.attr("height", height + margin.top + margin.bottom)
        .append("g")
        .attr("transform",
            "translate(" + margin.left + "," + margin.top + ")");

    // Build X scales and axis:
    var x = d3.scaleBand()
        .range([0, width])
        .domain(x_axis)
        .padding(0.01);
    svg.append("g")
    .attr("transform", "translate(0," + height + ")")
    .call(d3.axisBottom(x))

    // Build X scales and axis:
    var y = d3.scaleBand()
        .range([height, 0])
        .domain(y_axis)
        .padding(0.01);
    svg.append("g")
    .call(d3.axisLeft(y));

    // The color is not normalized in strict sense, but encoded.
    var colorDomain = d3.extent(data_items, function (d) {
            return parseInt(d.val);
        });

    var colorScale = d3.scaleLinear()
        .domain(colorDomain)
        .range(["#6badd8", "#005b83"]);

    //Heatmap representation. Note: Ensure for a scale.
    var legend_element = svg.append("g").attr("class", "legend")
        .attr("transform", "translate(" + 2 + ",-70)")

        var legend = d3.legendColor()
        .orient('horizontal')
        .shapeWidth(25)
        .cells([0, 1, 2, 3, 4, 5, 6, 7])
        .scale(colorScale);

    svg.select(".legend")
    .call(legend);

    svg.selectAll()
    .data(data_items, function (d) {
        return d.id + ':' + d.attr;
    })
    .enter()
    .append("rect")
    .attr("x", function (d) {
        return x(d.id)
    })
    .attr("y", function (d) {
        return y(d.attr)
    })
    .attr("width", x.bandwidth())
    .attr("height", y.bandwidth())
    .style("fill", function (d) {
        return colorScale(d.val)
    })

    /*
    for(var i=0;i<array.length;i++){
    obj = array[i]
    for(key in obj){
    x_axis.push(key)
    values.push(obj[key])
    }
    }
     */
    //A small issue had happened while converting to string at backend.
    //y_axis = Object.keys(values[0]);

    //Create the required items now.

}