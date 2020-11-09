$(document).ready(function(){
console.log("I enter here");

var legend_inner_color_scale = {
"range_scale": ["tschq12<=25%", "25% < tschq12 <= 50%", "50% < tschq12 <= 75%" , "tschq12 > 75%"],
"color_scale": ["#3c82ac", "#6badd8", "#98daff", "#c7ffff"],
"title": "Coloring range - Inner Circle"
};

var legend_zscores = {
"range_scale": [ -2.0, 0, 2.0 ],
"color_scale": ["#274B9F", "#fafcff", "#E50000"],
"title": "Color Range - z score",
"span_help": '<p>z-score > 0 denote tendency for outlierness </p>'
};


//Show grouping when static criteria is selected.
//Hide by default
$("#loader").hide();

// With JQuery
$('#kval').slider({
	formatter: function(value) {
		return 'Neighbors (k): ' + value;
	}
});

//Hide by default
$("#nn_slider").hide();

$("#dash-recom-form").on("submit", (function(e){
e.preventDefault();
$("#loader").show();
// disable button
$("#recom-sub-id").attr("disabled", true);

//Remove on every request and update on new response
removeExistingVis();
//var query_id = $("#patient-id").val();
//var selection_grp = $("#sim-sel-grp").val();
//var static_checked = $("#static-checkbox").val();
//var dynamic_checked = $("#dynamic-checkbox").val();

var formData = $('#dash-recom-form').serialize();
console.log('Posting the following: ', formData);

if(formData.includes("dynamic-checkbox") && formData.includes("static-checkbox")){
    //Load static visualization template for rendering
    $("#static_template").load("/app/static_vis");
    $("#dynamic_template").load("/app/dynamic_vis");
}

else if(formData.includes("static-checkbox")){
    //Load static visualization template for rendering
    if($("#dynamic_template").length > 0){
        $("#dynamic_template").empty();
    }
    $("#static_template").load("/app/static_vis");
}
else if(formData.includes("dynamic-checkbox")){
    //Load static visualization template for rendering
    if($("#static_template").length > 0){
        $("#static_template").empty();
    }
    $("#dynamic_template").load("/app/dynamic_vis");
}


$("#log").html("");
$("#log").show();

$.ajax({
  url: "/api/recommendations",
  type: "POST",
  data: (formData),
  cache:false,
  //processData:false,
  //contentType:"application/json",
  success: function(data, test)
        {
            $("#recom-sub-id").attr("disabled", false);
            $("#loader").hide();
            if("message" in data){
            $('#log').append("<span style='color:#EC7063;font-weight:bold;font-size:120%;'>"+JSON.stringify(data.message)+"</span>").fadeOut(15000,function(){
            });
            }
            console.log("api response " + data);
            //Static force-direct graphs
            //alert(JSON.stringify(data.static));

            patient_id = "Patient-".concat(document.getElementById('patient-id').value);

            //Plot the node-edge graph with weighting through force-direction
            if(("static" in data) && !("dynamic" in data)){
                sel_year = document.getElementById('sel_usr_year_box_select').value;
                set_choice("static");
                console.log("Calling Static visualization functions");
                get_force_direct_graph(data.static, data.combination, true);
                //Plot the query time series
                call_query_ts_plot(data.query_ts, true);
                populate_user_months(patient_id, sel_year);
                //Inner Circle
                legend_coloring(legend_inner_color_scale,true, "#innerCircle-data-info", "#zscore-data-info");
                //Z-score
                legend_coloring(legend_zscores, false, "#innerCircle-data-info", "#zscore-data-info");
                $("#nn_slider").show();
                var info_button = '<span class="distance_table_info badge badge-success">Distance Info</span>'
                //var color_info = '<span style="margin-left:2px;" class="vis_info badge badge-info">Color Info</span>'

                $("#click-table-info").empty().append(info_button);
                //$("#click-vis-info").empty().append(color_info);

                $("#table-data-info").empty().append(data.static["distance_table"]);

            }
            else if(!("static" in data) && ("dynamic" in data)){
                set_choice("dynamic");
                console.log("Calling Static dynamic functions");
                get_force_direct_graph(data.dynamic, "", false);
                //Plot the query time series
                call_query_ts_plot(data.query_ts, false);
                $("#nn_slider").show();
                //Inner Circle
                legend_coloring(legend_inner_color_scale, true, "#dyn-innerCircle-data-info", "#dyn-zscore-data-info");
                //Z-score
                legend_coloring(legend_zscores,false, "#dyn-innerCircle-data-info", "#dyn-zscore-data-info");
                //populate_user_months();
                var info_button = '<span class="distance_table1_info badge badge-success">Distance Info</span>'
                //var color_info = '<span style="margin-left:2px;" class="vis1_info badge badge-info">Color Info</span>'

                $("#click-table1-info").empty().append(info_button);
                //$("#click-vis1-info").empty().append(color_info);

                $("#dyn_table-data-info").empty().append(data.dynamic["distance_table"]);
            }
            else if(("static" in data) && ("dynamic" in data)){
                sel_year = document.getElementById('sel_usr_year_box_select').value;
                set_choice("both");
                console.log("Calling Static visualization functions first");
                get_force_direct_graph(data.static, data.combination, true);
                //Plot the query time series
                call_query_ts_plot(data.query_ts, true);
                populate_user_months(patient_id, sel_year);
                console.log("Calling Static dynamic functions second");
                get_force_direct_graph(data.dynamic, "", false);
                //Plot the query time series
                call_query_ts_plot(data.query_ts, false);
                console.log("Completed the visualizations");
                //Inner Circle
                legend_coloring(legend_inner_color_scale,true, "#innerCircle-data-info", "#zscore-data-info");
                legend_coloring(legend_inner_color_scale,true, "#dyn-innerCircle-data-info", "#dyn-zscore-data-info");
                //Z-score
                 legend_coloring(legend_zscores,false, "#innerCircle-data-info", "#zscore-data-info");
                legend_coloring(legend_zscores,false, "#dyn-innerCircle-data-info", "#dyn-zscore-data-info");
                $("#nn_slider").show();
                //populate_user_months();
                var info_button = '<span class="distance_table_info badge badge-success">Distance Info</span>'
                //var color_info = '<span  style="margin-left:2px;" class="vis_info badge badge-info">Color Info</span>'

                $("#click-table-info").empty().append(info_button);
                //$("#click-vis-info").empty().append(color_info);

                //$(".table-data-info").empty().append(data.static["distance_table"]);

                var info1_button = '<span class="distance_table1_info badge badge-success">Distance Info</span>'
                //var color1_info = '<span  style="margin-left:2px;" class="vis1_info badge badge-info">Color Info</span>'

                $("#click-table1-info").empty().append(info1_button);
                //$("#click-vis1-info").empty().append(color1_info);
                $("#table-data-info").empty().append(data.static["distance_table"]);
                $("#dyn_table-data-info").empty().append(data.dynamic["distance_table"]);
            }


            //Time series force-direct graphs
            //get_force_direct_graph(data.dynamic,false);
        },
  error: function(e)
        {
                      $("#recom-sub-id").attr("disabled", false);
                      $("#loader").hide();
                      $("#nn_slider").hide();
                      $('#log').append("<span style='color:#EC7063;font-weight:bold;font-size:120%;'>"+e+"</span>").fadeOut(30000,function(){
            });
        }
})
}));



//On change of selection hide the K change. (So that the submit can occur)
$("#static-checkbox, #dynamic-checkbox, #sim-sel-grp").change(function(){
    $("#nn_slider").hide();
});


//On change of the slider typical a mouse release event do something
$("#kvalSlider").mouseup(function(){

//var static_check = obtain_check_val("#static-checkbox");
//var dyn_check = obtain_check_val("#dynamic-checkbox");

var static_check = $("#static-checkbox").is(':checked');
var dyn_check = $("#dynamic-checkbox").is(':checked');

console.log("Static_check " + static_check);
console.log("Dyn_check " + dyn_check);

var combination = $("#sim-sel-grp").val();

var user_id = document.getElementById('patient-id').value
//var simulate_val = document.getElementById('simulate').value

var k_val = $("#kval").val();

post_data = {"user_id":user_id,
         "static_sim":static_check,
         "dyn_sim":dyn_check,
         "combination":combination,
         "k_val":k_val}
         //"simulate": simulate_val}

console.log(JSON.stringify(post_data));

//fire replot
$("#loader").show();
k_replot(post_data);


});

//End of document
});

$("body").delegate("#usr_vis_box_plot","click", function(){
$("#user_highlight_modal").modal("show");
});

$("body").delegate("#session_plot_info","click", function(){
$("#compare_plot_info").modal("show");
});

$("body").delegate(".distance_table_info","click", function(){
$(".distance_info").modal("show");
});

/*$("body").delegate(".vis_info","click", function(){
$(".visualization_info").modal("show");
});*/

$("body").delegate(".distance_table1_info","click", function(){
$(".dyn-distance_info").modal("show");
});

/*$("body").delegate(".vis1_info","click", function(){
$(".dyn-visualization_info").modal("show");
});*/



function obtain_check_val(id_check_val){
    $(id_check_val).change(function() {

       return $(this).is(':checked');
    });
}


// Populate months on the change of the year for the selected user.
$("body").delegate("#sel_usr_year_box_select", "change", (function(e){
e.preventDefault();
//Remove on every request and update on new response
//removeExistingVisBoxPlot();
$("#log").html("");
//$("#loader").show();
//populate months for the selected user
patient_id = "Patient-".concat(document.getElementById('patient-id').value);
sel_year = document.getElementById('sel_usr_year_box_select').value;
populate_user_months(patient_id, sel_year)
}));


$("body").delegate("#sel_usr_box_m_select", "change", (function(e){
e.preventDefault();
//Remove on every request and update on new response
//removeExistingVisBoxPlot();
$("#log").html("");
//$("#loader").show();
//populate months for the selected user
patient_id = "Patient-".concat(document.getElementById('patient-id').value);
sel_year = document.getElementById('sel_usr_year_box_select').value;
month = document.getElementById('sel_usr_box_m_select').value;
graph_user_boxplot(patient_id,"s03", sel_year, month);
}));

//Radio button as yesnoswitch
$('#radioBtn a').on('click', function(){
    var sel = $(this).data('title');
    var tog = $(this).data('toggle');
    $('#'+tog).prop('value', sel);

    $('a[data-toggle="'+tog+'"]').not('[data-title="'+sel+'"]').removeClass('active').addClass('notActive');
    $('a[data-toggle="'+tog+'"][data-title="'+sel+'"]').removeClass('notActive').addClass('active');
})


function removeSubVis(){
    d3.select("#static-legend").selectAll("svg").remove();
    d3.select("#force-direct").selectAll("svg").remove();
    d3.select("#dynamic-legend").selectAll("svg").remove();
    d3.select("#force-direct-ts").selectAll("svg").remove();
    d3.select("#heat-map").selectAll("svg").remove();
    d3.select("#ts-vis-statq").selectAll("svg").remove();
    d3.select("#ts-vis-statq").selectAll("h6").remove();
    d3.select("#ts-vis-statnn").selectAll("svg").remove();
    d3.select("#ts-vis-statnn").selectAll("h6").remove();
    d3.select("#ts-vis-query").selectAll("svg").remove();
    d3.select("#ts-vis-query").selectAll("h6").remove();
    d3.select("#ts-vis-target").selectAll("svg").remove();
    d3.select("#ts-vis-target").selectAll("h6").remove();
    d3.select("#fd_static").selectAll("p").remove();
    d3.select("#fd_dynamic").selectAll("p").remove();
    d3.select("#hm_static").selectAll("p").remove();
}

// Remove all svg before another ajax request so that no old plots stay and the static headers
function removeExistingVis(){
    d3.select("#static-legend").selectAll("svg").remove();
    d3.select("#force-direct").selectAll("svg").remove();
    d3.select("#dynamic-legend").selectAll("svg").remove();
    d3.select("#force-direct-ts").selectAll("svg").remove();
    d3.select("#heat-map").selectAll("svg").remove();
    d3.select("#ts-vis-statq").selectAll("svg").remove();
    d3.select("#ts-vis-statq").selectAll("h6").remove();
    d3.select("#ts-vis-statnn").selectAll("svg").remove();
    d3.select("#ts-vis-statnn").selectAll("h6").remove();
    d3.select("#ts-vis-query").selectAll("svg").remove();
    d3.select("#ts-vis-query").selectAll("h6").remove();
    d3.select("#ts-vis-target").selectAll("svg").remove();
    d3.select("#ts-vis-target").selectAll("h6").remove();
    d3.select("#query-ts-static").selectAll("svg").remove();
    d3.select("#query-ts-dynamic").selectAll("svg").remove();
    d3.select("#query_plot_static").selectAll("p").remove();
    d3.select("#query_plot_dynamic").selectAll("p").remove();
    d3.select("#fd_static").selectAll("p").remove();
    d3.select("#fd_dynamic").selectAll("p").remove();
    d3.select("#hm_static").selectAll("p").remove();
    d3.select(".visualization-data-info").selectAll("svg").remove();
}



$(function (){
$("#static-checkbox").click(function () {
            if ($(this).is(":checked")) {
                $("#sim-sel-grp").removeAttr("disabled");
            }
            else{
                $("#sim-sel-grp").attr("disabled", "disabled");
            }
        });
});


//Populate months
function populate_user_months(patient_id, sel_year){
$.ajax({
  url: "/populate_months",
  type: "GET",
  data: {
            'user_id': patient_id,
            "year": sel_year
    },
  async:false,
  //processData:false,
  contentType:"application/json;charset=UTF-8",
  //contentType: "application/json",
  success: function(data, test)
        {
            //$("#loader").hide();
            //$("#loader").css("display","none");
            //var data_obj = JSON.parse(months);
            console.log(data.months);
            // Populate the options for months, clear existing and change.
            $("#sel_usr_box_m_select").empty();
            for(var i=0;i<data.months.length;i++){
                $("#sel_usr_box_m_select").append(
                "<option>" + data.months[i] + "</option>"
                )
            }

            //After the success call
            if(data.months.length > 0){
                //Add the header and make the plot for the selected month and year
                graph_user_boxplot(patient_id, "s03", sel_year, data.months[0]);
            }
            else{
                $('#log').append("<span style='color:#EC7063;font-weight:bold;font-size:120%;'>"+"There are no values for the selection"+"</span>").fadeOut(30000,function(){
            });
            }

        },
  error: function(e)
        {
                        //$("#loader").hide();
                        //$("#loader").css("display","none");
                      $('#log').append("<span style='color:#EC7063;font-weight:bold;font-size:120%;'>"+e+"</span>").fadeOut(30000,function(){
            });
        }
});

}




// graph box plot
function graph_user_boxplot(user_id, col_name, year, month){
$.ajax({
  url: "/change_boxplot",
  type: "GET",
  data: {
            'user_id': user_id,
            "col_name": col_name,
            "year": year,
            "month": month
    },
  //processData:false,
  contentType:"application/json;charset=UTF-8",
  //contentType: "application/json",
  success: function(graph_data, test)
        {
               // $("#loader").hide();
            //$("#loader").css("display","none");
            var data_obj = JSON.parse(graph_data);
            var config = {displaylogo:false, responsive:true, editable:true};
             //, scrollZoom:true};
            console.log("api response " + data_obj[0].data)
            console.log("api response " + data_obj[1].data)
            Plotly.newPlot("sel_usr_boxplot",
                        data_obj[0].data,
                        data_obj[0].layout || {}, config);
        },
  error: function(e)
        {
                        //$("#loader").hide();
                        //$("#loader").css("display","none");
                      $('#log').append("<span style='color:#EC7063;font-weight:bold;font-size:120%;'>"+e+"</span>").fadeOut(30000,function(){
            });
        }
});

}

//Change k_replot
function k_replot(post_data){
$.ajax({
  url: "/api/replot",
  type: "GET",
  //data: {"user_id":user_id,
  //       "static_sim":static_check,
  //       "dyn_sim":dyn_check,
  //       "combination":combination,
  //       "k_val":k_val},
  data: post_data,
  contentType:"application/json;charset=UTF-8",
  success: function(data, test)
        {
            $("#recom-sub-id").attr("disabled", false);
            $("#loader").hide();
            $("#log").empty();
            if("message" in data){
            $('#log').append("<span style='color:#EC7063;font-weight:bold;font-size:120%;'>"+JSON.stringify(data.message)+"</span>").fadeOut(15000,function(){
            });
            $("#log").show();
            }
            console.log("api response " + data);
            //Static force-direct graphs
            //alert(JSON.stringify(data.static));

            patient_id = "Patient-".concat(document.getElementById('patient-id').value);


            //Plot the node-edge graph with weighting through force-direction
            if(("static" in data) && !("dynamic" in data)){
                sel_year = document.getElementById('sel_usr_year_box_select').value;
                set_choice("static");
                console.log("Calling Static visualization functions");
                removeSubVis();
                get_force_direct_graph(data.static, data.combination, true);
                //Plot the query time series
                //call_query_ts_plot(data.query_ts, true);
                //populate_user_months(patient_id, sel_year);
                //$("#kvalSlider").show();
                $("#table-data-info").empty().append(data.static["distance_table"]);
            }
            else if(!("static" in data) && ("dynamic" in data)){

                set_choice("dynamic");
                console.log("Calling Static dynamic functions");
                removeSubVis();
                get_force_direct_graph(data.dynamic, "", false);
                //Plot the query time series
                //call_query_ts_plot(data.query_ts, false);
                //$("#kvalSlider").show();
                //populate_user_months();
                $("#dyn_table-data-info").empty().append(data.dynamic["distance_table"]);
            }
            else if(("static" in data) && ("dynamic" in data)){
                sel_year = document.getElementById('sel_usr_year_box_select').value;
                set_choice("both");
                console.log("Calling Static visualization functions first");
                removeSubVis();
                get_force_direct_graph(data.static, data.combination, true);
                //Plot the query time series
                //call_query_ts_plot(data.query_ts, true);
                populate_user_months(patient_id, sel_year);
                console.log("Calling Static dynamic functions second");
                get_force_direct_graph(data.dynamic, "", false);
                //Plot the query time series
                //call_query_ts_plot(data.query_ts, false);
                console.log("Completed the visualizations");
                //$("#kvalSlider").show();
                $("#table-data-info").empty().append(data.static["distance_table"]);
                $("#dyn_table-data-info").empty().append(data.dynamic["distance_table"]);
            }


            //Time series force-direct graphs
            //get_force_direct_graph(data.dynamic,false);
        },
  error: function(e)
        {
                      $("#recom-sub-id").attr("disabled", false);
                      $("#loader").hide();
                     // $("#nn_slider").hide();
                      $('#log').append("<span style='color:#EC7063;font-weight:bold;font-size:120%;'>"+e+"</span>").fadeOut(30000,function(){
            });
        }
});

}

