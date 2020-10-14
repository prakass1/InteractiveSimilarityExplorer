$(document).ready(function(){
console.log("I enter here");
//$("#loader").css("display","none");
//Show grouping when static criteria is selected.
//Hide by default

// Nested user change dynamic plotting
$("#user_ids_boxplot_select").on("change", (function(e){
e.preventDefault();
//Remove on every request and update on new response
//removeExistingVisBoxPlot();

$("#log").html("");
$("#log").show();
$("#loader").show();

//$("#loader").css("display","block");
//populate months for the selected user
populate_months();
count_hits = 0;
day_val = 0;
}));

// Populate months on the change of the year for the selected user and variable.
$("#year_box_select").on("change", (function(e){
e.preventDefault();
//Remove on every request and update on new response
//removeExistingVisBoxPlot();
$("#log").html("");
$("#log").show();
$("#loader").show();
//populate months for the selected user
populate_months();
count_hits = 0;
day_val = 0;
}));


// Populate months on the change of the year for the selected user.
$("#year_nobsbox_select").on("change", (function(e){
e.preventDefault();
//Remove on every request and update on new response
//removeExistingVisBoxPlot();
$("#log").html("");
$("#log").show();
$("#loader").show();
//populate months for the selected user
populate_nobs_month();
count_hits = 0;
day_val = 0;
}));


$("#month_nobsbox_select").on("change", (function(e){
e.preventDefault();
//Remove on every request and update on new response
//removeExistingVisBoxPlot();

$("#log").html("");
$("#log").show();
$("#loader").show();

graph_plot();
count_hits = 0;
day_val = 0;
}));

$("#attr_boxplot_select, #month_box_select").on("change", (function(e){
e.preventDefault();
//Remove on every request and update on new response
//removeExistingVisBoxPlot();

$("#log").html("");
$("#log").show();
$("#loader").show();

//Box plot dynamic change for users
var user = $('#user_ids_boxplot_select').find(":selected").text();
var user_data = {"user_id": user}
//$("#loader").css("display","block");
            //After the success call
if($("#month_box_select").has("option").length > 0){
    //Add the header and make the plot for the selected month and year
    graph_boxplot();
}
else{
    $('#log').append("<span style='color:#EC7063;font-weight:bold;font-size:120%;'>"+"There are no values for the selection"+"</span>").fadeOut(30000,function(){});
    //Empty the graph div
    $("#loader").hide();
    $("#user_boxplot_vars").empty();
    $("#user_boxplot_hrs").empty();
}

}));

/*
//Box plot attribute change
$("#attr_boxplot_select").on("change", (function(e){
e.preventDefault();
//Remove on every request and update on new response
//removeExistingVisBoxPlot();

$("#log").html("");
$("#log").show();
$("#loader").show();
var user = $('#user_ids_boxplot_select').find(":selected").text();
var user_data = {"user_id": user}
//$("#loader").css("display","block");
graph_boxplot();
}));

//Box plot Year Level change
$("#year_box_select").on("change", (function(e){
e.preventDefault();
//Remove on every request and update on new response
//removeExistingVisBoxPlot();

$("#log").html("");
$("#log").show();
$("#loader").show();
var user = $('#user_ids_boxplot_select').find(":selected").text();
var user_data = {"user_id": user}
//$("#loader").css("display","block");
graph_boxplot();
}));

//Box plot Month Level change
$("#month_box_select").on("change", (function(e){
e.preventDefault();
//Remove on every request and update on new response
//removeExistingVisBoxPlot();

$("#log").html("");
$("#log").show();
$("#loader").show();
var user = $('#user_ids_boxplot_select').find(":selected").text();
var user_data = {"user_id": user}
//$("#loader").css("display","block");
graph_boxplot();
}));
*/

//Bar dynamic update
$("#user_ids_nobsbarplot_select").on("change", (function(e){
$("#log").html("");
$("#log").show();
$("#loader").show();
e.preventDefault();
//$("#loader").css("display","block");
populate_nobs_month();
count_hits = 0;
day_val = 0;
}));

$("#nobs_plot_type").on("change", (function(e){
$("#log").html("");
$("#log").show();
$("#loader").show();
//$("#loader").css("display","block");
e.preventDefault();
graph_plot();
}));

//$("#user_barplot_s03").off("click").click(function(){
//    var myPlot = document.getElementById('user_barplot_s03');
//    myPlot.on('plotly_click', function(data){

//    var pts = '';
//    var day = "";
//    for(var i=0; i < data.points.length; i++){
//        pts = 'x = '+ data.points[i].x +'\n y = '+
//            data.points[i].y.toPrecision(4) + '\n\n';
//        day = data.points[i].x;
//    }
    //Make the call
//    get_details_summary(day);
    //alert('Closest point clicked:\n\n'+pts);
//    console.log('Closest point clicked:\n\n',pts);
//    });

//});
//End of document
});
$("body").delegate("#nobs_alert_info", "click", (function(e){
    e.preventDefault();
    $("#details_nobs_bar").hide();
    count_hits = 0;
    day_val = 0;
}));

count_hits = 0;
day_val = 0;
// Populate months on the change of the year for the selected user.
$("body").delegate("#user_barplot_s03", "click", (function(e){
    e.preventDefault();
    var myPlot = document.getElementById('user_barplot_s03');
    myPlot.on('plotly_click', function(data){
    $("#user_barplot_s03").off("click");
    var pts = '';
    var day = "";
    for(var i=0; i < data.points.length; i++){
        pts = 'x = '+ data.points[i].x +'\n y = '+
            data.points[i].y.toPrecision(4) + '\n\n';
        day = data.points[i].x;
    }
    //Make the call
    if((count_hits >= 1) && (day_val == day)){
        console.log("Not making a call to ajax()");
        //alert("Please close the output dialog and reclick over the bar");
    }
    else{
        get_details_summary(day);
        count_hits = count_hits + 1;
        day_val = day;
        console.log('Closest point clicked:\n\n',pts);
    }
    //alert('Closest point clicked:\n\n'+pts);
});

}));

//bar_plot_nobs = document.getElementById('user_barplot_s03');


//Populate months
function populate_nobs_month(){
$.ajax({
  url: "/populate_months",
  type: "GET",
  data: {
            'user_id': document.getElementById('user_ids_nobsbarplot_select').value,
            "year": document.getElementById('year_nobsbox_select').value
    },
  async:false,
  //processData:false,
  contentType:"application/json;charset=UTF-8",
  //contentType: "application/json",
  success: function(data, test)
        {
            $("#loader").hide();
            //$("#loader").css("display","none");
            //var data_obj = JSON.parse(months);
            console.log(data.months);
            // Populate the options for months, clear existing and change.
            $("#month_nobsbox_select").empty();
            for(var i=0;i<data.months.length;i++){
                $("#month_nobsbox_select").append(
                "<option>" + data.months[i] + "</option>"
                )
            }

            //After the success call
            if(data.months.length > 0){
                //Add the header and make the plot for the selected month and year
                graph_plot();
            }
            else{
                $('#log').append("<span style='color:#EC7063;font-weight:bold;font-size:120%;'>"+"There are no values for the selection"+"</span>").fadeOut(30000,function(){
            });
            }

        },
  error: function(e)
        {
                        $("#loader").hide();
                        //$("#loader").css("display","none");
                      $('#log').append("<span style='color:#EC7063;font-weight:bold;font-size:120%;'>"+e+"</span>").fadeOut(30000,function(){
            });
        }
});

}



//Populate months
function populate_months(){
$.ajax({
  url: "/populate_months",
  type: "GET",
  data: {
            'user_id': document.getElementById('user_ids_boxplot_select').value,
            "year": document.getElementById('year_box_select').value
    },
  async:false,
  //processData:false,
  contentType:"application/json;charset=UTF-8",
  //contentType: "application/json",
  success: function(data, test)
        {
            $("#loader").hide();
            //$("#loader").css("display","none");
            //var data_obj = JSON.parse(months);
            console.log(data.months);
            // Populate the options for months, clear existing and change.
            $("#month_box_select").empty();
            for(var i=0;i<data.months.length;i++){
                $("#month_box_select").append(
                "<option>" + data.months[i] + "</option>"
                )
            }

            //After the success call
            if(data.months.length > 0){
                //Add the header and make the plot for the selected month and year
                graph_boxplot();
            }
            else{
                $('#log').append("<span style='color:#EC7063;font-weight:bold;font-size:120%;'>"+"There are no values for the selection"+"</span>").fadeOut(30000,function(){
            });
            }

        },
  error: function(e)
        {
                        $("#loader").hide();
                        //$("#loader").css("display","none");
                      $('#log').append("<span style='color:#EC7063;font-weight:bold;font-size:120%;'>"+e+"</span>").fadeOut(30000,function(){
            });
        }
});

}

// graph box plot
function graph_boxplot(){
$.ajax({
  url: "/change_boxplot",
  type: "GET",
  data: {
            'user_id': document.getElementById('user_ids_boxplot_select').value,
            "col_name": document.getElementById('attr_boxplot_select').value,
            "year": document.getElementById('year_box_select').value,
            "month": document.getElementById('month_box_select').value
    },
  //processData:false,
  contentType:"application/json;charset=UTF-8",
  //contentType: "application/json",
  success: function(graph_data, test)
        {
                $("#loader").hide();
            //$("#loader").css("display","none");
            var data_obj = JSON.parse(graph_data);
            var config = {displayModeBar: false, responsive:true, editable:true};
             //, scrollZoom:true};
            console.log("api response " + data_obj[0].data)
            console.log("api response " + data_obj[1].data)
            Plotly.newPlot("user_boxplot_vars",
                        data_obj[0].data,
                        data_obj[0].layout || {}, config);

            Plotly.newPlot("user_boxplot_hrs",
                        data_obj[1].data,
                        data_obj[1].layout || {}, config);
        },
  error: function(e)
        {
                        $("#loader").hide();
                        //$("#loader").css("display","none");
                      $('#log').append("<span style='color:#EC7063;font-weight:bold;font-size:120%;'>"+e+"</span>").fadeOut(30000,function(){
            });
        }
});

}

// Call for plotting ajax
function graph_plot(){
$.ajax({
  url: "/update_obs_plot",
  type: "GET",
  data: {
            'user_id': document.getElementById('user_ids_nobsbarplot_select').value,
            "plot_type": document.getElementById('nobs_plot_type').value,
            "year": document.getElementById('year_nobsbox_select').value,
            "month": document.getElementById('month_nobsbox_select').value
    },
  //processData:false,
  contentType:"application/json;charset=UTF-8",
  //contentType: "application/json",
  success: function(graph_data, test)
        {
            $("#loader").hide();
            //$("#loader").css("display","none");
            var data_obj_bar = JSON.parse(graph_data.bar_plot_graph);
            var data_obj_bar_overview = JSON.parse(graph_data.graph_monthly_overview);
            var config = {displayModeBar: false, responsive:true, editable:true};
             //, scrollZoom:true};
            console.log("api response plot data " + graph_data.bar_plot_graph);
            console.log("api response plot overview data " + graph_data.graph_monthly_overview);

            Plotly.newPlot("user_barplot_s03",
                        data_obj_bar[0].data,
                        data_obj_bar[0].layout || {}, config);

            Plotly.newPlot("usr_bar_m_overview",
                        data_obj_bar_overview.data,
                        data_obj_bar_overview.layout || {}, config);

            //Plotly.newPlot("user_barplot_s02",
            //            data_obj[1].data,
            //            data_obj[1].layout || {}, config);
        },
  error: function(e)
        {
                        $("#loader").hide();
                        //$("#loader").css("display","none");
                      $('#log').append("<span style='color:#EC7063;font-weight:bold;font-size:120%;'>"+e+"</span>").fadeOut(30000,function(){
            });
        }
});
}

// Call for plotting ajax
function get_details_summary(day){
$.ajax({
  url: "/get_details_nobs",
  type: "GET",
  data: {
            'user_id': document.getElementById('user_ids_nobsbarplot_select').value,
            "year": document.getElementById('year_nobsbox_select').value,
            "month": document.getElementById('month_nobsbox_select').value,
            "day": day
    },
  //processData:false,
  contentType:"application/json;charset=UTF-8",
  //contentType: "application/json",
  success: function(data, test)
        {
            $("#loader").hide();
            //$("#loader").css("display","none");
            console.log(JSON.stringify(data.user_data));
            //var data_obj = JSON.parse(data.user_data);
            data_obj = data.user_data;

            //After the success call
            if(data_obj.length > 0){
                //Add the header and make the plot for the selected month and year
                html_span="<p id='day_identifier'>Day - <span style='color:#EC7063;font-weight:bold;font-size:80%;'>" + data_obj[0].day + "</span></p>"
                $("#day_identifier").remove();
                $("#details_nobs_bar").append(html_span);
                var html_str = "<table id='info_bar_patient' class='table table-sm table-bordered'>" + "<thead><tr><th>Session</th><th>Time</th><th>Tinnitus Distress</th></tr></thead><tbody>";
                for(var i=0; i<data_obj.length;i++){
                    // Create the observations
                    tr = "<tr><td>" + data_obj[i].hour_bins + "</td><td>" + data_obj[i].hour + ":" + data_obj[i].minute + "</td>";

                    if(data_obj[i].s03 > 0.75){

                     tr  = tr + "<td><span style='color:#EC7063;font-weight:bold;font-size:120%;'>" + data_obj[i].s03 + "</span></td></tr>";

                    }
                    else{
                     tr  = tr + "<td>" + data_obj[i].s03 + "</td></tr>";
                    }

                    html_str = html_str + tr;
                }
                html_str = html_str + "</tbody></table>";
                console.log("html constructed -" + html_str);
                $("#info_bar_patient").remove();
                $("#details_nobs_bar").append(html_str);
                $("#details_nobs_bar").show();
            }
            else{
                $('#log').append("<span style='color:#EC7063;font-weight:bold;font-size:120%;'>"+"There are no observations recorded for the given day"+"</span>").fadeOut(15000,function(){
            });
        }

        },
  error: function(e)
        {
                        $("#loader").hide();
                        //$("#loader").css("display","none");
                      $('#log').append("<span style='color:#EC7063;font-weight:bold;font-size:120%;'>"+e+"</span>").fadeOut(30000,function(){
            });
        }
});
}




// Remove all svg before another ajax request so that no old plots stay and the static headers
function removeExistingVisBoxPlot(){
    $("#user_boxplot_s03").empty();
    $("#user_boxplot_s02").empty();
}


