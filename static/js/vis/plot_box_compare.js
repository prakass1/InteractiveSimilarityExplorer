function plot_comp_boxplots(graph_data, query_id, nn_id){
//Create the plots by utilizing plotly.js
    //$("#loader").css("display","none");
    var data_obj = JSON.parse(graph_data);
    var config = {displaylogo:false, responsive:true, editable:true, displayModeBar: false};
     //, scrollZoom:true};
    console.log("api response " + data_obj[0].data)

    Plotly.newPlot(query_id,
                data_obj[0].data,
                data_obj[0].layout || {}, config);

    //resize
    window.addEventListener('resize', function() { Plotly.Plots.resize(query_id); });

    Plotly.newPlot(nn_id,
                data_obj[1].data,
                data_obj[1].layout || {}, config);

    //resize
    window.addEventListener('resize', function() { Plotly.Plots.resize(nn_id); });

}