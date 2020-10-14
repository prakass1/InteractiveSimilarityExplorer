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
        "Patient Id:" + node.user_id + "<br/>" +
        "Gender:" + node.tschq02 + "<br/>" +
        "Loudness:" + node.tschq12 + "<br/>" +
        "family history of tinnitus:" + node["tschq04"] + "<br/>" +
        "Pattern:" + node.tschq08 + "<br/>" + //+
        "Initial Onset:" + node.tschq06 + "<br/>" + //+
        "Noise Tolerance:" + node.tschq28 + "<br/>" + "</span>" //+
        //"<b>Outcome:</b>" + node.tinnitus + "</span>"
    } else if (combination === "bg_tinnitus_history") {
        return "<span class='badge badge-secondary'>" +
        "Patient Id:" + node.user_id + "<br/>" +
        "Gender:" + node.tschq02 + "<br/>" +
        "Handedness:" + node.tschq03 + "<br/>" +
        "family history of tinnitus:" + node["tschq04"] + "<br/>" +
        "Loudness:" + parseFloat(node.tschq12) * 100 + "%" + "<br/>" +
        "Pattern:" + node.tschq08 + "<br/>" + //+
        "Initial Onset:" + node.tschq06 + "<br/>" + //+
        "Awake Time:" + parseFloat(node.tschq16) * 100 + "%" + "<br/>" + "</span>"
    } else if (combination === "related_conditions") {
        return "<span class='badge badge-secondary'>" +
        "Patient ID:" + node.user_id + "<br/>" +
        "Diagnosed hearing loss:" + node.hq03 + "<br/>" +
        "Hearing aids:" + node.hq04 + "<br/>" +
        "Noise Tolerance:" + node.tschq28 + "<br/>" + "</span>" //+
        //"<b>Outcome:</b>" + node.tinnitus + "</span>"
    } else if (combination === "modifying_influences") {
        return "<span class='badge badge-secondary'>" +
        "Patient ID:" + node.user_id + "<br/>" +
        "Effects of stress:" + node.tschq24 + "<br/>" +
        "Effects of loud noise:" + node.tschq20 + "<br/>" + "</span>" //+
        //"<b>Outcome:</b>" + node.tinnitus + "</span>"
    }
    else {
    return "<span class='badge badge-secondary'>" +
        "Patient Id:" + node.user_id + "<br/>" +
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
            "Patient Id:" + node.user_id + "<br/>" +
            "Gender:" + node.tschq02 + "<br/>" +
            "Loudness:" + node.tschq12 + "<br/>" +
            "family history of tinnitus:" + node["tschq04"] + "<br/>" +
            "Pattern:" + node.tschq08 + "<br/>" + //+
            "Initial Onset:" + node.tschq06 + "<br/>" + //+
            "Noise Tolerance:" + node.tschq28 + "<br/>" + "</span>" //+
            //"<b>Outcome:</b>" + node.tinnitus + "</span>"
        } else if (combination === "bg_tinnitus_history") {
            return "<span class='badge badge-secondary'>" +
            "Patient Id:" + node.user_id + "<br/>" +
            "Gender:" + node.tschq02 + "<br/>" +
            "Handedness:" + node.tschq03 + "<br/>" +
            "family history of tinnitus:" + node["tschq04"] + "<br/>" +
            "Loudness:" + parseFloat(node.tschq12) * 100 + "%" + "<br/>" +
            "Pattern:" + node.tschq08 + "<br/>" + //+
            "Initial Onset:" + node.tschq06 + "<br/>" + //+
            "Awake Time:" + parseFloat(node.tschq16) * 100 + "%" + "<br/>" + "</span>"
        } else if (combination === "related_conditions") {
            return "<span class='badge badge-secondary'>" +
            "Patient ID:" + node.user_id + "<br/>" +
            "Diagnosed hearing loss:" + node.hq03 + "<br/>" +
            "Hearing aids:" + node.hq04 + "<br/>" +
            "Noise Tolerance:" + node.tschq28 + "<br/>" + "</span>" //+
            //"<b>Outcome:</b>" + node.tinnitus + "</span>"
        } else if (combination === "modifying_influences") {
            return "<span class='badge badge-secondary'>" +
            "Patient ID:" + node.user_id + "<br/>" +
            "Effects of stress:" + node.tschq24 + "<br/>" +
            "Effects of loud noise:" + node.tschq20 + "<br/>" + "</span>" //+
            //"<b>Outcome:</b>" + node.tinnitus + "</span>"
        }
        else{
        return "<span class='badge badge-secondary'>" +
            "Patient Id:" + node.user_id + "<br/>" +
            "Gender:" + node.tschq02 + "<br/>" +
            "Loudness:" + node.tschq12 + "<br/>" +
            "family history of tinnitus:" + node["tschq04"] + "<br/>" +
            "Pattern:" + node.tschq08 + "<br/>" + //+
            "Initial Onset:" + node.tschq06 + "<br/>" + //+
            "Noise Tolerance:" + node.tschq28 + "<br/>" + "</span>" //+
            //"<b>Outcome:</b>" + node.tinnitus + "</span>"

        }

    } else {
        if ((combination === "all") || (combination === "overall")) {
            return "<span class='badge badge-secondary'>" +
            "Patient Id:" + node.user_id + "<br/>" +
            "Gender:" + node.tschq02 + "<br/>" +
            "Loudness:" + node.tschq12 + "<br/>" +
            "family history of tinnitus:" + node["tschq04"] + "<br/>" +
            "Pattern:" + node.tschq08 + "<br/>" + //+
            "Initial Onset:" + node.tschq06 + "<br/>" + //+
            "Noise Tolerance:" + node.tschq28 + "<br/>" + "</span>" //+
            //"<b>Outcome:</b>" + node.tinnitus + "</span>"
        } else if (combination === "bg_tinnitus_history") {
            return "<span class='badge badge-secondary'>" +
            "Patient Id:" + node.user_id + "<br/>" +
            "Gender:" + node.tschq02 + "<br/>" +
            "Handedness:" + node.tschq03 + "<br/>" +
            "family history of tinnitus:" + node["tschq04"] + "<br/>" +
            "Loudness:" + parseFloat(node.tschq12) * 100 + "%" + "<br/>" +
            "Pattern:" + node.tschq08 + "<br/>" + //+
            "Initial Onset:" + node.tschq06 + "<br/>" + //+
            "Awake Time:" + parseFloat(node.tschq16) * 100 + "%" + "<br/>" + "</span>"
        } else if (combination === "related_conditions") {
            return "<span class='badge badge-secondary'>" +
            "Patient ID:" + node.user_id + "<br/>" +
            "Diagnosed hearing loss:" + node.hq03 + "<br/>" +
            "Hearing aids:" + node.hq04 + "<br/>" +
            "Noise Tolerance:" + node.tschq28 + "<br/>" + "</span>" //+
            //"<b>Outcome:</b>" + node.tinnitus + "</span>"
        } else if (combination === "modifying_influences") {
            return "<span class='badge badge-secondary'>" +
            "Patient ID:" + node.user_id + "<br/>" +
            "Effects of stress:" + node.tschq24 + "<br/>" +
            "Effects of loud noise:" + node.tschq20 + "<br/>" + "</span>" //+
            //"<b>Outcome:</b>" + node.tinnitus + "</span>"
        }
        else{
        return "<span class='badge badge-secondary'>" +
            "Patient Id:" + node.user_id + "<br/>" +
            "Gender:" + node.tschq02 + "<br/>" +
            "Loudness:" + node.tschq12 + "<br/>" +
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

function colorInnerCircle(combination, node) {
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
    //Color palette for the rule must be explained - Blue combination
    /*
    if ((combination === "bg_tinnitus_history") ||
    (combination === "modifying_influences") ||
    (combination === "all") ||
    (combination === "overall") ||
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
    }*/
}

function fillRadiusInnerCircle(combination, node) {
    //Radius manipulation rules for loudness
    var r = 3
        if ((combination === "bg_tinnitus_history") ||
        (combination === "all") ||
        (combination === "overall") ||
        (combination === "related_conditions") ||
        (combination === "modifying_influences"))
        {
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
        //All others same radius is retained.
        return r
        }
}

