# Estimator Overview

The table below gives an overview of all estimators in sktime.

<p>
<label for="myInput"></label><input type="text" id="myInput" placeholder="Search the table ..." />
<br>
</p>

<script src="_static/js/dynamic_table.js"></script>
![](_static/estimator_overview_db.json)

<style>
.bd-article-container {
    max-width: 100em !important; 
}

.bd-sidebar-secondary {
    display: none;
}

.table-container {
    width: 100%;
    /* border: 4px solid #888;  */
    padding: 10px;
    overflow: auto;
}

#tableContainer {
    float: left; 
    border-collapse: collapse;
    table-layout: auto;
}

#tableContainer th {
    border: 2px solid #888; 
    text-align: center;
    word-wrap: break-word !important;
    max-width: 10% !important;
}

/* #tableContainer td {


} */
</style>

<div id="dropdownContainer">
<select id="filterOptions" onchange="filterTable()">
  <option value="all" selected>ALL</option>
  <option value="forecaster">Forecaster</option>
  <option value="transformer">Transformer</option>
  <option value="regressor">Regressor</option>
  <option value="aligner">Aligner</option>
  <option value="clusterer">Clusterer</option>
  <option value="classifier">Classifier</option>
</select>
<p>
<span id="checkboxContainer">
</span></p>
</div>
<div class="table-container">
<!-- Table to render estimators overview -->
<table id="tableContainer"></table>
<!-- Buttons for pages -->
</div> 

<script>
document.addEventListener("DOMContentLoaded", function () {
    var myInput = document.getElementById("myInput");
  
    myInput.addEventListener("keyup", function () {
      var value = this.value.toLowerCase();
      var rows = document.getElementsByTagName("tr");
  
      for (var i = 0; i < rows.length; i++) {
        var rowText = rows[i].textContent.toLowerCase();
        rows[i].style.display = rowText.indexOf(value) > -1 ? "" : "none";
      }
    });
  });
  

let filter = "all";
let tableAll = "";
let visibleTagsOfTypes = {};

filterTable();

function filterTable() {

    filter = document.getElementById("filterOptions").value;
    
    let header = ["Class Name", "Estimator Type", "Dependencies", "Import Path", "Maintainers"];

    const cachedData = sessionStorage.getItem("jsonData");
    
    if (cachedData) {
        const data = JSON.parse(cachedData);
        if (filter != "all") {
            let header = ["Class Name", "Dependencies", "Import Path", "Maintainers"];
            const filteredData = data.filter(row => row["Estimator Type"] === filter);
            
            if (visibleTagsOfTypes[filter]) {
                const tags = visibleTagsOfTypes[filter];
                header.push(Object.keys(tags).filter(key => tags[key]));

            } else {
                visibleTagsOfTypes[filter] = {};
                const tags = Object.keys(filteredData[0].Tags);
                tags.forEach(tag => {
                    visibleTagsOfTypes[filter][tag] = false;
                });
            }
            renderTable(filteredData, header); 
        } else {
            visibleTagsOfTypes[filter] = [];
            const table = document.getElementById("tableContainer");
            if (tableAll) {
              table.innerHTML = tableAll;
            } else {
              renderTable(data, header);
              tableAll = table.innerHTML;
            }
        }
        populateCheckboxes();
    } else {
        fetchAndRenderTable(header, filter);
    }
}

function renderTable(data, header) {
    const table = document.getElementById("tableContainer");

    
    table.innerHTML = "";
    console.log(header);
   // Table header
    let headerRow = "<tr>";
    header.forEach((headerItem, index) => {
        if (Array.isArray(headerItem)) {
            // Handle the case where headerItem is a list
            headerItem.forEach(item => {
                headerRow += `<th>${item}</th>`;
            });
        } else {
            headerRow += `<th>${headerItem}</th>`;
        }
    });
    headerRow += "</tr>";
    table.innerHTML += headerRow;
    console.log(filter)

    // Table rows
    data.forEach(rowData => {
        let rowContent = "<tr>";
        header.forEach(headerItem => {
            if (Array.isArray(headerItem)) {
                // Handle the case where headerItem is a list
                headerItem.forEach(item => {
                    rowContent += `<td>${rowData.Tags[item]}</td>`;
                });
            } else {
                rowContent += `<td>${rowData[headerItem]}</td>`;
            }
        });
        rowContent += "</tr>";
        table.innerHTML += rowContent;
    });
}

function fetchAndRenderTable(header) {
    fetch("_static/estimator_overview_db.json") 
      .then(response => response.json())
      .then(data => {
        renderTable(data, header)
        // Store data in sessionStorage
        sessionStorage.setItem("jsonData", JSON.stringify(data));
      })
      .catch(error => console.error("Error:", error));
}

// Function to populate the checkboxes
function populateCheckboxes() {
  const checkboxContainer = document.getElementById("checkboxContainer");
  checkboxContainer.innerHTML = "Check to Show Tags:"; 

  for (const [key, value] of Object.entries(visibleTagsOfTypes[filter])) {
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.id = key;
    checkbox.name = key;
    checkbox.checked = value;
    
    const label = document.createElement("label");
    label.htmlFor = key;
    label.textContent = key;

    checkboxContainer.appendChild(checkbox);
    checkboxContainer.appendChild(label);
  }
}

// Event listener for checkbox change
document.addEventListener("change", function(event) {
  const target = event.target;
  if (target.type === "checkbox") {
    visibleTagsOfTypes[filter][target.id] = target.checked;
    console.log(filter)
    console.log(visibleTagsOfTypes[filter])
    filterTable();
  }
});


 </script>

<!-- ```{include} estimator_overview_table.html
``` -->
