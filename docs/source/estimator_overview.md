<style>
.bd-article-container {
    max-width: 100em !important;
}

.bd-sidebar-secondary {
    display: none;
}

/* .table-container {
    width: 100%;
    padding: 10px;
    overflow-x: auto;
} */

#tableContainer {
    float: left;
    table-layout: fixed;
    border-collapse: collapse;
    overflow-x: auto;
}

#tableContainer th, #tableContainer td {
    border: 2px solid #888;
    text-align: center;
    word-break: break-word;
    width: 15vw;
}

</style>

# Estimator Overview

The table below gives an overview of all estimators in sktime.

<input type="text" id="searchInput" placeholder="Search the table ..." />

<div id="dropdownContainer">
<select id="filterOptions">
    <option value="all" selected>ALL</option>
    <option value="forecaster">Forecaster</option>
    <option value="transformer">Transformer</option>
    <option value="regressor">Regressor</option>
    <option value="aligner">Aligner</option>
    <option value="clusterer">Clusterer</option>
    <option value="classifier">Classifier</option>
</select>

<p>
<span id="checkboxContainer"></span>
</p>
</div>

<div class="table-container">
<!-- Table to render estimators overview -->
<table id="tableContainer"></table>
</div>

<script>

document.addEventListener("DOMContentLoaded", function () {

    // Event listener for search
    const searchInput = document.getElementById("searchInput");
    searchInput.addEventListener("keyup", function () {
        var value = this.value.toLowerCase();
        var rows = document.getElementsByTagName("tr");

        for (var i = 0; i < rows.length; i++) {
            var rowText = rows[i].textContent.toLowerCase();
            rows[i].style.display = rowText.indexOf(value) > -1 ? "" : "none";
        }
        // TODO: move this logic into filterTable
    });

    // Event listener for filter change
    const filterOptions = document.getElementById("filterOptions");
    filterOptions.addEventListener("change", function(event) {
        filterTable();
    })

    // Event listener for checkbox change
    // const checkboxContainer = document.getElementById("searchInput");
    document.addEventListener("change", function(event) {
        const filter = document.getElementById("filterOptions").value;
        const target = event.target;
        if (target.type === "checkbox") {
            visibleTagsOfTypes[filter][target.id] = target.checked;
            console.log(filter)
            console.log(visibleTagsOfTypes[filter])
            filterTable();
        };
    })

    // Initialize the table based on URL hash
    function initTableFromURL() {
        const params = new URLSearchParams(window.location.hash.slice(1));
        const filter = params.get('filter');
        const tags = params.get('tags');

        if (filter) {
            document.getElementById("filterOptions").value = filter;
        }
        if (tags) {
            visibleTagsOfTypes[filter] = JSON.parse(tags);
        }
    }

    initTableFromURL();

    filterTable();

});

let visibleTagsOfTypes = {};

//// Main logic
function filterTable() {

    const filter = document.getElementById("filterOptions").value;
    const header = ["Class Name", "Estimator Type", "Dependencies", "Maintainers"];

    // Process and render one type of estimators
    if (filter != "all") {
        const cachedData = sessionStorage.getItem("jsonData");
        if (cachedData) {
            let dynamicHeader = ["Class Name", "Dependencies", "Maintainers"];
            const data = JSON.parse(cachedData);
            const filteredData = data.filter(row => row["Estimator Type"] === filter);
            const tags = visibleTagsOfTypes[filter]
            if (tags) {
                dynamicHeader.push(Object.keys(tags).filter(key => tags[key]));
                // dynamicHeader.push(...Object.keys(tags).filter(key => tags[key]));
            } else {
                visibleTagsOfTypes[filter] = {};
                Object.keys(filteredData[0].Tags).forEach(tag => {
                    visibleTagsOfTypes[filter][tag] = false;
                });
            }
            renderTable(filteredData, dynamicHeader);
        } else {

            fetchJsonData();
        }
    // Render all estimators
    } else {
        visibleTagsOfTypes[filter] = {}; // TODO: add tags to ALL table (need or not?)
        const table = document.getElementById("tableContainer");
        const contentTableAll = sessionStorage.getItem("contentTableAll");
        if (contentTableAll) {
            table.innerHTML = contentTableAll;
        } else {

            fetchTableAll();
        }
    }

    populateCheckboxes();

    // Update URL hash
    updateURL(filter, visibleTagsOfTypes[filter]);
    function updateURL(filter, tags) {
        const params = new URLSearchParams();
        params.set('filter', filter);
        if (tags) {
            params.set('tags', JSON.stringify(tags));
        }
        window.location.hash = params.toString();
    }
}

//// Fetching
// Fetch json database of estimators
function fetchJsonData() {
    return fetch("_static/estimator_overview_db.json")
        .then(response => response.json())
        .then(data => {
        sessionStorage.setItem("jsonData", JSON.stringify(data));
        filterTable()
        })
        .catch(error => console.error("Error:", error));
}

// Fetch pre-rendered html of "all" table
function fetchTableAll() {
    return fetch('_static/table_all.html')
        .then(response => {
            if (response.ok) {
                return response.text();
            }
            throw new Error('Failed to fetch the HTML content.');
        })
        .then(html => {
            sessionStorage.setItem("contentTableAll", html);
            filterTable();
        })
        // .catch(error => {
        //     console.error('Error:', error);
        //     document.getElementById('content-area').innerHTML = '<p>Error loading content.</p>';
        // });
}

//// Rendering
// Render table
function renderTable(data, header) {
    const table = document.getElementById("tableContainer");

    table.innerHTML = "";
    console.log(header);

    // Table header
    let headerRow = "<tr>";
    header.forEach((headerItem, index) => {
        if (Array.isArray(headerItem)) {
            // Handle the case where headerItem is a list (of Tags)
            headerItem.forEach(item => {
                // i.e capability:inverse_transform => capability<br>inverse_transform
                // TODO: better way?
                headerRow += `<th>${item.replace(/:/g, '<br>')}</th>`;
            });
        } else {
            headerRow += `<th>${headerItem}</th>`;
        }
    });
    headerRow += "</tr>";
    table.innerHTML += headerRow;

    // Table rows
    data.forEach(rowData => {
        let rowContent = "<tr>";
        header.forEach(headerItem => {
            if (Array.isArray(headerItem)) {
                // Handle the case where headerItem is a list (of Tags)
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

// Populate the checkboxes
function populateCheckboxes() {

    const filter = document.getElementById("filterOptions").value;

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

function go2URL(primaryUrl, fallbackUrl, event) {
    event.preventDefault(); // Stop the link from navigating directly.
    fetch(primaryUrl)
        .then(response => {
            if (response.ok) {
                window.location.href = primaryUrl; // If primary URL is valid, go there.
            } else {
                window.location.href = fallbackUrl; // Otherwise, use the fallback URL.
            }
        })
        .catch(() => {
            window.location.href = fallbackUrl; // In case of any fetch error, use fallback.
        });
}

</script>

<!-- ```{include} estimator_overview_table.html
``` -->
