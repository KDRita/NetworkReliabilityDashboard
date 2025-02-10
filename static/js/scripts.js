document.addEventListener("DOMContentLoaded", function() {
    const form = document.getElementById("data-entry-form");
    form.addEventListener("submit", function(event) {
        event.preventDefault();
        const formData = new FormData(form);
        fetch("/api/data-entry/", {
            method: "POST",
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            console.log("Data successfully submitted:", data);
            if (data.status === "success") {
                const csvFile = formData.get('csv-file').name;
                fetch(`/api/view-csv/?csv-file=${csvFile}`)
                .then(response => response.json())
                .then(data => {
                    if (data.status === "success") {
                        displayCSVContent(data.data);
                    } else {
                        console.error("Error loading CSV content:", data.message);
                    }
                })
                .catch(error => {
                    console.error("Error loading CSV content:", error);
                });
            }
        })
        .catch(error => {
            console.error("Error submitting data:", error);
        });
    });

    function displayCSVContent(data) {
        const csvContent = document.getElementById("csv-content");
        csvContent.innerHTML = "<table class='table table-bordered'><thead><tr></tr></thead><tbody></tbody></table>";
        const thead = csvContent.querySelector("thead tr");
        const tbody = csvContent.querySelector("tbody");

        const headers = Object.keys(data[0]);
        headers.forEach(header => {
            const th = document.createElement("th");
            th.textContent = header;
            thead.appendChild(th);
        });

        data.forEach(row => {
            const tr = document.createElement("tr");
            headers.forEach(header => {
                const td = document.createElement("td");
                td.textContent = row[header];
                tr.appendChild(td);
            });
            tbody.appendChild(tr);
        });
    }

    window.buildModel = function(network) {
        fetch(`/api/build-model/?network=${network}`)
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    document.getElementById('model-accuracy').innerText = `Précision: ${data.accuracy}`;
                    document.getElementById('classification-report').innerText = data.classification_report;
                    document.getElementById('confusion-matrix').src = `/static/images/confusion_matrix.png`;
                    document.getElementById('bayesian-network-graph').src = `/static/images/${network}_model.png`;
                    document.getElementById('model-visualization').style.display = 'block';
                } else {
                    alert(data.message);
                }
            })
            .catch(error => console.error('Error:', error));
    }
    
    window.generateReport = function(networkType) {
        fetch(`/api/report/?network=${networkType}`)
        .then(response => response.json())
        .then(data => {
            const reportContent = document.getElementById("report-content");
            const inferenceData = document.getElementById("inference-data");
            if (data.status === "success") {
                reportContent.innerHTML = `<pre>${data.report}</pre>`;
                inferenceData.innerHTML = "<h4>Data d'Inference:</h4>";
                for (const [key, value] of Object.entries(data.inference_data)) {
                    inferenceData.innerHTML += `<p>${key}: ${value}</p>`;
                }
            } else {
                reportContent.innerHTML = `<p>Erreur lors de la génération du rapport : ${data.message}</p>`;
            }
        })
        .catch(error => {
            console.error("Error generating report:", error);
        });
    }
});

