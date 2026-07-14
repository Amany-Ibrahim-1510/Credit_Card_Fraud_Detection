document
  .getElementById("prediction-form-csv")
  .addEventListener("submit", function (event) {
    event.preventDefault();

    const formData = new FormData(event.target);

    fetch("http://127.0.0.1:8000/predict_csv", {
      method: "POST",
      body: formData,
    })
      .then((response) => {
        if (!response.ok) {
          return response.json().then((err) => {
            throw new Error(err.detail || `HTTP ${response.status}`);
          });
        }
        return response.json();
      })
      .then((results) => {
        const resultDiv = document.getElementById("result-csv");

        // compute summary
        const total = results.length;
        const fraudCount = results.filter(
          (r) => Number(r.prediction) === 1,
        ).length;
        const nonFraud = total - fraudCount;
        const fraudProbs = results
          .filter((r) => Number(r.prediction) === 1)
          .map((r) => Number(r.probability || 0));
        const avgFraudProb = fraudProbs.length
          ? fraudProbs.reduce((a, b) => a + b, 0) / fraudProbs.length
          : 0;

        let html = `<div class="csv-summary"><p><strong>Total rows:</strong> ${total} — <strong>Fraudulent:</strong> ${fraudCount} — <strong>Not fraud:</strong> ${nonFraud}</p><p><strong>Avg fraud probability:</strong> ${avgFraudProb.toFixed(4)}</p></div>`;

        // build table with colored rows for predictions
        let table = "<table><thead><tr>";
        const headers = Object.keys(results[0]);
        headers.forEach((h) => (table += `<th>${h}</th>`));
        table += "</tr></thead><tbody>";

        results.forEach((row) => {
          const isFraud = Number(row.prediction) === 1;
          table += `<tr class="${isFraud ? "fraud-row" : "ok-row"}">`;
          headers.forEach((h) => {
            let value = row[h];
            if (typeof value === "number") {
              value = value.toFixed(4);
            }
            // ensure undefined/null show as empty
            if (value === null || value === undefined) value = "";
            table += `<td>${value}</td>`;
          });
          table += "</tr>";
        });

        table += "</tbody></table>";
        resultDiv.innerHTML = html + table;
      })
      .catch((error) => {
        const resultDiv = document.getElementById("result-csv");
        resultDiv.innerHTML = `<h2>Error</h2><p>${error.toString()}</p>`;
        console.error("Error:", error);
      });
  });

document
  .getElementById("prediction-form")
  .addEventListener("submit", function (event) {
    event.preventDefault();

    const formData = new FormData(event.target);
    const data = {};
    formData.forEach((value, key) => {
      data[key] = parseFloat(value);
    });

    fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    })
      .then((response) => {
        if (!response.ok) {
          return response.json().then((err) => {
            throw new Error(err.detail || `HTTP ${response.status}`);
          });
        }
        return response.json();
      })
      .then((result) => {
        const resultDiv = document.getElementById("result");
        if (result[0].prediction === 1) {
          resultDiv.innerHTML = `<h2>Prediction: <span style="color:red;">Fraudulent</span></h2>
                                 <p>Probability: ${result[0].probability.toFixed(4)}</p>`;
        } else {
          resultDiv.innerHTML = `<h2>Prediction: <span style="color:green;">Not Fraudulent</span></h2>
                                 <p>Probability: ${result[0].probability.toFixed(4)}</p>`;
        }
      })
      .catch((error) => {
        const resultDiv = document.getElementById("result");
        resultDiv.innerHTML = `<h2>Error</h2><p>${error.toString()}</p>`;
        console.error("Error:", error);
      });
  });
