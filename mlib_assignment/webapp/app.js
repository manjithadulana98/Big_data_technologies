async function predictGenre() {
    const lyrics = document.getElementById("lyricsInput").value;
  
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ lyrics })
    });
  
    const data = await response.json();
    renderChart(data);
  }
  
  let genreChart = null;
  
  function renderChart(prediction) {
    console.log("Got prediction:", prediction);
  
    const safeProbs = prediction.probabilities.map(p => p > 0 ? p : 0.0001);
  
    const canvas = document.getElementById('genreChart');
    const ctx = canvas.getContext('2d');
  
    if (genreChart) {
      genreChart.destroy();
    }
  
    genreChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: prediction.labels,
        datasets: [{
          label: 'Prediction Probabilities',
          data: safeProbs,
          backgroundColor: 'rgba(0, 123, 255, 0.6)',
          borderColor: 'rgba(0, 123, 255, 1)',
          borderWidth: 1
        }]
      },
      options: {
        responsive: true,
        scales: {
          y: {
            beginAtZero: true,
            ticks: {
              stepSize: 0.1
            }
          }
        }
      }
    });
  
    console.log("Chart rendered!");
  }
  