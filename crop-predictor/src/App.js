import React, { useState } from "react";
import axios from "axios";

function App() {
  const [features, setFeatures] = useState("");
  const [prediction, setPrediction] = useState("");

  const handlePredict = async () => {
    try {
      const response = await axios.post("http://127.0.0.1:5000/predict", {
        features: features.split(",").map(Number),  // Convert input to numbers
      });
      setPrediction(response.data.prediction);
    } catch (error) {
      console.error("Error:", error);
      setPrediction("Error making prediction");
    }
  };

  return (
    <div style={{ textAlign: "center", marginTop: "50px" }}>
      <h1>Crop Production Prediction</h1>
      <input
        type="text"
        placeholder="Enter comma-separated features"
        value={features}
        onChange={(e) => setFeatures(e.target.value)}
      />
      <button onClick={handlePredict}>Predict</button>
      <h3>Prediction: {prediction}</h3>
    </div>
  );
}

export default App;
