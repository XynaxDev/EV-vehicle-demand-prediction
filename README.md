<div align="center">
  <img style="height:150px;" src="assets/car.png" alt="EV Forecast Dashboard">
  <h1>⚡ EV Charging Demand Prediction</h1>
  <p><em>A Data-Driven System to Forecast Electric Vehicle (EV) Charging Demand Across Washington State</em></p>
  
  <p>
    <img src="https://img.shields.io/badge/Status-Completed-gree?style=flat&logo=github" alt="Status">
    <img src="https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python" alt="Python">
    <img src="https://img.shields.io/badge/Streamlit-grey?style=flat&logo=streamlit" alt="Streamlit">
    <img src="https://img.shields.io/badge/License-MIT-green?style=flat" alt="License">
    <img src="https://img.shields.io/badge/AICTE-Skills4Future-blue?style=flat" alt="AICTE">
  </p>
</div>

## 📌 Project Overview

**AICTE Shell-Edunet Skills4Future Internship Project**

Electric Vehicles (EVs) are revolutionizing transportation, but efficient charging infrastructure is essential for sustainable adoption. This project leverages historical EV registration data to build a predictive model for forecasting adoption trends across Washington State counties.

![Dashboard Preview](./assets/dashboard.png)

## ✨ Key Features

- **County-Level Forecasting**: Predict EV adoption for any Washington State county
- **Interactive Dashboard**: Beautiful Streamlit interface with dark theme
- **3-Year Projections**: Visualize growth trends with historical context
- **Multi-County Comparison**: Analyze regional adoption patterns
- **Machine Learning Model**: RandomForest-based forecasting engine

## 🛠️ Tech Stack

| Component           | Technology                          |
|---------------------|-------------------------------------|
| Core Language       | Python 3.10                         |
| Data Processing     | pandas, numpy                       |
| Visualization       | matplotlib, Plotly                  |
| ML Framework        | scikit-learn (RandomForestRegressor)|
| Web Framework       | Streamlit                           |
| Deployment          | Render (via Procfile)               |

## 📂 Project Structure
```
EV-vehicle-demand-prediction/
├── assets/
│ ├── car.png
│ └── ev-car-factory.jpg
├── data/
│ ├── EV_Population_By_County.csv
│ └── preprocessed_ev_data.csv
├── notebook/
│ └── EV_DemandPrediction.ipynb
├── app.py
├── forecasting_ev_model.pkl
├── requirements.txt
├── runtime.txt
├── Procfile
├── LICENSE
└── README.md
```

## 🚀 Deployment Status

[![Render Status](https://img.shields.io/badge/Render-Deployed-46F2B5?logo=render&logoColor=white)](https://ev-demand-forecast.onrender.com)

<!-- [![Render Deployment Status](https://api.render.com/deploy/srv-d26cvvffte5s73enuqs0?type=badge)](https://ev-demand-forecast.onrender.com) -->

Deployed live on Render: [https://ev-demand-forecast.onrender.com](https://ev-demand-forecast.onrender.com)

## 💻 Local Setup
Follow these instructions to set up the project locally.

```bash
git clone https://github.com/XynaxDev/EV-vehicle-demand-prediction.git
cd EV-vehicle-demand-prediction
pip install -r requirements.txt
streamlit run app.py
```

## 📄 License

This project is licensed under the [MIT License](LICENSE).

## 🙏 Acknowledgements

- AICTE & Shell Edunet Skills4Future Internship Program
- Inspired by best practices from real-world EV infrastructure projects.

<br>
<br>
<div align="center"> Made with 💌 and Streamlit by Akash | © 2025 AICTE Internship Project </div>