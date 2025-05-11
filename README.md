# ⚡️ NeuroWatts: Neural Time Series for Power Consumption Forecasting

NeuroWatts is a deep learning-based time-series forecasting project built on the **Household Power Consumption** dataset. It leverages LSTM architecture, extensive feature exploration, and chunk-based training to deliver highly accurate electricity demand predictions across long temporal sequences.

---

## 📦 Dataset Overview

- **Dataset**: UCI Household Power Consumption
- **Time Range**: Dec 2006 – Nov 2010
- **Data Size**: ~2.07 million samples
- **Features**:
  - Global active & reactive power
  - Voltage, current, and sub-metering data
  - Timestamped at one-minute intervals

---

## 🔬 Exploratory Data Analysis (EDA)

Performed 15+ plots across:
- **Univariate** trends (power, voltage)
- **Bivariate** scatterplots (voltage vs power)
- **Multivariate** overlays and heatmaps
- **Resampling** by hour/day/week for seasonal pattern discovery
- **Rolling means**, outlier detection via boxplots, and distribution visualizations

---

## 🧪 Feature Engineering

- Created lag-based supervised learning sequences using a **60-timestep window**
- Resampled for time consistency
- Filled 7 columns with missing values using mean imputation
- Extracted time components (year, month, hour) as additional inputs

---

## 🧠 Model Architecture

Utilized a 2-layer LSTM architecture with dropout and dense layers:

```
Input → LSTM(50) → Dropout(0.2) → LSTM(50) → Dropout(0.2) → Dense(25) → Dense(1)
```


- Loss: `Mean Squared Error`
- Optimizer: `Adam`
- Epochs: Trained in **chunks of 100,000** samples due to memory constraints
- Validation split: 20%

---

## 📈 Training Performance

- **Mean Squared Error (Test)**: ~0.024
- **Mean Absolute Percentage Error (Test)**: ~6.2%
- **Training MAE**: Tracked over epochs using early stopping

### 📉 Loss & MAE Curves

<p align="center">
  <img src="https://github.com/mks2002/NeuroWatts-Neural-Time-Series-for-Power-Consumption/blob/main/LOSS.png" width="400" title="Loss Curves"/>
  <img src="https://github.com/mks2002/NeuroWatts-Neural-Time-Series-for-Power-Consumption/blob/main/MAE.png" width="400" title="MAE Curves"/>
  <img src="https://github.com/mks2002/NeuroWatts-Neural-Time-Series-for-Power-Consumption/blob/main/actual-Predicted.png" width="400" title="Actual VS Predicted Values"/>
</p>

---

## 📊 Error Analysis

### 🔹 Maximum Error

- **Absolute Error**: 3.81  
- **Percentage Error**: 67.85%  
- **Index**: 365708

### 🔹 Minimum Error

- **Absolute Error**: ~3.85e-07  
- **Percentage Error**: 0.00%  
- **Index**: 199702

### 🔸 Error Threshold Breakdown

| Threshold      | Points Exceeding |
|----------------|------------------|
| > 1%           | 349,541          |
| > 5%           | 209,754          |
| > 10%          | 96,055           |
| > 20%          | 43,685           |
| > 30%          | 25,479           |
| > 40%          | 16,351           |
| > 50%          | 11,382           |
| > 60%          | 8,198            |
| > 65%          | 6,974            |

---

## 🚀 Key Highlights

- Trained on **2M+ points** using chunk-wise strategy to fit limited memory resources
- Engineered supervised sequences with 60-step lookback
- Integrated advanced visual diagnostics before modeling
- Achieved high-precision short-term forecasts with under 7% MAPE

---

## 🛠 Tech Stack

- Python, pandas, NumPy
- TensorFlow (Keras)
- Scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebook

---

## 📁 Repository Structure

```
📁 NeuroWatts-Neural-Time-Series-for-Power-Consumption
├── household_power_consumption_lstm.ipynb
├── requirements.txt
├── actual-Predicted.png
├── LOSS.png
├── MAE.png
└── README.md
```


---

## 🧪 How to Run

1. Clone the repo:  
   `git clone https://github.com/yourusername/NeuroWatts-Neural-Time-Series-for-Power-Consumption.git`

2. Install dependencies:  
   `pip install -r requirements.txt`

3. Launch notebook:  
   `jupyter notebook household_power_consumption_lstm.ipynb`

---

## 📜 License

This project is licensed under the MIT License. See the LICENSE file for more details.

---

## ✍️ Author

**Mayuk Sarkar** – [@mayuksarkar](https://github.com/mks2002)

