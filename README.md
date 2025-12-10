# 📈 Stock Price Prediction Model

A deep learning-based stock price prediction system using LSTM (Long Short-Term Memory) neural networks to forecast Tesla (TSLA) stock prices.

## 🎯 Overview

This project implements a sequential LSTM model to predict stock prices based on historical data. The model analyzes 60-day price patterns to forecast future closing prices, helping identify market trends and potential price movements.

## ✨ Features

- **LSTM Neural Network**: Multi-layer LSTM architecture with dropout regularization
- **Data Preprocessing**: MinMaxScaler normalization optimized for LSTM models
- **Train/Test Split**: 80/20 split to ensure proper model validation
- **Early Stopping**: Prevents overfitting with automatic training halts
- **Comprehensive Metrics**: MAE, RMSE, MAPE, and R² score evaluation
- **Visual Analysis**: Real-time plotting of predictions vs actual prices
- **Data Leakage Prevention**: Scaler fitted only on training data

## 🛠️ Technologies Used

- **Python 3.x**
- **TensorFlow/Keras**: Deep learning framework
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation
- **Scikit-learn**: Data preprocessing and metrics
- **Matplotlib & Seaborn**: Data visualization

## 📋 Requirements

Install the required dependencies:

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn
```

Or use:

```bash
pip install -r requirements.txt
```

## 📊 Dataset

The model uses `TSLA.csv` containing Tesla stock market data with the following columns:
- **Date**: Trading date
- **Open**: Opening price
- **High**: Highest price of the day
- **Low**: Lowest price of the day
- **Close**: Closing price (target variable)
- **Volume**: Number of shares traded

## 🚀 Usage

1. **Clone the repository**:
```bash
git clone https://github.com/mahmmom/Stock_Prediction_Model.git
cd Stock_Prediction_Model
```

2. **Ensure the dataset is in place**:
- The `TSLA.csv` file should be in the project root directory

3. **Run the model**:
```bash
python stock_prediction_model.py
```

## 🏗️ Model Architecture

```
Layer (type)                 Output Shape              Params
================================================================
LSTM (1st layer)             (None, 60, 64)           16,896
LSTM (2nd layer)             (None, 32)               12,416
Dense (3rd layer)            (None, 128)              4,224
Dropout (regularization)     (None, 128)              0
Dense (output)               (None, 1)                129
================================================================
Total params: 33,665
```

### Model Configuration:
- **Sequence Length**: 60 days (sliding window)
- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error (MSE)
- **Batch Size**: 32
- **Max Epochs**: 50 (with early stopping)
- **Early Stopping Patience**: 5 epochs

## 📈 Model Performance

The model outputs the following evaluation metrics:
- **MAE (Mean Absolute Error)**: Average prediction error in dollars
- **RMSE (Root Mean Squared Error)**: Penalizes larger errors
- **MAPE (Mean Absolute Percentage Error)**: Percentage error
- **R² Score**: Model fit quality (closer to 1 is better)

## 📸 Visualization

The model generates a comprehensive plot showing:
- **Blue Line**: Training data (actual prices)
- **Orange Line**: Test data (actual prices)
- **Red Line**: Model predictions

## 🔍 Key Improvements

✅ Fixed data leakage by fitting scaler only on training data  
✅ Changed to MinMaxScaler (better for LSTM than StandardScaler)  
✅ Improved train/test split ratio (80/20 instead of 95/5)  
✅ Added dropout layer to prevent overfitting  
✅ Implemented early stopping callback  
✅ Changed loss function to MSE for better regression performance  
✅ Added comprehensive evaluation metrics  

## 🎓 How It Works

1. **Data Loading**: Reads historical stock data from CSV
2. **Preprocessing**: Normalizes data using MinMaxScaler
3. **Sequence Creation**: Creates 60-day sliding windows
4. **Model Training**: Trains LSTM on 80% of data
5. **Prediction**: Forecasts prices on remaining 20%
6. **Evaluation**: Calculates performance metrics
7. **Visualization**: Displays comparison plot

## 📝 Future Improvements

- [ ] Add more stock symbols support
- [ ] Implement real-time data fetching
- [ ] Add sentiment analysis from news
- [ ] Create web interface for predictions
- [ ] Add more technical indicators (RSI, MACD, etc.)
- [ ] Implement ensemble models
- [ ] Add hyperparameter tuning

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## ⚠️ Disclaimer

This model is for educational purposes only. Stock market predictions are inherently uncertain. Do not use this model as the sole basis for investment decisions. Always consult with financial advisors before making investment choices.

## 👤 Author

**Mahmmom**
- GitHub: [@mahmmom](https://github.com/mahmmom)

---

⭐ If you found this project helpful, please give it a star!
