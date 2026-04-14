import os
import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 1. Load Data
print("Downloading and loading PJM Energy Dataset...")
path = kagglehub.dataset_download("robikscube/hourly-energy-consumption")
df = pd.read_csv(os.path.join(path, 'PJME_hourly.csv'))
df['Datetime'] = pd.to_datetime(df['Datetime'])
df = df.set_index('Datetime').sort_index()

# 2. Preprocessing
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['PJME_MW']].values)

def create_sequences(data, window_size=24):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

print("Preparing sequences...")
X, y = create_sequences(scaled_data, window_size=24)
train_split = int(len(X) * 0.9)
X_train, X_test = X[:train_split], X[train_split:]
y_train, y_test = y[:train_split], y[train_split:]

# 3. Build & Train Model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(24, 1)),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

print("Training model...")
model.fit(X_train, y_train, epochs=5, batch_size=128, validation_data=(X_test, y_test), verbose=1)

# 4. Stress Testing Function
def evaluate_scenario(multiplier, scenario_label):
    test_input = X_test[-1].copy()
    stressed_input = (test_input * multiplier).reshape(1, 24, 1)
    pred_scaled = model.predict(stressed_input, verbose=0)
    pred_mw = scaler.inverse_transform(pred_scaled)[0][0]
    status = "CRITICAL" if pred_mw > 40000 else "STABLE"
    print(f"[{scenario_label}] Multiplier: {multiplier}x | Predicted: {pred_mw:,.0f} MW | Status: {status}")

print("\n--- Running Scenario Checks ---")
evaluate_scenario(1.0, "Baseline")
evaluate_scenario(1.2, "Heatwave (+20%)")
evaluate_scenario(0.85, "Saving (-15%)")

# 5. Visualization
print("\nGenerating Final Forecast Plot...")
predictions = model.predict(X_test[-168:], verbose=0)
actuals = scaler.inverse_transform(y_test[-168:].reshape(-1, 1))
preds = scaler.inverse_transform(predictions)

plt.figure(figsize=(15, 6))
plt.plot(actuals, label='Actual Demand', color='blue')
plt.plot(preds, label='LSTM AI Forecast', linestyle='--', color='orange')
plt.title('PJM Grid Energy Forecast (Last 7 Days)')
plt.xlabel('Time Steps (Hours)')
plt.ylabel('Megawatts (MW)')
plt.legend()
plt.grid(True, alpha=0.3)

# Save the image automatically for GitHub, then display it
plt.savefig('results.png') 
plt.show()
