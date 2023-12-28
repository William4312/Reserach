import pandas as pd
import numpy as np
import pywt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras import layers
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt 

df = pd.read_csv('s00.csv')
eeg_signal = df.values

def wavelet_packet_decomposition(signal, wavelet='db1', level=4):
    coeffs = pywt.wavedec(signal, wavelet, level=level, mode='symmetric')
    return coeffs

wavelet_level = 4
wavelet_coeffs = wavelet_packet_decomposition(eeg_signal, 'db1', wavelet_level)
max_length = max(arr.size for arr in wavelet_coeffs)
padded_coeffs = [np.pad(arr, (0, max_length - arr.size), 'constant', constant_values=0).flatten() for arr in wavelet_coeffs]
flattened_coeffs = np.concatenate([coeff.flatten() for coeff in wavelet_coeffs])

def plot_wavelet_packets(coeffs, title='Wavelet Packet Decomposition'):
    fig, axs = plt.subplots(len(coeffs) + 1, 1, figsize=(10, 2 * len(coeffs) + 2))
    fig.suptitle(title, y=1.02, fontsize=16)

    axs[0].plot(np.arange(len(coeffs[0])), coeffs[0], 'b-', label='Original Signal')
    axs[0].legend()

    for i in range(1, len(coeffs)):
        axs[i].plot(np.arange(len(coeffs[i])), coeffs[i], 'g-', label=f'Level {i}')
        axs[i].legend()

    plt.tight_layout()
    plt.show()

features = np.concatenate(flattened_coeffs)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(flattened_coeffs.reshape(-1, 1)).flatten()

num_time_steps = 128
num_features = scaled_features.shape[0] // num_time_steps

scaled_size = num_time_steps * num_features
scaled_features = scaled_features[:scaled_size]

X = scaled_features.reshape(-1, num_time_steps, num_features)
y = df['signal_1'].values[:len(X)]  # Use the length of X to match the number of samples

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, train_size=0.2, random_state=42)

model = keras.Sequential([
    layers.LSTM(64, input_shape=(num_time_steps, num_features), activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Assuming binary classification, adjust for multi-class
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)  # Assuming binary classification
test_accuracy = accuracy_score(y_test, y_pred_binary)

print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
plot_wavelet_packets(wavelet_coeffs, title=f'Wavelet Packet Decomposition - db1, Level {wavelet_level}')