import pandas as pd
import numpy as np
import pywt
from tensorflow import keras
from keras import layers
from keras import callbacks
from keras import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Load the EEG dataset
df = pd.read_csv('s00.csv')
eeg_signals = df.values

# Define the Wavelet Packet Decomposition function
def wpd(signal):
    wp = pywt.WaveletPacket(data=signal, wavelet='db4', mode='symmetric', maxlevel=4)
    return np.array([node.data for node in wp.get_level(4, 'natural')]).flatten()

# Apply WPD to each EEG signal
features = np.array([wpd(signal) for signal in eeg_signals])

# Define the number of timesteps and calculate the number of features per timestep
num_samples = features.shape[0]
num_total_features = features.shape[1]
num_timesteps = 128  # Adjust this based on your specific requirements
num_features_per_timestep = num_total_features // num_timesteps

# Reshape features for LSTM input
X = features.reshape(num_samples, num_timesteps, num_features_per_timestep)

# Generate dummy labels (replace with actual labels)
y = np.random.randint(0, 5, size=num_samples)  # Assuming 5 classes

# Split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train.reshape(-1, num_features_per_timestep)).reshape(-1, num_timesteps, num_features_per_timestep)
x_test = scaler.transform(x_test.reshape(-1, num_features_per_timestep)).reshape(-1, num_timesteps, num_features_per_timestep)

# Build LSTM Model
model = Sequential([
    layers.LSTM(64, input_shape=(num_timesteps, num_features_per_timestep), activation='relu'),
    layers.Dense(5, activation='softmax')  # Assuming 5 classes
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Fit the model and capture training history
history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the Model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
conf_matrix = confusion_matrix(y_test, y_pred_classes)
print("Confusion Matrix:\n", conf_matrix)

# Plotting Training History
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
