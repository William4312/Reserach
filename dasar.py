import pandas as pd
import numpy as np
import pywt
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.utils import to_categorical, plot_model
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Load the dataset
file_path = 's01.csv'
dataframe = pd.read_csv(file_path)

# Assuming the last column is the label
X = dataframe.iloc[:, :-1].values
y = dataframe.iloc[:, -1].values

# Label encoding (if labels are not numeric)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# One-hot encode labels
y_categorical = to_categorical(y_encoded)

# Apply Wavelet Packet Decomposition (WPD)
def apply_wpd(data, wavelet='db4', max_level=5):
    wpd_features = []
    for sample in data:
        wp = pywt.WaveletPacket(data=sample, wavelet=wavelet, mode='symmetric', maxlevel=max_level)
        concatenated_features = np.concatenate([node.data for node in wp.get_level(max_level, order='freq')])
        wpd_features.append(concatenated_features)
    return np.array(wpd_features)

# Apply WPD to each sample
X_wpd = apply_wpd(X)

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_wpd)

# Reshape the data for LSTM input
num_samples, num_features = X_scaled.shape
num_time_steps = 1  # This should be adjusted based on your EEG data structure
X_lstm = X_scaled.reshape((num_samples, num_time_steps, num_features))

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_lstm, y_categorical, test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(num_time_steps, num_features)))
model.add(LSTM(50))
model.add(Dense(y_categorical.shape[1], activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, callbacks=[early_stopping])

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

plt.figure(figsize=(12, 6))
for i in range(min(5, len(dataframe.columns) - 1)):  # Plot the first 5 channels
    plt.plot(dataframe.iloc[:100, i], label=f'Channel {i+1}')  # Plot first 100 data points

plt.title('EEG Data Visualization')
plt.xlabel('Time Points')
plt.ylabel('EEG Reading')
plt.legend()
plt.show()