import pandas as pd
import numpy as np
import pywt
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical

# Load the dataset
file_path = 's00.csv'
dataframe = pd.read_csv(file_path)

# Assuming the last column is the label
X = dataframe.iloc[:, :-1].values
y = dataframe.iloc[:, -1].values

# Check if labels are non-negative integers
if not np.issubdtype(y.dtype, np.integer) or np.any(y < 0):
    # Map labels to a zero-based integer range
    unique_labels = np.unique(y)
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    y = np.array([label_mapping[label] for label in y])

y_min = y.min()
if y_min != 0:
    print(f"Adjusting labels to be zero-based. Original min label: {y_min}")
    y = y - y_min

# Apply Wavelet Packet Decomposition (WPD)
def apply_wpd(data, wavelet='db4', max_level=5):
    wpd_features = []
    for sample in data:
        wp = pywt.WaveletPacket(data=sample, wavelet=wavelet, mode='symmetric', maxlevel=max_level)
        concatenated_features = np.concatenate([node.data for node in wp.get_level(max_level, order='freq')])
        wpd_features.append(concatenated_features)
    return np.array(wpd_features)

# Reshape the data appropriately for WPD
n_samples = len(X)  # Assuming each row is a separate sample
X_reshaped = X.reshape(n_samples, -1)  # The -1 infers the correct dimension

# Apply WPD
X_wpd = apply_wpd(X_reshaped)

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_wpd)

# One-hot encode labels
y_categorical = to_categorical(y)

# Reshape the data for LSTM input
num_time_steps = 1  # Adjust this based on your data's time steps per sample
num_features = X_scaled.shape[1]  # Number of features after WPD
X_lstm = X_scaled.reshape((-1, num_time_steps, num_features))

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_lstm, y_categorical, test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(num_time_steps, num_features)))
model.add(LSTM(20))
model.add(Dense(y_categorical.shape[1], activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")