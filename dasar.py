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

# Ensure no NaN values are present
if dataframe.isnull().values.any():
    # Handle NaN values appropriately, for example by filling or dropping them
    dataframe = dataframe.dropna()

# Assuming the last column is the label
X = dataframe.iloc[:, :-1].values
y = dataframe.iloc[:, -1].values

# Ensure labels start from 0 and are consecutive
y = pd.factorize(y)[0]

# Convert labels to one-hot encoding
y = to_categorical(y)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define a function to apply Wavelet Packet Decomposition (WPD)
def apply_wpd(data, wavelet='db4', level=5):
    wp_features = []
    for sample in data:
        wp = pywt.WaveletPacket(data=sample, wavelet=wavelet, mode='symmetric', maxlevel=level)
        wp_features.append(np.array([node.data for node in wp.get_level(level, 'freq')]).flatten())
    return np.array(wp_features)

# Apply WPD to each sample in your dataset
X_wpd = apply_wpd(X_scaled)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_wpd, y, test_size=0.2, random_state=42)

# Reshape the input data for LSTM: [samples, time steps, features]
# This is a crucial step and depends on your dataset's specifics
# For example, if each sample should have 100 time steps, you would reshape like this:
time_steps = 100  # This is an assumption, you need to adjust this according to your dataset
features_per_step = X_train.shape[1] // time_steps
X_train = X_train.reshape((-1, time_steps, features_per_step))
X_test = X_test.reshape((-1, time_steps, features_per_step))

# Create LSTM model
model = Sequential()
model.add(LSTM(units=100, input_shape=(time_steps, features_per_step), return_sequences=True))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=y.shape[1], activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")