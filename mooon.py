import pandas as pd # data handling
import numpy as np # numerical operations
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.model_selection import train_test_split
# import Matplotlib # data visualization

#load csv data
df = pd.read_csv('Ethereum_history.csv', delimiter=';', parse_dates=['timestamp'])

# Preview the data
print(df.head())

# Quick statistical summary
print(df.describe())

# Convert all relevant columns to datetime if not already
df['timeOpen'] = pd.to_datetime(df['timeOpen'])
df['timeClose'] = pd.to_datetime(df['timeClose'])
df['timeHigh'] = pd.to_datetime(df['timeHigh'])
df['timeLow'] = pd.to_datetime(df['timeLow'])

df['MA_7'] = df['close'].rolling(window=7).mean()

# Normalize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['open', 'high', 'low', 'close', 'volume']] = scaler.fit_transform(df[['open', 'high', 'low', 'close', 'volume']])

# Model selection and training
#start with lienar regression as a model type
# maybe try ARIMA, seasonal ARIMA, LSTM
# Assume 'df' is your preprocessed DataFrame and 'close' is your target variable

# Function to create sequences
def create_sequences(data, sequence_length):
    xs = []
    ys = []
    for i in range(len(data) - sequence_length):
        x = data.iloc[i:(i + sequence_length)].drop(['timestamp'], axis=1)  # drop non-numeric columns if present
        y = data.iloc[i + sequence_length]['close']
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Creating sequences
sequence_length = 24  # for example, use the last 24 hours to predict the next hour
X, y = create_sequences(df, sequence_length)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model architecture
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Fit the model
history = model.fit(
    X_train, y_train,
    epochs=50,  # You can adjust this
    batch_size=32,  # And this
    validation_data=(X_test, y_test),
    verbose=1
)

# Evaluate the model
test_loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")

import matplotlib.pyplot as plt

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# Model evaluation 