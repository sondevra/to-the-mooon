import pandas as pd
import numpy as np
import requests
import sklearn.preprocessing
import tweepy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta


api_key = '6a3f7b15edc3205ee1d44dfd73562ba034d2e4600f3f0abb179322d1f9c8f426'

#tweepy auth
tweepy_api_key = 'p3Qx5zqELgYSOOTvwVNBOuhet'
api_secret_key = 'KS3LDJQqaY94UfiZx68qHfSI0cU2qHRCGGeuDVfy79Knt9cJ9R'
access_token = '1783888692274176000-8at0wy3H0HCh45t81vTDM1LfHbW8H4'
access_token_secret = 'NWNHXtWDgGxud7BKjwx13zoTZCSSBbcodYIeR5z1JyRUJ'



# Check if elon musk has tweeted
def check_elon_tweets(tweepy_api_key, api_secret_key, access_token, access_token_secret):
    # Authenticate to Twitter
    auth = tweepy.OAuthHandler(tweepy_api_key, api_secret_key)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)

    # Prepare to count tweets
    score = 0
    keywords = ["bitcoin", "doge", "dogecoin", "crypto", "cryptocurrency"]
    today_date = datetime.utcnow().date()
    
    # Retrieve recent tweets from Elon Musk's Twitter
    tweets = api.user_timeline(screen_name='elonmusk', tweet_mode='extended', count=100)

    # Check each tweet
    for tweet in tweets:
        # Check if the tweet is from today
        if tweet.created_at.date() == today_date:
            # Check for keywords in the full text of the tweet
            tweet_text = tweet.full_text.lower()
            if any(keyword in tweet_text for keyword in keywords):
                score += 1

    return score

print('''
   /$$                       /$$     /$$                                                                   /$$
  | $$                      | $$    | $$                                                                  | $$
 /$$$$$$    /$$$$$$        /$$$$$$  | $$$$$$$   /$$$$$$        /$$$$$$/$$$$   /$$$$$$   /$$$$$$  /$$$$$$$ | $$
|_  $$_/   /$$__  $$      |_  $$_/  | $$__  $$ /$$__  $$      | $$_  $$_  $$ /$$__  $$ /$$__  $$| $$__  $$| $$
  | $$    | $$  \ $$        | $$    | $$  \ $$| $$$$$$$$      | $$ \ $$ \ $$| $$  \ $$| $$  \ $$| $$  \ $$|__/
  | $$ /$$| $$  | $$        | $$ /$$| $$  | $$| $$_____/      | $$ | $$ | $$| $$  | $$| $$  | $$| $$  | $$    
  |  $$$$/|  $$$$$$/        |  $$$$/| $$  | $$|  $$$$$$$      | $$ | $$ | $$|  $$$$$$/|  $$$$$$/| $$  | $$ /$$
   \___/   \______/          \___/  |__/  |__/ \_______/      |__/ |__/ |__/ \______/  \______/ |__/  |__/|__/
                                                                                                                                                                           
''')



# score = check_elon_tweets(tweepy_api_key, api_secret_key, access_token, access_token_secret)
print("Elon Musk has tweeted " + str(0) + " times about cryto today\n")

# Gets a list of all the coins tracked by cryptocompare
def get_coin_list(api_key):
    url = "https://min-api.cryptocompare.com/data/all/coinlist"
    params = {"api_key": api_key}
    response = requests.get(url, params=params)
    data = response.json()
    return data['Data']

# Get the coin ticker from the user 
coin_symbol = input("Enter the coin ticker you would like to fetch data for (e.g., BTC, ETH): ")
coins = get_coin_list(api_key)
# Check for valid ticker
if coin_symbol not in coins:
    print(f"Coin symbol {coin_symbol} not found in CryptoCompare database.")
    exit
# Set up get request
days = 365
url = "https://min-api.cryptocompare.com/data/v2/histoday"
params = {
    "fsym": coin_symbol,
    "tsym": "USD",
    "limit": days,
    "api_key": api_key
}
# Make the API request
response = requests.get(url, params=params)
data = response.json()['Data']['Data']
df = pd.DataFrame(data)
df['time'] = pd.to_datetime(df['time'], unit='s')
df.rename(columns={
    'time': 'timestamp',
    'high': 'high',
    'low': 'low',
    'open': 'open',
    'close': 'close',
    'volumeto': 'volume'
}, inplace=True)
    
# Select relevant columns
df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
df = df.sort_values(by='timestamp', ascending=False)

# Load CSV data
df['timestamp'] = pd.to_datetime(df['timestamp'])
# print(df['timestamp'])
df.sort_values(by='timestamp', inplace=True)
# print(df)

lag_cols = ['open', 'high', 'low', 'close', 'volume']
#lags = range(1, 6)  # Create lag features for the past 5 days
lags = range(1, 15)  
for col in lag_cols:
    for lag in lags:
        df[f'{col}_lag_{lag}'] = df[col].shift(lag)
# print(df[f'{col}_lag_{lag}'])

df.dropna(inplace=True)
# print(df)

features = ['open_lag_1', 'high_lag_1', 'low_lag_1', 'close_lag_1', 'volume_lag_1']
target_high = 'high'
target_low = 'low'
target_volume = 'volume'

X_train, X_test, y_train_high, y_test_high, y_train_low, y_test_low, y_train_volume, y_test_volume = train_test_split(
    df[features], df[target_high], df[target_low], df[target_volume], test_size=0.2, random_state=42)
# print(X_train)
# print(X_test)

# Normalize the features
scaler = sklearn.preprocessing.MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# print("X_Train_Scaled: \n", X_train_scaled)
# print("X_Test_Scaled: \n", X_test_scaled)



# High Model Training
# model_high = Sequential()
# model_high.add(Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)))
# model_high.add(Dense(32, activation='relu'))
# model_high.add(Dense(1))
model_high = Sequential()
model_high.add(Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)))
model_high.add(Dense(64, activation='relu'))
model_high.add(Dense(32, activation='relu'))
model_high.add(Dense(1))

# Low Model Training
# model_low = Sequential()
# model_low.add(Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)))
# model_low.add(Dense(32, activation='relu'))
# model_low.add(Dense(1))
model_low = Sequential()
model_low.add(Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)))
model_low.add(Dense(64, activation='relu'))
model_low.add(Dense(32, activation='relu'))
model_low.add(Dense(1))



# opt = Adam(learning_rate=0.001)
model_high.compile(optimizer='adam', loss='mean_squared_error')
model_high.fit(X_train_scaled, y_train_high, epochs=300, batch_size=32, verbose=1)
model_high.add(Dropout(0.1))
model_high.add(BatchNormalization())
model_high.add(LeakyReLU(alpha=0.01))


model_low.compile(optimizer='adam', loss='mean_squared_error')
model_low.fit(X_train_scaled, y_train_low, epochs=300, batch_size=32, verbose=1)
model_low.add(Dropout(0.1))
model_low.add(BatchNormalization())
model_low.add(LeakyReLU(alpha=0.01))



# Model Evaluation
loss_high = model_high.evaluate(X_test_scaled, y_test_high)
loss_low = model_low.evaluate(X_test_scaled, y_test_low)
print(f'High Prediction Loss: {loss_high}')
print(f'Low Prediction Loss: {loss_low}')



# High Model Prediction
# Predict high for the next day
last_day_features = df[features].tail(1)
last_day_features_scaled = scaler.transform(last_day_features)
predicted_high = model_high.predict(last_day_features_scaled)
print(f'Predicted High for the Next Day: {predicted_high}')

# Low Model Prediction
# Predict low for the next day
last_day_features = df[features].tail(1)
last_day_features_scaled = scaler.transform(last_day_features)
predicted_low = model_low.predict(last_day_features_scaled)
print(f'Predicted Low for the Next Day: {predicted_low}')



print('Program end')
