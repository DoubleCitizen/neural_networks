import keras as keras
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv('spx.csv', parse_dates=['date'], index_col='date')

train_size = int(len(df) * 0.95)  # 95 процентов данных под обучение
test_size = len(df) - train_size  # оставшиеся 5 процентов данных для теста
train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
print(train.shape, test.shape)


# Перемасштабировка
scaler = StandardScaler()
scaler = scaler.fit(train[['close']])
train['close'] = scaler.transform(train[['close']])
test['close'] = scaler.transform(test[['close']])

# создание набора данных
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

# создание последовательностей за 30 дней
TIME_STEPS = 30
# reshape to [samples, time_steps, n_features]
X_train, y_train = create_dataset(
  train[['close']],
  train.close,
  TIME_STEPS
)
X_test, y_test = create_dataset(
  test[['close']],
  test.close,
  TIME_STEPS
)
print(X_train.shape)


# Автоэнкодер
model = keras.Sequential()
model.add(keras.layers.LSTM(
    units=64,
    input_shape=(X_train.shape[1], X_train.shape[2])
))
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.RepeatVector(n=X_train.shape[1]))
model.add(keras.layers.LSTM(units=64, return_sequences=True))
model.add(keras.layers.Dropout(rate=0.2))
model.add(
  keras.layers.TimeDistributed(
    keras.layers.Dense(units=X_train.shape[2])
  )
)
model.compile(loss='mae', optimizer='adam')




history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    shuffle=False
)

#print(history)

# Нахождение аномалий

X_train_pred = model.predict(X_train)
train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)

# Порог аномалии

THRESHOLD = 0.65

X_test_pred = model.predict(X_test)
test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)

# Создадим DataFrame, содержащий промахи и порог

test_score_df = pd.DataFrame(index=test[TIME_STEPS:].index)
test_score_df['loss'] = test_mae_loss
test_score_df['threshold'] = THRESHOLD
test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold
test_score_df['close'] = test[TIME_STEPS:].close



anomalies = test_score_df[test_score_df.anomaly == True]

print(type(anomalies))
print(f"anomalies = \n{anomalies}")
print("+++++++")
print(anomalies['loss'])
print(anomalies.keys())
print(anomalies.values)

print(anomalies['anomaly'])
print(df)


anomalies.drop(anomalies[anomalies['anomaly'] != True].index, inplace=True)
print(anomalies['close'])
# compression_opts = dict(method='zip',
#                         archive_name='out.csv')
# anomalies.to_csv('out.zip', index=False,
#           compression=compression_opts)


plt.plot(anomalies['close'], ':o')
plt.plot(test.close)

plt.grid()
plt.title("Аномалии")


plt.show()