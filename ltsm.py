import keras as keras
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import sqlliteRead as data_coord_1
import model_data as data_coord_2

# df = data_coord_2.a
#
# df = df.abs()


# train_size = int(len(df) * 0.95)  # 95 процентов данных под обучение
# test_size = len(df) - train_size  # оставшиеся 5 процентов данных для теста
# train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
# print(train.shape, test.shape)

train = data_coord_2.a
test = data_coord_1.NSK1

df = pd.concat([train, test], axis=0)

print("fdfd")
print(df)
print("fdsfsd")



# Перемасштабировка
scaler = StandardScaler()
scaler = scaler.fit(df[['X']])
train['X'] = scaler.transform(train[['X']])
test['X'] = scaler.transform(test[['X']])

print("---------")
print(train['X'])
print("---------")


# создание набора данных
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

# создание последовательностей за 30 дней
TIME_STEPS = 10
# reshape to [samples, time_steps, n_features]
X_train, y_train = create_dataset(
  train[['X']],
  train.X,
  TIME_STEPS
)
X_test, y_test = create_dataset(
  test[['X']],
  test.X,
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

# THRESHOLD = 0.65
THRESHOLD = 0.65

X_test_pred = model.predict(X_test)
test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)



# Создадим DataFrame, содержащий промахи и порог

test_score_df = pd.DataFrame(index=test[TIME_STEPS:].index)
test_score_df['loss'] = test_mae_loss
test_score_df['threshold'] = THRESHOLD
test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold
test_score_df['X'] = test[TIME_STEPS:].X

print("++++++++")
print(test_score_df['loss'])
print("++++++++")

anomalies = test_score_df[test_score_df.anomaly == True]

print(type(anomalies))
print(f"anomalies = \n{anomalies}")
print("+++++++")
print(anomalies['loss'])
print(anomalies.keys())
print(anomalies.values)
anomalies.drop(anomalies[anomalies['anomaly'] != True].index, inplace=True)

print(anomalies['anomaly'])



plt.plot(anomalies['X'], ':o')
plt.plot(test.X)

plt.grid()
plt.title("Аномалии")


plt.show()
