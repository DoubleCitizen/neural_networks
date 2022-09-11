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

print(train)

pd.set_option("display.max.columns", None)

print("----")
print(df.head().values)
# print(df)
#
# plt.plot(df)
# plt.show()

# rng = np.arange(50)
# rnd = np.random.randint(0, 10, size=(1, rng.size))
# yrs = 1950 + rng
#
# fig, ax = plt.subplots(figsize=(5, 3))
# ax.stackplot(yrs, rnd + rng, labels=['Example'])
# ax.set_title('Combined debt growth over time')
# ax.legend(loc='upper left')
# ax.set_ylabel('Total debt')
# ax.set_xlim(xmin=yrs[0], xmax=yrs[-1])
# fig.tight_layout()
#
# plt.show()

a = np.array([(0.2, 2), (3, 4)])
b = np.array([('5', '6'), ('7', '8')])

c = np.hstack([a,b])
print(c)