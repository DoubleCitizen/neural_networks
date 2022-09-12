import sqlite3
import pandas as pd
from matplotlib import pyplot as plt

cnx = sqlite3.connect('fags_ppp_timeseries.sqlite3')
df = pd.read_sql_query("SELECT * FROM mark_coordinate", cnx)
df = df.set_index('code')

NSK1 = df.loc['NSK1']
# NSK1 = NSK1.reset_index()
# df.set_index


NSK1 = NSK1.set_index('epoch')

NSK1 = NSK1.drop(columns=['solution_type', 'Y', 'Z', 'CXX', 'CXY', 'CYY', 'CXZ', 'CYZ', 'CZZ'])




# XNSK1 = NSK1['X']
# timeNSK1 = NSK1['epoch']

# data_result =

if __name__ == "__main__":
    print(NSK1)

    plt.plot(NSK1.X)

    plt.grid()
    plt.title("NSK1")

    plt.show()
