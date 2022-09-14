import sqlite3

import numpy as np
import pandas as pd

a = np.load("./temp/trace.npy", mmap_mode=None, allow_pickle=True, fix_imports=True)

names = \
    [("id", "i4"), ("time", "f4"), ("day", "i1"), ("hours", "i2"), ("minutes", "i2"), ("screen_lock_status", "i1"),
     ("network_status", "i1"),
     ("battery_charge_status", "i1"), ("screen_status", "i1"), ("battery_level", "i1"),
     ("is_free", "i1"), ("idle", "i1")]

label = np.load("./temp/label_arr.npy", mmap_mode=None, allow_pickle=True, fix_imports=True).astype(int)

names = [i[0] for i in names]

print(a)

temp = [a[i] for i in names]

print(temp)

tab = pd.DataFrame(columns=[names])

for i, nam in enumerate(names, 0):
    tab[nam] = temp[i]

tab["idle"] = label

print(tab)

# tab.to_csv("./trace.csv", index=False)
sql = sqlite3.connect("../data/UserAction.db")

# tab = pd.read_csv("./trace.csv")

tab.to_sql("trace", con=sql,index=False)