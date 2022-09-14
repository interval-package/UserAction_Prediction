import sqlite3
import pandas as pd

sql = sqlite3.connect("../data/UserAction.db")

tab = pd.read_csv("./trace.csv")

tab.to_sql("trace", con=sql,index=False)