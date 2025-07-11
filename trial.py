import pandas as pd

df = pd.read_excel("user_data.xlsx", index_col = "ユーザID", engine = "openpyxl") 
print(df)