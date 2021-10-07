import pandas as pd
import numpy as np

import warnings 
warnings.filterwarnings("ignore")

from sklearn.preprocessing import RobustScaler

df = pd.read_csv("/home/ubuntu/git_workspace/Heart-Attack-Prediction/data/heart-attack-dataset.csv")
print("Data Shape:", df.shape)

dict = {}
for i in list(df.columns):
    dict[i] = df[i].value_counts().shape[0]

pd.DataFrame(dict,index=["unique count"]).transpose()

cat_cols = ['sex','exng','caa','cp','fbs','restecg','slp','thall']
con_cols = ["age","trtbps","chol","thalachh","oldpeak"]
target_col = ["output"]

print("Categorical cols :",cat_cols)
print("Continous cols :",con_cols)
print("Target variable :",target_col)

