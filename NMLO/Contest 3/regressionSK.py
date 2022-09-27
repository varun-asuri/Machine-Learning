import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
df_train = pd.read_csv("train.csv")
x = np.array(df_train[['ed', 'inc', 'pop']])
y = np.array(df_train[['cases']])
degree = 3
reg = make_pipeline(PolynomialFeatures(degree),LinearRegression(fit_intercept=False, n_jobs=3))
reg.fit(x, np.ravel(y))
df_test = pd.read_csv("test.csv")
x_test = np.array(df_test[['ed', 'inc', 'pop']])
answers = reg.predict(x_test)
df_test.insert(3, "cases", answers, True)
id_dict = {'id': [a for a in range(len(df_test))]}
idcol = pd.DataFrame(data = id_dict)
submission = pd.concat([idcol, df_test[['cases']]], axis=1, sort=False).astype({'id': int, 'cases': float})
submission.to_csv("submission.csv", index=False)