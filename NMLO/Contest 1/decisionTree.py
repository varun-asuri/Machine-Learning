import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
df_train = pd.read_csv("train.csv")
x = np.array(df_train[['age','gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']])
y = np.array(df_train[['cardio']])
clf = RandomForestClassifier(n_estimators=150, max_depth=4, random_state=1)
clf.fit(x, np.ravel(y))
df_test = pd.read_csv("test.csv")
x_test = np.array(df_test[['age','gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']])
answers = clf.predict(x_test)
df_test.insert(12, "cardio", answers, True)
submission = df_test[['id', 'cardio']]
submission.to_csv("submission.csv", index=False)