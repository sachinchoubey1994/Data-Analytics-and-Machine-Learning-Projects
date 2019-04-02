from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn import tree

df = pd.read_csv("train.csv")
d = {'low': 0, 'med': 1, 'high': 2}
e = {'no': 0, 'yes': 1}
df['maintenance'] = df['maintenance'].map(d)
df['price'] = df['price'].map(d)
df['airbag'] = df['airbag'].map(e)
df['profitable'] = df['profitable'].map(e)
features = list(df.columns[:4])
y_train = df["profitable"]
X_train = df[features]

clsfy_entrpy = tree.DecisionTreeClassifier(criterion = "entropy")
clsfy_entrpy = clsfy_entrpy.fit(X_train,y_train)

clsfy_gini = tree.DecisionTreeClassifier(criterion = "gini")
clsfy_gini = clsfy_gini.fit(X_train,y_train)

dt = pd.read_csv("test.csv")
dt['maintenance'] = dt['maintenance'].map(d)
dt['price'] = dt['price'].map(d)
dt['airbag'] = dt['airbag'].map(e)
dt['profitable'] = dt['profitable'].map(e)
y_test = dt["profitable"]
feature = list(dt.columns[:4])
X_test=dt[feature]
y_pred_entrpy=clsfy_entrpy.predict(X_test)

print("------------------------------------------------")
print("Info_gain of root node=", clsfy_entrpy.tree_.impurity[0])
print("% Accuracy_score for info_gain=", 100 * accuracy_score(y_test, y_pred_entrpy))
print(y_pred_entrpy, "(1=> profitable=yes and 0=> profitable=no)")

print("------------------------------------------------")
y_pred_gini=clsfy_gini.predict(X_test)
print("Gini of root node=", clsfy_gini.tree_.impurity[0])
print("% Accuracy_score for gini index=", 100 * accuracy_score(y_test, y_pred_gini))
print(y_pred_gini, "(1=> profitable=yes and 0=> profitable=no)")

tree.export_graphviz(clsfy_gini, out_file='tree.dot')