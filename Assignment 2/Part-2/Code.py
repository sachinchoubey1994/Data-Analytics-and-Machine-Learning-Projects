from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt

a=[]
trn_dt = pd.read_csv("traindata.txt", delimiter="\t", names=["docID", "wordID"])
trn_lb = pd.read_csv("trainlabel.txt", names=["newsgrop_label"])
y_train = trn_lb.sample(n=707)
X_train = trn_dt.sample(n=707)

tst_dt = pd.read_csv("testdata.txt", delimiter="\t", names=["docID", "wordID"])
tst_lb = pd.read_csv("trainlabel.txt", names=["newsgrop_label"])
y_test = tst_lb.sample(n=707)
X_test=tst_dt.sample(n=707)

for i in range(3,7):
    clf = tree.DecisionTreeClassifier(criterion = "entropy", max_depth=i)
    clf = clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    score=accuracy_score(y_test, y_pred)
    a.append(score)
    print("% Accuracy at max_depth", i ,"=", score)
    
plt.plot(range(3,7), a)
plt.suptitle('Accuracy vs Maximum depth')
plt.xlabel('Maximum depth')
plt.ylabel('Accuracy')
plt.show()