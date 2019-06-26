from pandas import read_csv, DataFrame
from numpy import random
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
import string
rem = string.punctuation
pattern = r"[{}]".format(rem)
def data_loader():
    dt=read_csv("Assignment_4_data.txt", delimiter='\t',  names=["label", "email"])
    return dt
def preprocess(dt):
    dt["label"] = dt["label"].map({'spam':1,'ham':0})
    dt['email'] = dt['email'].str.replace(pattern,'')
    dt['email'] = list(map(str, dt['email']))
    dt['email'].dropna(inplace=True)
    dt['email'] = dt['email'].apply(word_tokenize)
    def stopwords_n_stem(text):
        text = [t for t in text if t not in stopwords.words('english')]
        Stemmer= PorterStemmer()
        return [Stemmer.stem(word) for word in text]
    dt['email'] = dt['email'].apply(stopwords_n_stem)
    x_tr,x_ts,y_tr,y_ts = train_test_split(dt['email'],dt["label"],test_size=0.8)    
    x_tr = x_tr.to_frame('email')
    x_ts = x_ts.to_frame('email')
    mlb = MultiLabelBinarizer()
    d = DataFrame(mlb.fit_transform(x_tr.values), dt.index, mlb.classes_)
    x_tr = x_tr.drop('email', 1).join(d)
    mlb = MultiLabelBinarizer()
    d = DataFrame(mlb.fit_transform(x_ts.email.values), dt.index, mlb.classes_)
    x_ts = x_ts.drop('email', 1).join(d)
    return dt, x_tr,x_ts,y_tr,y_ts
def relu(x):
    return x * (x > 0)
def relu_devirate(x):
    return 1. * (x > 0)
def weit_initiat(inpt_nurn,hddn_nurn,otpt_nurn):
    net = []
    w1 = [{"weit" : [random.randn(inpt_nurn, hddn_nurn)]}]
    net.append(w1)
    w2 = [{"weit" : [random.randn(hddn_nurn, otpt_nurn)]}]
    net.append(w2)
    return net
def forward(net, row):
	inputs = row
	for layer in net:
		new_inputs = []
		for nurn in layer:
			nurn_activ = (nurn['weit'])
			nurn['val'] = relu(nurn_activ)
			new_inputs.append(nurn['val'])
		inputs = new_inputs
	return inputs
def backward(net, pred, row, lrnin_rate):
	for i in reversed(range(len(net))):
		layer = net[i]
		errs = list()
		if(i != len(net)-1):
			for j in range(len(layer)):
				err = 0.0
				for nurn in net[i+1]:
					err += (nurn['weit'][j] * nurn['del'])
				err.append(err)
		else:
			for j in range(len(layer)):
				nurn = layer[j]
				errs.append(pred[j] - nurn['val'])
		for j in range(len(layer)):
			nurn = layer[j]
			nurn['del'] = errs[j] * relu_devirate(nurn['val'])
		for i in range(len(net)):
			inputs = row[:-1]
			if(i != 0):
				inputs = [nurn['val'] for nurn in net[i-1]]
			for nurn in net[i]:
				for j in range(len(inputs)):
					nurn['weit'][j] += lrnin_rate * nurn['del'] * inputs[j]
				nurn['weit'][-1] += lrnin_rate * nurn['del']
def train(net, x_tr, lrnin_rate, n_epoch, otpt_nurns):
	for epoch in range(n_epoch):
		sm_err = 0
		for row in x_tr:
			outputs = forward(net, row)
			pred = [0 for i in range(otpt_nurns)]
			pred[row[-1]] = 1
			sm_err += sum([(pred[i]-outputs[i])**2 for i in range(len(pred))])
			backward(net, pred, row, lrnin_rate)
		print('>epoch=%d, test_error=%.3f' % (epoch, sm_err))   
def test(net, x_ts, lrnin_rate, n_epoch, otpt_nurns):
    for epoch in range(n_epoch):
        sm_err = 0
        for row in x_ts:
            outputs = forward(net, row)
            pred = [0 for i in range(otpt_nurns)]
            pred[row[-1]] = 1
            sm_err += sum([(pred[i]-outputs[i])**2 for i in range(len(pred))])
            backward(net, pred, row, lrnin_rate)
        print('>epoch=%d, train_error=%.3f' % (epoch, sm_err))
def predict(network, row):
	outputs = forward(network, row)
	return outputs.index(max(outputs))
dt, x_tr, x_ts, y_tr, y_ts = preprocess(data_loader())
inpt_nurn = len(x_tr[0]) - 1
otpt_nurn = len(set([row[-1] for row in y_tr]))
net = weit_initiat(inpt_nurn, 100, otpt_nurn)
for row in x_ts:
	prediction = predict(net, row)
	print('Expected=%d, Got=%d' % (row[-1], prediction))
print("Training set error over number of epochs")
train(net, x_tr, 0.1, 20, otpt_nurn)
print("Test set error over number of epochs")
test(net, x_ts, 0.1, 20, otpt_nurn)