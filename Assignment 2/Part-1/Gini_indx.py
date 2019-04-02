import numpy as np 
import pandas as pd
class NodPnt(object):
    def __init__(self, indxs = None, chld = [], gini = 0):
        self.indxs = indxs           
        self.gini = gini   
        self.splt_attrib = None 
        self.chld = chld 
        self.odr = None      
        self.lbl = None       
    def set_props(self, splt_attrib, odr):
        self.splt_attrib = splt_attrib
        self.odr = odr
    def set_lbl(self, lbl):
        self.lbl = lbl
def gini(freq):
    freq_0 = freq[np.array(freq).nonzero()[0]]
    prob_0 = freq_0/float(freq_0.sum())
    return (1-sum(prob_0**2))
class DT(object):
    def __init__(self):
        self.rt = None
        self.trn = 0
    def fit(self, dt, trgt):
        self.trn = dt.count()[0]
        self.dt = dt 
        self.attribs = list(dt)
        self.trgt = trgt 
        self.lbls = trgt.unique()
        indxs = range(self.trn)
        self.rt = NodPnt(indxs = indxs, gini = self._gini(indxs))
        queue = [self.rt]
        while queue:
            nod = queue.pop()
            nod.chld = self._splt(nod)
            if not nod.chld: 
                self._set_lbl(nod)
            queue += nod.chld
            self._set_lbl(nod)
    def _gini(self, indxs):
        if len(indxs) == 0: return 0
        indxs = [i+1 for i in indxs] 
        freq = np.array(self.trgt[indxs].value_counts())
        return gini(freq)
    def _set_lbl(self, nod):
        trgt_indxs = [i + 1 for i in nod.indxs] 
        nod.set_lbl(self.trgt[trgt_indxs].mode()[0]) 
    def _splt(self, nod):
        indxs = nod.indxs 
        bst_splt_qualty = 0
        bst_splts = []
        bst_attrib = None
        odr = None
        sub_dt = self.dt.iloc[indxs, :]
        for i, att in enumerate(self.attribs):
            vals = self.dt.iloc[indxs, i].unique().tolist()
            if len(vals) == 1: continue 
            splts = []
            for val in vals: 
                sub_indxs = sub_dt.index[sub_dt[att] == val].tolist()
                splts.append([sub_indx-1 for sub_indx in sub_indxs])
            splt_gini= 0
            for splt in splts:
                splt_gini += len(splt)*self._gini(splt)/len(indxs)
            splt_qualty = nod.gini - splt_gini
            if splt_qualty > bst_splt_qualty:
                bst_splt_qualty = splt_qualty 
                bst_splts = splts
                bst_attrib = att
                odr = vals
        nod.set_props(bst_attrib, odr)
        child_nods = [NodPnt(indxs = splt,
                     gini = self._gini(splt)) for splt in bst_splts]
        return child_nods
    def pred(self, nw_dt):
        npnts = nw_dt.count()[0]
        lbls = [None]*npnts
        for n in range(npnts):
            x = nw_dt.iloc[n, :] 
            nod = self.rt
            while nod.chld: 
                nod = nod.chld[nod.odr.index(x[nod.splt_attrib])]
            lbls[n] = nod.lbl
        return lbls
    def accrcy(self, y_pred, y_tst):
        flg=[i for i, j in zip(y_tst, y_pred) if i == j]
        return len(flg)/len(y_pred)
def gini_of_root(rows):
    counts = {}  
    for row in rows:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity
if __name__ == "__main__":
    df = pd.read_csv('train.csv')
    a = list(df.columns[:4])
    X_train=df[a]
    y_train = df["profitable"]
    tree = DT()
    tree.fit(X_train, y_train)
    d = pd.read_csv('test.csv')
    b = list(df.columns[:4])
    X_tst=d[b]
    y_tst = d["profitable"]
    y_pred=tree.pred(X_tst)
    print("---------------------------------------------")
    print("Gini of root node=", gini_of_root(X_train))
    print("Profitable prediction:", y_pred)
    accrcy_scor=100 * tree.accrcy(y_pred, y_tst)
    print("%Accuracy=", accrcy_scor)
    