import numpy as np 
import pandas as pd
import math
class NodPnt(object):
    def __init__(self, indxs = None, chld = [], entrpy = 0):
        self.indxs = indxs           
        self.entrpy = entrpy   
        self.splt_attrib = None 
        self.chld = chld 
        self.odr = None      
        self.lbl = None       
    def set_props(self, splt_attrib, odr):
        self.splt_attrib = splt_attrib
        self.odr = odr
    def set_lbl(self, lbl):
        self.lbl = lbl
def entrpy(freq):
    freq_0 = freq[np.array(freq).nonzero()[0]]
    prob_0 = freq_0/float(freq_0.sum())
    return -np.sum(prob_0*np.log(prob_0))
class DT(object):
    def __init__(self):
        self.rt = None
        self.trn = 0
    def fit(self, dt, trgt):
        self.trn = dt.count()[0]
        self.dt = dt 
        self.attribs = list(dt)
        self.trgt = trgt 
        self.lbls = trgt.values
        indxs = range(self.trn)
        self.rt = NodPnt(indxs = indxs, entrpy = self._entrpy(indxs))
        queue = [self.rt]
        while queue:
            nod = queue.pop()
            nod.chld = self._splt(nod)
            if not nod.chld: 
                self._set_lbl(nod)
            queue += nod.chld
            self._set_lbl(nod)
    def _entrpy(self, indxs):
        if len(indxs) == 0: return 0
        indxs = [i+1 for i in indxs] 
        freq = np.array(self.trgt[indxs].value_counts())
        return entrpy(freq)
    def _set_lbl(self, nod):
        trgt_indxs = [i + 1 for i in nod.indxs] 
        nod.set_lbl(self.trgt[trgt_indxs].mode()[0]) 
    def _splt(self, nod):
        indxs = nod.indxs 
        bst_info_gn = 0
        bst_splts = []
        bst_attrib = None
        odr = None
        sub_dt = self.dt.iloc[indxs, :]
        for i, att in enumerate(self.attribs):
            vals = self.dt.iloc[indxs, i].unique().tolist()
            if len(vals) == 1: continue # entrpy = 0
            splts = []
            for val in vals: 
                sub_indxs = sub_dt.index[sub_dt[att] == val].tolist()
                splts.append([sub_indx-1 for sub_indx in sub_indxs])
            wt_entrpy= 0
            for splt in splts:
                wt_entrpy += len(splt)*self._entrpy(splt)/len(indxs)
            info_gn = nod.entrpy - wt_entrpy 
            if info_gn > bst_info_gn:
                bst_info_gn = info_gn 
                bst_splts = splts
                bst_attrib = att
                odr = vals
        nod.set_props(bst_attrib, odr)
        child_nods = [NodPnt(indxs = splt,
                     entrpy = self._entrpy(splt)) for splt in bst_splts]
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
if __name__ == "__main__":
    a = pd.read_csv("traindata.txt", delimiter="\t", names=["docID", "wordID"])
    b = pd.read_csv("trainlabel.txt", names=["newsgrop_label"])
    y_train = a.sample(n=707)
    X_train = b.sample(n=707)
    tree = DT()
    tree.fit(X_train, y_train)
    d = pd.read_csv("testdata.txt", delimiter="\t", names=["docID", "wordID"])
    e = pd.read_csv("trainlabel.txt", names=["newsgrop_label"])
    y_test = d.sample(n=707)
    X_test= e.sample(n=707)
    y_pred=tree.pred(X_test)
    print("---------------------------------------------")
    print("Profitable prediction:", y_pred)
    accrcy_scor=100 * tree.accrcy(y_pred, y_test)
    print("%Accuracy=", accrcy_scor)