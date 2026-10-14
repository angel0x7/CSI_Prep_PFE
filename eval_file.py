import numpy as np, sys
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import eval_pipeline as E

W=Path(".")
variants={"A brut":(W/"out_raw","amp"),"B Hampel":(W/"out_full","amp_clean"),"C complet":(W/"out_full","amp"),"D passe-bande":(W/"out_bp","amp")}
def agg(X,y,g,fid):
    F,Y,G=[],[],[]
    for i in np.unique(fid):
        m=fid==i; x=X[m]
        F.append(np.r_[x.mean(0),x.max(0),x.std(0)]); Y.append(y[m][0]); G.append(g[m][0])
    return np.array(F),np.array(Y),np.array(G)
preds={}; 
for name,(root,key) in variants.items():
    F,Y,G=agg(*E.extract(root,key))
    # 3 graines pour stabiliser le RF
    P=np.zeros((len(Y),5))
    for seed in range(3):
        for tr,te in GroupKFold(7).split(F,Y,G):
            m=RandomForestClassifier(300,random_state=seed,n_jobs=-1).fit(F[tr],Y[tr]); P[te]+=m.predict_proba(F[te])
    pred=P.argmax(1); preds[name]=pred
    print(f"{name:10s} fichiers={len(Y)} acc={accuracy_score(Y,pred):.3f} F1={f1_score(Y,pred,average='macro'):.3f} AUC(chute)={roc_auc_score(Y==0,P[:,0]):.3f}")
    if name=="C complet": print(confusion_matrix(Y,pred))
# bootstrap par sujet de la différence d'accuracy
rng=np.random.default_rng(0); subj=np.unique(G)
def diff(a,b):
    d=[]
    for _ in range(2000):
        s=rng.choice(subj,len(subj)); idx=np.concatenate([np.where(G==k)[0] for k in s])
        d.append((preds[a][idx]==Y[idx]).mean()-(preds[b][idx]==Y[idx]).mean())
    return np.mean(d),np.percentile(d,[2.5,97.5])
for a,b in [("D passe-bande","A brut"),("D passe-bande","C complet"),("C complet","A brut"),("B Hampel","A brut")]:
    m,ci=diff(a,b); print(f"Δacc {a} - {b}: {100*m:+.1f} pts  IC95% [{100*ci[0]:+.1f}, {100*ci[1]:+.1f}]")
