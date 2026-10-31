import json, glob
from scipy.stats import spearmanr
P=[("ahc-vs-d48-g4",0.048),("ahc-vs-d48-s1",0.034),("ahc-hcrwsig-vs-d48",0.044),("ahc-hcs8-vs-d48",0.049),("ahc-hcs2-vs-d48",0.074),("amhc-vs-d48-g4",0.102),("amhc-mhcid-vs-d48",0.100),("amhc-mhcrwlin-vs-d48",0.080)]
def e(f):
    d=json.load(open(f))["dce"]; v=[x for k,x in d.items() if int(k)<=16]; return sum(v)/len(v), d
for kind in ("sep","sep-bbfrozen"):
    xs=[];ms=[];cs=[]
    for t,c in P:
        s,ds=e(f"has-{kind}-{t}.json"); m,_=e(f"ha-{kind}-{t}.json")
        xs.append(s);ms.append(m);cs.append(c)
        print(kind,t,c,"shuffle %.3f mean %.3f"%(s,m), "late shuffle", {k:round(v,3) for k,v in ds.items() if int(k)>=34})
    print(kind,"spearman shuffle(8)",spearmanr(xs,cs).correlation,"with 1-stream (10)",spearmanr(xs+[0,0],cs+[0.125,0.089]).correlation,"mean(10)",spearmanr(ms+[0,0],cs+[0.125,0.089]).correlation)
