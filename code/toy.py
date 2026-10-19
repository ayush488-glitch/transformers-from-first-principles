"""Toy transformer numbers for the sample chapter.
Every number printed in the PDF comes from this file (written to tex/numbers.tex).
Run: python3 code/toy.py  (fails loudly if any check breaks)."""
from fractions import Fraction as F
import math, pathlib, numpy as np

TOK = ["waste", "of", "time"]
X = np.array([[1,0,1,0],
              [0,1,0,0],
              [0,0,1,1]], dtype=float)            # (T=3, d=4)
WQ = np.eye(4)
WK = np.array([[0,0,1,1],
               [1,0,0,0],
               [0,0,1,1],
               [0,1,0,0]], dtype=float)
WV = np.array([[1,0,0,0],
               [0,0,0,1],
               [0,1,0,0],
               [0,0,1,0]], dtype=float)
T, d = X.shape
dk = 4

def attention_numpy(X):
    Q, K, V = X@WQ, X@WK, X@WV
    S = Q@K.T
    Ssc = S/np.sqrt(dk)
    mask = np.triu(np.ones((len(X),len(X))),1).astype(bool)
    Sm = np.where(mask, -np.inf, Ssc)
    A = np.exp(Sm - Sm.max(1,keepdims=True)); A /= A.sum(1,keepdims=True)
    return Q,K,V,S,Ssc,A,A@V

def attention_by_hand(X):
    """Independent pure-python version, exact fractions up to the exp."""
    Xf=[[F(int(v)) for v in r] for r in X]
    mm=lambda A,B:[[sum(A[i][k]*B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]
    WQf=[[F(int(v)) for v in r] for r in WQ]; WKf=[[F(int(v)) for v in r] for r in WK]; WVf=[[F(int(v)) for v in r] for r in WV]
    Q,K,V=mm(Xf,WQf),mm(Xf,WKf),mm(Xf,WVf)
    out=[]
    for i in range(len(Xf)):
        s=[sum(Q[i][c]*K[j][c] for c in range(dk))/2 for j in range(i+1)]   # sqrt(4)=2 exactly
        e=[math.exp(float(v)) for v in s]; z=sum(e); a=[v/z for v in e]
        out.append([sum(a[j]*float(V[j][c]) for j in range(i+1)) for c in range(dk)])
    return np.array(out)

Q,K,V,S,Ssc,A,O = attention_numpy(X)
assert np.allclose(O, attention_by_hand(X), atol=1e-12), "numpy != hand"
assert list(S[2]) == [4,0,2] and list(Ssc[2]) == [2,0,1]

# failure 1: uniform averaging
U = X.mean(0)
# failure 2 (cliffhanger): no notion of order. swap 'waste' and 'of'
Xswap = X[[1,0,2]]
_,_,_,_,_,A2,O2 = attention_numpy(Xswap)
assert np.allclose(O2[2], O[2]), "time's output should be order-blind"

# why sqrt(d): fixed-seed simulation
rng = np.random.default_rng(229)
sim = {}
for dd in (4, 64, 1024):
    dots, mx_raw, mx_sc = [], [], []
    for _ in range(4000):
        q = rng.standard_normal(dd); Kr = rng.standard_normal((8,dd))
        s = Kr@q; dots.append(s[0])
        for arr,div in ((mx_raw,1.0),(mx_sc,math.sqrt(dd))):
            z=s/div; p=np.exp(z-z.max()); p/=p.sum(); arr.append(p.max())
    sim[dd]=(np.std(dots), np.mean(mx_raw), np.mean(mx_sc))

# ---- write macros ----
def f(v,n=3): return f"{v:.{n}f}"
L=[]
def m(name,val): L.append(f"\\newcommand{{\\{name}}}{{{val}}}")
def mat(M, n=0):
    rows=[" & ".join((str(int(v)) if n==0 else f(v,n)) if np.isfinite(v) else "-\\infty" for v in r) for r in M]
    return "\\begin{bmatrix}"+" \\\\ ".join(rows)+"\\end{bmatrix}"
m("Xmat",mat(X)); m("WQmat",mat(WQ)); m("WKmat",mat(WK)); m("WVmat",mat(WV))
m("Qmat",mat(Q)); m("Kmat",mat(K)); m("Vmat",mat(V)); m("Smat",mat(S))
mask=np.triu(np.ones((3,3)),1).astype(bool)
m("SmaskMat",mat(np.where(mask,-np.inf,Ssc)))
m("Amat",mat(A,3)); m("Omat",mat(O,3))
for j,t in enumerate(["Waste","Of","Time"]):
    m(f"a{t}",f(A[2,j])); m(f"apct{t}",f(100*A[2,j],1)); m(f"s{t}",str(int(S[2,j]))); m(f"ss{t}",str(int(Ssc[2,j])))
    m(f"e{t}",f(math.exp(Ssc[2,j]),3))
m("oOf","("+", ".join(f(v) for v in O[1])+")")
m("eSum",f(sum(math.exp(v) for v in Ssc[2]),3))
m("oTime","("+", ".join(f(v) for v in O[2])+")")
m("uTime","("+", ".join(f(v) for v in U)+")")
m("oTimeSwap","("+", ".join(f(v) for v in O2[2])+")")
for dd,(sd,mr,ms) in sim.items():
    k={4:"Four",64:"SixtyFour",1024:"Thousand"}[dd]
    m(f"sd{k}",f(sd,2)); m(f"mxRaw{k}",f(mr,3)); m(f"mxSc{k}",f(ms,3)); m(f"sqrt{k}",f(math.sqrt(dd),1))

# the mask makes earlier rows blind to later tokens: change 'time' and rows 0,1 must not move
Xchg = X.copy(); Xchg[2] = [5,-3,2,7]
assert np.allclose(attention_numpy(Xchg)[-1][:2], O[:2])
# ---- geometry of the dot product (2-D picture) ----
a2, b2 = np.array([3.,1.]), np.array([1.,2.])
dot2 = a2@b2; na, nb = np.linalg.norm(a2), np.linalg.norm(b2)
ang2 = math.degrees(math.acos(dot2/(na*nb)))
assert dot2 == 5 and abs(ang2-45) < 1e-9
m("geoDot", str(int(dot2))); m("geoNa", f(na)); m("geoNb", f(nb)); m("geoAng", f"{ang2:.0f}")
m("geoShadow", f(dot2/na))   # length of b's shadow on a
# angles between time's question and every key
qt = Q[2]
for j,t in enumerate(["Waste","Of","Time"]):
    c = qt@K[j]/(np.linalg.norm(qt)*np.linalg.norm(K[j]))
    m(f"cos{t}", f(c)); m(f"ang{t}", f"{math.degrees(math.acos(np.clip(c,-1,1))):.1f}")
    m(f"len{t}", f(np.linalg.norm(K[j])))
m("lenQ", f(np.linalg.norm(qt)))
# softmax does not care if every score moves by the same amount
z = Ssc[2]; p1 = np.exp(z)/np.exp(z).sum(); p2 = np.exp(z+10)/np.exp(z+10).sum()
assert np.allclose(p1,p2)
# a longer query arrow = a more decisive softmax (same direction, double length)
z2 = (2*Q[2])@K.T/np.sqrt(dk); p3 = np.exp(z2)/np.exp(z2).sum()
m("dblScores", "("+", ".join(str(int(v)) for v in z2)+")")
for j,t in enumerate(["Waste","Of","Time"]): m(f"dbl{t}", f(p3[j]))
# ---- a batch of two sentences, processed in one go ----
TOK2 = ["not","very","useful"]
X2 = np.array([[0,1,0,1],[1,0,0,0],[0,0,1,0]], dtype=float)
B = np.stack([X, X2])                          # (B=2, T=3, d=4)
Qb, Kb, Vb = B@WQ, B@WK, B@WV                  # (2,3,4) each
Sb = Qb@Kb.transpose(0,2,1)/np.sqrt(dk)        # (2,3,3)
Sb = np.where(np.triu(np.ones((3,3)),1).astype(bool), -np.inf, Sb)
Ab = np.exp(Sb-Sb.max(-1,keepdims=True)); Ab /= Ab.sum(-1,keepdims=True)
Ob = Ab@Vb
assert np.allclose(Ob[0], O)                   # sentence 1 unchanged by batching
for b_ in range(2):                            # and each sentence alone gives the same answer
    assert np.allclose(Ob[b_], attention_numpy(B[b_])[-1])
m("XtwoMat", mat(X2)); m("AtwoMat", mat(Ab[1],3)); m("StwoMat", mat(Qb[1]@Kb[1].T))
m("kWasteRow","("+", ".join(str(int(v)) for v in K[0])+")")
m("Scells", ",".join(f"{i}/{j}/{int(S[i,j])}" for i in range(3) for j in range(3)))
m("nScores", str(Sb.size)); m("nPreds", str(B.shape[0]*B.shape[1]))

# bar heights for pgfplots
m("barCoords"," ".join(f"({t},{f(A[2,j],4)})" for j,t in enumerate(TOK)))
open(pathlib.Path(__file__).resolve().parent.parent/"tex"/"num_toy.tex","w").write("% AUTO-GENERATED by code/toy.py -- do not edit\n"+"\n".join(L)+"\n")
print("ok", A[2], O[2], {k:tuple(round(x,3) for x in v) for k,v in sim.items()})
