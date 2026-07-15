"""
Replicates average_price_generator.py -> balanced_variance_reduction()
in pure stdlib (no pandas/numpy/scipy available in this env).

Method (matching the original, recommended params):
  Round      = ceil(overall_ADP / 12)          # round 1 = ADP 1-12, etc.
  group      = (Position, Round)
  outliers   : IQR x 1.8 removal, ONLY if group size > 4   (sensitivity 0.7)
  avg        : recency-weighted  {2023:1, 2024:2, 2025:3}  (was 2022/23/24)
  std        : recency-weighted; James-Stein shrink 0.3 toward position std if n<5
  rounding   : ceil(x*10)/10
"""
import csv, json, re, math
from pathlib import Path
from collections import defaultdict
D=Path('mfl_data'); OUT=Path('external_data/standardized')
def lst(x): return [] if x is None else (x if isinstance(x,list) else [x])
YEAR_W={2023:1.0, 2024:2.0, 2025:3.0}
IQR_MULT=1.8; SHRINK=0.3; MIN_FOR_OUTLIER=4

# ---- build merged data: (Position, Round, Dollar_Amount, Year) ----
price={}
for y in (2023,2024,2025):
    for a in lst(json.load(open(D/str(y)/'auctionResults.json'))['auctionResults'].get('auctionUnit',{}).get('auction')):
        price[(y,a['player'])]=float(a.get('winningBid') or 0)
merged=[]  # dict rows
for r in csv.DictReader(open(OUT/'adp_history.csv')):
    y=int(r['season']); mid=r['mfl_id']; pos=r['pos']
    if pos not in ('QB','RB','WR','TE') or not mid: continue
    if (y,mid) not in price: continue          # only players actually auctioned
    rnd=math.ceil(int(r['adp_rank'])/12)
    merged.append(dict(Position=pos,Round=rnd,Dollar_Amount=price[(y,mid)],Year=y,
                       Player=r['player'],ADP=int(r['adp_rank'])))
with open('external_data/merged_fantasy_data.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['Position','Round','Dollar_Amount','Year','Player','ADP']); w.writeheader(); [w.writerow(m) for m in merged]

# ---- helpers ----
def percentile(sorted_vals,p):  # numpy 'linear' default
    n=len(sorted_vals)
    if n==1: return sorted_vals[0]
    rank=p/100*(n-1); lo=math.floor(rank); frac=rank-lo
    if lo+1<n: return sorted_vals[lo]+frac*(sorted_vals[lo+1]-sorted_vals[lo])
    return sorted_vals[lo]
def sample_std(vals):
    n=len(vals)
    if n<2: return 0.0
    m=sum(vals)/n
    return math.sqrt(sum((v-m)**2 for v in vals)/(n-1))
def ceil1(x): return math.ceil(x*10)/10

# ---- outlier removal per (Position,Round) ----
groups=defaultdict(list)
for m in merged: groups[(m['Position'],m['Round'])].append(m)
cleaned=defaultdict(list)
for key,rows in groups.items():
    vals=[r['Dollar_Amount'] for r in rows]
    if len(vals)<=MIN_FOR_OUTLIER:
        cleaned[key]=rows; continue
    s=sorted(vals); q1=percentile(s,25); q3=percentile(s,75); iqr=q3-q1
    if iqr==0: cleaned[key]=rows; continue
    lo=q1-IQR_MULT*iqr; hi=q3+IQR_MULT*iqr
    cleaned[key]=[r for r in rows if lo<=r['Dollar_Amount']<=hi]

# ---- position-level std (after outlier removal) for shrinkage ----
pos_vals=defaultdict(list)
for key,rows in cleaned.items():
    for r in rows: pos_vals[key[0]].append(r['Dollar_Amount'])
pos_std={p:sample_std(v) for p,v in pos_vals.items()}

# ---- weighted stats per group ----
out=[]
for (pos,rnd),rows in cleaned.items():
    if not rows: continue
    vals=[r['Dollar_Amount'] for r in rows]; ws=[YEAR_W[r['Year']] for r in rows]
    sw=sum(ws)
    wavg=sum(w*v for w,v in zip(ws,vals))/sw
    wvar=sum(w*(v-wavg)**2 for w,v in zip(ws,vals))/sw
    wstd=math.sqrt(wvar)
    method='Standard'
    if len(vals)<5:
        wstd=(1-SHRINK)*wstd+SHRINK*pos_std.get(pos,wstd); method=f'Shrinkage_{SHRINK}'
    out.append(dict(Position=pos,Round=rnd,Avg_Dollar_Amount=ceil1(wavg),Std_Dollar_Amount=ceil1(wstd),
                    Player_Count=len(rows),Min_Price=min(vals),Max_Price=max(vals),Method_Used=method))
order={'RB':0,'WR':1,'QB':2,'TE':3}
out.sort(key=lambda r:(order.get(r['Position'],9),r['Round']))
with open(OUT/'auction_prices_by_round.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['Position','Round','Avg_Dollar_Amount','Std_Dollar_Amount','Player_Count','Min_Price','Max_Price','Method_Used'])
    w.writeheader(); [w.writerow(r) for r in out]

print(f"merged rows (auctioned players w/ ADP): {len(merged)}")
print("saved -> standardized/auction_prices_by_round.csv\n")
print(f"{'Pos':<4}{'Rd':>3}{'Avgize':>8}{'Std':>7}{'n':>4}{'min':>5}{'max':>5}  method")
for r in out:
    if r['Round']<=8:
        print(f"{r['Position']:<4}{r['Round']:>3}{r['Avg_Dollar_Amount']:>8.1f}{r['Std_Dollar_Amount']:>7.1f}{r['Player_Count']:>4}{r['Min_Price']:>5.0f}{r['Max_Price']:>5.0f}  {r['Method_Used']}")
