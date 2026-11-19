"""Prepare and freeze all data rules and partitions before baseline fitting."""
from pathlib import Path
import sys, shutil, json, hashlib
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT=Path(__file__).resolve().parents[1]
AUDIT=ROOT.parent/'Thesis_B_Audited'
for name in ['inputs/code','inputs/data/raw','data','protocol','results','models','figures','verification']:
    (ROOT/name).mkdir(parents=True,exist_ok=True)
if (ROOT/'protocol/frozen.json').exists():raise SystemExit('Frozen protocol exists. Refusing to replace it.')
for src,dest in [(AUDIT/'code/reproduce.py',ROOT/'inputs/code/reproduce.py'),(AUDIT/'code/reference_constants.json',ROOT/'inputs/code/reference_constants.json')]+[(p,ROOT/'inputs/data/raw'/p.name) for p in (AUDIT/'data/raw').glob('*.csv')]:
    shutil.copy2(src,dest)
sys.path.insert(0,str(ROOT/'inputs/code'))
import reproduce as b
r,c=b.load_data('corrected')
SPECS={
 'depth':dict(model='NN',features=b.REG_CONFIGS['lpbf_baseline'],target=b.TARGET['depth'],file='regression',rationale='NN architecture retained; pre-build eight-feature configuration excludes measured-width absorptivity. Not the paper headline configuration.'),
 'width':dict(model='GB',features=b.REG_CONFIGS['baseline+elemental'],target=b.TARGET['width'],file='regression',rationale='Thesis B headline elemental representation retained; SLM restricted.'),
 'length':dict(model='GB',features=b.REG_CONFIGS['baseline+hatch'],target=b.TARGET['length'],file='regression',rationale='Thesis B headline hatch configuration retained; SLM restricted.'),
 'classification':dict(model='RFC',features=b.CLS_CONFIGS['baseline+abs1'],target='label',file='classification',rationale='Pre-build RFC with corrected equation-2 absorption and audited static material lookup. Four-class task; spatter excluded explicitly, as in Thesis B.')}
overview=[];counts=[];classcounts=[];budgets=[];overlaps=[];splits=[];future=[];exclusions=[]
for task,s in SPECS.items():
    orig=(c if task=='classification' else r).copy()
    d=orig[orig['Sub-process'].eq('SLM')].copy()
    for col in s['features']:d[col]=pd.to_numeric(d[col],errors='coerce')
    if task!='classification':d[s['target']]=pd.to_numeric(d[s['target']],errors='coerce')
    d=d.replace([np.inf,-np.inf],np.nan)
    valid=d[s['features']+[s['target']]].notna().all(axis=1)
    for _,x in d[~valid].iterrows():
        missing=[col for col in s['features']+[s['target']] if pd.isna(x[col])]
        exclusions.append(dict(task=task,row_id=int(x.row_id),reason='missing_or_invalid:'+','.join(missing),raw_label=x.get('meltpool shape')))
    d=d[valid].copy()
    d['alloy']=d.Material.astype(str).str.strip()
    d['source']=d['paper ID'].astype(str).str.strip()
    assert d.row_id.is_unique and d.alloy.notna().all()
    d.to_csv(ROOT/'data'/f'{task}.csv',index=False)
    group=d.groupby('alloy').size()
    overview.append(dict(task=task,n=len(d),alloys=len(group),sources=d.source.nunique(),primary_alloys=int((group>=20).sum()),budget9_eval20=int((group>=29).sum()),budget20_eval20=int((group>=40).sum()),budget40_eval30=int((group>=70).sum())))
    for alloy,g in d.groupby('alloy',sort=True):
        counts.append(dict(task=task,alloy=alloy,n=len(g),sources=g.source.nunique(),source_ids='|'.join(sorted(g.source.unique())),primary=len(g)>=20))
        other=d[d.alloy.ne(alloy)]
        overlaps.append(dict(task=task,alloy=alloy,n=len(g),sources_shared_with_training=len(set(g.source)&set(other.source)),test_rows_from_shared_sources=int(g.source.isin(other.source).sum())))
        if task=='classification':
            for label,n in g.label.value_counts().items():classcounts.append(dict(alloy=alloy,label=label,n=int(n)))
        for budget,nt in [(9,20),(20,20),(40,30)]:
            budgets.append(dict(task=task,alloy=alloy,budget=budget,min_eval=nt,n=len(g),eligible=len(g)>=budget+nt))
        if len(g)>=29:
            pool,te=train_test_split(g.row_id.to_numpy(),test_size=20,random_state=20260923)
            for repeat in range(20):
                ordered=np.random.default_rng(71000+repeat).permutation(pool)
                future.append(dict(task=task,alloy=alloy,repeat=repeat,test_row_ids=sorted(map(int,te)),pool_row_ids=sorted(map(int,pool)),calibration9=list(map(int,ordered[:9])),calibration20=list(map(int,ordered[:20])) if len(pool)>=20 else None,primary_budget_comparison=len(pool)>=20))
    def make_split(protocol,seed,trpool,te,alloy=''):
        labels=d.set_index('row_id')[s['target']]
        strat=labels.loc[trpool].to_numpy() if task=='classification' else None
        # Primary data must allow class-stratified training/calibration; do not silently fall back.
        tr,cal=train_test_split(trpool,test_size=.2,random_state=1000+seed,stratify=strat)
        splits.append(dict(task=task,protocol=protocol,seed=seed,alloy=alloy,primary=protocol=='random' or len(te)>=20,train=list(map(int,tr)),calibration=list(map(int,cal)),test=list(map(int,te))))
    for seed in range(5):
        pool,te=train_test_split(d.row_id.to_numpy(),test_size=.2,random_state=seed,stratify=d[s['target']] if task=='classification' else None)
        make_split('random',seed,pool,te)
        for alloy,g in d.groupby('alloy',sort=True):
            if len(g)<2:continue
            make_split('leave_one_alloy_out',seed,d.loc[d.alloy.ne(alloy),'row_id'].to_numpy(),g.row_id.to_numpy(),alloy)
    s['parameters']=b.estimator(s['model'],task,0,'corrected').get_params()
    s['target_scaling']=task=='depth'
    s['n']=len(d)
for name,records in [('feasibility',overview),('alloy_counts',counts),('class_counts',classcounts),('calibration_budgets',budgets),('alloy_source_overlap',overlaps),('exclusions',exclusions)]:
    pd.DataFrame(records).to_csv(ROOT/'data'/f'{name}.csv',index=False)
def dump(path,obj):path.write_text(json.dumps(obj,indent=2,default=str),encoding='utf-8')
dump(ROOT/'protocol/splits.json',splits)
dump(ROOT/'protocol/future_calibration_splits.json',future)
hashes={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['inputs','data','protocol'] for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in str(p)}
dump(ROOT/'protocol/frozen.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),status='fixed_before_model_fitting',tasks=SPECS,seeds=list(range(5)),process='SLM only',random_partition='64% training /16% reserved initial calibration /20% test approximately',group_partition='exclude entire test alloy; remaining rows 80% training/20% reserved calibration',primary_group_min_n=20,small_groups='n 2 to 19: exploratory only; singleton excluded from model scoring',primary_recalibration_budgets=[9,20],repeats=20,future_test_n=20,metrics=['MAE','RMSE','R2','accuracy','macro_f1_fixed_four_classes','balanced_accuracy','confusion_matrix'],no_tuning=True,scope='Weeks 1 and 2 only; no intervals fitted, no interface built',input_hashes=hashes))
print(pd.DataFrame(overview).to_string(index=False))
print('Frozen jobs:',len(splits))
