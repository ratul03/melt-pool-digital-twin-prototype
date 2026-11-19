"""Audited, portable Thesis B benchmark. No Thesis C methods.

historical reproduces the submitted implementation. corrected recalculates physics
features and quarantines the eleven documented suspect rows (does not invent units).
Abs2 models remain diagnostic/post-build, including in corrected mode.
Every seed has saved test row IDs and predictions; seed 0 has a reload-checked model.
"""
from pathlib import Path
import argparse, json, hashlib, time, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, roc_auc_score, f1_score
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
CONSTANTS=json.loads((ROOT/'code/reference_constants.json').read_text())
BASE=['Power','Velocity','density','Cp','k','melting T']
PROC=['proc_PBF','proc_DED']
ELEMENTAL=['elem_atomic_number','elem_atomic_volume','elem_ionization_energy','elem_heat_of_fusion','elem_electron_affinity']
TARGET={'depth':'depth of meltpool','width':'width of melt pool','length':'length of melt pool'}
CLS_BASE=['Power','Velocity','beam D','density','Cp','k','melting T']
LABELS=['LOF','balling','desirable','keyhole']
REG_CONFIGS={
 'baseline':BASE+PROC, 'baseline+hatch':BASE+PROC+['Hatch spacing'],
 'baseline+elemental':BASE+PROC+ELEMENTAL,
 'lpbf_baseline':BASE+['beam D','layer thickness'],
 'lpbf+abs1':BASE+['beam D','layer thickness','absorptivity_1'],
 'lpbf+abs2':BASE+['beam D','layer thickness','absorptivity_2'],
 'lpbf+abs1+abs2':BASE+['beam D','layer thickness','absorptivity_1','absorptivity_2'],
 'lpbf+abs2+hatch':BASE+['beam D','layer thickness','absorptivity_2','Hatch spacing']}
CLS_CONFIGS={'baseline':CLS_BASE,'baseline+abs1':CLS_BASE+['absorptivity_1'],'baseline+abs1+hatch':CLS_BASE+['absorptivity_1','Hatch spacing']}
SUSPECT_IDS=list(range(580,583))+list(range(837,845))

def numeric(df,col):return pd.to_numeric(df[col],errors='coerce')

def abs1(df,eta,T0=298.):
    p,v,rho,cp,t,d=[numeric(df,c) for c in ['Power','Velocity','density','Cp','melting T','beam D']]
    x=.6*eta*p/((t-T0)*np.pi*rho*cp*(v/1000)*(d/2e6)**2)
    with np.errstate(invalid='ignore',divide='ignore',over='ignore'):
        value=-.7*np.expm1(-x)
    return value.where((p>0)&(v>0)&(rho>0)&(cp>0)&(t>T0)&(d>0)&(eta>=0)&(eta<=1))

def abs2(df,T0=298.):
    p,v,rho,cp,t,k,w=[numeric(df,c) for c in ['Power','Velocity','density','Cp','melting T','k','width of melt pool']]
    w=w/1e6
    return (np.pi*k*(t-T0)*w+np.e*np.pi*rho*cp*(t-T0)*(v/1000)*w*w/8)/p

def load_data(mode):
    r=pd.read_csv(ROOT/'data/raw/meltpoolnet_regression.csv')
    c=pd.read_csv(ROOT/'data/raw/meltpoolnet_classification.csv')
    r['row_id']=r.index;c['row_id']=c.index
    props=CONSTANTS['ELEMENT_PROPS'];cols=[];mat=[]
    for el,values in props.items():
        col=next(x for x in [f'{el} (wt%)',f'{el} (wt.%)'] if x in r)
        cols.append(col);mat.append(values)
    # Historical paper-style composition approximation; missing fractions are zero.
    mix=r[cols].apply(pd.to_numeric,errors='coerce').fillna(0).to_numpy()/100@np.asarray(mat)
    for i,name in enumerate(ELEMENTAL):r[name]=mix[:,i]
    for process in ['PBF','DED']:r['proc_'+process]=r.Process.astype(str).str.strip().eq(process).astype(int)
    r['energy_density_linear']=numeric(r,'Power')/numeric(r,'Velocity')
    h=numeric(r,'Hatch spacing')/1000;t=numeric(r,'layer thickness')/1000
    r['energy_density_volumetric']=(numeric(r,'Power')/(numeric(r,'Velocity')*h*t)).where((h>0)&(t>0))
    r['absorptivity_1_calc']=abs1(r,numeric(r,'minimum absorptivity'))
    r['absorptivity_2_calc']=abs2(r)
    c['label']=c['meltpool shape'].astype(str).str.strip().where(lambda x:x.isin(LABELS))
    if mode=='historical':
        r['absorptivity_1']=numeric(r,'absorption coefficient')
        r['absorptivity_2']=numeric(r,'absorption coefficient 2').fillna(r.absorptivity_2_calc)
        c['absorptivity_1']=numeric(c,'absorption coefficient')
    else:
        # Static material lookup from regression input properties, never from target labels.
        lookup={m:float(g.dropna().iloc[0]) for m,g in r.groupby('Material')['minimum absorptivity'] if len(g.dropna()) and g.dropna().nunique()==1}
        c['eta_m_material']=c.Material.map(lookup)
        c['absorptivity_1']=abs1(c,c.eta_m_material)
        r['absorptivity_1']=r.absorptivity_1_calc
        r['absorptivity_2']=r.absorptivity_2_calc
        r=r[~r.row_id.isin(SUSPECT_IDS)].copy()
    return r,c

def jobs():
    for model,configs in [('GB',list(REG_CONFIGS)),('NN',['baseline','baseline+hatch','lpbf+abs2','lpbf+abs2+hatch']),('RF',['baseline','lpbf+abs2+hatch'])]:
        for target in TARGET:
            for config in configs:yield model,target,config
    for config in CLS_CONFIGS:yield 'RFC','classification',config

def slug(model,target,config):return f'{model}_{target}_{config.replace("+","_")}'

def estimator(model,target,seed,mode):
    if model=='GB':return GradientBoostingRegressor(n_estimators=CONSTANTS['GB_N_ESTIMATORS'][target],random_state=seed)
    if model=='RF':return RandomForestRegressor(n_estimators=500,random_state=seed,n_jobs=1)
    if model=='RFC':return RandomForestClassifier(n_estimators=160,random_state=seed,n_jobs=1)
    hidden,alpha=CONSTANTS['NN_PARAMS'][target]
    return MLPRegressor(hidden_layer_sizes=tuple(hidden),alpha=alpha,activation='relu',solver='adam',max_iter=3000 if mode=='historical' else 20000,random_state=seed)

def run_job(mode,model,target,config,r,c,overwrite=False):
    out=ROOT/'results'/mode; out.mkdir(parents=True,exist_ok=True)
    name=slug(model,target,config); done=out/(name+'.json')
    if done.exists() and not overwrite:
        raise FileExistsError(f'{done} already exists. Use --overwrite to deliberately regenerate this mode; existing results are never silently reused.')
    iscls=model=='RFC'; feats=(CLS_CONFIGS if iscls else REG_CONFIGS)[config];tc='label' if iscls else TARGET[target]
    d=c.copy() if iscls else r.copy()
    # Historic "lpbf" actually selects all PBF; corrected mode explicitly selects SLM.
    if not iscls and config.startswith('lpbf'):
        d=d[d.Process.eq('PBF') if mode=='historical' else d['Sub-process'].eq('SLM')]
    d[feats]=d[feats].apply(pd.to_numeric,errors='coerce')
    if not iscls:d[tc]=pd.to_numeric(d[tc],errors='coerce')
    d=d.replace([np.inf,-np.inf],np.nan).dropna(subset=feats+[tc])
    X=d[feats].to_numpy(dtype=float);y=d[tc].to_numpy();ids=d.row_id.to_numpy()
    if len(y)<60: return None
    runs=[];predictions=[];splits=[]
    t0=time.time()
    for seed in range(5):
        tr,te=train_test_split(np.arange(len(d)),test_size=.2,random_state=seed,stratify=y if iscls else None)
        assert not set(ids[tr])&set(ids[te])
        sc=StandardScaler().fit(X[tr]);ysc=None; yt=y[tr]
        if model=='NN':ysc=StandardScaler().fit(yt.reshape(-1,1));yt=ysc.transform(yt.reshape(-1,1)).ravel()
        m=estimator(model,target,seed,mode)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');m.fit(sc.transform(X[tr]),yt)
        p=m.predict(sc.transform(X[te]))
        if ysc is not None:p=ysc.inverse_transform(p.reshape(-1,1)).ravel()
        assert np.isfinite(p).all() if not iscls else len(p)==len(te)
        row=dict(seed=seed,n_train=len(tr),n_test=len(te),warnings=[str(w.message) for w in caught],n_iter=getattr(m,'n_iter_',None))
        if iscls:
            proba=m.predict_proba(sc.transform(X[te]))
            row.update(accuracy=accuracy_score(y[te],p),auc=roc_auc_score(y[te],proba,multi_class='ovr',average='macro',labels=m.classes_),macro_f1=f1_score(y[te],p,average='macro'))
        else:row.update(R2=r2_score(y[te],p),MAE=mean_absolute_error(y[te],p),negative_predictions=int((p<0).sum()))
        runs.append(row)
        for j,i in enumerate(te):
            record=dict(seed=seed,row_id=int(ids[i]),actual=y[i],predicted=p[j])
            if iscls:record.update({'prob_'+str(cl):float(proba[j,k]) for k,cl in enumerate(m.classes_)})
            predictions.append(record)
        splits.append(dict(seed=seed,train_row_ids=ids[tr].tolist(),test_row_ids=ids[te].tolist()))
        if seed==0:
            mdir=ROOT/'models'/mode;mdir.mkdir(parents=True,exist_ok=True)
            usage='post_build_diagnostic' if 'abs2' in config else ('unresolved_supplied_feature' if mode=='historical' and 'abs1' in config else 'prebuild_research_baseline')
            artifact=dict(model=m,scaler=sc,y_scaler=ysc,features=feats,target=tc,config=config,mode=mode,split_seed=seed,train_row_ids=ids[tr].tolist(),test_row_ids=ids[te].tolist(),usage=usage,raw_data_sha256=hashlib.sha256((ROOT/'data/raw'/('meltpoolnet_classification.csv' if iscls else 'meltpoolnet_regression.csv')).read_bytes()).hexdigest(),physical_guarantee=False)
            joblib.dump(artifact,mdir/(name+'.joblib'),compress=3)
            loaded=joblib.load(mdir/(name+'.joblib'));p2=loaded['model'].predict(loaded['scaler'].transform(X[te]))
            if ysc is not None:p2=loaded['y_scaler'].inverse_transform(p2.reshape(-1,1)).ravel()
            assert np.array_equal(p,p2),'Reload predictions differ'
    summary=dict(model=model,target=target,config=config,n=len(d),n_runs=5,mode=mode,features=feats,seconds=round(time.time()-t0,2),runs=runs)
    for metric in ['accuracy','auc','macro_f1'] if iscls else ['R2','MAE']:
        summary[metric+'_mean']=float(np.mean([x[metric] for x in runs]));summary[metric+'_std']=float(np.std([x[metric] for x in runs],ddof=1))
    pd.DataFrame(predictions).to_csv(out/(name+'_predictions.csv'),index=False)
    (out/(name+'_splits.json')).write_text(json.dumps(splits),encoding='utf-8')
    done.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(mode,name,len(d),{k:round(v,5) for k,v in summary.items() if k.endswith('_mean')},'seconds',summary['seconds'],flush=True)
    return summary

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['historical','corrected'],required=True);ap.add_argument('--models',default='GB,RF,RFC,NN');ap.add_argument('--overwrite',action='store_true');a=ap.parse_args()
    r,c=load_data(a.mode)
    r.to_csv(ROOT/f'data/processed/{a.mode}_regression.csv',index=False);c.to_csv(ROOT/f'data/processed/{a.mode}_classification.csv',index=False)
    with threadpool_limits(limits=1):
        for model,target,config in jobs():
            if model in a.models.split(','):run_job(a.mode,model,target,config,r,c,overwrite=a.overwrite)
    records=[json.loads(f.read_text()) for f in (ROOT/'results'/a.mode).glob('*.json') if not f.name.endswith('_splits.json')]
    pd.DataFrame([{k:v for k,v in x.items() if k not in ['runs','features']} for x in records]).to_csv(ROOT/'results'/a.mode/'summary.csv',index=False)

if __name__=='__main__':main()
