"""Reproducible RAVDESS / personal speech emotion comparison.
Run: python run_experiment.py --ravdess ../data/ravdess --personal ../personal
Personal files must be named angry.wav, calm.wav, disgust.wav, etc.
"""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import pandas as pd
import librosa
import sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

EMOTIONS = dict(zip(range(1, 9), ['neutral','calm','happy','sad','angry','fearful','disgust','surprised']))
LABELS = sorted(EMOTIONS.values())

def get_features(path):
    # Same rate and channel handling for both datasets; no normalization or silence trimming.
    waveform, sr = librosa.load(path, sr=22050, mono=True)
    spectrum = np.abs(librosa.stft(waveform))
    chroma = librosa.feature.chroma_stft(S=spectrum, sr=sr).mean(axis=1)
    mel = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=128, fmax=8000).mean(axis=1)
    mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=40).mean(axis=1)
    result = np.hstack([chroma, mel, mfcc])
    if result.shape != (180,) or not np.isfinite(result).all():
        raise ValueError(f'Invalid features: {path}')
    return result

def summarize(X, dataset):
    rows=[]
    for name, lo, hi in [('Chroma',0,12),('Mel',12,140),('MFCC',140,180)]:
        v=X[:,lo:hi]
        rows.append(dict(dataset=dataset,group=name,min=float(v.min()),max=float(v.max()),mean=float(v.mean()),std=float(v.std())))
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--ravdess',default='data/ravdess')
    ap.add_argument('--personal',default='personal')
    ap.add_argument('--output',default='results')
    ap.add_argument('--jobs',type=int,default=4)
    args=ap.parse_args()
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    files=sorted(Path(args.ravdess).rglob('*.wav'))
    if not files: raise ValueError('No RAVDESS WAV files found')
    cache=out/'ravdess_features.npz'
    filenames=[str(p.relative_to(args.ravdess)) for p in files]
    if cache.exists():
        saved=np.load(cache)
        if saved['filenames'].tolist()!=filenames: raise ValueError('Cache does not match dataset')
        X,y=saved['X'],saved['y']
    else:
        print(f'Extracting {len(files)} RAVDESS samples',flush=True)
        X=np.array(Parallel(n_jobs=args.jobs,verbose=10)(delayed(get_features)(str(p)) for p in files))
        y=np.array([EMOTIONS[int(p.stem.split('-')[2])] for p in files])
        np.savez_compressed(cache,X=X,y=y,filenames=filenames)
    personal=sorted(Path(args.personal).glob('*.wav'))
    yp=np.array([p.stem.lower().replace('disgusted','disgust') for p in personal])
    if len(personal)!=8 or set(yp)!=set(LABELS): raise ValueError('Require one personal WAV for each of eight emotions')
    Xp=np.array([get_features(str(p)) for p in personal])
    pd.DataFrame(X).to_csv(out/'featuresRavdess.csv',index=False)
    pd.DataFrame({'emotion':y}).to_csv(out/'emotionsRavdess.csv',index=False)
    pd.DataFrame(Xp).to_csv(out/'personal_features.csv',index=False)
    pd.DataFrame(summarize(X,'RAVDESS')+summarize(Xp,'Personal')).to_csv(out/'feature_summary.csv',index=False)
    train,test=train_test_split(np.arange(len(y)),test_size=.2,random_state=42,stratify=y)
    pd.DataFrame({'file':filenames,'emotion':y,'split':np.where(np.isin(np.arange(len(y)),test),'test','train')}).to_csv(out/'split_manifest.csv',index=False)
    constructors={
        'SVM RBF':lambda:SVC(C=3,gamma='scale',kernel='rbf'),
        'KNN':lambda:KNeighborsClassifier(n_neighbors=5,weights='distance'),
        'Random Forest':lambda:RandomForestClassifier(n_estimators=300,random_state=42,n_jobs=args.jobs),
        'Logistic Regression':lambda:LogisticRegression(C=1,max_iter=10000,solver='lbfgs'),
    }
    rows=[]; predictions=[]
    for name,ctor in constructors.items():
        for scale in [False,True]:
            model=make_pipeline(StandardScaler(),ctor()) if scale else ctor()
            model.fit(X[train],y[train])
            for dataset,xx,yy in [('RAVDESS',X[test],y[test]),('Personal',Xp,yp)]:
                pred=model.predict(xx)
                precision,recall,f1,_=precision_recall_fscore_support(yy,pred,labels=LABELS,average='macro',zero_division=0)
                rows.append(dict(model=name,scaling='Standard' if scale else 'None',dataset=dataset,n=len(yy),accuracy=accuracy_score(yy,pred),macro_precision=precision,macro_recall=recall,macro_f1=f1,training_accuracy=accuracy_score(y[train],model.predict(X[train]))))
                if dataset=='Personal':
                    predictions.extend(dict(model=name,scaling='Standard' if scale else 'None',file=p.name,true=t,predicted=q) for p,t,q in zip(personal,yy,pred))
                pd.DataFrame(confusion_matrix(yy,pred,labels=LABELS),index=LABELS,columns=LABELS).to_csv(out/f"cm_{name.replace(' ','_')}_{scale}_{dataset}.csv")
            print(name,'scaled' if scale else 'unscaled',rows[-2]['accuracy'],rows[-1]['accuracy'],flush=True)
    df=pd.DataFrame(rows); df.to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(predictions).to_csv(out/'personal_predictions.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(10,3.3),sharey=True)
    for ax,dataset in zip(axes,['RAVDESS','Personal']):
        subset=df[df.dataset==dataset]
        for shift,scaling,color in [(-.18,'None','#98a5b5'),(.18,'Standard','#206b9b')]:
            vals=[subset[(subset.model==m)&(subset.scaling==scaling)].accuracy.iloc[0]*100 for m in constructors]
            bars=ax.bar(np.arange(4)+shift,vals,width=.36,label=scaling,color=color)
            ax.bar_label(bars,fmt='%.1f',fontsize=8)
        ax.set_xticks(range(4),['SVM','KNN','Forest','Logistic']);ax.set_ylim(0,100);ax.set_title(f'{dataset} test (n={len(test) if dataset=="RAVDESS" else 8})');ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Accuracy (%)');axes[1].legend(title='Scaling',fontsize=8)
    fig.tight_layout();fig.savefig(out/'accuracy_comparison.png',dpi=180);plt.close(fig)
    metadata=dict(ravdess_n=len(y),train_n=len(train),test_n=len(test),personal_n=len(yp),rate=22050,features=180,random_state=42,versions=dict(librosa=librosa.__version__,sklearn=sklearn.__version__,numpy=np.__version__),method='stratified file-level 80/20 split; shared actors possible; scalers fitted on training only')
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    print(df.to_string(index=False),flush=True)

if __name__=='__main__': main()
