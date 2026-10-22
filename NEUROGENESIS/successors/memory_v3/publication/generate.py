"""Generate publication figures and tables only from verified retained artifacts."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiment import analyze,read,sha,verify,write
from verify_real_outcome import verify as verify_real_outcome


NAMES={'candidate':'Memory gate','baseline':'RLS baseline','all_update':'Always-update bank','capacity_one':'Capacity 1','uniform_bank':'Uniform bank','no_bank':'No bank (RLS)','replay':'FIFO replay','knn':'FIFO kNN','frozen':'Frozen zero','last_observation':'Last observation'}
COLORS={'candidate':'#147d92','baseline':'#ab4e2d','all_update':'#6654a3','frozen':'#808080'}


def save(fig,path):
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(path.with_suffix('.png'),bbox_inches='tight',dpi=160)
    plt.close(fig)


def generate(runs,output):
    runs=Path(runs).resolve();output=Path(output);output.mkdir(parents=True,exist_ok=False)
    lock=read(runs/'freeze.json')
    confirmation=runs/'confirmation-02'
    a=analyze(confirmation,lock)
    if a['verification']['failed']:raise ValueError('failed cells cannot make a publication table')
    real_receipt=verify_real_outcome(runs/'real-transfer-03',ROOT)
    real=read(runs/'real-transfer-03/summary.json')['results']
    rows=[read(p) for p in sorted((confirmation/'attempts').glob('*/result.json'))]
    methods=list(lock['configs']);conditions=lock['protocol']['conditions']
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','pdf.fonttype':42})
    # Figure 1: all condition outcomes, no favorable-cell filtering.
    heat=np.array([[a['per_condition_mse'][c][m]/a['per_condition_mse'][c]['baseline'] for c in conditions] for m in methods])
    fig,ax=plt.subplots(figsize=(8,4.5));im=ax.imshow(np.log2(heat),cmap='RdBu_r',vmin=-1,vmax=2,aspect='auto')
    ax.set_xticks(range(len(conditions)),['Base','Novel','Delay 16','Noise .2','Distractors','16 dims','Horizon x2'],rotation=25,ha='right')
    ax.set_yticks(range(len(methods)),[NAMES[m] for m in methods])
    for i in range(len(methods)):
        for j in range(len(conditions)):ax.text(j,i,f'{heat[i,j]:.2f}',ha='center',va='center',fontsize=8,color='black')
    ax.set_title('Held-out MSE relative to RLS (1.00 = baseline; lower is better)')
    fig.colorbar(im,ax=ax,label='log2 MSE ratio',shrink=.8);save(fig,output/'condition_heatmap')
    # Figure 2: all-seed rolling prequential mean, base recurring condition.
    fig,ax=plt.subplots(figsize=(8,3.2))
    for method in ('candidate','baseline','all_update'):
        curves=[]
        for path in sorted((confirmation/'attempts').glob(f'base-*__{method}/trace.jsonl.gz')):
            e=np.array([r['squared_error'] for r in map(json.loads,gzip.open(path,'rt'))])
            cs=np.cumsum(np.r_[0,e]);curve=np.array([(cs[i+1]-cs[max(0,i-23)])/(i+1-max(0,i-23)) for i in range(len(e))]);curves.append(curve)
        ax.plot(np.mean(curves,axis=0),label=NAMES[method],color=COLORS[method],lw=1.4)
    for t in range(120,960,120):ax.axvline(t,color='#bbbbbb',ls=':',lw=.7)
    ax.set(xlabel='Prediction step (zero-based)',ylabel='24-step mean squared error',title='Fresh-seed learning curves: all 20 base-condition tasks');ax.legend(ncol=3,fontsize=8);save(fig,output/'learning_curves')
    # Figure 3: update accounting and primary-condition MSE.
    primary=lock['protocol']['primary_conditions']
    primary_rows=[r for r in rows if r['task_id'].rsplit('-',1)[0] in primary]
    table=[]
    fig,axes=plt.subplots(1,2,figsize=(9,3.5))
    for method in methods:
        subset=[r for r in primary_rows if r['method']==method]
        item={'method':method,'mse':statistics.mean(r['mse'] for r in subset),'updates':statistics.mean(r['usage']['updates'] for r in subset),'coordinate_ops':statistics.mean(r['usage']['coordinate_ops'] for r in subset),'peak_state_bytes':max(r['peak_state_bytes'] for r in subset),'wall_seconds':statistics.mean(r['wall_seconds'] for r in subset),'attempts':len(subset)}
        table.append(item)
        for ax,key in zip(axes,('updates','coordinate_ops')):
            ax.scatter(item[key],item['mse'],s=38,color=COLORS.get(method,'#7f8994'))
            if method!='no_bank':
                offset=(-52,20) if method=='capacity_one' else (6,4)
                label='RLS / no bank' if method=='baseline' else NAMES[method]
                ax.annotate(label,(item[key],item['mse']),xytext=offset,textcoords='offset points',fontsize=6,
                            arrowprops={'arrowstyle':'-','color':'#888888','lw':.5} if method=='capacity_one' else None)
    axes[0].set(xlabel='Mean weight updates',ylabel='Mean primary-condition MSE',title='Updates are not total computation')
    axes[1].set(xlabel='Counted coordinate arithmetic',ylabel='Mean primary-condition MSE',title='Reference arithmetic / accuracy tradeoff')
    save(fig,output/'efficiency')
    # Figure 4: real transfer temporal blocks; no IID confidence claim.
    fig,ax=plt.subplots(figsize=(8,3.2))
    for method in ('replay','knn','frozen','last_observation'):
        ax.plot(range(1,8),real[method]['test_block_mse_original_units'],marker='o',label=NAMES[method],color=COLORS.get(method,'#747474'))
    ax.set(xlabel='Contiguous test block',ylabel='CO MSE (original units squared)',title='Completed controls only; six RLS-based methods failed');ax.legend(fontsize=8,ncol=2);save(fig,output/'real_transfer')
    # Figure 5: exact computational flow, no generated illustration.
    fig,ax=plt.subplots(figsize=(9,2.8));ax.set_xlim(0,10);ax.set_ylim(0,3);ax.axis('off')
    boxes=[(.15,1.7,1.7,'Current features'),(2.2,1.7,2.4,'Active RLS + frozen bank'),(5,1.7,1.65,'Legal loss gate'),(7.1,1.7,1.6,'Prediction'),(6.5,.25,3.2,'Delayed label + saved outputs'),(4,.25,2.1,'Update EW credit'),(.1,.25,3.4,'RLS update or skip; FIFO archive')]
    for x,y,w,label in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,.65,boxstyle='round,pad=.04',facecolor='#eaf3f5',edgecolor='#356573'))
        ax.text(x+w/2,y+.325,label,ha='center',va='center',fontsize=8)
    for start,end in [((1.9,2.025),(2.15,2.025)),((4.65,2.025),(4.95,2.025)),((6.7,2.025),(7.05,2.025)),((7.9,1.65),(7.9,.95)),((6.45,.575),(6.15,.575)),((3.95,.575),(3.55,.575)),((1.8,.95),(3.4,1.65))]:
        ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'->','color':'#536979'})
    save(fig,output/'architecture')
    with (output/'primary_results.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    with (output/'real_results.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['method','mse_original_units','scored','test_updates'])
        for name,result in real.items():w.writerow([name,result.get('test_mse_original_units','FAILED'),result.get('test_scored',''),result.get('test_updates','')])
    p=a['primary'];effects=[100*x for x in p['ci95']]
    candidate=next(r for r in table if r['method']=='candidate');baseline=next(r for r in table if r['method']=='baseline')
    real_change=None
    lines=[r'\subsection{Frozen synthetic evaluation}',f'The completed matrix contains {len(rows):,} attempts and no failed cells. The predefined synthetic disposition is \\textbf{{{a["disposition"]}}}. Across the primary conditions, the mean paired seed-level relative MSE change is {100*p["relative_mse_change"]:.2f}\\% (95\\% bootstrap interval [{effects[0]:.2f}, {effects[1]:.2f}]\\%). Mean weight-update reduction is {100*p["mean_update_reduction"]:.2f}\\%. These quantities use the frozen aggregation rule; the table below reports pooled means over the same balanced conditions.',r'\begin{table}[ht]\centering\small\begin{tabular}{lrrrr}\toprule Method & MSE & Updates & Arithmetic & State bytes\\\midrule']
    for row in table:lines.append(f'{NAMES[row["method"]]} & {row["mse"]:.4f} & {row["updates"]:.1f} & {row["coordinate_ops"]:.0f} & {row["peak_state_bytes"]} \\\\')
    lines.extend([r'\bottomrule\end{tabular}\caption{Primary-condition means over 100 task instances per method. State is maximum numerical state, not process RSS.}\end{table}',r'\begin{figure}[ht]\centering\includegraphics[width=\linewidth]{figures/condition_heatmap.pdf}\caption{Complete condition matrix. Values are mean-MSE ratios, not paired-effect confidence bounds.}\end{figure}',r'\begin{figure}[ht]\centering\includegraphics[width=\linewidth]{figures/learning_curves.pdf}\caption{Rolling errors averaged over all 20 fresh base-condition seeds. Vertical lines mark generator changes, hidden from the learner.}\end{figure}'])
    c=a['comparisons']['all_update'];lines.append(f'Against the all-update bank, the candidate relative MSE change is {100*c["relative_mse_change"]:.2f}\\% (95\\% interval [{100*c["ci95"][0]:.2f}, {100*c["ci95"][1]:.2f}]\\%). This comparison is descriptive mechanism evidence and is distinct from the primary baseline test.')
    lines.extend([r'\begin{figure}[ht]\centering\includegraphics[width=\linewidth]{figures/efficiency.pdf}\caption{Accuracy versus actual update counts and counted arithmetic. Additional prediction work can offset gradient savings.}\end{figure}',r'\subsection{Chronological real-data transfer}',r'All ten registered methods were attempted. Six RLS-based methods failed because the covariance update lost positive definiteness: the candidate, baseline, all-update bank, capacity-one bank, uniform bank, and no-bank RLS. The candidate failed after 6,702 predictions; the baseline failed after 710, before the test boundary. Their partial errors are not substitutes for a full-transfer comparison. Four controls completed, each scoring 4,668 test targets. Real-transfer efficacy is therefore \textbf{not established}. The overall evidence disposition is \textbf{MIXED}: supported on the registered synthetic endpoint, with a numerical failure preventing the real-data comparison.',r'\begin{table}[ht]\centering\small\begin{tabular}{lrr}\toprule Method & CO MSE & Test updates\\\midrule'])
    for name,result in real.items():
        if result['status']=='COMPLETED':lines.append(f'{NAMES[name]} & {result["test_mse_original_units"]:.6f} & {result["test_updates"]} \\\\')
        else:lines.append(f'{NAMES[name]} & FAILED & --- \\\\')
    lines.extend([r'\bottomrule\end{tabular}\caption{All real-transfer methods, including a last-observation control.}\end{table}',r'\begin{figure}[ht]\centering\includegraphics[width=\linewidth]{figures/real_transfer.pdf}\caption{Seven contiguous temporal blocks are descriptive, not independent replications.}\end{figure}',r'\subsection{Evidence boundary}',r'These are executed local CPU results. Kaggle execution and independent external replication are separate release gates; this manuscript does not infer them from the presence of a notebook or package. The v1 protected evaluation remains untouched.'])
    (output/'results.tex').write_text('\n\n'.join(lines)+'\n')
    claim={'overall_disposition':'MIXED','synthetic_disposition':a['disposition'],'primary_relative_mse_change':p,'real_relative_mse_change_percent':real_change,'candidate':candidate,'baseline':baseline,'confirmation_verification':a['verification'],'real_artifact_verification':real_receipt,'input_hashes':{str(path.relative_to(ROOT)):sha(path) for path in [runs/'freeze.json',runs/'analysis.json',confirmation/'manifest.json',runs/'real-transfer-03/summary.json']}}
    write(output/'claims.json',claim)
    write(output/'manifest.json',{p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()})
    print(json.dumps({'status':'GENERATED','disposition':a['disposition'],'primary_mse_change_percent':100*p['relative_mse_change'] if isinstance(p,dict) else 100*a['primary']['relative_mse_change'],'real_mse_change_percent':real_change}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();generate(a.runs,a.output)
