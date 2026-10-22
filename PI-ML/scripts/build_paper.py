"""Generate manuscript numbers, tables, figures and claim links from saved evidence."""
import argparse
import hashlib
import json
from pathlib import Path


LABELS = {'direct':'Direct Thomas','zero_jacobi':'Zero + Jacobi', 'quadratic_jacobi':'Quadratic + Jacobi',
          'zero_laplace':'Zero + Laplacian','physics_laplace':'Physics + Laplacian',
          'quadratic_laplace':'Quadratic + Laplacian','linear_laplace':'Degree 1 + Laplacian',
          'structured_laplace':'Structured + Laplacian','shuffled_laplace':'Shuffled + Laplacian'}


def generate(development, confirmation, paper):
    dev = json.loads((development/'s2-development/analysis.json').read_text())
    run = confirmation/'s2-confirmatory'
    analysis = json.loads((run/'analysis.json').read_text())
    config = json.loads((run/'config.json').read_text())
    receipt = json.loads((run/'receipt.json').read_text())
    env = json.loads((run/'environment.json').read_text())
    s1 = json.loads((confirmation/'s1-confirmatory/analysis.json').read_text())
    checks = json.loads((run/'verification.json').read_text())
    if checks['status']!='PASS':raise ValueError('Refusing unverified evidence')
    gate=config['primary'];grid=gate['grid'];tol=gate['rtol']
    rows=[s for s in analysis['summaries'] if s['grid']==grid and s['rtol']==tol and s['regime']=='iid']
    comp=[c for c in analysis['comparisons'] if c['grid']==grid and c['rtol']==tol and c['regime']=='iid' and c['proposed']=='structured_laplace']
    by={c['baseline']:c for c in comp};means={r['method']:r for r in rows}
    zero,physics=by['zero_laplace'],by['physics_laplace']
    status=analysis['disposition'];percent=lambda x:f'{100*x:.2f}'
    ci=lambda c:f"[{percent(c['ci95'][0])}, {percent(c['ci95'][1])}]"
    ratio=means['structured_laplace']['time_ms_mean']/means['direct']['time_ms_mean']
    primary_n=means['structured_laplace']['n']
    s1iid=s1['paired_primary']['iid_test'];s1ood=s1['paired_primary']['ood_test']
    all_failures=sum(s['failures'] for s in analysis['summaries'])
    paper.mkdir(parents=True,exist_ok=True);(paper/'figures').mkdir(exist_ok=True)
    (paper/'abstract.tex').write_text(r'''\begin{abstract}
Learned initial guesses can improve prediction error without reducing accepted numerical correction work. We study this distinction in synthetic one-dimensional elliptic systems, preserving a failed quadratic-ensemble predecessor and evaluating a separately frozen source-linear successor. The original strict-tolerance confirmatory iteration reduction is '''+percent(s1iid['mean'])+r'''\% on IID cases. In the successor's predeclared 129-point, $10^{-6}$-tolerance IID cell, five independent training/evaluation blocks yield '''+percent(zero['iteration_reduction'])+r'''\% mean paired iteration reduction against zero-initialized Laplacian PCG and '''+percent(physics['iteration_reduction'])+r'''\% against a physics-only initial field. The corresponding descriptive block-bootstrap intervals are '''+ci(zero)+r'''\% and '''+ci(physics)+r'''\%. Complete online time improves by '''+percent(zero['online_time_reduction'])+r'''\% and '''+percent(physics['online_time_reduction'])+r'''\%, respectively. The frozen primary rule is '''+('satisfied' if status=='SUPPORTED' else 'not satisfied')+r'''. However, direct tridiagonal solving remains '''+f'{ratio:.2f}'+r''' times faster by the ratio of mean times. The full successor matrix contains '''+str(receipt['metric_rows'])+r''' condition-level rows, with '''+str(all_failures)+r''' acceptance failures. These findings support only the stated iterative-solver comparison; they do not establish a new solver family, multidimensional superiority, or independent investigator reproduction.
\end{abstract}
''')
    table=[r'\begin{tabular}{lrrr}',r'\toprule',r'Method & Mean iterations & Mean time (ms) & Mean rel. error \\',r'\midrule']
    for r in rows:
        table.append(f"{LABELS[r['method']]} & {r['iterations_mean']:.3f} & {r['time_ms_mean']:.3f} & {r['relative_error_mean']:.2e} \\\\")
    table.extend([r'\bottomrule',r'\end{tabular}'])
    (paper/'primary_table.tex').write_text('\n'.join(table)+'\n')
    comparison=[r'\begin{tabular}{lrrr}',r'\toprule',r'Baseline & Iteration reduction & 95\% block interval & Online saving \\',r'\midrule']
    for c in comp:
        effect='--' if c['iteration_reduction'] is None else percent(c['iteration_reduction'])+r'\%'
        bounds='--' if c['ci95'] is None else ci(c)+r'\%'
        comparison.append(f"{LABELS[c['baseline']]} & {effect} & {bounds} & {percent(c['online_time_reduction'])}\\% \\\\")
    comparison.extend([r'\bottomrule',r'\end{tabular}'])
    (paper/'comparison_table.tex').write_text('\n'.join(comparison)+'\n')
    regimes=[]
    for regime,label in [('coefficient_ood','coefficient shift'),('forcing_ood','forcing shift')]:
        c=next(c for c in analysis['comparisons'] if c['grid']==grid and c['rtol']==tol and c['regime']==regime and c['proposed']=='structured_laplace' and c['baseline']=='physics_laplace')
        regimes.append(f"Under the {label}, the paired iteration reduction versus the physics baseline is {percent(c['iteration_reduction'])}\\%, with online saving {percent(c['online_time_reduction'])}\\%.")
    strict=next(c for c in analysis['comparisons'] if c['grid']==grid and c['rtol']==1e-10 and c['regime']=='iid' and c['proposed']=='structured_laplace' and c['baseline']=='physics_laplace')
    results=r'''\section{Results}
\subsection{S1: preserved failure and prospective confirmation}
The original S1 development result is unchanged: all 64 IID and 64 coefficient-OOD cases require 63 PCG iterations under both learned and zero starts. The diagnostic tolerance sweep finds an IID benefit of 12.86\% at $10^{-2}$, but an OOD change of $-4.26$\%; the advantage disappears or reverses at more stringent settings. This diagnostic is not substituted for the frozen endpoint.

The original frozen S1 confirmation yields '''+percent(s1iid['mean'])+r'''\% IID and '''+percent(s1ood['mean'])+r'''\% coefficient-OOD mean iteration reduction. Its recorded disposition is \texttt{'''+s1['decision']['disposition']+r'''}. The source-linear successor is therefore not presented as a successful rerun of S1: it changes the predictor, preconditioning controls and evaluation protocol and uses disjoint data.

\subsection{S2: primary outcome and cost}
Table~\ref{tab:primary} reports the primary cell, with '''+str(primary_n)+r''' paired cases across five independent training blocks. Table~\ref{tab:comparisons} gives paired effect estimates; a mean of paired ratios differs from a ratio of mean times. The predeclared joint iteration rule is '''+('satisfied' if status=='SUPPORTED' else 'not satisfied')+r''', producing \texttt{'''+status+r'''} for this narrow primary hypothesis. All '''+str(receipt['metric_rows'])+r''' successor condition-level rows are retained; '''+str(all_failures)+r''' fail true-residual acceptance.
\begin{table}[ht]
\centering\small\input{primary_table}
\caption{Frozen S2 primary cell: IID, 129 grid points, relative tolerance $10^{-6}$. Online time includes assembly, inference, setup, correction and residual certification. Training and loading are excluded.}\label{tab:primary}
\end{table}
\begin{table}[ht]
\centering\small\input{comparison_table}
\caption{Structured predictor plus Laplacian PCG versus each listed baseline. Negative online saving means slower. Direct solving has no meaningful relative-iteration metric. Five-block bootstrap intervals have limited resolution.}\label{tab:comparisons}
\end{table}
\begin{figure}[ht]
\centering\includegraphics[width=\linewidth]{figures/primary.pdf}
\caption{Accepted correction work and complete online time. The classical preconditioner changes the scale of correction work; direct tridiagonal solving remains the fastest measured path.}\label{fig:primary}
\end{figure}

The direct method is '''+f'{ratio:.2f}'+r''' times faster than the proposed method by the ratio of mean online times. Therefore the evidence does not support replacing direct 1D solving with this learned hybrid for throughput. The positive iterative comparison remains meaningful only when the permitted baseline class is iterative and the offline learning cost is separately justified.

\subsection{Ablation, sensitivity and distribution shifts}
The degree-one and shuffled-label controls appear in Table~\ref{tab:primary}. Both use the same PCG acceptance implementation. The shuffled control preserves the proposed feature count and fitting procedure while removing the input/target pairing. The degree ablation measures the additional coefficient-quadratic features; the unconstrained quadratic comparison also changes parameter count and cannot isolate structure alone.

'''+ '\n'.join(regimes)+r'''
At the stricter IID tolerance $10^{-10}$, the iteration reduction against the physics baseline is '''+percent(strict['iteration_reduction'])+r'''\%. These are descriptive checks, not replacements for the primary endpoint. Figure~\ref{fig:matrix} shows every grid/tolerance/regime comparison against both classical iterative starts; the appendix retains all method-level summaries.
\begin{figure}[ht]
\centering\includegraphics[width=.95\linewidth]{figures/matrix.pdf}
\caption{All S2 cells. Columns distinguish paired iteration and online-time reductions against zero and physics Laplacian starts. Negative values are retained.}\label{fig:matrix}
\end{figure}
\begin{figure}[ht]
\centering\includegraphics[width=\linewidth]{figures/tolerance_scaling.pdf}
\caption{Left: the development-only S1 tolerance diagnostic. Right: IID S2 iteration scaling at $10^{-6}$ with separately fitted models per grid. The two panels concern different, explicitly versioned studies.}\label{fig:sensitivity}
\end{figure}

\subsection{Resources and evidence verification}
The S2 producer records '''+f"{receipt['cpu_seconds']:.2f}"+r''' CPU seconds, '''+f"{receipt['wall_seconds']:.2f}"+r''' wall seconds, and peak RSS '''+f"{receipt['max_rss_bytes']/2**20:.2f}"+r''' MiB before the subsequent verifier. These producer measurements exclude packaging, remote scheduling and later artifact replay. The runtime is Python '''+env['python'].split()[0]+r''' and NumPy '''+env['numpy']+r''' on Kaggle CPU with one BLAS/OpenMP thread. Offline fit and training-data generation measurements are retained per block, grid and model in \texttt{offline\_costs.json}; they are not hidden inside an assumed amortization.
'''
    (paper/'results.tex').write_text(results)
    discussion=(paper/'discussion.tex').read_text()
    discussion=discussion.split(r'\section{Conclusion}')[0]
    discussion+=r'''\section{Conclusion}
The frozen predecessor remains a negative result. The source-linear successor '''+('meets' if status=='SUPPORTED' else 'does not meet')+r''' its predeclared iteration-reduction rule against both strong classical iterative starts on the specified IID cell. Complete online timing is reported separately, and the direct tridiagonal method remains faster. The resulting research claim is deliberately narrow: structural learning and preconditioning can interact usefully in this tested iterative setting, while a broader fastest-solver or multidimensional claim is unsupported. Preserved failures, fixed source identities and full raw artifacts make this conclusion inspectable and reproducible.
'''
    (paper/'discussion.tex').write_text(discussion)
    full=[r'\section{Complete S2 matrix}',r'\small',r'\begin{longtable}{rlllrrr}',r'\toprule',r'$N$ & $\tau$ & Regime & Method & Iterations & ms & Failures \\',r'\midrule\endhead']
    short={'coefficient_ood':'coeff. OOD','forcing_ood':'source OOD','iid':'IID'}
    for r in analysis['summaries']:
        full.append(f"{r['grid']} & {r['rtol']:.0e} & {short[r['regime']]} & {LABELS[r['method']]} & {r['iterations_mean']:.2f} & {r['time_ms_mean']:.3f} & {r['failures']} \\\\")
    full.extend([r'\bottomrule\end{longtable}',r'\normalsize'])
    (paper/'full_matrix.tex').write_text('\n'.join(full)+'\n')
    appendix=(paper/'appendix.tex').read_text()
    if r'\input{full_matrix}' not in appendix:
        (paper/'appendix.tex').write_text(appendix+'\n'+r'\input{full_matrix}'+'\n')
    # Scientific plots use conventional plotting tools, never generated imagery.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})
    order=list(LABELS);labels=[LABELS[m] for m in order]
    colors=['#7c8798' if m!='structured_laplace' else '#007a78' for m in order]
    fig,axes=plt.subplots(1,2,figsize=(10,4.2),layout='constrained')
    for ax,field,title in zip(axes,['iterations_mean','time_ms_mean'],['Mean accepted iterations','Mean online time (ms)']):
        ax.barh(labels,[means[m][field] for m in order],color=colors);ax.invert_yaxis();ax.set_xlabel(title);ax.grid(axis='x',alpha=.15)
    fig.savefig(paper/'figures/primary.pdf');fig.savefig(paper/'figures/primary.png',dpi=180);plt.close(fig)
    matrix=[];rowlabels=[]
    for grid2 in config['grids']:
        for t in config['rtols']:
            for regime in config['regimes']:
                selected={c['baseline']:c for c in analysis['comparisons'] if c['grid']==grid2 and c['rtol']==t and c['regime']==regime and c['proposed']=='structured_laplace'}
                matrix.append([100*selected[b][metric] for metric in ['iteration_reduction','online_time_reduction'] for b in ['zero_laplace','physics_laplace']])
                rowlabels.append(f"N={grid2}, {t:.0e}, {short[regime]}")
    fig,ax=plt.subplots(figsize=(8,5.5),layout='constrained')
    arr=np.array(matrix);bound=max(5,float(np.abs(arr).max()))
    im=ax.imshow(arr,cmap='RdYlGn',vmin=-bound,vmax=bound,aspect='auto')
    ax.set_yticks(range(len(rowlabels)),rowlabels);ax.set_xticks(range(4),['Iterations\nvs zero','Iterations\nvs physics','Online time\nvs zero','Online time\nvs physics'])
    for i,j in np.ndindex(arr.shape):ax.text(j,i,f'{arr[i,j]:.1f}%',ha='center',va='center',fontsize=8,color='black')
    fig.colorbar(im,ax=ax,label='Paired mean reduction (%)',shrink=.8)
    fig.savefig(paper/'figures/matrix.pdf');fig.savefig(paper/'figures/matrix.png',dpi=180);plt.close(fig)
    sweep=json.loads((development/'s1-tolerance-audit.json').read_text())['records']
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    for split in ['iid_test','ood_test']:
        rr=sorted([r for r in sweep if r['split']==split],key=lambda r:r['tolerance'])
        axes[0].semilogx([r['tolerance'] for r in rr],[100*r['relative_iteration_reduction_mean'] for r in rr],marker='o',label=split.replace('_test','').upper())
    axes[0].axhline(0,color='black',lw=.7);axes[0].set_xlabel('S1 relative residual tolerance');axes[0].set_ylabel('Iteration reduction (%)');axes[0].legend()
    for method in ['zero_jacobi','zero_laplace','physics_laplace','structured_laplace']:
        rr=sorted([r for r in analysis['summaries'] if r['method']==method and r['regime']=='iid' and r['rtol']==tol],key=lambda r:r['grid'])
        axes[1].plot([r['grid'] for r in rr],[r['iterations_mean'] for r in rr],marker='o',label=LABELS[method])
    axes[1].set_xlabel('S2 total grid points');axes[1].set_ylabel('Mean iterations');axes[1].legend(fontsize=8)
    fig.savefig(paper/'figures/tolerance_scaling.pdf');fig.savefig(paper/'figures/tolerance_scaling.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,2.2),layout='constrained');ax.set_axis_off()
    boxes=[(.1,'Training only\n192 exact discrete solutions\nper independent block'),(.38,'Source-linear features\n60 features, ridge fit\nSave fitted coefficients'),(.66,'New coefficient + forcing\nPredict initial field\nLaplacian PCG'),(.91,'True residual\naccept or retain failure\nComplete online cost')]
    for x,label in boxes:ax.text(x,.5,label,ha='center',va='center',transform=ax.transAxes,bbox=dict(boxstyle='round,pad=.6',fc='#eaf3f4',ec='#007a78'),fontsize=8)
    for (x,_),(y,_) in zip(boxes,boxes[1:]):ax.annotate('',xy=(y-.12,.5),xytext=(x+.12,.5),xycoords='axes fraction',arrowprops=dict(arrowstyle='->',color='#334155'))
    fig.savefig(paper/'figures/pipeline.pdf');fig.savefig(paper/'figures/pipeline.png',dpi=180);plt.close(fig)
    claims={'primary_disposition':status,'primary_against_zero':zero,'primary_against_physics':physics,
        'direct_ratio_of_mean_times':ratio,'s1_confirmatory':s1['decision'],
        'artifact_chain':{'raw':'s2-confirmatory/metrics.jsonl','analysis':'s2-confirmatory/analysis.json','table':'paper/primary_table.tex','figure':'paper/figures/primary.pdf'},
        'scope':'Synthetic 1D fixed family; no SOTA, real-world data or independent investigator reproduction claim',
        'source_artifact_hashes':{str(p.relative_to(confirmation)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [run/'analysis.json',run/'metrics.jsonl',run/'config.json',run/'receipt.json',confirmation/'s1-confirmatory/analysis.json']}}
    (paper/'evidence_facts.json').write_text(json.dumps(claims,indent=2)+'\n')
    print(json.dumps({'disposition':status,'primary_cases':primary_n,'rows':receipt['metric_rows'],'failures':all_failures,'direct_speed_ratio':ratio},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('development',type=Path);p.add_argument('confirmation',type=Path);p.add_argument('paper',type=Path)
    a=p.parse_args();generate(a.development,a.confirmation,a.paper)
