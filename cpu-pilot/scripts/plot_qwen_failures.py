"""Plot recorded Qwen decisions separately from downstream TalkNet frame errors."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'
q = json.loads((OUT / 'qwen_agent_pilot.json').read_text())
m = json.loads((OUT / 'agent_pilot_measurements.json').read_text())
lookup = {c['id']: c for c in m['cases']}
cases = q['cases']
assert len(cases) == 8 and all(c['status'] == 'completed' for c in cases)
confusion = np.zeros((2, 2), dtype=int)
counts = []
for c in cases:
    original = lookup[c['id']]
    truth = original['label']
    prediction = int(c['submission']['speaking'])
    confusion[truth, prediction] += 1
    assert c['evaluation']['majority_answer_correct'] == (truth == prediction)
    scores = np.array(original['candidates'][c['submission']['candidate']]['probability'])
    errors = int(np.sum((scores >= .5) != truth))
    assert abs(errors / 24 - c['evaluation']['selected_frame_error']) < 1e-9
    counts.append(errors)
assert confusion.tolist() == [[4, 0], [1, 3]]

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titleweight': 'bold', 'savefig.facecolor': '#f7f9fc'})
fig = plt.figure(figsize=(18, 8.5), facecolor='#f7f9fc')
grid = fig.add_gridspec(1, 3, left=.065, right=.975, bottom=.29, top=.73,
                      width_ratios=[1, 1.35, 1.2], wspace=.43)
fig.text(.065, .94, 'Qwen pilot: one wrong speaking answer; no frame-error improvement',
         fontsize=23, weight='bold', color='#14263d')
fig.text(.065, .89, '8 completed tool-agent cases  |  2 source videos  |  4 tracks × 2 noise levels',
         fontsize=15, color='#526477')
fig.text(.065, .81, '1 / 8 wrong answers', fontsize=19, weight='bold', color='#bb433b')
fig.text(.39, .81, '35 / 192 selected frame decisions wrong', fontsize=19, weight='bold', color='#14263d')
fig.text(.765, .81, '0.50 confidence on the wrong answer', fontsize=14, weight='bold', color='#526477')

ax = fig.add_subplot(grid[0, 0])
ax.set_title('A. Qwen’s final clip answers', loc='left', pad=16)
for row in range(2):
    for col in range(2):
        correct = row == col
        color = '#dceee8' if correct else ('#f3c9c5' if confusion[row, col] else '#edf0f4')
        ax.add_patch(plt.Rectangle((col-.5, row-.5), 1, 1, facecolor=color, edgecolor='white', linewidth=4))
        ax.text(col, row-.08, str(confusion[row, col]), ha='center', va='center', fontsize=30,
                weight='bold', color='#216858' if correct else '#a5332e')
        label = [['Correct silent', 'False speaking'], ['Missed speaker', 'Correct speaking']][row][col]
        ax.text(col, row+.23, label, ha='center', va='center', fontsize=10, color='#33465b')
ax.set(xlim=(-.5,1.5), ylim=(1.5,-.5), xticks=[0,1], yticks=[0,1],
       xticklabels=['Silent','Speaking'], yticklabels=['Silent','Speaking'],
       xlabel='Qwen answer', ylabel='Published ground truth')
ax.set_aspect('equal')
for spine in ax.spines.values(): spine.set_visible(False)
ax.tick_params(length=0)

ax = fig.add_subplot(grid[0, 1])
ax.set_title('B. Errors in Qwen-selected evidence', loc='left', pad=16)
labels=[]
for c in cases:
    row=lookup[c['id']]
    source='A' if row['source']=='kgVz_AXYvn0' else 'B'
    labels.append(f"{source} · {'speaking' if row['label'] else 'silent'} · σ{row['sigma']}")
colors=['#bb433b' if not c['evaluation']['majority_answer_correct'] else '#687f9c' for c in cases]
ax.barh(range(8), counts, color=colors, height=.64)
for i,n in enumerate(counts): ax.text(n+.45, i, f'{n}/24', va='center', fontsize=11, color='#263b53')
ax.set(yticks=range(8), yticklabels=labels, xlim=(0,29), xticks=[0,6,12,18,24],
       xlabel='Wrong TalkNet frame decisions')
ax.invert_yaxis(); ax.set_axisbelow(True); ax.grid(axis='x', alpha=.15)

ax = fig.add_subplot(grid[0, 2])
ax.set_title('C. Selection baselines', loc='left', pad=16)
names=['MUSIQ', 'Keep captured', 'Qwen', 'Fixed safeguard', 'BRISQUE']
values=[m['summary']['mean_error'][key]*100 for key in ['musiq','retain']] + [sum(counts)/192*100] + [m['summary']['mean_error'][key]*100 for key in ['guard','brisque']]
ax.barh(range(5), values, color=['#2b8873','#9ca9ba','#bb433b','#9ca9ba','#9ca9ba'], height=.6)
for i,v in enumerate(values): ax.text(v+.5,i,f'{v:.2f}%',va='center',fontsize=12,weight='bold' if i==2 else 'normal')
ax.set(yticks=range(5),yticklabels=names,xlim=(0,37),xticks=[0,10,20,30],xlabel='Selected-evidence frame error (%)')
ax.invert_yaxis();ax.set_axisbelow(True);ax.grid(axis='x',alpha=.15)

fig.text(.065,.205,'What failed:',fontsize=14,weight='bold',color='#14263d')
fig.text(.155,.205,'Qwen missed one speaking clip; the underlying detector also missed it in the clean reference.',fontsize=14,color='#33465b')
fig.text(.065,.165,'What this does not show:',fontsize=13,weight='bold',color='#14263d')
fig.text(.255,.165,'a highly confident Qwen error, or proof that restoration caused its wrong clip answer.',fontsize=13,color='#33465b')
fig.text(.065,.105,'Qwen used tool measurements, not direct video/audio. Panels B–C measure TalkNet errors in selected video, not Qwen frame predictions.',fontsize=11,color='#526477')
fig.text(.065,.073,'Small descriptive pilot: noise variants and adjacent frames are correlated. No population-level confidence interval is claimed.',fontsize=11,color='#526477')
fig.text(.065,.041,'Source: saved Qwen/Groq traces and CPU pilot measurements • 15 October 2026',fontsize=10,color='#6d7d8d')
for ext in ('png','pdf','svg'):
    fig.savefig(OUT/f'qwen_failure_summary.{ext}',dpi=220)
plt.close(fig)
(OUT/'qwen_failure_plot_data.json').write_text(json.dumps(dict(
    model=q['model'],confusion_matrix=confusion.tolist(),frame_error_counts=counts,
    selection_error_percent=dict(zip(names,values)),case_ids=[c['id'] for c in cases],
    source_aliases={'A':'kgVz_AXYvn0','B':'EEey6Pt0rqI'},
    interpretation='Qwen clip decisions and selected-evidence TalkNet frame decisions are distinct endpoints.'
),indent=2))
print('Generated PNG, PDF, SVG and plot data; verified 1/8 wrong answers and 35/192 wrong selected frames.')
