"""Generate all figures for the paper — true grayscale output."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

def save_grayscale(filename):
    """Save current figure as true grayscale PNG."""
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    # Convert to true grayscale and back to RGB so LaTeX renders correctly
    img = Image.open(filename).convert('L').convert('RGB')
    img.save(filename)
    print(f'Saved {filename} (grayscale)')


# ── Shared style — grayscale only ────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'axes.prop_cycle': plt.cycler(color=['#000000', '#555555', '#999999', '#cccccc']),
})

# Grayscale hatches for bars — visually distinct without colour
HATCH = ['', '///', '...', 'xxx']
GRAY  = ['#222222', '#666666', '#aaaaaa', '#dddddd']

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: A/B Study Results (3-panel)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
fig.suptitle('A/B Study --- Mean Results (n=4)', fontweight='bold', fontsize=11)

# Panel 1: Accuracy
ax = axes[0]
conditions = ['Control', 'Treatment']
acc = [44.0, 56.0]
err = [8.5, 13.1]
bars = ax.bar(conditions, acc, yerr=err, color=[GRAY[2], GRAY[0]],
              hatch=[HATCH[2], HATCH[0]], edgecolor='black',
              capsize=5, width=0.5, error_kw={'ecolor': 'black'})
ax.axhline(85, color='black', linestyle='--', linewidth=0.8, label='Secondary target (85%)')
ax.set_ylabel('Mean Accuracy (%)')
ax.set_title('Diagnostic Accuracy')
ax.set_ylim(0, 100)
for bar, val, e in zip(bars, acc, err):
    ax.text(bar.get_x() + bar.get_width()/2, val + e + 1,
            f'{val}%\n±{e}', ha='center', va='bottom', fontsize=8)
ax.legend(fontsize=7)

# Panel 2: Latency
ax = axes[1]
lat = [42.0, 22.0]
lat_err = [22.7, 15.4]
bars = ax.bar(conditions, lat, yerr=lat_err, color=[GRAY[2], GRAY[0]],
              hatch=[HATCH[2], HATCH[0]], edgecolor='black',
              capsize=5, width=0.5, error_kw={'ecolor': 'black'})
ax.set_ylabel('Mean time per case (s)')
ax.set_title('Diagnostic Time')
ax.set_ylim(0, 80)
ax.annotate('−20.0s\n(47.5%)', xy=(0.5, 32), xytext=(0.5, 55),
            ha='center', arrowprops=dict(arrowstyle='->', color='black'),
            fontsize=8, color='black')
for bar, val, e in zip(bars, lat, lat_err):
    ax.text(bar.get_x() + bar.get_width()/2, val + e + 1,
            f'{val}s\n±{e}', ha='center', va='bottom', fontsize=8)

# Panel 3: Latency Reduction
ax = axes[2]
ax.bar(['Treatment\nvs Control'], [47.0], yerr=[18.2], color=GRAY[0],
       hatch=HATCH[0], edgecolor='black',
       capsize=5, width=0.4, error_kw={'ecolor': 'black'})
ax.axhline(30, color='black', linestyle='--', linewidth=1.2,
           label='H1 threshold (30%)')
ax.set_ylabel('Mean latency reduction (%)')
ax.set_title('Latency Reduction')
ax.set_ylim(0, 90)
ax.text(0, 47 + 18.2 + 2, '47.0%\n±18.2', ha='center', va='bottom', fontsize=8)
ax.legend(fontsize=7)

plt.tight_layout()
save_grayscale('ab_results.png')

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: K-means sweep
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 3.5))
ks = list(range(3, 11))
sil_umap    = [0.41, 0.4111, 0.395, 0.383, 0.370, 0.362, 0.351, 0.338]
sil_fulldim = [0.035, 0.039, 0.038, 0.036, 0.034, 0.033, 0.031, 0.029]

ax.plot(ks, sil_umap,    'k-o',  label='UMAP 50-dim', linewidth=1.5, markersize=5)
ax.plot(ks, sil_fulldim, 'k--s', label='Full 4096-dim', linewidth=1.5, markersize=5,
        markerfacecolor='white')
ax.axhline(0.45, color='gray', linestyle=':', linewidth=0.8, label='H2 target 0.45')
ax.axvline(4, color='black', linestyle=':', linewidth=1.2, label='k*=4')
ax.set_xlabel('k (number of clusters)')
ax.set_ylabel('Silhouette Score')
ax.set_title('K-means Sweep: UMAP vs Full-Dimensional')
ax.legend(fontsize=8)
ax.set_ylim(-0.05, 0.55)
ax.set_xticks(ks)
plt.tight_layout()
save_grayscale('clustering_sweep.png')

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Explainability comparison
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 3.5))
metrics = ['FP Accuracy', 'FN Accuracy', 'Overall Accuracy']
phi_blind = [40, 3, 22]
phi_gt    = [84, 20, 52]
proto     = [80, 100, 90]

x = np.arange(len(metrics))
w = 0.25
b1 = ax.bar(x - w, phi_blind, w, label='Phi-3.5-Blind',
            color=GRAY[3], hatch=HATCH[3], edgecolor='black')
b2 = ax.bar(x,     phi_gt,    w, label='Phi-3.5-GT',
            color=GRAY[2], hatch=HATCH[2], edgecolor='black')
b3 = ax.bar(x + w, proto,     w, label='Prototype',
            color=GRAY[0], hatch=HATCH[0], edgecolor='black')

ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylabel('Accuracy (%)')
ax.set_title('Explainability Comparison on 50-Case Benchmark')
ax.set_ylim(0, 115)
ax.legend(fontsize=8)

for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1, f'{h}%',
                ha='center', va='bottom', fontsize=7)

plt.tight_layout()
save_grayscale('explainability_comparison.png')

# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Decision geometry (box per quadrant)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
quadrants = ['TP\n(n=187)', 'TN\n(n=4504)', 'FP\n(n=194)', 'FN\n(n=197)']
mean_dist   = [0.00040, 0.00034, 0.00037, 0.00037]
mean_margin = [0.00096, 0.00114, 0.00109, 0.00099]
grays = [GRAY[0], GRAY[1], GRAY[2], GRAY[3]]
hatches = HATCH

ax = axes[0]
bars = ax.bar(quadrants, [v*1e4 for v in mean_dist],
              color=grays, hatch=hatches, edgecolor='black')
ax.set_ylabel('Mean cosine distance (×10⁻⁴)')
ax.set_title('Cosine Distance by Quadrant')

ax = axes[1]
bars = ax.bar(quadrants, [v*1e4 for v in mean_margin],
              color=grays, hatch=hatches, edgecolor='black')
ax.set_ylabel('Mean prototype margin (×10⁻⁴)')
ax.set_title('Prototype Margin by Quadrant')

plt.suptitle('Decision Geometry: TP/TN/FP/FN are Geometrically Indistinguishable',
             fontsize=10, fontweight='bold')
plt.tight_layout()
save_grayscale('decision_geometry.png')

# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: LTL trust property coverage / precision
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 3.5))
props = [r'$\phi_\mathrm{trust}$', r'$\phi_\mathrm{incoherent}$',
         r'$\phi_\mathrm{ambiguous}$', r'$\phi_\mathrm{polarised}$']
coverage  = [93, 33, 17, 53]
precision = [92.3, 100.0, 100.0, 87.3]

x = np.arange(len(props))
w = 0.35
b1 = ax.bar(x - w/2, coverage,  w, label='Coverage (%)',
            color=GRAY[2], hatch=HATCH[2], edgecolor='black')
b2 = ax.bar(x + w/2, precision, w, label='Precision (%)',
            color=GRAY[0], hatch=HATCH[0], edgecolor='black')

ax.set_xticks(x)
ax.set_xticklabels(props, fontsize=11)
ax.set_ylabel('Percentage (%)')
ax.set_title('LTL Trust Properties: Coverage vs Precision')
ax.set_ylim(0, 115)
ax.legend(fontsize=9)
ax.axhline(100, color='gray', linestyle='--', linewidth=0.6)

for bar, val in zip(b1, coverage):
    ax.text(bar.get_x() + bar.get_width()/2, val + 1, f'{val}%',
            ha='center', va='bottom', fontsize=8)
for bar, val in zip(b2, precision):
    ax.text(bar.get_x() + bar.get_width()/2, val + 1, f'{val}%',
            ha='center', va='bottom', fontsize=8)

plt.tight_layout()
save_grayscale('ltl_properties.png')

# ─────────────────────────────────────────────────────────────────────────────
# Figure 6: Architecture diagram (black and white)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 7))
ax.set_xlim(0, 12)
ax.set_ylim(0, 7)
ax.axis('off')

def box(ax, x, y, w, h, label, fill, lw=1.2, fontsize=9, bold=False):
    rect = mpatches.FancyBboxPatch((x, y), w, h,
                                   boxstyle='round,pad=0.1',
                                   facecolor=fill, edgecolor='black',
                                   linewidth=lw)
    ax.add_patch(rect)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, label, ha='center', va='center',
            fontsize=fontsize, fontweight=weight, multialignment='center')

def arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.2))

# Banners
box(ax, 0.2, 5.5, 11.6, 0.7, 'TRAINING (offline, once)', '#dddddd', lw=1.5, fontsize=11, bold=True)
box(ax, 0.2, 3.2, 11.6, 0.7, 'INFERENCE (real-time, per prompt)', '#f0f0f0', lw=1.5, fontsize=11, bold=True)
box(ax, 0.2, 0.5, 11.6, 0.7, 'REVIEW (on flagged cases)', '#e8e8e8', lw=1.5, fontsize=11, bold=True)

# Training row
box(ax, 0.3, 4.3, 1.8, 1.0, 'ToxicChat\n5,082 prompts', '#eeeeee')
box(ax, 2.3, 4.3, 2.2, 1.0, 'Llama-Guard-3-8B\nExtract hidden states\n(4,096-dim)', '#ffffff')
box(ax, 4.7, 4.3, 2.2, 1.0, 'UMAP + K-Means\n4 attack clusters\nS=0.41 (+945%)', '#eeeeee')
box(ax, 7.1, 4.3, 2.0, 1.0, 'Human Labels\n4 prototypes +\nexemplars', '#ffffff')
box(ax, 9.3, 4.3, 2.0, 1.0, 'Persist\ntaxonomy.json\numap_reducer.pkl', '#cccccc', bold=True)
for x1, x2 in [(2.1, 2.3), (4.5, 4.7), (6.9, 7.1), (9.1, 9.3)]:
    arrow(ax, x1, 4.8, x2, 4.8)

# Inference row
box(ax, 0.3, 2.0, 1.6, 1.0, 'New\nPrompt', '#eeeeee', bold=True)
box(ax, 2.1, 2.0, 2.2, 1.0, 'Llama-Guard\nDecision +\nHidden State', '#ffffff')
box(ax, 4.5, 2.0, 2.4, 1.0, 'Prototype Match\nUMAP → cosine\n→ nearest P0-P3', '#eeeeee')
box(ax, 7.1, 2.0, 2.2, 1.0, 'LTL Monitor\nφ₁–φ₅ trust\nproperties', '#cccccc', bold=True)
box(ax, 9.5, 2.3, 1.2, 0.6, 'CLEAN\nor\nFLAGGED', '#e0e0e0', bold=True)
for x1, x2 in [(1.9, 2.1), (4.3, 4.5), (6.9, 7.1), (9.3, 9.5)]:
    arrow(ax, x1, 2.5, x2, 2.5)

# Review row
box(ax, 0.3, -0.3, 2.0, 0.9, 'LTL-Flagged\nCases', '#dddddd')
box(ax, 2.8, -0.3, 2.2, 0.9, 'Prototype\nExplanation\n(top-3)', '#cccccc', bold=True)
box(ax, 5.2, -0.3, 2.5, 0.9, 'Developer Review\nDiagnose root cause\nRecord time+conf.', '#ffffff')
box(ax, 8.0, -0.3, 2.0, 0.9, 'Results\n46% faster\n90% accuracy', '#eeeeee', bold=True)
box(ax, 10.2, -0.3, 1.6, 0.9, 'Fine-tune\nData Recipe', '#dddddd')
for x1, x2 in [(2.3, 2.8), (5.0, 5.2), (7.7, 8.0), (10.0, 10.2)]:
    arrow(ax, x1, 0.15, x2, 0.15)

# Cross-phase dashed arrows
ax.annotate('', xy=(5.7, 2.0), xytext=(10.3, 4.3),
            arrowprops=dict(arrowstyle='->', color='gray',
                            connectionstyle='arc3,rad=0.2',
                            linestyle='dashed', lw=1.2))
ax.text(8.5, 3.4, 'loads at startup', fontsize=7, color='gray', rotation=-25)

ax.annotate('', xy=(1.3, -0.3), xytext=(9.5, 2.3),
            arrowprops=dict(arrowstyle='->', color='black',
                            connectionstyle='arc3,rad=0.3',
                            linestyle='dashed', lw=1.2))
ax.text(4.5, 1.3, 'flagged case', fontsize=7, color='black')

plt.title('Prototype-Driven Guardrail Auditing Architecture',
          fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
save_grayscale('architecture.png')

print('\nAll figures generated (black and white).')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Shared style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: A/B Study Results (3-panel: accuracy, latency, latency reduction)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
fig.suptitle('A/B Study --- Mean Results (n=4)', fontweight='bold', fontsize=11)

# Panel 1: Accuracy
ax = axes[0]
conditions = ['Control', 'Treatment']
acc = [44.0, 56.0]
err = [8.5, 13.1]
bars = ax.bar(conditions, acc, yerr=err, color=['#4472C4', '#70AD47'],
              capsize=5, width=0.5)
ax.axhline(85, color='gray', linestyle='--', linewidth=0.8, label='Secondary target (85%)')
ax.set_ylabel('Mean Accuracy (%)')
ax.set_title('Diagnostic Accuracy')
ax.set_ylim(0, 100)
for bar, val, e in zip(bars, acc, err):
    ax.text(bar.get_x() + bar.get_width()/2, val + e + 1,
            f'{val}%\n±{e}', ha='center', va='bottom', fontsize=8)
ax.legend(fontsize=7)

# Panel 2: Latency
ax = axes[1]
lat = [42.0, 22.0]
lat_err = [22.7, 15.4]
bars = ax.bar(conditions, lat, yerr=lat_err, color=['#4472C4', '#70AD47'],
              capsize=5, width=0.5)
ax.set_ylabel('Mean time per case (s)')
ax.set_title('Diagnostic Time')
ax.set_ylim(0, 80)
ax.annotate('−20.0s\n(47.5%)', xy=(0.5, 32), xytext=(0.5, 55),
            ha='center', arrowprops=dict(arrowstyle='->', color='black'),
            fontsize=8, color='black')
for bar, val, e in zip(bars, lat, lat_err):
    ax.text(bar.get_x() + bar.get_width()/2, val + e + 1,
            f'{val}s\n±{e}', ha='center', va='bottom', fontsize=8)

# Panel 3: Latency Reduction
ax = axes[2]
ax.bar(['Treatment\nvs Control'], [47.0], yerr=[18.2], color='#70AD47',
       capsize=5, width=0.4)
ax.axhline(30, color='#D55E00', linestyle='--', linewidth=1.2,
           label='H1 threshold (30%)')
ax.set_ylabel('Mean latency reduction (%)')
ax.set_title('Latency Reduction')
ax.set_ylim(0, 90)
ax.text(0, 47 + 18.2 + 2, '47.0%\n±18.2', ha='center', va='bottom', fontsize=8)
ax.legend(fontsize=7)

plt.tight_layout()
save_grayscale('ab_results.png')

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: K-means sweep (silhouette vs k)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 3.5))
ks = list(range(3, 11))
sil_umap   = [0.41, 0.4111, 0.395, 0.383, 0.370, 0.362, 0.351, 0.338]
sil_fulldim = [0.035, 0.039, 0.038, 0.036, 0.034, 0.033, 0.031, 0.029]

ax.plot(ks, sil_umap,   'b-o', label='UMAP 50-dim', linewidth=1.5, markersize=5)
ax.plot(ks, sil_fulldim,'r--s', label='Full 4096-dim', linewidth=1.5, markersize=5)
ax.axhline(0.45, color='gray', linestyle='--', linewidth=0.8, label='H2 target 0.45')
ax.axvline(4, color='green', linestyle=':', linewidth=1.2, label='k*=4')
ax.set_xlabel('k (number of clusters)')
ax.set_ylabel('Silhouette Score')
ax.set_title('K-means Sweep: UMAP vs Full-Dimensional')
ax.legend(fontsize=8)
ax.set_ylim(-0.05, 0.55)
ax.set_xticks(ks)
plt.tight_layout()
save_grayscale('clustering_sweep.png')

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Explainability comparison bar chart
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 3.5))
metrics = ['FP Accuracy', 'FN Accuracy', 'Overall Accuracy']
phi_blind = [40, 3, 22]
phi_gt    = [84, 20, 52]
proto     = [80, 100, 90]

x = np.arange(len(metrics))
w = 0.25
b1 = ax.bar(x - w, phi_blind, w, label='Phi-3.5-Blind', color='#C0504D')
b2 = ax.bar(x,     phi_gt,    w, label='Phi-3.5-GT',    color='#F79646')
b3 = ax.bar(x + w, proto,     w, label='Prototype',      color='#70AD47')

ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylabel('Accuracy (%)')
ax.set_title('Explainability Comparison on 50-Case Benchmark')
ax.set_ylim(0, 115)
ax.legend(fontsize=8)

for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1, f'{h}%',
                ha='center', va='bottom', fontsize=7)

plt.tight_layout()
save_grayscale('explainability_comparison.png')

# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Decision geometry (violin / box per quadrant)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

quadrants = ['TP\n(n=187)', 'TN\n(n=4504)', 'FP\n(n=194)', 'FN\n(n=197)']
mean_dist   = [0.00040, 0.00034, 0.00037, 0.00037]
mean_margin = [0.00096, 0.00114, 0.00109, 0.00099]

colors = ['#70AD47', '#4472C4', '#C0504D', '#F79646']

ax = axes[0]
bars = ax.bar(quadrants, [v*1e4 for v in mean_dist], color=colors)
ax.set_ylabel('Mean cosine distance (×10⁻⁴)')
ax.set_title('Cosine Distance by Quadrant')
for bar, val in zip(bars, mean_dist):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f'{val:.5f}', ha='center', va='bottom', fontsize=7, rotation=45)

ax = axes[1]
bars = ax.bar(quadrants, [v*1e4 for v in mean_margin], color=colors)
ax.set_ylabel('Mean prototype margin (×10⁻⁴)')
ax.set_title('Prototype Margin by Quadrant')
for bar, val in zip(bars, mean_margin):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f'{val:.5f}', ha='center', va='bottom', fontsize=7, rotation=45)

plt.suptitle('Decision Geometry: TP/TN/FP/FN are Geometrically Indistinguishable',
             fontsize=10, fontweight='bold')
plt.tight_layout()
save_grayscale('decision_geometry.png')

# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: LTL trust property coverage / precision
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 3.5))
props = [r'$\phi_\mathrm{trust}$', r'$\phi_\mathrm{incoherent}$',
         r'$\phi_\mathrm{ambiguous}$', r'$\phi_\mathrm{polarised}$']
coverage  = [93, 33, 17, 53]
precision = [92.3, 100.0, 100.0, 87.3]

x = np.arange(len(props))
w = 0.35
b1 = ax.bar(x - w/2, coverage,  w, label='Coverage (%)',  color='#4472C4', alpha=0.85)
b2 = ax.bar(x + w/2, precision, w, label='Precision (%)', color='#70AD47', alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(props, fontsize=11)
ax.set_ylabel('Percentage (%)')
ax.set_title('LTL Trust Properties: Coverage vs Precision')
ax.set_ylim(0, 115)
ax.legend(fontsize=9)
ax.axhline(100, color='gray', linestyle='--', linewidth=0.6)

for bar, val in zip(b1, coverage):
    ax.text(bar.get_x() + bar.get_width()/2, val + 1, f'{val}%',
            ha='center', va='bottom', fontsize=8)
for bar, val in zip(b2, precision):
    ax.text(bar.get_x() + bar.get_width()/2, val + 1, f'{val}%',
            ha='center', va='bottom', fontsize=8)

plt.tight_layout()
save_grayscale('ltl_properties.png')

# ─────────────────────────────────────────────────────────────────────────────
# Figure 6: Architecture diagram (programmatic)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 7))
ax.set_xlim(0, 12)
ax.set_ylim(0, 7)
ax.axis('off')

def box(ax, x, y, w, h, label, color, fontsize=9, bold=False):
    rect = mpatches.FancyBboxPatch((x, y), w, h,
                                   boxstyle='round,pad=0.1',
                                   facecolor=color, edgecolor='#333',
                                   linewidth=1.2)
    ax.add_patch(rect)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, label, ha='center', va='center',
            fontsize=fontsize, fontweight=weight, wrap=True,
            multialignment='center')

def arrow(ax, x1, y1, x2, y2, label=''):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#444', lw=1.2))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my + 0.1, label, ha='center', fontsize=7, color='#555')

# Banners
box(ax, 0.2, 5.5, 11.6, 0.7, 'TRAINING (offline, once)', '#dae8fc', fontsize=11, bold=True)
box(ax, 0.2, 3.2, 11.6, 0.7, 'INFERENCE (real-time, per prompt)', '#d5e8d4', fontsize=11, bold=True)
box(ax, 0.2, 0.5, 11.6, 0.7, 'REVIEW (on flagged cases)', '#ffe6cc', fontsize=11, bold=True)

# Training row
box(ax, 0.3, 4.3, 1.8, 1.0, 'ToxicChat\n5,082 prompts', '#fff2cc')
box(ax, 2.3, 4.3, 2.2, 1.0, 'Llama-Guard-3-8B\nExtract hidden states\n(4,096-dim)', '#dae8fc')
box(ax, 4.7, 4.3, 2.2, 1.0, 'UMAP + K-Means\n4 attack clusters\nS=0.41 (+945%)', '#d5e8d4')
box(ax, 7.1, 4.3, 2.0, 1.0, 'Human Labels\n4 prototypes +\nexemplars', '#fff2cc')
box(ax, 9.3, 4.3, 2.0, 1.0, 'Persist\ntaxonomy.json\numap_reducer.pkl', '#f8cecc', bold=True)
for x1, x2 in [(2.1, 2.3), (4.5, 4.7), (6.9, 7.1), (9.1, 9.3)]:
    arrow(ax, x1, 4.8, x2, 4.8)

# Inference row
box(ax, 0.3, 2.0, 1.6, 1.0, 'New\nPrompt', '#fff2cc', bold=True)
box(ax, 2.1, 2.0, 2.2, 1.0, 'Llama-Guard\nDecision +\nHidden State', '#dae8fc')
box(ax, 4.5, 2.0, 2.4, 1.0, 'Prototype Match\nUMAP → cosine sim\n→ nearest P0-P3', '#d5e8d4')
box(ax, 7.1, 2.0, 2.2, 1.0, 'LTL Monitor\nφ₁–φ₅ trust\nproperties', '#f8cecc', bold=True)
box(ax, 9.5, 2.3, 1.2, 0.6, 'CLEAN\nor\nFLAGGED', '#ffe6cc', bold=True)
for x1, x2 in [(1.9, 2.1), (4.3, 4.5), (6.9, 7.1), (9.3, 9.5)]:
    arrow(ax, x1, 2.5, x2, 2.5)

# Review row
box(ax, 0.3, -0.3, 2.0, 0.9, 'LTL-Flagged\nCases', '#f8cecc')
box(ax, 2.8, -0.3, 2.2, 0.9, 'Prototype\nExplanation\n(top-3)', '#e1d5e7', bold=True)
box(ax, 5.2, -0.3, 2.5, 0.9, 'Developer Review\nDiagnose root cause\nRecord time+conf.', '#dae8fc')
box(ax, 8.0, -0.3, 2.0, 0.9, 'Results\n46% faster\n90% accuracy', '#d5e8d4', bold=True)
box(ax, 10.2, -0.3, 1.6, 0.9, 'Fine-tune\nData Recipe', '#fff2cc')
for x1, x2 in [(2.3, 2.8), (5.0, 5.2), (7.7, 8.0), (10.0, 10.2)]:
    arrow(ax, x1, 0.15, x2, 0.15)

# Cross-phase dashed arrows
ax.annotate('', xy=(5.7, 2.0), xytext=(10.3, 4.3),
            arrowprops=dict(arrowstyle='->', color='#9673a6',
                            connectionstyle='arc3,rad=0.2',
                            linestyle='dashed', lw=1.2))
ax.text(8.5, 3.4, 'loads at startup', fontsize=7, color='#9673a6', rotation=-25)

ax.annotate('', xy=(1.3, -0.3), xytext=(9.5, 2.3),
            arrowprops=dict(arrowstyle='->', color='#b85450',
                            connectionstyle='arc3,rad=0.3',
                            linestyle='dashed', lw=1.2))
ax.text(4.5, 1.3, 'flagged case', fontsize=7, color='#b85450')

plt.title('Prototype-Driven Guardrail Auditing Architecture', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
save_grayscale('architecture.png')

print('\nAll figures generated successfully.')
