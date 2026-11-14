"""Czytelne diagramy zapisywane jako PNG bezpośrednio w wynikach notatnika."""
import io
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'phishing-mpl'))
os.environ.setdefault('XDG_CACHE_HOME', str(Path(tempfile.gettempdir()) / 'phishing-cache'))
Path(os.environ['XDG_CACHE_HOME']).mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from IPython.display import Image, display
import numpy as np

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.titleweight': 'bold', 'figure.facecolor': 'white',
    'axes.facecolor': 'white', 'savefig.facecolor': 'white',
})
BLUE, GREEN, ORANGE = '#2563a6', '#16816a', '#c45a15'


def render(fig):
    """Embed a portable image; no interactive backend or external files needed."""
    output = io.BytesIO()
    fig.savefig(output, format='png', dpi=150, bbox_inches='tight')
    display(Image(data=output.getvalue(), width=round(fig.get_figwidth() * 90)))
    plt.close(fig)


def workflow(title, steps, caption):
    fig, ax = plt.subplots(figsize=(13, 3.3))
    ax.set(xlim=(0, 12), ylim=(0, 2.1))
    ax.axis('off')
    width = 12 / len(steps)
    for i, text in enumerate(steps):
        left = i * width + .10
        ax.add_patch(FancyBboxPatch((left, .55), width - .45, 1.0,
                     boxstyle='round,pad=0.08', facecolor='#edf4fb',
                     edgecolor=BLUE, linewidth=1.5))
        ax.text(left + (width - .45) / 2, 1.05, text, ha='center', va='center',
                fontsize=11, linespacing=1.6)
        if i < len(steps) - 1:
            ax.annotate('', xy=((i + 1) * width + .04, 1.05),
                        xytext=(left + width - .35, 1.05),
                        arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.8))
    ax.set_title(title, fontsize=16, loc='left', pad=12)
    ax.text(0, .05, caption, fontsize=10, va='bottom', color='#475569')
    render(fig)


def metric_bars(frame, metrics, title, labels=None):
    """Side-by-side panels, common 0..1 scale, labels on every bar."""
    fig, axes = plt.subplots(1, len(metrics), figsize=(13, max(3.5, len(frame) * .6 + 1.5)),
                              layout='constrained', squeeze=False)
    labels = frame['method'].tolist() if labels is None else list(labels)
    for ax, (column, label, lower_better) in zip(axes[0], metrics):
        values = frame[column].to_numpy(dtype=float)
        ax.barh(np.arange(len(frame)), values, color=ORANGE if lower_better else BLUE,
                height=.6)
        ax.set_yticks(np.arange(len(frame)), labels)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.18)
        ax.set_xticks([0, .25, .5, .75, 1])
        if column == 'silhouette':
            ax.set_xlim(-1, 1.18)
            ax.set_xticks([-1, -.5, 0, .5, 1])
        direction = '' if lower_better is None else (' ↓' if lower_better else ' ↑')
        ax.set_title(label + direction, fontsize=12)
        ax.grid(axis='x', alpha=.18)
        ax.set_axisbelow(True)
        for i, value in enumerate(values):
            if np.isfinite(value):
                ax.text(value + .02, i, f'{value:.3f}', va='center', fontsize=10)
            else:
                ax.text(.02, i, 'brak wartości', va='center', fontsize=10)
    fig.suptitle(title, fontsize=15, fontweight='bold')
    render(fig)


def heatmap_grid(summary, metrics, title):
    """Each metric keeps its own colorbar and units; no cross-unit color ranking."""
    methods = ['SVM', 'Naive Bayes', 'Random Forest']
    representations = ['word 1-gram', 'word 1-2-gram', 'char 3-5-gram']
    names = ['Słowa\n1-gram', 'Słowa\n1–2-gram', 'Znaki\n3–5-gram']
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4.5),
                              layout='constrained', squeeze=False)
    for ax, (column, label, lower_better) in zip(axes[0], metrics):
        table = summary.pivot(index='method', columns='representation', values=column)
        values = table.reindex(index=methods, columns=representations).to_numpy()
        assert np.isfinite(values).all()
        quality = column in ('f1', 'recall', 'fpr')
        image = ax.imshow(values, cmap='YlOrBr' if lower_better else 'Blues',
                          vmin=0 if quality else None, vmax=1 if quality else None,
                          aspect='auto')
        ax.set_xticks(range(3), names)
        ax.set_yticks(range(3), methods)
        ax.set_title(label + (' ↓' if lower_better else ' ↑'), fontsize=12)
        low, high = image.get_clim()
        for row in range(3):
            for col in range(3):
                value = values[row, col]
                fraction = (value - low) / max(high - low, 1e-12)
                ax.text(col, row, f'{value:.3f}', ha='center', va='center',
                        color='white' if fraction > .6 else '#172033', fontsize=11)
        fig.colorbar(image, ax=ax, shrink=.75)
    fig.suptitle(title, fontsize=15, fontweight='bold')
    render(fig)
