"""
PRL Figure Generation for Memory-DFT Paper
===========================================

Generates publication-quality figures for:
  "History-Dependent Quantum Dynamics from Direct Schrödinger Evolution"

Figures:
  Fig. 1: γ Distance Decomposition (Static evidence)
  Fig. 2: Path-Dependent λ Evolution (Dynamic evidence)  
  Fig. 3: Memory Indicators Comparison (Summary)

Author: Masamichi Iizumi, Tamaki Iizumi
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec

# PRL style settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 1.0,
    'lines.linewidth': 1.5,
})

# Color scheme (colorblind-friendly)
COLORS = {
    'local': '#0077BB',      # Blue
    'total': '#CC3311',      # Red
    'memory': '#EE7733',     # Orange (shaded region)
    'path1': '#009988',      # Teal
    'path2': '#EE3377',      # Magenta
    'dft': '#BBBBBB',        # Gray (for DFT comparison)
}


def fig1_gamma_decomposition(L_values, gamma_local_data, gamma_total_data,
                              gamma_local_fit=1.388, gamma_total_fit=2.604,
                              save_path=None, show=True):
    """
    Figure 1: γ Distance Decomposition
    
    Shows that 46.7% of correlations are non-local (memory).
    
    Parameters
    ----------
    L_values : array-like
        System sizes [6, 8, 10, 12]
    gamma_local_data : array-like
        γ values for r≤2 at each L
    gamma_total_data : array-like
        γ values for r→∞ at each L
    gamma_local_fit : float
        Fitted γ_local value
    gamma_total_fit : float
        Fitted γ_total value
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to display figure
        
    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    fig, ax = plt.subplots(figsize=(4, 3.5))
    
    N_values = np.array(L_values)
    
    # Plot data points
    ax.scatter(N_values, gamma_local_data, c=COLORS['local'], s=60, 
               marker='o', label=r'$\gamma_{\mathrm{local}}$ ($r \leq 2$)', zorder=3)
    ax.scatter(N_values, gamma_total_data, c=COLORS['total'], s=60, 
               marker='s', label=r'$\gamma_{\mathrm{total}}$ ($r \to \infty$)', zorder=3)
    
    # Fit lines (horizontal for converged values)
    ax.axhline(y=gamma_local_fit, color=COLORS['local'], linestyle='--', 
               alpha=0.7, linewidth=1.2)
    ax.axhline(y=gamma_total_fit, color=COLORS['total'], linestyle='--', 
               alpha=0.7, linewidth=1.2)
    
    # Shade memory region
    ax.fill_between([L_values[0]-0.5, L_values[-1]+0.5], 
                    gamma_local_fit, gamma_total_fit,
                    color=COLORS['memory'], alpha=0.25, 
                    label=r'$\gamma_{\mathrm{memory}}$ (46.7%)')
    
    # Annotation
    mid_gamma = (gamma_local_fit + gamma_total_fit) / 2
    ax.annotate(r'$\gamma_{\mathrm{memory}} = 1.216$', 
                xy=(L_values[-1]+0.3, mid_gamma),
                fontsize=10, ha='left', va='center',
                color=COLORS['memory'])
    
    # Labels and formatting
    ax.set_xlabel('System size $L$')
    ax.set_ylabel(r'Correlation exponent $\gamma$')
    ax.set_xlim(L_values[0]-0.5, L_values[-1]+1.5)
    ax.set_ylim(0.5, 3.5)
    ax.set_xticks(L_values)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=':')
    
    # Title (optional for PRL)
    # ax.set_title('Distance Decomposition of Electronic Correlations')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"✅ Fig. 1 saved to {save_path}")
    
    if show:
        plt.show()
    
    return fig, ax


def fig2_path_evolution(times1, lambdas1, times2, lambdas2,
                         path1_name="A→B", path2_name="B→A",
                         event_times=None, 
                         save_path=None, show=True):
    """
    Figure 2: Path-Dependent λ Evolution
    
    Main figure showing that identical final structures 
    yield different stability parameters.
    
    Parameters
    ----------
    times1, lambdas1 : array-like
        Time and λ data for path 1
    times2, lambdas2 : array-like
        Time and λ data for path 2
    path1_name, path2_name : str
        Names for legend
    event_times : list of (time, label) tuples
        Event markers
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to display figure
        
    Returns
    -------
    fig, axes : matplotlib figure and axes
    """
    fig = plt.figure(figsize=(6, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 2])
    
    # ===== Panel (a): Schematic =====
    ax_schem = fig.add_subplot(gs[0])
    ax_schem.set_xlim(0, 10)
    ax_schem.set_ylim(0, 10)
    ax_schem.axis('off')
    ax_schem.set_title('(a) Reaction paths', fontsize=10, loc='left')
    
    # Path 1: A→B
    ax_schem.text(1, 8.5, 'Path A→B:', fontsize=9, fontweight='bold', 
                  color=COLORS['path1'])
    ax_schem.text(1, 7.5, '(1) A adsorbs', fontsize=8, color=COLORS['path1'])
    ax_schem.text(1, 6.5, '(2) B adsorbs', fontsize=8, color=COLORS['path1'])
    
    # Path 2: B→A  
    ax_schem.text(1, 4.5, 'Path B→A:', fontsize=9, fontweight='bold',
                  color=COLORS['path2'])
    ax_schem.text(1, 3.5, '(1) B adsorbs', fontsize=8, color=COLORS['path2'])
    ax_schem.text(1, 2.5, '(2) A adsorbs', fontsize=8, color=COLORS['path2'])
    
    # Arrow to "Same final state"
    ax_schem.annotate('', xy=(8, 5.5), xytext=(5, 7.5),
                      arrowprops=dict(arrowstyle='->', color=COLORS['path1'], lw=1.5))
    ax_schem.annotate('', xy=(8, 5.5), xytext=(5, 3.5),
                      arrowprops=dict(arrowstyle='->', color=COLORS['path2'], lw=1.5))
    
    # Final state box
    bbox = FancyBboxPatch((6.5, 4.5), 3, 2, boxstyle="round,pad=0.1",
                          facecolor='white', edgecolor='black', linewidth=1.5)
    ax_schem.add_patch(bbox)
    ax_schem.text(8, 5.5, 'Same\nfinal\nstate', fontsize=8, ha='center', va='center')
    
    # But different λ!
    ax_schem.text(5, 1, r'But $\Delta\lambda = 1.59$!', fontsize=10, 
                  fontweight='bold', ha='center', color='red')
    
    # ===== Panel (b): λ(t) evolution =====
    ax_evol = fig.add_subplot(gs[1])
    
    # Plot both paths
    ax_evol.plot(times1, lambdas1, color=COLORS['path1'], linewidth=2,
                 label=f'Path {path1_name}')
    ax_evol.plot(times2, lambdas2, color=COLORS['path2'], linewidth=2,
                 label=f'Path {path2_name}')
    
    # Event markers
    if event_times:
        for t, label in event_times:
            ax_evol.axvline(x=t, color='gray', linestyle=':', alpha=0.5)
            ax_evol.text(t, ax_evol.get_ylim()[1]*0.95, label, 
                        fontsize=7, ha='center', va='top', rotation=90)
    
    # Highlight final difference
    t_final = times1[-1]
    lambda1_final = lambdas1[-1]
    lambda2_final = lambdas2[-1]
    
    # Arrow showing ΔΛ
    ax_evol.annotate('', xy=(t_final+0.3, lambda2_final), 
                     xytext=(t_final+0.3, lambda1_final),
                     arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax_evol.text(t_final+0.6, (lambda1_final+lambda2_final)/2,
                 r'$\Delta\lambda$', fontsize=10, color='red', va='center')
    
    ax_evol.set_xlabel('Time $t$ (a.u.)')
    ax_evol.set_ylabel(r'Stability parameter $\lambda$')
    ax_evol.set_title('(b) Time evolution', fontsize=10, loc='left')
    ax_evol.legend(loc='upper left', framealpha=0.9)
    ax_evol.grid(True, alpha=0.3, linestyle=':')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"✅ Fig. 2 saved to {save_path}")
    
    if show:
        plt.show()
    
    return fig, (ax_schem, ax_evol)


def fig3_memory_comparison(tests, dft_values, dse_values,
                            test_labels=None, 
                            save_path=None, show=True):
    """
    Figure 3: DFT vs DSE Memory Comparison
    
    Bar chart showing DFT cannot distinguish paths while DSE can.
    
    Parameters
    ----------
    tests : list of str
        Test names
    dft_values : array-like
        ΔΛ values from DFT (should be ~0)
    dse_values : array-like
        ΔΛ values from DSE
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to display figure
        
    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    fig, ax = plt.subplots(figsize=(4, 3))
    
    x = np.arange(len(tests))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, dft_values, width, label='Standard DFT',
                   color=COLORS['dft'], edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, dse_values, width, label='DSE (This work)',
                   color=COLORS['path1'], edgecolor='black', linewidth=1)
    
    # Add value labels
    for bar, val in zip(bars2, dse_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    # DFT = 0 annotation
    ax.text(x[0] - width/2, 0.15, '≡ 0', ha='center', va='bottom', 
            fontsize=9, color='gray')
    ax.text(x[1] - width/2, 0.15, '≡ 0', ha='center', va='bottom', 
            fontsize=9, color='gray')
    
    ax.set_ylabel(r'Path difference $\Delta\lambda$')
    ax.set_xticks(x)
    ax.set_xticklabels(test_labels if test_labels else tests)
    ax.legend(loc='upper left')
    ax.set_ylim(0, max(dse_values) * 1.2)
    ax.grid(True, alpha=0.3, linestyle=':', axis='y')
    
    # Annotation
    ax.text(0.95, 0.95, 'DFT cannot\ndistinguish paths!',
            transform=ax.transAxes, fontsize=9, ha='right', va='top',
            color='gray', style='italic')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"✅ Fig. 3 saved to {save_path}")
    
    if show:
        plt.show()
    
    return fig, ax


def generate_all_prl_figures(results_dict, output_dir='./figures'):
    """
    Generate all PRL figures from test results.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary containing:
        - 'gamma': {L_values, gamma_local, gamma_total}
        - 'adsorption': {times1, lambdas1, times2, lambdas2, delta_lambda}
        - 'reaction': {delta_lambda}
    output_dir : str
        Directory to save figures
        
    Returns
    -------
    figs : dict of figures
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    figs = {}
    
    # Fig. 1: γ decomposition
    if 'gamma' in results_dict:
        g = results_dict['gamma']
        fig1, _ = fig1_gamma_decomposition(
            g['L_values'], g['gamma_local'], g['gamma_total'],
            save_path=os.path.join(output_dir, 'fig1_gamma_decomposition.pdf'),
            show=False
        )
        figs['fig1'] = fig1
    
    # Fig. 2: Path evolution
    if 'adsorption' in results_dict:
        a = results_dict['adsorption']
        fig2, _ = fig2_path_evolution(
            a['times1'], a['lambdas1'],
            a['times2'], a['lambdas2'],
            event_times=[(2.0, 'Event 1'), (5.0, 'Event 2')],
            save_path=os.path.join(output_dir, 'fig2_path_evolution.pdf'),
            show=False
        )
        figs['fig2'] = fig2
    
    # Fig. 3: Comparison
    if 'adsorption' in results_dict and 'reaction' in results_dict:
        tests = ['Adsorption\norder', 'Reaction\nsequence']
        dft_values = [0, 0]  # DFT gives identical results
        dse_values = [
            results_dict['adsorption']['delta_lambda'],
            results_dict['reaction']['delta_lambda']
        ]
        fig3, _ = fig3_memory_comparison(
            tests, dft_values, dse_values,
            save_path=os.path.join(output_dir, 'fig3_memory_comparison.pdf'),
            show=False
        )
        figs['fig3'] = fig3
    
    print(f"\n✅ All figures saved to {output_dir}/")
    return figs


# =============================================================================
# Demo / Test
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("PRL Figure Generation Demo")
    print("="*60)
    
    # Demo data (replace with actual test results)
    
    # Fig. 1: γ decomposition
    L_values = [6, 8, 10, 12]
    gamma_local = [1.45, 1.40, 1.38, 1.37]  # Example data
    gamma_total = [2.65, 2.60, 2.59, 2.58]
    
    print("\n📊 Generating Fig. 1: γ Decomposition...")
    fig1_gamma_decomposition(L_values, gamma_local, gamma_total,
                             save_path='/home/claude/fig1_demo.png', show=False)
    
    # Fig. 2: Path evolution
    times = np.linspace(0, 10, 100)
    lambdas1 = 4 + 0.5*np.tanh((times-2)/0.5) + 2*np.tanh((times-5)/0.5)
    lambdas2 = 4 + 0.3*np.tanh((times-2)/0.5) + 3*np.tanh((times-5)/0.5)
    
    print("\n📊 Generating Fig. 2: Path Evolution...")
    fig2_path_evolution(times, lambdas1, times, lambdas2,
                        event_times=[(2.0, '1st event'), (5.0, '2nd event')],
                        save_path='/home/claude/fig2_demo.png', show=False)
    
    # Fig. 3: Comparison
    print("\n📊 Generating Fig. 3: Memory Comparison...")
    fig3_memory_comparison(
        ['Test 1', 'Test 2'],
        [0, 0],
        [1.59, 2.18],
        test_labels=['Adsorption\norder', 'Reaction\nsequence'],
        save_path='/home/claude/fig3_demo.png', show=False
    )
    
    print("\n✅ Demo complete! Check /home/claude/ for PNG files.")
