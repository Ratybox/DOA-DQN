"""
DOA-DQN Visualization Suite
Professional plotting for research paper & comparison with SOTA

Author: DOA-DQN Research Project
Date: December 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import json
from pathlib import Path
from typing import Dict, List, Tuple
import seaborn as sns

# Set style for academic papers
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 11
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.family'] = 'sans-serif'


class DOADQNVisualizer:
    """Professional visualization suite for DOA-DQN results."""
    
    def __init__(self, output_dir: str = './doa_dqn_visualizations'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.colors = {
            'doa': '#1f77b4',      # Blue
            'optuna': '#ff7f0e',   # Orange
            'random': '#2ca02c',   # Green
            'bohb': '#d62728',     # Red
            'pbt': '#9467bd',      # Purple
        }
    
    # ========================================================================
    # VISUALIZATION 1: DOA Convergence Curve (Main optimization progress)
    # ========================================================================
    
    def plot_doa_convergence(self, history_best: np.ndarray, history_mean: np.ndarray,
                             title: str = 'DOA Convergence Curve', filename: str = 'v1_doa_convergence.png'):
        """
        Plot DOA optimization convergence.
        
        Shows:
        - Best fitness found (lower bound, tight)
        - Mean population fitness (exploration breadth)
        - Demonstrates exploration→exploitation transition
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        iterations = np.arange(len(history_best))
        
        ax.plot(iterations, -history_best, 'o-', color=self.colors['doa'], 
                linewidth=2.5, markersize=5, label='Best fitness', zorder=3)
        ax.fill_between(iterations, -history_best, -history_mean, 
                        alpha=0.2, color=self.colors['doa'], label='Population diversity')
        ax.plot(iterations, -history_mean, '--', color=self.colors['doa'], 
                linewidth=1.5, alpha=0.6, label='Mean fitness')
        
        # Add exploration/exploitation boundary
        t_d = int(0.9 * len(history_best))
        ax.axvline(x=t_d, color='red', linestyle=':', linewidth=2, alpha=0.7, label='Exploitation phase starts')
        
        ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
        ax.set_ylabel('Reward (Mean Evaluation)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()
    
    # ========================================================================
    # VISUALIZATION 2: Performance Comparison Bar Chart (vs SOTA)
    # ========================================================================
    
    def plot_performance_comparison(self, methods_scores: Dict[str, Tuple[float, float]],
                                   title: str = 'Performance Comparison: DOA-DQN vs SOTA',
                                   env_name: str = 'CartPole-v1',
                                   filename: str = 'v2_performance_comparison.png'):
        """
        Plot performance comparison with error bars.
        
        methods_scores: {method_name: (mean_score, std_dev)}
        Shows absolute performance + confidence intervals
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        methods = list(methods_scores.keys())
        means = [methods_scores[m][0] for m in methods]
        stds = [methods_scores[m][1] for m in methods]
        
        colors_list = [self.colors.get(m, '#444444') for m in methods]
        
        x_pos = np.arange(len(methods))
        bars = ax.bar(x_pos, means, yerr=stds, capsize=8, color=colors_list,
                     alpha=0.8, edgecolor='black', linewidth=1.5, error_kw={'linewidth': 2})
        
        # Add value labels on bars
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 2,
                   f'{mean:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # Highlight DOA bar
        bars[0].set_edgecolor('gold')
        bars[0].set_linewidth(3)
        
        ax.set_xlabel('Method', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Reward', fontsize=12, fontweight='bold')
        ax.set_title(f'{title} ({env_name})', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods, rotation=15, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()
    
    # ========================================================================
    # VISUALIZATION 3: Efficiency Frontier (Performance vs Time)
    # ========================================================================
    
    def plot_efficiency_frontier(self, methods_efficiency: Dict[str, Tuple[float, float, float]],
                                title: str = 'Efficiency Frontier: Performance vs Computational Cost',
                                filename: str = 'v3_efficiency_frontier.png'):
        """
        Plot performance vs wall-clock time / number of evaluations.
        
        methods_efficiency: {method_name: (mean_reward, time_minutes, n_evals)}
        Shows which method gives best reward per unit time
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Panel 1: Performance vs Time
        for method, (reward, time_min, n_evals) in methods_efficiency.items():
            color = self.colors.get(method, '#444444')
            ax1.scatter(time_min, reward, s=300, alpha=0.7, color=color, 
                       edgecolors='black', linewidth=2, label=method, zorder=3)
            ax1.annotate(method, (time_min, reward), xytext=(5, 5), 
                        textcoords='offset points', fontsize=9, fontweight='bold')
        
        ax1.set_xlabel('Wall-Clock Time (minutes)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Mean Reward', fontsize=12, fontweight='bold')
        ax1.set_title('Performance vs Time', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        
        # Panel 2: Performance vs Evaluations
        for method, (reward, time_min, n_evals) in methods_efficiency.items():
            color = self.colors.get(method, '#444444')
            ax2.scatter(n_evals, reward, s=300, alpha=0.7, color=color,
                       edgecolors='black', linewidth=2, label=method, zorder=3)
            ax2.annotate(method, (n_evals, reward), xytext=(5, 5),
                        textcoords='offset points', fontsize=9, fontweight='bold')
        
        ax2.set_xlabel('Number of Function Evaluations', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Mean Reward', fontsize=12, fontweight='bold')
        ax2.set_title('Performance vs Evaluations', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()
    
    # ========================================================================
    # VISUALIZATION 4: Multi-Environment Comparison (Radar/Spider)
    # ========================================================================
    
    def plot_radar_comparison(self, envs_methods: Dict[str, Dict[str, float]],
                             title: str = 'Multi-Environment Performance (Normalized)',
                             filename: str = 'v4_radar_comparison.png'):
        """
        Radar plot showing performance across multiple environments.
        
        envs_methods: {env_name: {method_name: normalized_score [0-100]}}
        Visualizes robustness across different benchmarks
        """
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        environments = list(envs_methods.keys())
        n_envs = len(environments)
        angles = np.linspace(0, 2 * np.pi, n_envs, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        # Get all methods
        all_methods = set()
        for env_scores in envs_methods.values():
            all_methods.update(env_scores.keys())
        
        # Plot each method
        for method in all_methods:
            values = [envs_methods[env].get(method, 0) for env in environments]
            values += values[:1]  # Complete the circle
            
            color = self.colors.get(method, '#444444')
            ax.plot(angles, values, 'o-', linewidth=2.5, label=method, color=color, markersize=8)
            ax.fill(angles, values, alpha=0.15, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(environments, fontsize=11, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.set_ylabel('Normalized Score', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()
    
    # ========================================================================
    # VISUALIZATION 5: Box Plot - Robustness Across Seeds
    # ========================================================================
    
    def plot_robustness_boxplot(self, methods_seeds: Dict[str, List[float]],
                               title: str = 'Robustness: Distribution Across Seeds',
                               filename: str = 'v5_robustness_boxplot.png'):
        """
        Box plot showing score distribution across multiple seeds.
        
        methods_seeds: {method_name: [seed1_score, seed2_score, ...]}
        Shows variance, median, outliers for each method
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        methods = list(methods_seeds.keys())
        data = [methods_seeds[m] for m in methods]
        colors_list = [self.colors.get(m, '#444444') for m in methods]
        
        bp = ax.boxplot(data, labels=methods, patch_artist=True, 
                       widths=0.6, showmeans=True,
                       meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add individual points
        for i, (method, scores) in enumerate(methods_seeds.items(), 1):
            y = np.array(scores)
            x = np.random.normal(i, 0.04, size=len(y))
            ax.scatter(x, y, alpha=0.4, s=30, color=self.colors.get(method, '#444444'))
        
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()
    
    # ========================================================================
    # VISUALIZATION 6: Learning Curves During Training
    # ========================================================================
    
    def plot_training_curves(self, methods_curves: Dict[str, Tuple[np.ndarray, np.ndarray]],
                            title: str = 'Training Curves: Online Performance',
                            filename: str = 'v6_training_curves.png'):
        """
        Plot training curves showing cumulative reward during DQN training.
        
        methods_curves: {method_name: (steps_array, rewards_array)}
        Shows learning speed and stability
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for method, (steps, rewards) in methods_curves.items():
            color = self.colors.get(method, '#444444')
            ax.plot(steps, rewards, linewidth=2, label=method, color=color, alpha=0.8)
        
        ax.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
        ax.set_ylabel('Episode Reward (Moving Average)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()
    
    # ========================================================================
    # VISUALIZATION 7: Hyperparameter Sensitivity Heatmap
    # ========================================================================
    
    def plot_hyperparameter_heatmap(self, hp_sensitivity: Dict[str, Dict[str, float]],
                                   title: str = 'Hyperparameter Sensitivity Analysis',
                                   filename: str = 'v7_hp_sensitivity_heatmap.png'):
        """
        Heatmap showing sensitivity of final performance to each HP.
        
        hp_sensitivity: {hp_name: {hp_value: final_score}}
        Shows which HPs matter most
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        hp_names = list(hp_sensitivity.keys())
        hp_values_lists = [list(hp_sensitivity[hp].keys()) for hp in hp_names]
        
        # Create matrix (HPs × Values)
        max_values = max(len(v) for v in hp_values_lists)
        sensitivity_matrix = np.zeros((len(hp_names), max_values))
        
        for i, hp in enumerate(hp_names):
            for j, value in enumerate(hp_values_lists[i]):
                sensitivity_matrix[i, j] = hp_sensitivity[hp][value]
        
        im = ax.imshow(sensitivity_matrix, cmap='RdYlGn', aspect='auto')
        
        ax.set_xlabel('Hyperparameter Value Index', fontsize=12, fontweight='bold')
        ax.set_ylabel('Hyperparameter', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_yticks(np.arange(len(hp_names)))
        ax.set_yticklabels(hp_names)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Final Score', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()
    
    # ========================================================================
    # VISUALIZATION 8: Population Diversity Evolution
    # ========================================================================
    
    def plot_population_diversity(self, diversity_history: np.ndarray,
                                 title: str = 'Population Diversity Evolution',
                                 filename: str = 'v8_population_diversity.png'):
        """
        Plot population diversity metric over iterations.
        
        diversity_history: array of diversity metrics [0, 1]
        Shows exploration breadth over time
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        iterations = np.arange(len(diversity_history))
        
        ax.fill_between(iterations, 0, diversity_history, alpha=0.3, color=self.colors['doa'])
        ax.plot(iterations, diversity_history, 'o-', color=self.colors['doa'], 
               linewidth=2.5, markersize=6)
        
        ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
        ax.set_ylabel('Population Diversity (Entropy)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()
    
    # ========================================================================
    # VISUALIZATION 9: Summary Dashboard (All-in-one)
    # ========================================================================
    
    def plot_comprehensive_dashboard(self, results: Dict, filename: str = 'v9_comprehensive_dashboard.png'):
        """
        Create a comprehensive dashboard with multiple subplots.
        
        Combines key metrics in one figure for paper supplement
        """
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. DOA Convergence
        ax1 = fig.add_subplot(gs[0, :2])
        if 'doa_history' in results:
            ax1.plot(results['doa_history'], 'o-', color=self.colors['doa'], linewidth=2.5)
            ax1.set_title('DOA Convergence', fontweight='bold')
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Best Fitness')
            ax1.grid(True, alpha=0.3)
        
        # 2. Performance Bars
        ax2 = fig.add_subplot(gs[0, 2])
        if 'method_scores' in results:
            methods = list(results['method_scores'].keys())
            scores = list(results['method_scores'].values())
            colors_list = [self.colors.get(m, '#444') for m in methods]
            ax2.barh(methods, scores, color=colors_list, alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Score')
            ax2.set_title('Performance', fontweight='bold')
            ax2.grid(axis='x', alpha=0.3)
        
        # 3. Training Curve
        ax3 = fig.add_subplot(gs[1, :2])
        if 'training_curve' in results:
            steps, rewards = results['training_curve']
            ax3.plot(steps, rewards, linewidth=2, color=self.colors['doa'])
            ax3.fill_between(steps, rewards, alpha=0.2, color=self.colors['doa'])
            ax3.set_title('Training Progress', fontweight='bold')
            ax3.set_xlabel('Steps')
            ax3.set_ylabel('Reward')
            ax3.grid(True, alpha=0.3)
        
        # 4. HP Distribution
        ax4 = fig.add_subplot(gs[1, 2])
        if 'hp_importance' in results:
            hp_names = list(results['hp_importance'].keys())[:5]  # Top 5
            importance = list(results['hp_importance'].values())[:5]
            ax4.barh(hp_names, importance, color=self.colors['doa'], alpha=0.7, edgecolor='black')
            ax4.set_xlabel('Importance')
            ax4.set_title('Top Hyperparameters', fontweight='bold')
            ax4.grid(axis='x', alpha=0.3)
        
        # 5. Efficiency
        ax5 = fig.add_subplot(gs[2, :2])
        if 'efficiency_data' in results:
            methods = list(results['efficiency_data'].keys())
            times = [results['efficiency_data'][m][0] for m in methods]
            scores = [results['efficiency_data'][m][1] for m in methods]
            colors_list = [self.colors.get(m, '#444') for m in methods]
            ax5.scatter(times, scores, s=200, c=colors_list, alpha=0.7, edgecolors='black', linewidth=2)
            for i, method in enumerate(methods):
                ax5.annotate(method, (times[i], scores[i]), xytext=(5, 5),
                           textcoords='offset points', fontsize=9)
            ax5.set_xlabel('Time (minutes)')
            ax5.set_ylabel('Performance')
            ax5.set_title('Efficiency Frontier', fontweight='bold')
            ax5.set_xscale('log')
            ax5.grid(True, alpha=0.3)
        
        # 6. Statistics
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.axis('off')
        if 'stats' in results:
            stats_text = "📊 Optimization Summary\n" + "─" * 30 + "\n"
            for key, value in results['stats'].items():
                stats_text += f"{key}: {value}\n"
            ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes,
                    fontsize=10, verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('DOA-DQN Comprehensive Results Dashboard', fontsize=16, fontweight='bold', y=0.995)
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()
    
    # ========================================================================
    # VISUALIZATION 10: SOTA Comparison Table (as image)
    # ========================================================================
    
    def plot_sota_table(self, sota_data: Dict[str, Dict], filename: str = 'v10_sota_table.png'):
        """
        Create a comparison table as an image.
        
        sota_data: {method: {metric: value}}
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('off')
        
        methods = list(sota_data.keys())
        metrics = list(sota_data[methods[0]].keys()) if methods else []
        
        # Build table data
        table_data = [[m] + [f"{sota_data[m][metric]:.2f}" for metric in metrics] for m in methods]
        
        # Create table
        table = ax.table(cellText=table_data,
                        colLabels=['Method'] + metrics,
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.2] + [0.15] * len(metrics))
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)
        
        # Style header
        for i in range(len(metrics) + 1):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Highlight DOA row
        if 'DOA-DQN' in methods:
            doa_idx = methods.index('DOA-DQN') + 1
            for j in range(len(metrics) + 1):
                table[(doa_idx, j)].set_facecolor('#FFEB3B')
                table[(doa_idx, j)].set_text_props(weight='bold')
        
        # Alternate row colors
        for i in range(1, len(methods) + 1):
            color = '#f0f0f0' if i % 2 == 0 else 'white'
            for j in range(len(metrics) + 1):
                table[(i, j)].set_facecolor(color)
        
        plt.title('SOTA Comparison: DOA-DQN vs Benchmark Methods', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    visualizer = DOADQNVisualizer('./doa_dqn_visualizations')
    
    print("=" * 70)
    print("DOA-DQN Visualization Suite")
    print("=" * 70)
    
    # Example 1: DOA Convergence
    print("\n[1/10] Plotting DOA Convergence...")
    history_best = np.array([100, 95, 88, 82, 78, 75, 73, 72, 71, 70])
    history_mean = np.array([50, 55, 60, 65, 70, 71.5, 72, 72.3, 72.5, 72.6])
    visualizer.plot_doa_convergence(history_best, history_mean)
    
    # Example 2: Performance Comparison
    print("[2/10] Plotting Performance Comparison...")
    methods_scores = {
        'DOA-DQN': (437.5, 18.2),
        'Optuna': (445.0, 22.1),
        'Random': (380.0, 45.3),
        'BOHB': (452.0, 16.5),
    }
    visualizer.plot_performance_comparison(methods_scores)
    
    # Example 3: Efficiency Frontier
    print("[3/10] Plotting Efficiency Frontier...")
    methods_efficiency = {
        'DOA-DQN': (437.5, 6.0, 15),
        'Optuna': (445.0, 40.0, 20),
        'Random': (380.0, 8.0, 20),
        'BOHB': (452.0, 15.0, 20),
    }
    visualizer.plot_efficiency_frontier(methods_efficiency)
    
    # Example 4: Radar Comparison
    print("[4/10] Plotting Radar Comparison...")
    envs_methods = {
        'CartPole': {'DOA-DQN': 87.4, 'Optuna': 89.0, 'Random': 76.0},
        'LunarLander': {'DOA-DQN': 80.0, 'Optuna': 85.0, 'Random': 60.0},
        'Pong': {'DOA-DQN': 85.0, 'Optuna': 88.0, 'Random': 65.0},
    }
    visualizer.plot_radar_comparison(envs_methods)
    
    # Example 5: Robustness Boxplot
    print("[5/10] Plotting Robustness Boxplot...")
    methods_seeds = {
        'DOA-DQN': [435, 440, 438, 442, 436],
        'Optuna': [443, 447, 441, 445, 446],
        'Random': [380, 390, 375, 385, 395],
    }
    visualizer.plot_robustness_boxplot(methods_seeds)
    
    # Example 6: Training Curves
    print("[6/10] Plotting Training Curves...")
    steps = np.linspace(0, 50000, 100)
    methods_curves = {
        'DOA-DQN': (steps, 50 + 100 * (1 - np.exp(-steps / 10000))),
        'Optuna': (steps, 50 + 105 * (1 - np.exp(-steps / 12000))),
        'Random': (steps, 50 + 80 * (1 - np.exp(-steps / 20000))),
    }
    visualizer.plot_training_curves(methods_curves)
    
    # Example 7: Hyperparameter Sensitivity
    print("[7/10] Plotting Hyperparameter Sensitivity...")
    hp_sensitivity = {
        'learning_rate': {1e-4: 400, 5e-4: 435, 1e-3: 420, 5e-3: 300},
        'gamma': {0.90: 380, 0.95: 420, 0.99: 437, 0.999: 430},
        'batch_size': {32: 430, 64: 437, 128: 435, 256: 420},
    }
    visualizer.plot_hyperparameter_heatmap(hp_sensitivity)
    
    # Example 8: Population Diversity
    print("[8/10] Plotting Population Diversity...")
    diversity = np.array([0.95, 0.92, 0.88, 0.82, 0.75, 0.68, 0.60, 0.55, 0.52, 0.50])
    visualizer.plot_population_diversity(diversity)
    
    # Example 9: Comprehensive Dashboard
    print("[9/10] Plotting Comprehensive Dashboard...")
    dashboard_data = {
        'doa_history': np.array([100, 95, 88, 82, 78, 75, 73, 72, 71, 70]),
        'method_scores': {'DOA-DQN': 437.5, 'Optuna': 445, 'Random': 380},
        'training_curve': (steps, 50 + 100 * (1 - np.exp(-steps / 10000))),
        'hp_importance': {'learning_rate': 0.25, 'gamma': 0.22, 'batch_size': 0.18},
        'efficiency_data': {
            'DOA-DQN': (6, 437),
            'Optuna': (40, 445),
            'Random': (8, 380),
        },
        'stats': {
            'Evaluations': 15,
            'Time (h)': 0.1,
            'Best Score': 437.5,
        }
    }
    visualizer.plot_comprehensive_dashboard(dashboard_data)
    
    # Example 10: SOTA Table
    print("[10/10] Plotting SOTA Table...")
    sota_data = {
        'DOA-DQN': {'CartPole': 437.5, 'LunarLander': 200, 'Pong': 18.5},
        'Optuna': {'CartPole': 445.0, 'LunarLander': 205, 'Pong': 19.0},
        'BOHB': {'CartPole': 452.0, 'LunarLander': 208, 'Pong': 19.2},
        'Random': {'CartPole': 380.0, 'LunarLander': 150, 'Pong': 12.5},
    }
    visualizer.plot_sota_table(sota_data)
    
    print("\n" + "=" * 70)
    print("✅ All visualizations saved!")
    print("=" * 70)