import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Callable, Optional, Dict, List
from datetime import datetime
import json
import os
from matplotlib import pyplot as plt

"""
    evaluation.py
    Evaluation module for RL Tower Defense
    COMP4010 - Group 12
"""


@dataclass
class EpisodeMetrics:
    """metrics from a single episode"""
    total_reward: float = 0.0
    waves_completed: int = 0
    enemies_killed: int = 0
    towers_placed: int = 0
    survived: bool = False


@dataclass 
class TrainingResult:
    """stores training curve data for an algorithm"""
    algorithm_name: str
    eval_returns: List[float]  # average returns at each evaluation point
    eval_episodes: List[int]   # episode numbers where evaluation occurred
    evaluate_every: int        # how often evaluation was done
    

class AlgorithmComparator:
    """compare multiple algorithms"""
    
    def __init__(self):
        self.results: Dict[str, TrainingResult] = {}
    
    def add_result(self, name, eval_returns, evaluate_every, total_episodes=None):
        """add training results for an algorithm"""
        if total_episodes is None:
            total_episodes = len(eval_returns) * evaluate_every
        
        eval_episodes = [(i + 1) * evaluate_every for i in range(len(eval_returns))]
        
        self.results[name] = TrainingResult(
            algorithm_name=name,
            eval_returns=list(eval_returns),
            eval_episodes=eval_episodes,
            evaluate_every=evaluate_every
        )
    
    def print_comparison(self):
        """print comparison table"""
        if not self.results:
            print("No results to compare.")
            return
        
        print("\n" + "=" * 70)
        print("  ALGORITHM COMPARISON")
        print("=" * 70)
        
        header = f"{'Algorithm':<15} {'Avg Return':>15} {'Final Return':>15} {'Episodes':>12}"
        print(header)
        print("-" * 70)
        
        for name, result in self.results.items():
            avg_return = np.mean(result.eval_returns)
            final_return = result.eval_returns[-1] if result.eval_returns else 0
            total_eps = result.eval_episodes[-1] if result.eval_episodes else 0
            row = f"{name:<15} {avg_return:>15.2f} {final_return:>15.2f} {total_eps:>12}"
            print(row)
        
        print("=" * 70 + "\n")
    
    def get_comparison_dataframe(self):
        """get comparison as dataframe"""
        data = []
        for name, result in self.results.items():
            data.append({
                "algorithm": name,
                "avg_return": np.mean(result.eval_returns),
                "final_return": result.eval_returns[-1] if result.eval_returns else 0,
                "total_episodes": result.eval_episodes[-1] if result.eval_episodes else 0,
                "evaluate_every": result.evaluate_every
            })
        return pd.DataFrame(data)
    
    def plot_learning_curves(self, output_path=None, show=True, title=None):
        """plot learning curves for all algorithms"""
        if not self.results:
            print("No results to compare.")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for name, result in self.results.items():
            ax.plot(result.eval_episodes, result.eval_returns, label=name, linewidth=2)
        
        ax.set_xlabel("Episode")
        ax.set_ylabel("Average Return")
        if title:
            ax.set_title(title)
        else:
            ax.set_title("Learning Curves - Average Return over Training")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {output_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_comparison_bar(self, output_path=None, show=True, title=None):
        """bar chart comparing final average returns"""
        if not self.results:
            print("No results to compare.")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        algorithms = list(self.results.keys())
        avg_returns = [np.mean(self.results[a].eval_returns) for a in algorithms]
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(algorithms)))
        ax.bar(algorithms, avg_returns, color=colors)
        
        ax.set_xlabel("Algorithm")
        ax.set_ylabel("Average Return")
        if title:
            ax.set_title(title)
        else:
            ax.set_title("Average Return Comparison")
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {output_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def save_comparison(self, output_dir):
        """save comparison results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # save summary csv
        df = self.get_comparison_dataframe()
        csv_path = os.path.join(output_dir, "algorithm_comparison.csv")
        df.to_csv(csv_path, index=False)
        
        # save learning curves data
        for name, result in self.results.items():
            curve_df = pd.DataFrame({
                "episode": result.eval_episodes,
                "avg_return": result.eval_returns
            })
            curve_path = os.path.join(output_dir, f"{name}_learning_curve.csv")
            curve_df.to_csv(curve_path, index=False)
        
        # save plots
        self.plot_learning_curves(
            output_path=os.path.join(output_dir, "learning_curves.png"),
            show=False
        )
        self.plot_comparison_bar(
            output_path=os.path.join(output_dir, "comparison_bar.png"),
            show=False
        )
        
        print(f"Results saved to {output_dir}/")


def plot_single_algorithm(eval_returns, evaluate_every, algorithm_name, 
                          output_path=None, show=True, title=None):
    """plot learning curve for a single algorithm"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    episodes = [(i + 1) * evaluate_every for i in range(len(eval_returns))]
    ax.plot(episodes, eval_returns, linewidth=2)
    
    ax.set_xlabel(f"Evaluation Number (Every {evaluate_every} Episodes)")
    ax.set_ylabel("Average Evaluated Return")
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"{algorithm_name} - Average Return over Training")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_hyperparameter_comparison(results_dict, param_name, evaluate_every,
                                   algorithm_name, output_path=None, show=True):
    """
    plot learning curves for different hyperparameter values
    results_dict: {param_value: eval_returns_list}
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for param_value, eval_returns in results_dict.items():
        episodes = [(i + 1) * evaluate_every for i in range(len(eval_returns))]
        ax.plot(episodes, eval_returns, label=f"{param_name} = {param_value}", linewidth=2)
    
    ax.set_xlabel(f"Evaluation Number (Every {evaluate_every} Episodes)")
    ax.set_ylabel("Average Evaluated Return")
    ax.set_title(f"{algorithm_name} - {param_name} Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    # demo with dummy data
    print("=" * 60)
    print("  Evaluation Module Demo")
    print("=" * 60)
    
    comparator = AlgorithmComparator()
    
    # add some dummy results
    dummy_returns_1 = np.cumsum(np.random.randn(100)) + np.linspace(0, 500, 100)
    dummy_returns_2 = np.cumsum(np.random.randn(100)) + np.linspace(0, 300, 100)
    
    comparator.add_result("Algorithm A", dummy_returns_1, evaluate_every=50)
    comparator.add_result("Algorithm B", dummy_returns_2, evaluate_every=50)
    
    comparator.print_comparison()
    comparator.plot_learning_curves(show=True)
    
    print("\nDone.")
