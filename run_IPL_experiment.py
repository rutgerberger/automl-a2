import argparse
import ConfigSpace
import logging
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from IPL import IPL
from surrogate_model import SurrogateModel


def parse_args():
    parser = argparse.ArgumentParser(description='Run IPL hyperparameter optimization')
    parser.add_argument('--config_space_file', type=str, default='lcdb_config_space_knn.json')
    parser.add_argument('--configurations_performance_file', type=str, default='config_performances_dataset-6.csv')
    parser.add_argument('--minimal_anchor', type=int, default=16)
    parser.add_argument('--max_anchor_size', type=int, default=16000)
    parser.add_argument('--num_iterations', type=int, default=50)
    parser.add_argument('--anchor_schedule', type=str, default='16,32,64,128,256,512,1024', 
                       help='Comma-separated list of anchor sizes')
    parser.add_argument('--plot_results', action='store_true', 
                       help='Generate visualization plots of IPL performance')
    parser.add_argument('--plot', type=str, default='true', 
                       choices=['true', 'none'])

    return parser.parse_args()


def create_ipl_plots(all_curves_data, best_so_far, plot='true'):
    """Create visualization plots for IPL results"""
    
    if plot == 'none':
        return
    
    completed_count = sum(1 for curve in all_curves_data if curve['stopped_at'] is None)
    stopped_count = len(all_curves_data) - completed_count
    
    if plot == 'true':
        
        for curve in all_curves_data:
            x_vals = curve['x_values']
            y_vals = curve['y_values']
            stopped_at = curve['stopped_at']
            
            if stopped_at is None:
                # Completed configuration
                plt.plot(x_vals, y_vals, 'o-', color='green', alpha=0.4, linewidth=1, markersize=2)
            else:
                # Stopped configuration
                stop_idx = x_vals.index(stopped_at)
                plt.plot(x_vals[:stop_idx+1], y_vals[:stop_idx+1], 'o-', color='red', alpha=0.4, linewidth=1, markersize=2)
                plt.plot(stopped_at, y_vals[stop_idx], 'X', color='darkred', markersize=6)
        
        if best_so_far is not None:
            plt.axhline(y=best_so_far, color='blue', linestyle='--', linewidth=2, 
                       label=f'Best Error: {best_so_far:.4f}')
        
        plt.xlabel('Anchor Size')
        plt.ylabel('Error Rate')
        plt.title(f'IPL Learning Curves Summary\n({completed_count} Completed, {stopped_count} Stopped Early)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xscale('log')


    plt.tight_layout()
    plt.show()


def run(args):
    # Load configuration space and data
    config_space = ConfigSpace.ConfigurationSpace.from_json(args.config_space_file)
    df = pd.read_csv(args.configurations_performance_file)

    dataset_max_anchor = df['anchor_size'].max()
    dataset_min_anchor = df['anchor_size'].min()
    unique_anchors = sorted(df['anchor_size'].unique())

    print(f"Dataset anchor sizes range from {dataset_min_anchor} to {dataset_max_anchor}")
    print(f"Unique anchor sizes in dataset: {unique_anchors}")
    
    # Train surrogate model
    surrogate_model = SurrogateModel(config_space)
    surrogate_model.fit(df)
    
    # Parse anchor schedule and filter by max_anchor_size
    all_anchors = [int(x.strip()) for x in args.anchor_schedule.split(',')]
    anchors = [a for a in all_anchors if a <= args.max_anchor_size]
    
    # Initialize IPL with just config_space and surrogate_model
    ipl_evaluator = IPL(config_space, surrogate_model, anchors=anchors, max_anchor_size=dataset_max_anchor)
    
    best_so_far = None
    evaluation_history = []
    all_curves_data = []
    
    for idx in range(args.num_iterations):
        # Sample a new configuration
        theta_new = dict(config_space.sample_configuration())
        
        print(f"\n--- Iteration {idx+1}/{args.num_iterations} ---")
        print(f"Configuration: {theta_new}")
        
        # Evaluate with IPL
        result = ipl_evaluator.evaluate(theta_new)
        
        # Get the evaluation data from IPL's history
        if hasattr(ipl_evaluator, 'evaluation_history') and ipl_evaluator.evaluation_history:
            last_eval = ipl_evaluator.evaluation_history[-1]
            
            if (last_eval['configuration'] == theta_new and 
                last_eval['observed_anchors'] and 
                last_eval['observed_errors']):
                
                x_values = last_eval['observed_anchors']
                y_values = last_eval['observed_errors']
                stopped_at = last_eval['stopped_at_anchor']
                final_pred = last_eval['final_prediction']
                
                # Store for plotting (if plotting is enabled)
                if args.plot_results or args.plot != 'none':
                    curve_data = {
                        'iteration': idx,
                        'x_values': x_values,
                        'y_values': y_values,
                        'stopped_at': stopped_at,
                        'final_pred': final_pred,
                        'configuration': theta_new
                    }
                    all_curves_data.append(curve_data)
        
        if result is None:
            # Configuration was stopped early
            print("→ STOPPED EARLY")
            evaluation_history.append({
                'iteration': idx,
                'configuration': theta_new,
                'final_error': None,
                'status': 'stopped_early'
            })
        else:
            # Configuration completed evaluation
            final_error = result
            print(f"→ COMPLETED: Final error = {final_error:.4f}")
            
            # Update best performance
            if best_so_far is None or final_error < best_so_far:
                old_best = best_so_far if best_so_far is not None else float('inf')
                best_so_far = final_error
                print(f"NEW BEST: {final_error:.4f} (improved from {old_best:.4f})")
            
            evaluation_history.append({
                'iteration': idx,
                'configuration': theta_new,
                'final_error': final_error,
                'status': 'completed'
            })

    # Generate plots if requested
    if args.plot_results or args.plot != 'none':
        print("\nGenerating IPL visualization plots...")
        create_ipl_plots(all_curves_data, best_so_far, args.plot)

    # Final statistics
    stats = ipl_evaluator.get_evaluation_stats()
    
    print(f"\n{'='*60}")
    print("IPL EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    print(f"Best error found: {best_so_far:.4f}")
    print(f"Total iterations: {args.num_iterations}")
    print(f"Configurations completed: {stats['completed']}")
    print(f"Configurations stopped early: {stats['stopped_early']}")
    print(f"Early stopping rate: {stats['early_stopping_rate']:.1%}")
    
    # Additional insights if we have curve data
    if all_curves_data:
        completed_count = sum(1 for curve in all_curves_data if curve['stopped_at'] is None)
        stopped_count = len(all_curves_data) - completed_count
        print(f"\nVISUALIZATION DATA:")
        print(f"- Configurations with learning curves: {len(all_curves_data)}")
        print(f"- Completed evaluations: {completed_count}")
        print(f"- Stopped early: {stopped_count}")
    
    return evaluation_history, best_so_far, all_curves_data


if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    args = parse_args()
    history, best_error, curves_data = run(args)