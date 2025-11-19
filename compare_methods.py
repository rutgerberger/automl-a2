import argparse
import ConfigSpace
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Import your existing classes
from IPL import IPL
from lccv import LCCV
from surrogate_model import SurrogateModel

def parse_args():
    parser = argparse.ArgumentParser(description='Compare HPO Methods')
    parser.add_argument('--config_space_file', type=str, default='lcdb_config_space_knn.json')
    parser.add_argument('--configurations_performance_file', type=str, default='config_performances_dataset-6.csv')
    parser.add_argument('--num_iterations', type=int, default=50)
    # The image mentions 16000 as the full evaluation cost
    parser.add_argument('--max_anchor_size', type=int, default=16000)
    # Start small for LCCV
    parser.add_argument('--minimal_anchor', type=int, default=16)
    return parser.parse_args()

def calculate_mean_results(all_seed_runs):
    """Averages costs and errors across multiple seed runs."""
    if not all_seed_runs: return None
    
    num_runs = len(all_seed_runs)
    num_points = len(all_seed_runs[0]['LCCV']['costs'])
    
    mean_results = {'LCCV': {'costs': np.zeros(num_points), 'errors': np.zeros(num_points)},
                    'IPL': {'costs': np.zeros(num_points), 'errors': np.zeros(num_points)}}

    for run in all_seed_runs:
        for method in ['LCCV', 'IPL']:
            mean_results[method]['costs'] += np.array(run[method]['costs'])
            mean_results[method]['errors'] += np.array(run[method]['errors'])
    
    # Divide by number of runs and convert back to list for consistency
    for method in ['LCCV', 'IPL']:
        mean_results[method]['costs'] = (mean_results[method]['costs'] / num_runs).tolist()
        mean_results[method]['errors'] = (mean_results[method]['errors'] / num_runs).tolist()
        
    return mean_results


def run_comparison(args):
    datasets = ['config_performances_dataset-1457.csv']
    random_seeds = [1,2,3,4]
    for dataset in datasets:
        df = pd.read_csv(dataset)
        anchors = []
        current_anchor = args.minimal_anchor
        max_anchor = max(df['anchor_size'])
        while current_anchor < max_anchor:
            anchors.append(current_anchor)
            current_anchor *= 2 
        # Ensure the max anchor is always the last item
        if max_anchor not in anchors:
            anchors.append(max_anchor)

        dataset_results_by_seed = []
        for seed in random_seeds:


            config_space = ConfigSpace.ConfigurationSpace.from_json(args.config_space_file)
            config_space.seed = seed
            surrogate_model = SurrogateModel(config_space)
            surrogate_model.fit(df)
            lccv = LCCV(surrogate_model, 16, args.max_anchor_size)
            ipl = IPL(config_space, surrogate_model, anchors=anchors, max_anchor_size=args.max_anchor_size)
            # We track specific state for the loop
            lccv_best_error = None
            # IPL tracks its own best_error internally, but we track it here for plotting consistency
            ipl_best_error = float('inf') 
            results = { 
                'LCCV': {'costs': [0], 'errors': [1.0]},
                'IPL': {'costs': [0], 'errors': [1.0]}
            }

            print(f"{'='*60}")
            print(f"Starting Comparison: {args.num_iterations} iterations")
            print(f"{'='*60}")

            for i in range(args.num_iterations):
                theta_new = dict(config_space.sample_configuration())
                #LCCV
                lccv_curve = lccv.evaluate_model(lccv_best_error, theta_new)
                # sum of all anchors evaluated in this curve
                cost_lccv = sum([step[0] for step in lccv_curve])
                # Logic: Did it finish?
                last_step = lccv_curve[-1]
                if last_step[0] == args.max_anchor_size:
                    # It didn't prune: Check if it's a new best.
                    actual_error = last_step[1]
                    if lccv_best_error is None or actual_error < lccv_best_error:
                        lccv_best_error = actual_error
                results['LCCV']['costs'].append(results['LCCV']['costs'][-1] + cost_lccv)
                results['LCCV']['errors'].append(lccv_best_error)
                #IPL
                ipl.evaluate(theta_new)
                last_run = ipl.evaluation_history[-1]
                cost_ipl = sum(last_run['observed_anchors'])
                if last_run['stopped_at_anchor'] is None:
                    # It completed. The "observed_errors" list has the actual errors.
                    # The last one is the error at max_anchor_size.
                    actual_error = last_run['observed_errors'][-1]
                    if actual_error < ipl_best_error:
                        ipl_best_error = actual_error
                
                results['IPL']['costs'].append(results['IPL']['costs'][-1] + cost_ipl)
                results['IPL']['errors'].append(ipl_best_error)

                if (i+1) % 10 == 0:
                    print(f"Iter {i+1}: LCCV Best={lccv_best_error:.4f}, IPL Best={ipl_best_error:.4f}")

            dataset_results_by_seed.append(results)
            ipl.reset()
    
        mean_results = calculate_mean_results(dataset_results_by_seed)
        print_results_table(mean_results, dataset)
        plot_convergence(results)

# def print_results_table(results, iterations):
#     print("\n" + "="*60)
#     print("2. THE SUMMARY: RESULTS TABLE")
#     print("="*60)
#     print(f"{'Method':<25} | {'Final Best Perf':<18} | {'Total Cost':<15}")
#     print("-" * 65)
    
#     for method in ['LCCV', 'IPL']:
#         final_perf = results[method]['errors'][-1]
#         total_cost = results[method]['costs'][-1]
#         print(f"{method:<25} | {final_perf:.5f}            | {total_cost:,.0f}")
#     print("-" * 65)


def print_results_table(results, dataset):
    print("\n" + "="*60)
    print(f"THE SUMMARY for dataset {dataset}: (Averaged over 4 seeds)")
    print("="*60)
    
    for method in ['LCCV', 'IPL']:
        if method in results:
            final_perf = results[method]['errors'][-1]
            total_cost = results[method]['costs'][-1]
            print(f"{method:<25} | Final Best Perf: {final_perf:.5f} | Total Cost: {total_cost:,.0f}")
        else:
            print(f"{method:<25} | Data Missing")

def plot_convergence(results):
    plt.figure(figsize=(10, 6))
    # Define styles for clear distinction
    styles = {
        'LCCV':     {'color': 'blue',  'ls': '--', 'lw': 2, 'label': 'LCCV'},
        'IPL':      {'color': 'red',   'ls': '-.', 'lw': 2, 'label': 'IPL'}
    }
    all_y_values = []
    for method in ['LCCV', 'IPL']:
        x_data = results[method]['costs'][1:]
        y_data = results[method]['errors'][1:]
        if not x_data: continue
        all_y_values.extend(y_data)
        plt.plot(x_data, y_data, 
                 color=styles[method]['color'], 
                 linestyle=styles[method]['ls'], 
                 linewidth=styles[method]['lw'], 
                 label=styles[method]['label'])
    plt.xscale('log')
    # if all_y_values:
    #     y_min = min(all_y_values)
    #     y_max = max(all_y_values)
    #     margin = (y_max - y_min) * 0.1
    #     if margin == 0: margin = 0.01
        #plt.ylim(max(0, y_min - margin), y_max + margin)

    plt.xlabel('Cumulative Computational Cost (Log Scale)')
    plt.ylabel('Best-so-far Error Rate')
    plt.title('Cost vs. Quality: Method Comparison')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend()
    
    #plt.tight_layout()
    plt.show()
if __name__ == '__main__':
    args = parse_args()
    run_comparison(args)
