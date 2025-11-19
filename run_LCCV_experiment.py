import argparse
import ConfigSpace
import logging
import matplotlib.pyplot as plt
import pandas as pd
from lccv import LCCV
from IPL import IPL
from surrogate_model import SurrogateModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_space_file', type=str, default='lcdb_config_space_knn.json')
    parser.add_argument('--configurations_performance_file', type=str, default='config_performances_dataset-1457.csv')
    # max_anchor_size: connected to the configurations_performance_file. The max value upon which anchors are sampled
    parser.add_argument('--minimal_anchor', type=int, default=64)
    parser.add_argument('--num_iterations', type=int, default=50)

    return parser.parse_args()


def run(args):
    datasets = ['config_performances_dataset-6.csv', 'config_performances_dataset-11.csv','config_performances_dataset-1457.csv']
    min_anchors = [32, 32, 64]
    config_space = ConfigSpace.ConfigurationSpace.from_json(args.config_space_file)
    for idy, dataset in enumerate(datasets):
        df = pd.read_csv(dataset)
        surrogate_model = SurrogateModel(config_space)
        surrogate_model.fit(df)
        max_anchor = max(df['anchor_size'])
        min_anchor = min_anchors[idy]
        lccv = LCCV(surrogate_model, min_anchor, max_anchor)
        best_so_far = None
        
        for idx in range(args.num_iterations):
            theta_new = dict(config_space.sample_configuration())
            result = lccv.evaluate_model(best_so_far, theta_new)
            final_result = result[-1][1]
            if best_so_far is None or final_result < best_so_far:
                best_so_far = final_result
            x_values = [i[0] for i in result]
            y_values = [i[1] for i in result]
            plt.plot(x_values, y_values, "-o")

        plt.xscale('log', base=2)
        plt.title(f"Performance LCCV on {dataset}")
        plt.show()


if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    run(parse_args())
