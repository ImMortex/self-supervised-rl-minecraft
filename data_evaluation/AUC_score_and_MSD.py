import logging
import coloredlogs
import numpy as np
import pandas as pd
import glob
import os

def get_area_under_graph(graph_x, graph_y):
    """
    Integral calculation to determine the area between graph and x-axis
    """
    A = np.trapz(graph_y, x=graph_x, dx=None) # Integrate along the given axis using the composite trapezoidal rule. Trapezregel
    return A


def calc_AUC_100_for_each_training_variant_sorted(best_score_for_env, graph_columns, all_x_values, steps_begin = 0, steps_end=10_000_000):
    print("area between graph and x-axis for steps " + str(steps_begin) + " to " + str(steps_end))
    graphs_dict: dict = {}
    for graph in graph_columns:
        all_graph_y = df[graph].tolist()

        # filter out nan
        graph_x = []
        graph_y = []

        for i in range(len(all_x_values)):
            if all_x_values[i] < steps_begin:
                continue
            if str(all_graph_y[i]) != "nan":
                graph_x.append(all_x_values[i])
                graph_y.append(all_graph_y[i])
            if all_x_values[i] > steps_end:
                break

        # Mean square deviation from the mean value (MSD)
        mean_value = np.mean(graph_y)

        sum = 0
        for val in graph_y:
            sum += (val-mean_value)** 2

        msd = 1/len(graph_y) * sum

        area = float(get_area_under_graph(graph_x, graph_y))
        max_possible_area = best_score_for_env * (steps_end-steps_begin)  # rectangle
        score = area / max_possible_area  # AUC-100 https://arxiv.org/pdf/1507.00814.pdf p. 11
        graphs_dict[graph] = {"area": score, "msd": msd}
    tuples = []
    for dict_key in graphs_dict:
        tuples.append([dict_key, graphs_dict[dict_key]["area"], graphs_dict[dict_key]["msd"]])
    sorted_list = sorted(tuples, key=lambda x: x[1], reverse=True)
    for o in sorted_list:
        print(o[0], "score: " + str(o[1]), "MSD: " + str(o[2]))
        #print(o[0], format(o[1], ".1E"))


if __name__ == '__main__':
    coloredlogs.install(level='INFO')
    best_score_for_env_breakout = 401.2  # best DQN score value for Breakout as reference see https://arxiv.org/pdf/1507.00814.pdf [3]
    best_score_for_env_minecraft = 128  # Minecraft best possible score in a Epoch of max n steps
    dir_name = "../tmp/csv_input/"
    csv_files: [] = glob.glob(dir_name + '*.csv')
    if len(csv_files) == 0:
        logging.error("no csv files in " + dir_name + " found!")

    for csv_file in csv_files:
        file_base_name: str = os.path.basename(csv_file)
        key = file_base_name.split(".")[0]

        print("\n\n\n")
        logging.info(key)
        df = pd.read_csv(csv_file, ",")
        all_x_values = df[df.columns[1]].tolist()

        graph_columns = df.columns.tolist()[2:] # exclude index column and x column

        # area under curve scores
        if "minecraft" not in file_base_name.lower():
            # Breakout
            unit = 1_000_000
            calc_AUC_100_for_each_training_variant_sorted(best_score_for_env_breakout, graph_columns, all_x_values, 1* unit, 10* unit)

        else:
            # Minecraft
            unit = 51200
            calc_AUC_100_for_each_training_variant_sorted(best_score_for_env_minecraft, graph_columns, all_x_values,
                                                          0*unit,
                                                          1*unit)
            calc_AUC_100_for_each_training_variant_sorted(best_score_for_env_minecraft, graph_columns, all_x_values,
                                                          1 * unit,
                                                          2 * unit)
            calc_AUC_100_for_each_training_variant_sorted(best_score_for_env_minecraft, graph_columns, all_x_values,
                                                          2 * unit,
                                                          3 * unit)
            calc_AUC_100_for_each_training_variant_sorted(best_score_for_env_minecraft, graph_columns, all_x_values,
                                                          3 * unit,
                                                          4 * unit)
            calc_AUC_100_for_each_training_variant_sorted(best_score_for_env_minecraft, graph_columns, all_x_values,
                                                          4 * unit,
                                                          5 * unit)
            calc_AUC_100_for_each_training_variant_sorted(best_score_for_env_minecraft, graph_columns, all_x_values,
                                                          0 * unit,
                                                          5 * unit)

