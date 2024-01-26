import os

import pandas as pd
import wandb
import numpy as np

api = wandb.Api()
entity, project = "stcngurs", "mc_agent"
runs = api.runs(entity + "/" + project)


current_base_name = ""
dfs = []

run_names = []


def get_run_name(run):
    base_name = get_base_name(run)

    if "pretrain_mode" not in run.config["parameters"]:
        return base_name

    return base_name + " " + str(run.config["parameters"]["pretrain_mode"]["values"][0])


def get_base_name(run):
    base_name = "wandb_api " + "Minecraft" + " seq len" + str(1) \
                + " " + "NatureCNN" + " agent " + str(run.config["parameters"]["agent_id"]["value"])
    base_name = base_name.replace("/", "_")
    base_name = base_name.replace(":", " ")
    return base_name


for run in runs:
    # .summary contains output keys/values for
    # metrics such as accuracy.
    #  We call ._json_dict to omit large files
    if "finished" in run.tags:
        name = get_run_name(run)

        run_names.append(name)

run_names.sort()
for run_name in run_names:
    for run in runs:

        name = get_run_name(run)
        base_name = get_base_name(run)

        if current_base_name == "":
            current_base_name = base_name

        if "finished" in run.tags and run_name == name:

            dirname = "../tmp/"
            filename = dirname + base_name + ".csv"

            print(name)

            if os.path.isfile(filename):
                continue

            if current_base_name != base_name:
                dfs = []

            current_base_name = base_name


            df_filename = dirname +  name + ".csv"
            if not os.path.isfile(df_filename):
                hist = run.scan_history(keys=["score", "reward", "action_id"])

                run_score = [row["score"] for row in hist]
                run_reward = [row["reward"] for row in hist]
                run_action_id = [row["action_id"] for row in hist]
                run_steps = np.arange(start=0, stop=len(run_score)).tolist()

                #run_steps = [0,1]
                #run_rewards = [3, 4]

                df = pd.DataFrame(
                    {"steps": run_steps,
                     name + "_score": run_score,  name + "_reward": run_reward,  name + "_action_id": run_action_id}
                )
                df.to_csv(df_filename)
            else:
                df = pd.read_csv(df_filename)

            dfs.append(df)

            if len(dfs) == 8:

                merged_df = pd.merge(dfs[0], dfs[1], on='steps', how='outer')
                merged_df = merged_df.sort_values(by='steps')

                print(filename)
                merged_df.to_csv(filename)