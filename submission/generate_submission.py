import configparser
import pathlib as path

from dataset_loader import load_dataset
import numpy as np
import pandas as pd

import json
import lightgbm as lgb

def main(cfg):
    # parse config
    USER_ID = cfg["COLUMNS"]["USER_ID"]
    PREDICTION = cfg["COLUMNS"]["PREDICTION"]
    SUBMISSION_FILE = path.Path(cfg["SUBMISSION"]["FilePath"])

    with open('state_dict.json', mode='r') as inp:
        state_dict = json.load(inp)

    columns_meta = state_dict['columns_meta']
    ds, *_ = load_dataset(cfg, True, columns_meta)

    models_cnt = len(state_dict['models'])
    preds = np.zeros(len(ds))
    for d in state_dict['models']:
        model = d['model']
        thr = d['model_thr']
        probs = lgb.Booster(model_file=model).predict(ds)
        preds += probs > thr

    submission = pd.DataFrame({
        USER_ID: ds.index,
        PREDICTION: (preds >= models_cnt // 2).astype(np.int32)
    })
    submission.to_csv(SUBMISSION_FILE, index=False)


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("./config.ini")
    main(cfg=config)
