import configparser
import pathlib as path

import numpy as np
import pandas as pd
import joblib


def main(cfg):
    # parse config
    DATA_FOLDER = path.Path(cfg["DATA"]["DatasetPath"])
    USER_ID = cfg["COLUMNS"]["USER_ID"]
    PREDICTION = cfg["COLUMNS"]["PREDICTION"]
    MODEL_PATH = path.Path(cfg["MODEL"]["FilePath"])
    SUBMISSION_FILE = path.Path(cfg["SUBMISSION"]["FilePath"])

    funnel = pd.read_csv(f'{DATA_FOLDER}/{cfg["DATA"]["UsersFile"]}').set_index('client_id')
    client = pd.read_csv(f'{DATA_FOLDER}/{cfg["DATA"]["SocdemFile"]}').set_index('client_id')

    drop_columns = []
    drop_columns.extend(cfg['COLUMNS']['FUNNEL_DROP_FEATS'].split(','))
    drop_columns.extend(cfg['COLUMNS']['CLIENT_DROP_FEATS'].split(','))

    pd_X = funnel.join(client)
    pd_X.drop(drop_columns, axis=1, inplace=True)

    model = joblib.load(MODEL_PATH)
    pred = (model.predict_proba(pd_X)[:, 1] > 0.5).astype(np.int32)

    submission = pd.DataFrame({
        USER_ID: pd_X.index,
        PREDICTION: pred
    })
    submission.to_csv(SUBMISSION_FILE, index=False)


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("./config.ini")
    main(cfg=config)
