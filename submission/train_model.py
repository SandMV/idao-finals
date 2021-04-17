import configparser
import pathlib as path
import logging

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

import pandas as pd
import joblib

logging.basicConfig(format='%(asctime)s %(message)s', filename='training.log', level=logging.DEBUG)

RF_PARAMS = {
    'n_estimators': 500,
    'criterion': 'entropy',
    'min_samples_leaf': 10,
    'max_samples': 0.6
}

def main(cfg):
    # parse config
    DATA_FOLDER = path.Path(cfg["DATA"]["DatasetPath"])
    MODEL_PATH = path.Path(cfg["MODEL"]["FilePath"])

    funnel = pd.read_csv(f'{DATA_FOLDER}/{cfg["DATA"]["UsersFile"]}').set_index('client_id')
    client = pd.read_csv(f'{DATA_FOLDER}/{cfg["DATA"]["SocdemFile"]}').set_index('client_id')
    client['region'][(client.region > 60) | (client.region == -1)] = 61
    client['city'][(client.city > 1000) | (client.city == -1)] = 1001

    # first, join all the data frames
    pd_train = funnel.join(client)
    target = pd_train[cfg['COLUMNS']['FUNNEL_TARGET']]

    drop_columns = ['sale_amount', 'contacts', cfg['COLUMNS']['FUNNEL_TARGET']]
    drop_columns.extend(cfg['COLUMNS']['FUNNEL_DROP_FEATS'].split(','))
    drop_columns.extend(cfg['COLUMNS']['CLIENT_DROP_FEATS'].split(','))

    cat_columns = []
    # cat_columns.extend(cfg['COLUMNS']['FUNNEL_CAT_FEATS'].split(','))
    cat_columns.extend(cfg['COLUMNS']['CLIENT_CAT_FEATS'].split(','))

    con_columns = []
    con_columns.extend(cfg['COLUMNS']['FUNNEL_CON_FEATS'].split(','))
    con_columns.extend(cfg['COLUMNS']['CLIENT_CON_FEATS'].split(','))

    pd_train.drop(drop_columns, axis=1, inplace=True)
    con_prep = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    cat_prep = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    prep = ColumnTransformer(
        transformers=[
            ('con', con_prep, con_columns),
            ('cat', cat_prep, cat_columns)
        ]
    )

    model = Pipeline([
        ('prep', prep),
        ('clf', RandomForestClassifier(**RF_PARAMS))
    ])

    model.fit(pd_train, target)
    logging.info("model was trained")
    joblib.dump(model, MODEL_PATH)


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("./config.ini")
    main(cfg=config)
