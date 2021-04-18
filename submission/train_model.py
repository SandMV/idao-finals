import configparser
import logging

from operator import itemgetter
from dataset_loader import load_dataset
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
import numpy as np

import joblib
import lightgbm as lgb
import json

logging.basicConfig(format='%(asctime)s %(message)s', filename='training.log', level=logging.DEBUG)

LGM_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.05,
    'subsample': 0.7,
    'class_weight': 'balanced',
    'colsample_bytree': 0.7,
    'max_depth': 5,
    'num_leaves': 256,
}

RF_PARAMS = {
    'n_estimators': 500,
    'criterion': 'entropy',
    'min_samples_leaf': 10,
    'max_samples': 0.6,
}


def find_threshold(booster, dts, gains, n_calls):
    def mean_income(labels, gains, n_calls):
        return (labels * (gains - 4000 * n_calls)).mean()

    scores = booster.predict_proba(dts)
    if scores.shape[1] == 2:
        scores = scores[:, 1]
    else:
        # because of fucked up rfc implementation!
        scores = scores[:, 0]
    thrs = np.linspace(0, 1, 1000)
    return max(
        ((t, mean_income(scores > t, gains, n_calls)) for t in thrs),
        key=itemgetter(1)
    )


def main(cfg):
    # parse config
    dataset, trg = load_dataset(cfg, False, None)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    # let's remember fold for future threshold optimization
    logging.info(f'Dataset shape: {dataset.shape}')

    folds = [(tr, te) for tr, te in kfold.split(dataset, trg.sale_flg)]
    # X_train = lgb.Dataset(
    #     data=dataset,
    #     label=trg.sale_flg.to_numpy().astype(np.int8),
    # )
    #
    # eval_hist = lgb.cv(
    #     LGM_PARAMS, X_train,
    #     show_stdv=False,
    #     verbose_eval=False,
    #     num_boost_round=1000,
    #     early_stopping_rounds=50,
    #     return_cvbooster=True,
    #     folds=folds
    # )
    # logging.info(f'Mean AUC: {eval_hist["auc-mean"][-1]:4.3f}')
    # boosters = eval_hist.pop('cvbooster', None).boosters

    state_dict = {
        'columns_meta': dataset.columns.to_list(),
        'models': []
    }

    gains = trg[cfg['COLUMNS']['GAINS']].to_numpy()
    n_calls = trg[cfg['COLUMNS']['N_CALLS']].to_numpy()

    for i, (tr, te) in enumerate(folds):
        model_name = f'model_{i}.bst'
        model = RandomForestClassifier(**RF_PARAMS)
        model.fit(dataset.iloc[tr], trg.sale_flg.iloc[tr].to_numpy())
        model_thr, model_sc = find_threshold(model, dataset.iloc[te], gains[te], n_calls[te])
        state_dict['models'].append({
            'model': model_name,
            'model_thr': 0.1
        })
        print(f'Found threshold {model_thr:4.3f} with score {model_sc:10.3f} for model with index {i}')
        joblib.dump(model, model_name)
        # b.save_model(model_name)

    with open('state_dict.json', mode='w') as out:
        json.dump(state_dict, out)

    logging.info("model was trained")


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("./config.ini")
    main(cfg=config)
