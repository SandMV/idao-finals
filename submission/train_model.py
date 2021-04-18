import configparser
import logging

from operator import itemgetter
from dataset_loader import load_dataset
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import numpy as np

import lightgbm as lgb
import catboost as cbm
import json

logging.basicConfig(format='%(asctime)s %(message)s', filename='training.log', level=logging.DEBUG)

RF_PARAMS = {
    'n_estimators': 500,
    'criterion': 'entropy',
    'min_samples_leaf': 10,
    'max_samples': 0.6,
}


def find_threshold(booster, dts, gains, n_calls):
    def mean_income(labels, gains, n_calls):
        return (labels * (gains - 4000 * n_calls)).mean()

    scores = np.array(booster.predict(data=dts, prediction_type='Probability')[:, 1])
    thrs = np.linspace(0, 1, 1000)
    return max(
        ((t, mean_income(scores > t, gains, n_calls)) for t in thrs),
        key=itemgetter(1)
    )


def main(cfg):
    # parse config
    dataset, trg = load_dataset(cfg, False, None)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    print(f'Dataset shape: {dataset.shape}')
    print('Dataset:')
    print(dataset.head())

    print('Target:')
    print(trg.head())
    print()
    print('Columns:')
    print('\n'.join(dataset.columns.to_list()))

    dataset.to_csv('train_ds.csv', sep=',')
    trg.to_csv('train_trg.csv', sep=',')

    folds = [(tr, te) for tr, te in kfold.split(dataset, trg.sale_flg)]
    # X_train = cbm.Pool(
    #     data=dataset,
    #     label=trg.sale_flg.to_numpy().astype(np.int32),
    # )

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

    # eval_hist = cbm.cv(
    #     LGM_PARAMS, X_train,
    #     show_stdv=True,
    #     verbose_eval=True,
    #     num_boost_round=1000,
    #     return_cvbooster=True,
    #     early_stopping_rounds = 50,
    #     folds=folds
    # )

    # print(f'Mean AUC: {eval_hist["auc-mean"][-1]:4.3f}')
    # boosters = eval_hist.pop('cvbooster', None).boosters

    state_dict = {
        'columns_meta': dataset.columns.to_list(),
        'models': []
    }

    gains = trg[cfg['COLUMNS']['GAINS']].to_numpy()
    n_calls = trg[cfg['COLUMNS']['N_CALLS']].to_numpy()

    for i, (tr, te) in enumerate(folds):
        b = cbm.CatBoostClassifier(
            depth=6,
            iterations=1000,
            learning_rate=0.05,
            subsample=0.7,
            best_model_min_trees=100,
            eval_metric='AUC'
        ).fit(dataset.iloc[tr], trg.sale_flg.iloc[tr])
        model_name = f'model_{i}.bst'
        model_thr, model_sc = find_threshold(b, dataset.iloc[te], gains[te], n_calls[te])
        state_dict['models'].append({
            'model': model_name,
            'model_thr': model_thr,
            'model_sc': model_sc,
            'model_auc': roc_auc_score(
                trg.sale_flg.iloc[te], b.predict(dataset.iloc[te], prediction_type='Probability')[:, 1]
            )
        })
        b.save_model(model_name)

    for md in state_dict['models']:
        print(f'Found threshold {md["model_thr"]:4.3f} with score {md["model_sc"]:10.3f} for model {md["model"]}')
        print(f'AUC {md["model_auc"]} for model {md["model"]}')

    with open('state_dict.json', mode='w') as out:
        json.dump(state_dict, out)

    logging.info("model was trained")


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("./config.ini")
    main(cfg=config)
