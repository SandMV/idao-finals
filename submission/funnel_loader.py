from typing import Union, Tuple
import pandas as pd

TARGET_COLUMNS = ['sale_flg', 'sale_amount', 'contacts']


def load_funnel(funnel_file: str, in_test: False) -> Tuple[pd.DataFrame, Union[None, pd.DataFrame]]:
    funnel = pd.read_csv(funnel_file, sep=',').set_index('client_id').sort_index()
    funnel.drop(columns=['region_cd'], inplace=True)

    target_c = None
    if not in_test:
        target_c = funnel[TARGET_COLUMNS]
        target_c.sale_amount.fillna(0, inplace=True)
        funnel.drop(columns=TARGET_COLUMNS, inplace=True)

    funnel.fillna(0, inplace=True)

    # rare features
    funnel['feature_4_0'] = funnel.feature_4 < 1e-10
    funnel['feature_4_1'] = funnel.feature_4 > 1e-10

    funnel['feature_5_0'] = funnel.feature_5 < 1e-10
    funnel['feature_5_1'] = funnel.feature_5 > 1e-10

    return funnel, target_c
