from typing import Union, Tuple
import pandas as pd




def load_funnel(funnel_file: str, in_test: False) -> Tuple[pd.DataFrame, Union[None, pd.DataFrame]]:
    funnel = pd.read_csv(funnel_file, sep=',').set_index('client_id').sort_index()
    funnel.drop(columns=['region_cd'], inplace=True)

    funnel.fillna(0, inplace=True)

    # rare features
    funnel['feature_4_0'] = funnel.feature_4 < 1e-10
    funnel['feature_4_1'] = funnel.feature_4 > 1e-10

    funnel['feature_5_0'] = funnel.feature_5 < 1e-10
    funnel['feature_5_1'] = funnel.feature_5 > 1e-10

    return funnel
