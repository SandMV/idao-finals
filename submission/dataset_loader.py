from typing import List, Tuple, Union
from funnel_loader import load_funnel
from client_loader import load_client
from trxn_loader import load_trxn
from configparser import ConfigParser

import pandas as pd
import numpy as np


def trim_to_meta(ds: pd.DataFrame, columns_meta: List[str]) -> pd.DataFrame:
    ds_c = set(ds.columns)
    meta_c = set(columns_meta)

    extra_c = ds_c - meta_c
    ds.drop(columns=extra_c, inplace=True)

    missing_c = meta_c - ds_c
    for c in missing_c:
        ds[c] = 0

    return ds


def load_dataset(cfg: ConfigParser, in_test=False, columns_meta: List[str] = None) \
        -> Tuple[np.ndarray, Union[pd.DataFrame, None]]:
    fun, trg = load_funnel(cfg['DATA']['Users'], in_test)
    socdem = load_client(cfg['DATA']['Socdem'])

    ds = fun.join(socdem, how='left')
    del fun
    del socdem

    # trxn = load_trxn(cfg['DATA']['Trxn'], cfg['DATA']['MccDict'], 1. if in_test else .95)
    # ds = ds.join(trxn, how='left')
    # del trxn

    na_c = ds.columns[ds.isna().sum(axis=0) > 0]
    for c in na_c:
        ds[f'isna_{c}'] = ds[c].isna()
        ds[c].fillna(0, inplace=True)

    if in_test and columns_meta:
        ds = trim_to_meta(ds, columns_meta)

    return ds, trg
