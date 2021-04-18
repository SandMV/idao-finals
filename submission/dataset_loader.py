from typing import List, Tuple, Union
from funnel_loader import load_funnel
from client_loader import load_client
from trxn_loader import load_trxn
from payment_loader import load_payment
from com_loader import load_com
from configparser import ConfigParser

import pandas as pd
import numpy as np

from os import path

def trim_to_meta(ds: pd.DataFrame, columns_meta: List[str]) -> pd.DataFrame:
    ds_c = set(ds.columns.to_list())
    meta_c = set(columns_meta)

    extra_c = ds_c - meta_c
    ds.drop(columns=extra_c, inplace=True)

    missing_c = meta_c - ds_c
    for c in missing_c:
        ds[c] = 0

    return ds


def load_dataset(cfg: ConfigParser, in_test=False, columns_meta: List[str] = None) \
        -> Tuple[np.ndarray, Union[pd.DataFrame, None]]:
    ds, trg = load_funnel(cfg['DATA']['Users'], in_test)

    if path.exists(cfg['DATA']['Socdem']):
        socdem = load_client(cfg['DATA']['Socdem'])
        ds['has_socdem'] = ds.index.isin(set(socdem.index))
        ds = ds.join(socdem, how='left')
        del socdem

    if path.exists(cfg['DATA']['Trxn']) and path.exists(cfg['DATA']['MccDict']):
        trxn = load_trxn(cfg['DATA']['Trxn'], cfg['DATA']['MccDict'], 1. if in_test else .95)
        ds['has_trxn'] = ds.index.isin(set(trxn.index))
        ds = ds.join(trxn, how='left')
        del trxn

    if path.exists(cfg['DATA']['Pmnts']):
        pmnts = load_payment(cfg['DATA']['Pmnts'])
        ds['has_pmts'] = ds.index.isin(set(pmnts.index))
        ds = ds.join(pmnts, how='left')
        del pmnts

    if path.exists(cfg['DATA']['Com']):
        com = load_com(cfg['DATA']['Com'])
        ds['has_com'] = ds.index.isin(set(com.index))
        ds = ds.join(com, how='left')
        del com

    ds.fillna(0, inplace=True)

    if in_test and columns_meta:
        ds = trim_to_meta(ds, columns_meta)

    return ds, trg
