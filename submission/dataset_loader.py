from typing import List, Tuple, Union
from funnel_loader import load_funnel
from client_loader import load_client
from trxn_loader import load_trxn
from payment_loader import load_payment
from com_loader import load_com
from balance_loader import load_balance
from aum_loader import load_aum
from appl_loader import load_appl
from configparser import ConfigParser

import pandas as pd
import numpy as np

from os import path

TARGET_COLUMNS = ['sale_flg', 'sale_amount', 'contacts']


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
    ds = load_funnel(cfg['DATA']['Users'], in_test)

    if path.exists(cfg['DATA']['Trxn']) and path.exists(cfg['DATA']['MccDict']):
        heavy_trxn = load_trxn(cfg['DATA']['Trxn'], cfg['DATA']['MccDict'], False, 1. if in_test else .95)
        ds['has_heavy_trxn'] = ds.index.isin(set(heavy_trxn.index))
        ds = ds.join(heavy_trxn, how='left')
        del heavy_trxn

    if path.exists(cfg['DATA']['Trxn']) and path.exists(cfg['DATA']['MccDict']):
        light_trxn = load_trxn(cfg['DATA']['Trxn'], cfg['DATA']['MccDict'], True, 1. if in_test else .95)
        ds_cols = set(ds.columns)
        light_cols = set(light_trxn.columns)
        light_trxn.drop(columns=(ds_cols & light_cols), inplace=True)
        ds['has_light_trxn'] = ds.index.isin(set(light_trxn.index))
        ds = ds.join(light_trxn, how='left')
        del light_trxn

    if path.exists(cfg['DATA']['Socdem']):
        socdem = load_client(cfg['DATA']['Socdem'])
        ds['has_socdem'] = ds.index.isin(set(socdem.index))
        ds = ds.join(socdem, how='left')
        del socdem

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

    if path.exists(cfg['DATA']['Aum']):
        aum = load_aum(cfg['DATA']['Aum'])
        ds['has_aum'] = ds.index.isin(set(aum.index))
        ds = ds.join(aum, how='left')
        del aum

    if path.exists(cfg['DATA']['Balance']):
        balance = load_balance(cfg['DATA']['Balance'])
        ds['has_balance'] = ds.index.isin(set(balance.index))
        ds = ds.join(balance, how='left')
        del balance

    # if path.exists(cfg['DATA']['Appl']):
    #     appl = load_appl(cfg['DATA']['Appl'])
    #     ds['has_appl'] = ds.index.isin(set(appl.index))
    #     ds = ds.join(appl, how='left')
    #     del appl

    ds.fillna(0, inplace=True)

    target_c = None
    if not in_test:
        target_c = ds[TARGET_COLUMNS]
        target_c['sale_amount'].fillna(0, inplace=True)
        ds.drop(columns=TARGET_COLUMNS, inplace=True)

    if in_test and columns_meta:
        ds = trim_to_meta(ds, columns_meta)

    return ds, target_c
