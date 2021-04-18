import pandas as pd
import numpy as np
from itertools import starmap
from operator import or_
from functools import reduce

MCC_CODES_RARE = {
    'mcc_cd_-1',
    'mcc_cd_1520',
    'mcc_cd_1711',
    'mcc_cd_1731',
    'mcc_cd_1740',
    'mcc_cd_1750',
    'mcc_cd_1761',
    'mcc_cd_1799',
    'mcc_cd_2741',
    'mcc_cd_2791',
    'mcc_cd_2842',
    'mcc_cd_3007',
    'mcc_cd_3008',
    'mcc_cd_3011',
    'mcc_cd_3015',
    'mcc_cd_3026',
    'mcc_cd_3034',
    'mcc_cd_3042',
    'mcc_cd_3047',
    'mcc_cd_3183',
    'mcc_cd_3217',
    'mcc_cd_3245',
    'mcc_cd_3246',
    'mcc_cd_3301',
    'mcc_cd_3355',
    'mcc_cd_3501',
    'mcc_cd_3502',
    'mcc_cd_3503',
    'mcc_cd_3504',
    'mcc_cd_3509',
    'mcc_cd_3512',
    'mcc_cd_3513',
    'mcc_cd_3530',
    'mcc_cd_3533',
    'mcc_cd_3543',
    'mcc_cd_3553',
    'mcc_cd_3579',
    'mcc_cd_3583',
    'mcc_cd_3598',
    'mcc_cd_3612',
    'mcc_cd_3634',
    'mcc_cd_3640',
    'mcc_cd_3642',
    'mcc_cd_3649',
    'mcc_cd_3665',
    'mcc_cd_3690',
    'mcc_cd_3692',
    'mcc_cd_3710',
    'mcc_cd_3750',
    'mcc_cd_4011',
    'mcc_cd_4119',
    'mcc_cd_4214',
    'mcc_cd_4225',
    'mcc_cd_4411',
    'mcc_cd_4457',
    'mcc_cd_4582',
    'mcc_cd_4722',
    'mcc_cd_4784',
    'mcc_cd_4829',
    'mcc_cd_4899',
    'mcc_cd_5013',
    'mcc_cd_5021',
    'mcc_cd_5044',
    'mcc_cd_5045',
    'mcc_cd_5046',
    'mcc_cd_5047',
    'mcc_cd_5051',
    'mcc_cd_5065',
    'mcc_cd_5072',
    'mcc_cd_5074',
    'mcc_cd_5085',
    'mcc_cd_5094',
    'mcc_cd_5099',
    'mcc_cd_5111',
    'mcc_cd_5122',
    'mcc_cd_5131',
    'mcc_cd_5137',
    'mcc_cd_5139',
    'mcc_cd_5169',
    'mcc_cd_5172',
    'mcc_cd_5192',
    'mcc_cd_5193',
    'mcc_cd_5198',
    'mcc_cd_5199',
    'mcc_cd_5231',
    'mcc_cd_5300',
    'mcc_cd_5309',
    'mcc_cd_5310',
    'mcc_cd_5521',
    'mcc_cd_5531',
    'mcc_cd_5532',
    'mcc_cd_5542',
    'mcc_cd_5551',
    'mcc_cd_5561',
    'mcc_cd_5571',
    'mcc_cd_5599',
    'mcc_cd_5655',
    'mcc_cd_5681',
    'mcc_cd_5697',
    'mcc_cd_5698',
    'mcc_cd_5713',
    'mcc_cd_5718',
    'mcc_cd_5733',
    'mcc_cd_5734',
    'mcc_cd_5735',
    'mcc_cd_5811',
    'mcc_cd_5815',
    'mcc_cd_5817',
    'mcc_cd_5818',
    'mcc_cd_5931',
    'mcc_cd_5932',
    'mcc_cd_5933',
    'mcc_cd_5940',
    'mcc_cd_5946',
    'mcc_cd_5948',
    'mcc_cd_5950',
    'mcc_cd_5960',
    'mcc_cd_5962',
    'mcc_cd_5963',
    'mcc_cd_5965',
    'mcc_cd_5966',
    'mcc_cd_5967',
    'mcc_cd_5968',
    'mcc_cd_5969',
    'mcc_cd_5970',
    'mcc_cd_5971',
    'mcc_cd_5973',
    'mcc_cd_5975',
    'mcc_cd_5976',
    'mcc_cd_5978',
    'mcc_cd_5983',
    'mcc_cd_5994',
    'mcc_cd_5996',
    'mcc_cd_5998',
    'mcc_cd_6050',
    'mcc_cd_6051',
    'mcc_cd_6211',
    'mcc_cd_6513',
    'mcc_cd_6532',
    'mcc_cd_6537',
    'mcc_cd_6540',
    'mcc_cd_7012',
    'mcc_cd_7032',
    'mcc_cd_7033',
    'mcc_cd_7210',
    'mcc_cd_7211',
    'mcc_cd_7216',
    'mcc_cd_7217',
    'mcc_cd_7221',
    'mcc_cd_7251',
    'mcc_cd_7261',
    'mcc_cd_7273',
    'mcc_cd_7277',
    'mcc_cd_7278',
    'mcc_cd_7296',
    'mcc_cd_7297',
    'mcc_cd_7298',
    'mcc_cd_7311',
    'mcc_cd_7333',
    'mcc_cd_7338',
    'mcc_cd_7339',
    'mcc_cd_7349',
    'mcc_cd_7361',
    'mcc_cd_7372',
    'mcc_cd_7375',
    'mcc_cd_7379',
    'mcc_cd_7392',
    'mcc_cd_7393',
    'mcc_cd_7394',
    'mcc_cd_7395',
    'mcc_cd_7399',
    'mcc_cd_742',
    'mcc_cd_7512',
    'mcc_cd_7513',
    'mcc_cd_7519',
    'mcc_cd_7523',
    'mcc_cd_7531',
    'mcc_cd_7534',
    'mcc_cd_7535',
    'mcc_cd_7622',
    'mcc_cd_7623',
    'mcc_cd_7629',
    'mcc_cd_763',
    'mcc_cd_7631',
    'mcc_cd_7641',
    'mcc_cd_7692',
    'mcc_cd_7699',
    'mcc_cd_780',
    'mcc_cd_7829',
    'mcc_cd_7841',
    'mcc_cd_7911',
    'mcc_cd_7929',
    'mcc_cd_7932',
    'mcc_cd_7933',
    'mcc_cd_7941',
    'mcc_cd_7991',
    'mcc_cd_7993',
    'mcc_cd_7994',
    'mcc_cd_7995',
    'mcc_cd_7996',
    'mcc_cd_7998',
    'mcc_cd_7999',
    'mcc_cd_8041',
    'mcc_cd_8042',
    'mcc_cd_8049',
    'mcc_cd_8050',
    'mcc_cd_8071',
    'mcc_cd_8111',
    'mcc_cd_8211',
    'mcc_cd_8220',
    'mcc_cd_8241',
    'mcc_cd_8244',
    'mcc_cd_8249',
    'mcc_cd_8351',
    'mcc_cd_8398',
    'mcc_cd_8641',
    'mcc_cd_8661',
    'mcc_cd_8699',
    'mcc_cd_8734',
    'mcc_cd_8931',
    'mcc_cd_9211',
    'mcc_cd_9223',
    'mcc_cd_9405',
}

MCC_CODES_RARE = {int(e.rsplit('_')[-1]) for e in MCC_CODES_RARE}


def create_aggregates_amount_by(group, by):
    kd = group.groupby(by)['tran_amt_rur'].agg(['sum', 'count', 'min', 'max', 'std'])
    kd['std'] = kd['std'].fillna(0)
    kd['mean'] = kd['sum'] / kd['count']

    features = {}

    for col in kd.columns:
        features_ = {kd.index.name + '_' + str(k) + '_amnt_' + col: v
                     for k, v in kd[col].to_dict().items()}
        features.update(features_)

    return features


def create_transaction_features(name, group):
    info = {'client_id': name}

    group = group.sort_values(by='tran_time')

    info['num_transactions'] = group.shape[0]

    tran_time_min = group['tran_time'].min()
    tran_time_max = group['tran_time'].max()

    group['days_before'] = (group['tran_time'] - tran_time_max).dt.days

    info['num_weekofyear'] = group['tran_time'].dt.isocalendar().week.nunique()

    info['num_days_tran_diff'] = (tran_time_max - tran_time_min).days
    info['num_days_tran_uniq'] = group['tran_time_day_offset'].nunique()
    info['num_days_tran_median'] = group['tran_time_day_offset'].median()
    info['num_days_tran_max'] = group['tran_time_day_offset'].max()
    info['num_days_tran_std'] = group['tran_time_day_offset'].std()

    hour_diff = group['tran_time'].diff().dt.total_seconds() / 3600

    info['hour_mean'] = group['tran_time'].dt.hour.mean()
    info['hour_std'] = group['tran_time'].dt.hour.std()
    info['hour_q05'], info['hour_q50'], info['hour_q95'] = \
        group['tran_time'].dt.hour.quantile([0.05, 0.5, 0.95])

    info['hour_diff_q50'] = hour_diff.median()
    info['hour_diff_max'] = hour_diff.max()
    info['hour_diff_std'] = hour_diff.std()

    info['weekend_amnt_mean'] = group.loc[group['tran_time'].dt.weekday >= 5, 'tran_amt_rur'].mean()

    amnt = group['tran_amt_rur']

    info['amnt_mean'] = amnt.mean()
    info['amnt_max'] = amnt.max()
    info['amnt_std'] = amnt.std()

    info['amnt_q10'], info['amnt_q50'], info['amnt_q95'] = \
        amnt.quantile([0.1, 0.5, 0.95])

    info['amnt_mean_last_month'] = amnt[group['days_before'] < 30].mean()
    if np.isnan(info['amnt_mean_last_month']):
        info['amnt_mean_last_month'] = 0

    info['div(amnt_mean_last_month__amnt_mean)'] = \
        info['amnt_mean_last_month'] / (info['amnt_mean'] + 1)

    amnt_diff = group['tran_amt_rur'].diff()

    info['amnt_diff_std'] = amnt_diff.std()
    info['amnt_diff_q10'], info['amnt_diff_q50'], info['amnt_diff_q95'] = \
        amnt_diff.quantile([0.1, 0.5, 0.95])

    info['num_cards'] = group['card_id'].nunique()

    # info.update(create_aggregates_amount_by(group, by='mcc_cd'))
    # info.update(create_aggregates_amount_by(group, by='mcc_category'))
    # info.update(create_aggregates_amount_by(group, by='txn_comment_1_cat'))
    # info.update(create_aggregates_amount_by(group, by='txn_comment_2_cat'))

    return info


def load_trxn(trxn_file: str, dict_mcc_file: str, rarity_thr: float = 1.) -> pd.DataFrame:
    dict_mcc = pd.read_csv(dict_mcc_file, sep=',')
    dict_mcc['mcc_category'] = dict_mcc['brs_mcc_group'].astype('category').cat.codes

    trxn = pd.read_csv(trxn_file, sep=',')
    trxn['tran_time'] = pd.to_datetime(trxn['tran_time'])
    trxn['tran_time_day_offset'] = (trxn['tran_time'] - trxn['tran_time'].min()).dt.days
    trxn['mcc_cd'] = trxn['mcc_cd'].fillna(-1).astype(int)

    df = pd.merge(trxn, dict_mcc[['mcc_cd', 'mcc_category']], how='left', on='mcc_cd')
    df['mcc_category'] = df['mcc_category'].fillna(-1).astype(int)
    df['txn_comment_1_cat'] = df['txn_comment_1'].astype('category').cat.codes
    df['txn_comment_2_cat'] = df['txn_comment_2'].astype('category').cat.codes
    df.loc[df['mcc_cd'].isin(MCC_CODES_RARE), 'mcc_cd'] = -1

    df_grouped = df.groupby('client_id')
    df_features = starmap(create_transaction_features, df_grouped)
    df_features = pd.DataFrame(df_features)

    counts = df_features.isna().sum(axis=0) / df_features.shape[0]
    columns = df_features.columns[counts >= rarity_thr]
    df_features.drop(columns=columns, inplace=True)

    mask = reduce(or_, [
        df_features.columns.str.startswith('txn_comment_1_cat_1'),
        df_features.columns.str.startswith('txn_comment_2_cat_1'),
    ])
    columns = df_features.columns[mask]

    df_features.drop(columns=columns, inplace=True)

    return df_features.set_index('client_id')
