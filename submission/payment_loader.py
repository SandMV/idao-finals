import pandas as pd


def load_payment(payment_file: str) -> pd.DataFrame:
    payments = pd.read_csv(payment_file, sep=',')

    pmnts_sts = payments.groupby(['client_id', 'pmnts_name']).agg(['mean', 'count'])
    pmnts_sts.columns = ['_'.join(col) for col in pmnts_sts.columns]
    pmnts_sts = pmnts_sts.reset_index()

    features_pool = []
    for values in ['sum_rur_mean', 'sum_rur_count']:
        df_features_ = pd.pivot_table(
            pmnts_sts,
            values=values,
            index='client_id',
            columns='pmnts_name',
        )
        df_features_.columns = df_features_.columns.str.replace(' ', '_')
        df_features_ = df_features_.add_prefix('pmnts_name_' + values + '_')
        features_pool.append(df_features_)

    feats = pd.concat(features_pool, axis=1).sort_index()
    feats.fillna(0, inplace=True)
    feats['pmnts_name_sum_rur_count_Pension_receipts'] = feats['pmnts_name_sum_rur_count_Pension_receipts'].astype(int)
    feats['pmnts_name_sum_rur_count_Salary_receipts'] = feats['pmnts_name_sum_rur_count_Salary_receipts'].astype(int)
    return feats
