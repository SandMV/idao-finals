import pandas as pd

def load_com(com_file: str) -> pd.DataFrame:
    com = pd.read_csv(com_file, sep=',')
    com = com[com['prod'].isin({'Credit Card', 'Cash Loan', 'Debit Card', })]
    com['channel_prod_comm'] = com['channel'] + '_' + com['prod']

    com_cnt = com.groupby(['client_id', 'prod']).agg('sum').reset_index()

    features_pool = []

    for values in ['agr_flg', 'otkaz', 'dumaet', 'ring_up_flg', 'not_ring_up_flg', 'count_comm']:
        df_features_ = pd.pivot_table(
            com_cnt,
            values=values,
            index='client_id',
            columns='prod'
        )
        df_features_.columns = df_features_.columns.str.replace(' ', '_')
        df_features_ = df_features_.add_prefix('prod_' + values + '_')
        features_pool.append(df_features_)

    return pd.concat(features_pool, axis=1).sort_index()