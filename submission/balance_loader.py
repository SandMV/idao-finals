import numpy as np
import pandas as pd


def load_balance(balance_file: str) -> pd.DataFrame:
    balance = pd.read_csv(balance_file, sep=',')
    balance.sort_values(by='month_end_dt', inplace=True)

    mask = np.isclose(balance[['min_bal_sum_rur', 'max_bal_sum_rur']].values, 0).all(axis=1)
    balance = balance[~mask]
    balance['prod_group_name'] = balance['prod_group_name'].replace('Credit card 120 days', 'Credit card other')
    balance = balance[~balance['prod_group_name'].isin({'Car loans', 'Mortgage', 'Technical cards'})]

    df_grouped = balance.groupby(['client_id', 'prod_group_name']).tail(1)

    features_pool = []

    for values in ['eop_bal_sum_rur', 'min_bal_sum_rur', 'max_bal_sum_rur', 'avg_bal_sum_rur']:
        df_features_ = pd.pivot_table(
            df_grouped,
            values=values,
            index='client_id',
            columns='prod_group_name',
        )
        df_features_.columns = df_features_.columns.str.replace(' ', '_')
        df_features_ = df_features_.add_prefix('prod_group_name' + '_' + values + '_').add_suffix('_last_month')
        features_pool.append(df_features_)

    prod_features = pd.concat(features_pool, axis=1)

    df_grouped = balance.groupby(['client_id', 'prod_group_name']).agg({
        'min_bal_sum_rur': 'min',
        'max_bal_sum_rur': 'max',
    }).reset_index()

    features_pool = []

    for values in ['min_bal_sum_rur', 'max_bal_sum_rur']:
        df_features_ = pd.pivot_table(
            df_grouped,
            values=values,
            index='client_id',
            columns='prod_group_name',
        )

        df_features_.columns = df_features_.columns.str.replace(' ', '_')
        df_features_ = df_features_.add_prefix('prod_group_name_' + values + '_').add_suffix('_total')
        features_pool.append(df_features_)

    agg_features = pd.concat(features_pool, axis=1)
    return pd.concat([prod_features, agg_features], axis=1).sort_index()
