import pandas as pd


def load_aum(aum_file: str) -> pd.DataFrame:
    aum = pd.read_csv(aum_file, sep=',')
    grouped = aum.sort_values(by='month_end_dt') \
        .groupby(['client_id', 'product_code'])['balance_rur_amt'].apply(lambda r: r.iloc[-1])
    grouped = grouped.unstack('product_code').fillna(0)
    grouped['sum'] = grouped.sum(axis=1).values
    grouped = grouped.add_prefix('aum_amnt_')
    return grouped.reset_index().set_index('client_id').sort_index()
