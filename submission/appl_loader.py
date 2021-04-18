import pandas as pd


def load_appl(appl_file: str) -> pd.DataFrame:
    appl = pd.read_csv(appl_file, sep=',')
    columns = ['appl_prod_group_name', 'appl_prod_type_name', 'appl_stts_name_dc', 'appl_sale_channel_name']

    return appl.dropna(subset=['appl_stts_name_dc'], axis=0).sort_values('month_end_dt') \
        .groupby('client_id')[columns].apply(lambda r: r.iloc[-1]) \
        .reset_index().set_index('client_id').sort_index()
