import pandas as pd


def load_client(client_data: str) -> pd.DataFrame:
    client = pd.read_csv(client_data, sep=',')
    client.education.fillna('MISSING', inplace=True)
    client.job_type.fillna('MISSING', inplace=True)
    client.citizenship.fillna('MISSING', inplace=True)

    client.fillna(0, inplace=True)

    client.loc[(client.city > 1000) | (client.city == -1), 'city'] = 1001
    client.loc[(client.region > 60) | (client.region == -1), 'region'] = 61
    client = pd.get_dummies(
        client,
        columns=['education', 'region', 'city', 'gender', 'citizenship', 'job_type']
    )
    return client.set_index('client_id').sort_index()
