import pandas as pd


def load_client(client_data: str) -> pd.DataFrame:
    client = pd.read_csv(client_data, sep=',')
    client.drop(columns=['job_type', 'citizenship'], inplace=True)
    client.age.fillna(0, inplace=True)
    client.gender.fillna(0, inplace=True)
    client.education.fillna('MISSING', inplace=True)
    client.loc[(client.city > 1000) | (client.city == -1), 'city'] = 1001
    client.loc[(client.region > 60) | (client.region == -1), 'region'] = 61
    client = pd.get_dummies(
        client,
        columns=['education', 'region', 'city', 'gender']
    )
    return client.set_index('client_id')
