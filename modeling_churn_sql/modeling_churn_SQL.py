# first, the imports. if you're in Jupyter, you can always !pip install new libraries
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import numpy as np

# load registration data: reg_data.csv
authentications = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "debs2x/gamelytics-mobile-analytics-challenge",
    "auth_data.csv",
    pandas_kwargs = {'sep':';'}
)
# preprocessing timestamp integer -> calendar day
authentications['authentication_day'] = pd.to_datetime(authentications.auth_ts, \
                                                    unit='s').dt.date
authentications['date'] = authentications.authentication_day.to_numpy().astype('datetime64[M]')
authentications.date = pd.to_datetime(authentications.date)
authentications = authentications.loc[authentications.date < authentications.date.max()]

authentications['prior_date'] = authentications.groupby('uid').date.shift()
authentications['next_date'] = authentications.groupby('uid').date.shift(-1)
authentications['cohort_date'] = authentications.groupby('uid').date.transform('min')
authentications['date_index'] = authentications.date.dt.to_period('M').astype('int') - authentications.cohort_date.dt.to_period('M').astype('int')
authentications['n_til_next_seen'] = np.where(authentications.next_date.notna(), authentications.next_date.dt.to_period('M').astype('int') - authentications.date.dt.to_period('M').astype('int'), None)
authentications['n_since_last_seen'] = np.where(authentications.prior_date.notna(), authentications.date.dt.to_period('M').astype('int') - authentications.prior_date.dt.to_period('M').astype('int'), None)
authentications['forwards_retention'] = np.where((authentications.next_date.isna())|(authentications.n_til_next_seen > 3), 'Lapsed', 'Retained')

backwards_conditions = [
    # new users
    authentications.date == authentications.cohort_date,
    # seen in past 3 months = Retained
    authentications.n_since_last_seen <= 3,
    # seen > 3 months ago
    authentications.n_til_next_seen > 3
]

labels = ['New', 'Retained', 'Reactivated']

authentications['backwards_retention'] = np.select(backwards_conditions, labels, default='unhandled_case')

# add labels
countries = ['United States', 'Japan', 'Germany', 'United Kingdom', 'Brazil']

authentications['did_iap'] = pd.NA
authentications['did_iap'] = authentications['did_iap'].apply(lambda row: np.random.choice([True, False], p=[0.15, 0.85]))

authentications['country'] = pd.NA
authentications['country'] = authentications['country'].apply(lambda row: np.random.choice(countries, p=[0.25, 0.04, 0.33, 0.20, 0.18]))

output = authentications.groupby(['date', 'country', 'cohort_date', 'date_index', 'did_iap', 'forwards_retention', 'backwards_retention'], as_index=False).agg(unique_users=('uid', 'nunique'))

output.to_csv('authentication_retention.csv', index=False)

