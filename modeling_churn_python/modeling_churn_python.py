# first, the imports. if you're in Jupyter, you can always !pip install new libraries
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
np.random.seed(42)

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

# set your date markers
date_tracking = authentications[['uid', 'date']].drop_duplicates()
date_tracking['prior_date'] = date_tracking.groupby('uid').date.shift()
date_tracking['next_date'] = date_tracking.groupby('uid').date.shift(-1)
date_tracking['cohort_date'] = date_tracking.groupby('uid').date.transform('min')
date_tracking['date_index'] = date_tracking.date.dt.to_period('M').astype('int') - date_tracking.cohort_date.dt.to_period('M').astype('int')
date_tracking['n_til_next_seen'] = np.where(date_tracking.next_date.notna(), date_tracking.next_date.dt.to_period('M').astype('int') - date_tracking.date.dt.to_period('M').astype('int'), None)
date_tracking['n_since_last_seen'] = np.where(date_tracking.prior_date.notna(), date_tracking.date.dt.to_period('M').astype('int') - date_tracking.prior_date.dt.to_period('M').astype('int'), None)
date_tracking['forwards_retention'] = np.where((date_tracking.next_date.isna())|(date_tracking.n_til_next_seen > 3), 'Lapsed', 'Retained')

# apply bidirectional churn labels
backwards_conditions = [
    # new users
    date_tracking.date == date_tracking.cohort_date,
    # seen in past 3 months = Retained
    date_tracking.n_since_last_seen <= 3,
    # seen > 3 months ago
    date_tracking.n_til_next_seen > 3
]

labels = ['New', 'Retained', 'Reactivated']

date_tracking['backwards_retention'] = np.select(backwards_conditions, labels, default='unhandled_case')

authentications = authentications.merge(date_tracking, how='left', on=['uid', 'date'])


# create some dummy categorical data for segmentation
tiers = ['No Subscription', 'Basic Ad Supported', 'Premium Ad Free', 'Ultimate']
countries = ['United States', 'Japan', 'Germany', 'United Kingdom', 'Brazil']
channel_skus = ['ABC', 'DEF', 'GHI']

ids = authentications[['uid']].drop_duplicates()

# 1. Assign Subscription Tier based on engagement (active months)
quantiles = ids.merge(date_tracking.groupby(['uid', 'cohort_date'], as_index=False).agg(active_months=('date', 'nunique')), on='uid', how='left')
# sharp left skew towards single month
bin_edges = [1, 2, 5, 20, 254]
quantiles['subscription_tier'] = pd.cut(quantiles['active_months'], bins=bin_edges, labels=tiers, include_lowest=True)

# 2. Assign Channel SKU based on cohort effect (time-based stratification)
quantiles['channel_sku'] = pd.cut(cohort_bins, bins=[0, 0.2, 0.7, 1], labels=channel_skus)

# 3. Assign Country based on retention behavior (models regional expansion of an app or service)
cohort_bins = quantiles['cohort_date'].rank(method='first', pct=True)  # Convert cohort_date into a percentile
quantiles['country'] = pd.cut(cohort_bins, bins=[0, 0.2, 0.5, 0.7, 0.9, 1], labels=countries)

ids = ids.merge(quantiles[['uid', 'subscription_tier', 'channel_sku', 'country']], on='uid', how='left')

# generate a left-skew synthetic distribution to represent the purchasing habits of the imagined user group

from scipy.stats import skewnorm
fig, ax = plt.subplots(1, 1)
a = 4
mean, var, skew, kurt = skewnorm.stats(a, moments='mvsk')
x = np.linspace(skewnorm.ppf(0.01, a),
                skewnorm.ppf(0.99, a), 100)
ax.plot(x, skewnorm.pdf(x, a),
       'r-', lw=5, alpha=0.6, label='skewnorm pdf')

rv = skewnorm(a)
ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')
vals = skewnorm.ppf([0.001, 0.5, 0.999], a)
np.allclose([0.001, 0.5, 0.999], skewnorm.cdf(vals, a))
# generate random numbers
r = skewnorm.rvs(a, size=len(ids))
r_min, r_max = r.min(), r.max()
r_rescaled = np.round((r - r_min) / (r_max - r_min) * 9).astype(int)

ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
ax.set_xlim([x[0], x[-1]])
ax.legend(loc='best', frameon=False)
plt.show()

ids['lifetime_transactions'] = r_rescaled

auth_merged = authentications.merge(ids, how='left', on='uid')

# Loop through months and aggregate in sequence - simple solution to potential memory overflow
output_list = []
for date in auth_merged.date.drop_duplicates().values.tolist():
    curr = auth_merged.loc[auth_merged.date == date].copy()
    curr_output = curr.groupby(['date', 'cohort_date', 'date_index', 'country', 'subscription_tier', 'channel_sku', 'lifetime_transactions', 'forwards_retention', 'backwards_retention'], as_index=False, observed=True).agg(unique_users=('uid', 'nunique'))
    output_list.append(curr_output)
    del curr_output

output = pd.concat(output_list)

output.to_csv('authentication_retention_extended_labels.csv', index=False)

