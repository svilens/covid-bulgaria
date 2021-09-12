import pandas as pd
from datetime import datetime
import pytz

from func_logging import *
from func_read_vaccines_web import *

logger = create_logger(__name__)
logger.info('Obtaining vaccines data')

vaccines_df = get_vaccines_data_web(vaccines_url, chromedriver_dir)
vaccines_df['date'] = pd.to_datetime(datetime.now(pytz.timezone('Europe/Sofia'))).date()

# load the existing data
vaccines_df_old = pd.read_csv('./dash_data/vaccines.csv')

vaccines_df = pd.merge(
    vaccines_df_old[['code', 'province', 'pop']].drop_duplicates(),
    vaccines_df,
    on='code'
).sort_values(by='province', ascending=True).reset_index(drop=True)

# check if the vaccines data has been updated in the source
vacc_old_compare = vaccines_df_old.loc[vaccines_df_old.date == vaccines_df_old.date.max(),['province','total']].sort_values(by='province', ascending=True).reset_index(drop=True)

if len(vaccines_df[['province','total']].compare(vacc_old_compare)) != 0:
    vacc_cols = ['date', 'province', 'code', 'pop', 'total', 'new_pfizer', 'new_astrazeneca', 'new_moderna', 'new_johnson', 'second_dose']
    pd.concat([vaccines_df_old[vacc_cols], vaccines_df[vacc_cols]]).to_csv('./dash_data/vaccines.csv', header=True, index=False)
    logger.info('Vaccines data was added successfully.')
    logger.info('Starting git push')
    from func_git import git_push
    git_push()
elif vaccines_df_old['date'].max() == datetime.now(pytz.timezone('Europe/Sofia')).strftime(format="%Y-%m-%d"):
    logger.info('Vaccines data is up to date!')
else:
    logger.info('WARNING: Vaccines data has not been updated in the source!')


