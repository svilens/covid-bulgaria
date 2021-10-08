import geopandas as gpd
import pandas as pd
from datetime import datetime
import pytz
import warnings
warnings.simplefilter('ignore')
from func_logging import *

global_start = datetime.now()
logger = create_logger('app.log')
logger.info('STARTED')

###############
# Download data
###############
from func_download_data_api import *

api_request = get_covid_data()
if api_request == 'old':
    logger.info('The last date on the source data is different than the current date. Check if the source has been updated today. Process terminated!')
    sys.exit()
elif api_request == 'error':
    logger.info('There was an error while communicating with the API. Process terminated!')
    sys.exit()

logger.info('Finished downloading the new COVID-19 data!')


####################
# Process COVID data
####################
logger.info('Reading general stats file')
from func_read_covid import *

covid_general = read_covid_general('./data/COVID_general.csv', 'Дата')
covid_general['week'] = covid_general.date.dt.isocalendar().week
covid_general_weekly = covid_general.groupby('week')[['new_cases', 'new_deaths', 'new_recoveries']].sum()
covid_general_weekly['new_cases_pct_change'] = covid_general_weekly['new_cases'].pct_change()
# removing the first week as it starts on Saturday, and the last week, as it would be incomplete in most cases
covid_general_weekly = covid_general_weekly[1:-1]

covid_general['total_cases_7days_ago'] = covid_general['total_cases'].shift(periods=7)
covid_general['total_cases_14days_ago'] = covid_general['total_cases'].shift(periods=14)
covid_general['death_rate'] = (covid_general['total_deaths'] / covid_general['total_cases_14days_ago']).round(4)
covid_general['recovery_rate'] = (covid_general['total_recoveries'] / covid_general['total_cases_14days_ago']).round(4)
covid_general['hospitalized_rate'] = (covid_general['hospitalized'] / covid_general['active_cases']).round(4)
covid_general['intensive_care_rate'] = (covid_general['intensive_care'] / covid_general['hospitalized']).round(4)
covid_general['tests_positive_rate'] = (covid_general['new_cases'] / covid_general['daily_tests']).round(4)


######################
# Process spatial data
######################
logger.info('Reading spatial data file')
from func_read_nsi import *
from func_read_spatial import *

geodf = read_spatial_data('./shape/BGR_adm1.shp', codes_spatial)
covid_by_province = read_covid_by_province('./data/COVID_provinces.csv', date_col='Дата')
pop_by_province = read_population_data('./data/Pop_6.1.1_Pop_DR.xls', worksheet_name='2019',
                                           col_num=2, col_names=['municipality','pop'],
                                           skip=5, codes=codes_pop)
covid_pop = (
	covid_by_province.set_index('code')
    .join(pop_by_province.set_index('code'))
    .join(geodf[['code','province']].set_index('code'))
)
covid_pop['new_per_100k'] = (100000*covid_pop['new_cases']/covid_pop['pop']).round(2)
covid_pop['total_per_100k'] = (100000*covid_pop['ALL']/covid_pop['pop']).round(2)
covid_pop['active_per_100k'] = (100000*covid_pop['ACT']/covid_pop['pop']).round(2)
covid_pop = gpd.GeoDataFrame(covid_pop)
covid_pop.reset_index()[['date', 'province', 'ALL', 'new_cases']].rename(columns={'ALL':'total_cases'}).to_csv('./dash_data/ts_data.csv', header=True)
covid_pop_sorted = covid_pop.sort_values(by=['date', 'ALL'])


geodf['geometry'] = geodf['geometry'].simplify(tolerance=0.00001, preserve_topology=True)

covid_yesterday = gpd.GeoDataFrame(
    covid_pop.loc[covid_pop.date == max(covid_pop.date)]
    .rename(columns={'ALL':'total cases', 'ACT':'active cases', 'new_cases':'new cases'})
    .join(geodf[['code','geometry']].set_index('code'))
)


##############
# Get vaccines
##############
logger.info('Scraping vaccines data')
from func_read_vaccines_web import *

vaccines_df = get_vaccines_data_web(vaccines_url, chromedriver_dir)
vaccines_df['date'] = pd.to_datetime(datetime.now(pytz.timezone('Europe/Sofia'))).date()

# load the existing data
vaccines_df_old = pd.read_csv('./dash_data/vaccines.csv')
# create backup
vaccines_df_old.to_csv('./dash_data/backup_vaccines.csv', index=False)
if vaccines_df_old.columns[-1] != 'booster_jab':
    vaccines_df_old['booster_jab'] = 0

# add codes and pop
vaccines_df = pd.merge(
    vaccines_df_old[['code', 'province', 'pop']].drop_duplicates(),
    vaccines_df,
    on='code'
).sort_values(by='province', ascending=True).reset_index(drop=True)

# check if the vaccines data has been updated in the source
vacc_old_compare = vaccines_df_old.loc[vaccines_df_old.date == vaccines_df_old.date.max(),['province','total']].sort_values(by='province', ascending=True).reset_index(drop=True)

if len(vaccines_df[['province','total']].compare(vacc_old_compare)) != 0:
    vacc_cols = ['date','province','code','pop','total','new_pfizer','new_astrazeneca','new_moderna','new_johnson','second_dose', 'booster_jab']
    pd.concat([vaccines_df_old[vacc_cols], vaccines_df[vacc_cols]]).fillna(0).to_csv('./dash_data/vaccines.csv', header=True, index=False)
    logger.info('Vaccines data was added successfully.')
else:
    logger.info('WARNING: Vaccines data has not been updated in the source!')


#####################
# Reproduction number
#####################
logger.info('Starting Rt processing')
from func_reproduction_number import *

provinces = covid_pop_sorted[['province', 'date', 'ALL']].groupby(['province','date']).ALL.sum()
bg_total_new = provinces.groupby('date').sum()

# smoothed new cases
logger.info('preparing smoothed new cases for Bulgaria')
orig_bg, smoothed_bg = prepare_cases(bg_total_new)
orig_bg.to_csv('./dash_data/r0_bg_original.csv', header=True)
smoothed_bg.to_csv('./dash_data/r0_bg_smoothed.csv', header=True)

# smoothed new recoveries
logger.info('preparing smoothed recovered cases for Bulgaria')
bg_recovered = covid_general[['date', 'total_recoveries']].groupby('date')['total_recoveries'].sum()
orig_recovered_bg, smoothed_recovered_bg = prepare_cases(bg_recovered)
orig_recovered_bg.to_csv('./dash_data/r0_bg_original_recovered.csv', header=True)
smoothed_recovered_bg.to_csv('./dash_data/r0_bg_smoothed_recovered.csv', header=True)

# smoothed deaths
logger.info('preparing smoothed death cases for Bulgaria')
bg_deaths = covid_general[['date', 'total_deaths']].groupby('date')['total_deaths'].sum()
orig_deaths_bg, smoothed_deaths_bg = prepare_cases(bg_deaths)
orig_deaths_bg.to_csv('./dash_data/r0_bg_original_deaths.csv', header=True)
smoothed_deaths_bg.to_csv('./dash_data/r0_bg_smoothed_deaths.csv', header=True)


logger.info('preparing posteriors for Bulgaria')
sigmas = np.linspace(1/15, 1, 15)
result_bg = {}
 # Holds all posteriors with every given value of sigma
result_bg['posteriors'] = []
# Holds the log likelihood across all k for each value of sigma
result_bg['log_likelihoods'] = []

for sigma in sigmas:
    posteriors, log_likelihood = get_posteriors(smoothed_bg, sigma=sigma)
    result_bg['posteriors'].append(posteriors)
    result_bg['log_likelihoods'].append(log_likelihood)
    
# Each index of this array holds the total of the log likelihoods for the corresponding index of the sigmas array.
total_log_likelihoods_bg = np.zeros_like(sigmas)

# Loop through each state's results and add the log likelihoods to the running total.
for result in result_bg.items():
    total_log_likelihoods_bg += result_bg['log_likelihoods']

# Select the index with the largest log likelihood total
max_likelihood_index_bg = total_log_likelihoods_bg.argmax()

# Select the value that has the highest log likelihood
sigma_bg = sigmas[max_likelihood_index_bg]

logger.info('calculating Rt for Bulgaria')
posteriors_bg, log_likelihood_bg = get_posteriors(smoothed_bg, sigma=sigma_bg)

# Note that this takes a while to execute - it's not the most efficient algorithm
hdis_bg = highest_density_interval(posteriors_bg, p=.9)

most_likely_bg = posteriors_bg.idxmax().rename('Estimated')

# Look into why you shift -1
result_bg = pd.concat([most_likely_bg, hdis_bg], axis=1)

result_bg.to_csv('./dash_data/r0_bg_r0.csv', header=True)

index_bg = result_bg['Estimated'].index.get_level_values('date')
values_bg = result_bg['Estimated'].values

lowfn_bg = interp1d(date2num(index_bg),
                     result_bg['Low_90'].values,
                     bounds_error=False,
                     fill_value='extrapolate')
    
highfn_bg = interp1d(date2num(index_bg),
                      result_bg['High_90'].values,
                      bounds_error=False,
                      fill_value='extrapolate')

extended_bg = pd.date_range(start=index_bg[0],
                             end=index_bg[-1])


logger.info('Calculating smoothed cases for provinces')
start = datetime.now()

sigmas = np.linspace(1/15, 1, 15)

provinces_to_process = provinces

results = {}
#r0_provinces_original = pd.DataFrame()
r0_provinces_smoothed = pd.DataFrame()

for province_name, cases in provinces_to_process.groupby(level='province'):
    logger.info(f'processing province: {province_name}')
    new, smoothed = prepare_cases(cases)

    # for dash
    new_dash = new.to_frame()
    smoothed_dash = smoothed.to_frame()
    new_dash.columns = ['new_cases']
    smoothed_dash.columns = ['new_cases']
 #   r0_provinces_original = pd.concat([r0_provinces_original, new_dash])
#    r0_provinces_smoothed = pd.concat([r0_provinces_smoothed, smoothed_dash])
    if province_name == 'Blagoevgrad':
        new_dash.to_csv('./dash_data/r0_provinces_original.csv', header=True)
        smoothed_dash.to_csv('./dash_data/r0_provinces_smoothed.csv', header=True)
    else:
        new_dash.to_csv('./dash_data/r0_provinces_original.csv', header=False, mode='a')
        smoothed_dash.to_csv('./dash_data/r0_provinces_smoothed.csv', header=False, mode='a')
    # end of dash

    result = {}
    # Holds all posteriors with every given value of sigma
    result['posteriors'] = []
    # Holds the log likelihood across all k for each value of sigma
    result['log_likelihoods'] = []
    for sigma in sigmas:
        posteriors, log_likelihood = get_posteriors(smoothed, sigma=sigma)
        result['posteriors'].append(posteriors)
        result['log_likelihoods'].append(log_likelihood)
    # Store all results keyed off of province name
    results[province_name] = result

# for dash
#logger.info('Outputting smoothed cases by province')
#r0_provinces_original.to_csv('./dash_data/r0_provinces_original.csv', header=True)
#r0_provinces_smoothed.to_csv('./dash_data/r0_provinces_smoothed.csv', header=True)

logger.info('Getting the best log likelihood by province')
# Each index of this array holds the total of the log likelihoods for the corresponding index of the sigmas array.
total_log_likelihoods = np.zeros_like(sigmas)

# Loop through each state's results and add the log likelihoods to the running total.
for province_name, result in results.items():
    total_log_likelihoods += result['log_likelihoods']

# Select the index with the largest log likelihood total
max_likelihood_index = total_log_likelihoods.argmax()

# Select the value that has the highest log likelihood
sigma = sigmas[max_likelihood_index]

final_results = None

logger.info('calculating final Rt by province')

for province_name, result in results.items():
    logger.info(f'Rt for {province_name}')
    posteriors = result['posteriors'][max_likelihood_index].fillna(0.1)
    hdis_90 = highest_density_interval(posteriors, p=.9)
    hdis_50 = highest_density_interval(posteriors, p=.5)
    most_likely = posteriors.idxmax().rename('Estimated')
    result = pd.concat([most_likely, hdis_90, hdis_50], axis=1)
    if province_name == 'Blagoevgrad':
        result.to_csv('./dash_data/r0_provinces_r0.csv', header=True)
    else:
        result.to_csv('./dash_data/r0_provinces_r0.csv', header=False, mode='a')
    if final_results is None:
        final_results = result
    else:
        final_results = pd.concat([final_results, result])

logger.info(f'Provinces Rt runtime: {datetime.now() - start}')

#final_results.to_csv('./dash_data/r0_provinces_r0.csv', header=True)

mr = final_results.groupby(level=0)[['Estimated', 'High_90', 'Low_90']].last()

mr['diff_up'] = mr['High_90'] - mr['Estimated']
mr['diff_down'] = mr['Estimated'] - mr['Low_90']
# create status column
mr_conditions = [(mr.High_90 <= 1), (mr.Low_90 >= 1)]
mr_values = ['Likely under control', 'Likely not under control']
mr['status'] = np.select(mr_conditions, mr_values)
mr_colors = ['green', 'crimson']
mr['colors'] = np.select(mr_conditions, mr_colors)
mr.loc[mr.status=="0", 'colors'] = 'grey'


#######
# ARIMA
#######
# ARIMA IS DISABLED

#logger.info('Starting ARIMA')
from func_predictive_models import *

#ts_data = pd.read_csv('./dash_data/ts_data.csv')

#from statsmodels.tools.sm_exceptions import ConvergenceWarning
#warnings.simplefilter('ignore', ConvergenceWarning)
#warnings.simplefilter('ignore')
#logger.info('Getting ARIMA predictions for provinces')
#start_arima = datetime.now()
#arima_provinces_df = pd.DataFrame()

#for province in ts_data['province'].unique():
#    print(province)
#    data = ts_data.loc[ts_data['province'] == province]
#    arima_results = arima_province(data, province, 'total_cases')
#    arima_preds = [*arima_results[5][:-15], *arima_results[0]]
#    arima_errors = [arima_results[3] for x in range(len(arima_preds))]
#    arima_indexes = arima_results[4]
#    arima_true_values = arima_results[5]
#    arima_temp_df = pd.DataFrame({'date':arima_indexes, 'province':province, 'pred':arima_preds, 'error':arima_errors, 'value':arima_true_values})
#    arima_provinces_df = pd.concat([arima_provinces_df, arima_temp_df])

#logger.info(f'Provinces ARIMA runtime: {datetime.now() - start_arima}')

#arima_provinces_df.to_csv('./dash_data/arima_provinces.csv', header=True)

logger.info(f'Processing is done. Total runtime: {datetime.now() - global_start}')
#logger.info('Starting git push')
#from func_git import git_push
#git_push_result = git_push_automation()
#logger.info(git_push_result)
#git_push()

logger.info('FINISHED!')
