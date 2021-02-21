import geopandas as gpd
import pandas as pd
from datetime import datetime
import warnings
warnings.simplefilter('ignore')

# from external modules
from func_download_data_api import *
from func_read_spatial import *
from func_read_covid import *
from func_read_nsi import *
from func_logging import *
from func_read_vaccines_web import *

global_start = datetime.now()
logger = get_logger(__name__)
logger.info('STARTED')


# Download data
api_request = get_covid_data()
if api_request == 'old':
    logger.info('The last date on the source data is different than the current date. Check if the source has been updated today. Process terminated!')
    #sys.exit()
elif api_request == 'error':
    logger.info('There was an error while communicating with the API. Process terminated!')
    sys.exit()

logger.info('Finished downloading the new COVID-19 data!')


logger.info('Reading general stats file')
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


logger.info('Reading spatial data file')

geodf = read_spatial_data('./shape/BGR_adm1.shp', codes_spatial)
covid_by_province = read_covid_by_province('./data/COVID_provinces.csv', date_col='Дата')
pop_by_province = read_population_data('./data/Pop_6.1.1_Pop_DR.xls', worksheet_name='2019',
                                           col_num=2, col_names=['municipality','pop'],
                                           skip=5, codes=codes_pop)
covid_pop = (covid_by_province.set_index('code')
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


logger.info('Scraping vaccines data')
vaccines_df = get_vaccines_data_web(vaccines_url, chromedriver_dir)
vaccines_df['date'] = pd.to_datetime(datetime.now()).date()

vaccines_df = pd.merge(
            covid_pop[['province', 'pop']].drop_duplicates().reset_index(),
            vaccines_df,
            on='code').sort_values(by='province', ascending=True).reset_index(drop=True)

# check if the vaccines data has been updated in the source
vaccines_df_old = pd.read_csv('./dash_data/vaccines.csv')
vacc_old_compare = vaccines_df_old.loc[vaccines_df_old.date == vaccines_df_old.date.max(),['province','total']].sort_values(by='province', ascending=True).reset_index(drop=True)

if len(vaccines_df[['province','total']].compare(vacc_old_compare)) != 0:
    vacc_cols = ['date','province','code','pop','total','new_pfizer','new_astrazeneca','new_moderna','second_dose']
    vaccines_df[vacc_cols].to_csv('./dash_data/vaccines.csv', header=True, index=False)
    logger.info('Vaccines data was added successfully.')
else:
    logger.info('WARNING: Vaccines data has not been updated in the source!')


logger.info('Reading age bands')
covid_by_age_band = (pd.read_csv('./data/COVID_age_bands.csv', parse_dates=['Дата'])
                     .rename(columns={'Дата':'date'}))


logger.info('Starting Rt processing')

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.dates import date2num, num2date
from matplotlib import dates as mdates
from matplotlib import ticker
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from scipy import stats as sps
from scipy.interpolate import interp1d

provinces = covid_pop_sorted[['province', 'date', 'ALL']].groupby(['province','date']).ALL.sum()

bg_total_new = provinces.groupby('date').sum()

def prepare_cases(cases):
    new_cases = cases.diff()

    smoothed = new_cases.rolling(9,
        win_type='gaussian',
        min_periods=1,
        center=True).mean(std=3).round()
    
    idx_start = np.searchsorted(smoothed, 10)
    smoothed = smoothed.iloc[idx_start:]
    original = new_cases.loc[smoothed.index]
    
    return original, smoothed

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


# smoothed new cases by age band
logger.info('preparing smoothed cases by age band for Bulgaria')
covid_by_age_band_diff_smoothed = covid_by_age_band.set_index('date').diff().rolling(9,
        win_type='gaussian',
        min_periods=1,
        center=True).mean(std=3).round()


# age band new cases per 100k pop
pop_by_age_band = read_nsi_age_bands('./data/Pop_6.1.2_Pop_DR.xls', worksheet_name='2019', col_num=2, col_names=['age_band', 'pop'], skip=5, rows_needed=22)
covid_by_age_band_diff_smoothed_per100k = covid_by_age_band_diff_smoothed.copy()
for col in covid_by_age_band_diff_smoothed_per100k.columns:
    covid_by_age_band_diff_smoothed_per100k[col] = (100000*covid_by_age_band_diff_smoothed_per100k[col]/pop_by_age_band.loc[pop_by_age_band.covid_age_band == col, 'pop'].values).round(0)

logger.info('preparing smoothed cases by province')
# provinces
provinces_list = covid_pop[['province', 'pop']].drop_duplicates().sort_values(by='pop', ascending=False).province.values


r0_provinces_original = pd.DataFrame()
r0_provinces_smoothed = pd.DataFrame()

# add charts for provinces
for i, province in list(enumerate(provinces_list)):
    cases = provinces.xs(province).rename(f"{province} cases")
    original, smoothed = prepare_cases(cases)
    original = original.to_frame()
    smoothed = smoothed.to_frame()
    original.columns = ['new_cases']
    smoothed.columns = ['new_cases']
    original['province'] = province
    smoothed['province'] = province
    r0_provinces_original = pd.concat([r0_provinces_original, original])
    r0_provinces_smoothed = pd.concat([r0_provinces_smoothed, smoothed])
    
r0_provinces_original.to_csv('./dash_data/r0_provinces_original.csv', header=True)
r0_provinces_smoothed.to_csv('./dash_data/r0_provinces_smoothed.csv', header=True)


# We create an array for every possible value of Rt
R_T_MAX = 12
r_t_range = np.linspace(0, R_T_MAX, R_T_MAX*100+1)

# Gamma is 1/serial interval
# https://wwwnc.cdc.gov/eid/article/26/7/20-0282_article
# https://www.nejm.org/doi/full/10.1056/NEJMoa2001316
GAMMA = 1/7


def get_posteriors(sr, sigma=0.15):

    # (1) Calculate Lambda
    lam = sr[:-1].values * np.exp(GAMMA * (r_t_range[:, None] - 1))

    
    # (2) Calculate each day's likelihood
    likelihoods = pd.DataFrame(
        data = sps.poisson.pmf(sr[1:].values, lam),
        index = r_t_range,
        columns = sr.index[1:])
    
    # (3) Create the Gaussian Matrix
    process_matrix = sps.norm(loc=r_t_range,
                              scale=sigma
                             ).pdf(r_t_range[:, None]) 

    # (3a) Normalize all rows to sum to 1
    process_matrix /= process_matrix.sum(axis=0)
    
    # (4) Calculate the initial prior
    prior0 = sps.gamma(a=4).pdf(r_t_range)
    prior0 /= prior0.sum()

    # Create a DataFrame that will hold our posteriors for each day
    # Insert our prior as the first posterior.
    posteriors = pd.DataFrame(
        index=r_t_range,
        columns=sr.index,
        data={sr.index[0]: prior0}
    )
    
    # We said we'd keep track of the sum of the log of the probability
    # of the data for maximum likelihood calculation.
    log_likelihood = 0.0

    # (5) Iteratively apply Bayes' rule
    for previous_day, current_day in zip(sr.index[:-1], sr.index[1:]):

        #(5a) Calculate the new prior
        current_prior = process_matrix @ posteriors[previous_day]
        
        #(5b) Calculate the numerator of Bayes' Rule: P(k|R_t)P(R_t)
        numerator = likelihoods[current_day] * current_prior
        
        #(5c) Calcluate the denominator of Bayes' Rule P(k)
        denominator = np.sum(numerator)
        
        # Execute full Bayes' Rule
        posteriors[current_day] = numerator/denominator
        
        # Add to the running sum of log likelihoods
        log_likelihood += np.log(denominator)
    
    return posteriors, log_likelihood


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

def highest_density_interval(pmf, p=.9):
    # If we pass a DataFrame, just call this recursively on the columns
    if(isinstance(pmf, pd.DataFrame)):
        return pd.DataFrame([highest_density_interval(pmf[col], p=p) for col in pmf],
                            index=pmf.columns)
    
    cumsum = np.cumsum(pmf.values)
    best = None
    for i, value in enumerate(cumsum):
        for j, high_value in enumerate(cumsum[i+1:]):
            if (high_value-value > p) and (not best or j<best[1]-best[0]):
                best = (i, i+j+1)
                break
            
    low = pmf.index[best[0]]
    high = pmf.index[best[1]]
    return pd.Series([low, high], index=[f'Low_{p*100:.0f}', f'High_{p*100:.0f}'])

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


logger.info('Calculating Rt for provinces')

sigmas = np.linspace(1/15, 1, 15)

provinces_to_process = provinces

results = {}

for province_name, cases in provinces_to_process.groupby(level='province'):
    print(f'processing province: {province_name}')
    new, smoothed = prepare_cases(cases)
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

# Each index of this array holds the total of the log likelihoods for the corresponding index of the sigmas array.
total_log_likelihoods = np.zeros_like(sigmas)

# Loop through each state's results and add the log likelihoods to the running total.
for province_name, result in results.items():
    total_log_likelihoods += result['log_likelihoods']

# Select the index with the largest log likelihood total
max_likelihood_index = total_log_likelihoods.argmax()

# Select the value that has the highest log likelihood
sigma = sigmas[max_likelihood_index]

start = datetime.now()
final_results = None

logger.info('calculating final Rt by province')

for province_name, result in results.items():
    print(province_name)
    posteriors = result['posteriors'][max_likelihood_index]
    hdis_90 = highest_density_interval(posteriors, p=.9)
    hdis_50 = highest_density_interval(posteriors, p=.5)
    most_likely = posteriors.idxmax().rename('Estimated')
    result = pd.concat([most_likely, hdis_90, hdis_50], axis=1)
    if final_results is None:
        final_results = result
    else:
        final_results = pd.concat([final_results, result])

logger.info(f'Provinces Rt runtime: {datetime.now() - start}')

final_results.to_csv('./dash_data/r0_provinces_r0.csv', header=True)

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


logger.info('Starting ARIMA')

import itertools
import pandas as pd
#import seaborn as sns
import math
import matplotlib
import numpy as np
#import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA


ts_data = pd.read_csv('./dash_data/ts_data.csv')

#Function for making a time series on a province and plotting the rolled mean and standard deviation
def roll(ts_data, region, column='total_cases'):
    test_s=ts_data.loc[(ts_data['province']==region)]  
    test_s=test_s[['date',column]]
    test_s=test_s.set_index('date')
    test_s.astype('int64')
    a=len(test_s.loc[(test_s[column]>=10)])
    test_s=test_s[-a:]
    return (test_s.rolling(window=7,center=False).mean().dropna())

def split(ts, forecast_days=15):
    #size = int(len(ts) * math.log(0.80))
    size=-forecast_days
    train= ts[:size]
    test = ts[size:]
    return(train,test)

def mape(y1, y_pred): 
    y1, y_pred = np.array(y1), np.array(y_pred)
    return np.mean(np.abs((y1 - y_pred) / y1)) * 100

def arima_province(ts_data, province, column='total_cases', forecast_days=15):
    rolling = roll(ts_data, province, column)
    train, test = split(rolling.values, forecast_days)
    p=d=q=range(0,7)
    a=99999
    pdq=list(itertools.product(p,d,q))
    
    #Determining the best parameters
    for var in pdq:
        try:
            model = ARIMA(train, order=var)
            result = model.fit(disp=0)
            if (result.aic<=a) :
                a=result.aic
                param=var
        except:
            continue
        
    #Modeling
    model = ARIMA(train, order=param)
    result = model.fit(disp=0)
    
    pred=result.forecast(steps=len(test))[0]
    #Printing the error metrics
    model_error = mape(test,pred)
    #Plotting results
    #fig = go.Figure()
    fig = None

    return (pred, result, fig, model_error, rolling.index, rolling[column])


logger.info('Getting ARIMA predictions for provinces')

#from statsmodels.tools.sm_exceptions import ConvergenceWarning
#warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore')

start_arima = datetime.now()
arima_provinces_df = pd.DataFrame()

for province in ts_data['province'].unique():
    print(province)
    data = ts_data.loc[ts_data['province'] == province]
    arima_results = arima_province(data, province, 'total_cases')
    arima_preds = [*arima_results[5][:-15], *arima_results[0]]
    arima_errors = [arima_results[3] for x in range(len(arima_preds))]
    arima_indexes = arima_results[4]
    arima_true_values = arima_results[5]
    arima_temp_df = pd.DataFrame({'date':arima_indexes, 'province':province, 'pred':arima_preds, 'error':arima_errors, 'value':arima_true_values})
    arima_provinces_df = pd.concat([arima_provinces_df, arima_temp_df])

logger.info(f'Provinces ARIMA runtime: {datetime.now() - start_arima}')

arima_provinces_df.to_csv('./dash_data/arima_provinces.csv', header=True)

logger.info(f'Processing is done. Total runtime: {datetime.now() - global_start}')
logger.info('Starting git push')
from func_git import *
git_push_result = git_push_automation()
logger.info(git_push_result)

logger.info('FINISHED!')
