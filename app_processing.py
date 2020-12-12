import geopandas as gpd
import pandas as pd
from datetime import datetime

# from external modules
from func_download_data import *
from func_read_spatial import *
from func_read_covid import *
from func_read_nsi import *
from func_logging import *

logger = get_logger(__name__)
logger.info('STARTED')

covid_files = ['Обща статистика за разпространението.csv', 'Разпределение по дата и по области.csv',
               'Разпределение по дата и по възрастови групи.csv']

for filename in covid_files:
    backup_existing_file(filename)

logger.info('COVID-19 files backup is done!')

covid_urls = ['https://data.egov.bg/data/resourceView/e59f95dd-afde-43af-83c8-ea2916badd19', # general
              'https://data.egov.bg/data/resourceView/cb5d7df0-3066-4d7a-b4a1-ac26525e0f0c', # province
              'https://data.egov.bg/data/resourceView/8f62cfcf-a979-46d4-8317-4e1ab9cbd6a8'] # age bands

#for url in covid_urls:
#    download_files(url, "chrome")
download_files(covid_urls[0], "chrome")
download_files(covid_urls[1], "chrome")
download_files(covid_urls[2], "chrome")

logger.info('Finished downloading the new COVID-19 files!')

logger.info('Reading general stats file')
covid_general = read_covid_general('./data/Обща статистика за разпространението.csv', 'Дата')
covid_general.head()

import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "plotly_dark"
from plotly.offline import plot

fig_gen_stats = go.Figure()
fig_gen_stats.add_trace(go.Scatter(x=covid_general.date, y=covid_general.total_cases, name='Confirmed'))
fig_gen_stats.add_trace(go.Scatter(x=covid_general.date, y=covid_general.active_cases, line=dict(color='yellow'), name='Active'))
fig_gen_stats.add_trace(go.Scatter(x=covid_general.date, y=covid_general.total_recoveries, line=dict(color='green'), name='Recovered'))
fig_gen_stats.add_trace(go.Scatter(x=covid_general.date, y=covid_general.total_deaths, line=dict(color='red'), name='Deaths'))
fig_gen_stats.update_layout(title='Number of cases over time (cumulative)')
#fig_gen_stats.show()
fig_gen_stats = fig_gen_stats.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

covid_general['week'] = covid_general.date.dt.isocalendar().week
covid_general_weekly = covid_general.groupby('week')[['new_cases', 'new_deaths', 'new_recoveries']].sum()
covid_general_weekly['new_cases_pct_change'] = covid_general_weekly['new_cases'].pct_change()
# removing the first week as it starts on Saturday, and the last week, as it would be incomplete in most cases
covid_general_weekly = covid_general_weekly[1:-1]


logger.info('Charting weekly stats')
from plotly.subplots import make_subplots

fig_gen_stats_weekly = make_subplots(specs=[[{"secondary_y": True}]])
fig_gen_stats_weekly.add_trace(go.Scatter(x=covid_general_weekly.index[1:], y=covid_general_weekly.new_cases[1:], name='New confirmed cases'), secondary_y=False)
fig_gen_stats_weekly.add_trace(go.Bar(x=covid_general_weekly.index[1:], y=covid_general_weekly.new_deaths[1:], name='New death cases'), secondary_y=True)
fig_gen_stats_weekly.add_trace(go.Scatter(x=covid_general_weekly.index[1:], y=covid_general_weekly.new_recoveries[1:], name='New recoveriers'), secondary_y=True)
fig_gen_stats_weekly.update_layout(title = 'New cases per week')
fig_gen_stats_weekly.update_xaxes(title_text="week number")
fig_gen_stats_weekly.update_yaxes(title_text="Confirmed cases", secondary_y=False)
fig_gen_stats_weekly.update_yaxes(title_text="Deaths / recoveries", secondary_y=True)
#fig_gen_stats_weekly.show()
fig_gen_stats_weekly = fig_gen_stats_weekly.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')


fig_gen_stats_weekly_new_pct = go.Figure()
fig_gen_stats_weekly_new_pct.add_trace(go.Scatter(x=covid_general_weekly.index[1:], y=covid_general_weekly.new_cases_pct_change[1:], line=dict(color='orange'), line_shape='spline', name='Confirmed % change'))
fig_gen_stats_weekly_new_pct.add_trace(go.Scatter(x=[25, 25], y=[-0.2,0.5],
                             mode='lines', line=dict(dash='dash'), name='Borders reopening', marker_color='white'))
fig_gen_stats_weekly_new_pct.add_trace(go.Scatter(x=[27, 27], y=[-0.2,0.5],
                             mode='lines', line=dict(dash='dash'), name='Football Cup final', marker_color='yellow'))
fig_gen_stats_weekly_new_pct.add_trace(go.Scatter(x=[28, 28], y=[-0.2,0.5],
                             mode='lines', line=dict(dash='dash'), name='Start of protests', marker_color='cyan'))
fig_gen_stats_weekly_new_pct.add_trace(go.Scatter(x=[36, 36], y=[-0.2,0.5],
                             mode='lines', line=dict(dash='dash'), name='1st mass protest', marker_color='brown'))
fig_gen_stats_weekly_new_pct.add_trace(go.Scatter(x=[38, 38], y=[-0.2,0.5],
                             mode='lines', line=dict(dash='dash'), name='Schools opening', marker_color='red'))
fig_gen_stats_weekly_new_pct.update_layout(title='New cases over time - weekly % change')
#fig_gen_stats_weekly_new_pct.show()
fig_gen_stats_weekly_new_pct = fig_gen_stats_weekly_new_pct.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')


fig_gen_stats_weekly_events = go.Figure()
fig_gen_stats_weekly_events.add_trace(go.Scatter(x=covid_general_weekly.index[1:], 
                                                 y=covid_general_weekly.new_cases[1:],
                                                 name='New confirmed cases'))
fig_gen_stats_weekly_events.add_trace(go.Scatter(x=[25, 25], y=[0,10000],
                             mode='lines', line=dict(dash='dash'), name='Borders reopening', marker_color='white'))
fig_gen_stats_weekly_events.add_trace(go.Scatter(x=[27, 27], y=[0,10000],
                             mode='lines', line=dict(dash='dash'), name='Football Cup final', marker_color='yellow'))
fig_gen_stats_weekly_events.add_trace(go.Scatter(x=[28, 28], y=[0,10000],
                             mode='lines', line=dict(dash='dash'), name='Start of protests', marker_color='cyan'))
fig_gen_stats_weekly_events.add_trace(go.Scatter(x=[36, 36], y=[0,10000],
                             mode='lines', line=dict(dash='dash'), name='1st mass protest', marker_color='purple'))
fig_gen_stats_weekly_events.add_trace(go.Scatter(x=[38, 38], y=[0,10000],
                             mode='lines', line=dict(dash='dash'), name='Schools opening', marker_color='red'))
fig_gen_stats_weekly_events.update_layout(title='New confirmed cases per week + summer events')
fig_gen_stats_weekly_events.update_xaxes(range=[24, 43])
fig_gen_stats_weekly_events.update_yaxes(range=[0, 6000])
#fig_gen_stats_weekly_events.show()
fig_gen_stats_weekly_events = fig_gen_stats_weekly_events.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')


covid_general['total_cases_7days_ago'] = covid_general['total_cases'].shift(periods=7)
covid_general['total_cases_14days_ago'] = covid_general['total_cases'].shift(periods=14)
covid_general['death_rate'] = (covid_general['total_deaths'] / covid_general['total_cases_14days_ago']).round(4)
covid_general['recovery_rate'] = (covid_general['total_recoveries'] / covid_general['total_cases_14days_ago']).round(4)
covid_general['hospitalized_rate'] = (covid_general['hospitalized'] / covid_general['active_cases']).round(4)
covid_general['intensive_care_rate'] = (covid_general['intensive_care'] / covid_general['hospitalized']).round(4)
covid_general['tests_positive_rate'] = (covid_general['new_cases'] / covid_general['daily_tests']).round(4)

logger.info('Charting rates')

fig_rates_mort_rec = go.Figure()
fig_rates_mort_rec.add_trace(go.Scatter(x=covid_general.date, y=covid_general.death_rate,
                               line_shape='spline', line=dict(color='red'), name='Mortality rate'))
fig_rates_mort_rec.add_trace(go.Scatter(x=covid_general.date, y=covid_general.recovery_rate,
                               line_shape='spline', line=dict(color='green'), name='Recovery rate', visible='legendonly'))
fig_rates_mort_rec.update_layout(title='COVID-19 mortality and recovery rates over time')

fig_rates_hospitalized = go.Figure()
fig_rates_hospitalized.add_trace(go.Scatter(x=covid_general.date, y=covid_general.hospitalized_rate,
                               line_shape='spline', line=dict(color='yellow'), name='Hospitalized rate'))
fig_rates_hospitalized.add_trace(go.Scatter(x=covid_general.date, y=covid_general.intensive_care_rate,
                               line_shape='spline', line=dict(color='orange'), name='Intensive care rate'))
fig_rates_hospitalized.update_layout(title="COVID-19 hospitalized and intensive care rates over time")

fig_rates_positive_tests = go.Figure()
fig_rates_positive_tests.add_trace(go.Scatter(x=covid_general.date, y=covid_general.tests_positive_rate,
                               line_shape='spline', line=dict(color='cyan'), name='Tests positive rate'))
fig_rates_positive_tests.update_layout(title="COVID-19 positive tests rate")
# for the dashboard
for f in [fig_rates_mort_rec, fig_rates_hospitalized, fig_rates_positive_tests]:
    f.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')


logger.info('Reading spatial data file')

geodf = read_spatial_data('./shape/BGR_adm1.shp', codes_spatial)
covid_by_province = read_covid_by_province('./data/Разпределение по дата и по области.csv', date_col='Дата')
pop_by_province = read_population_data('./data/Pop_6.1.1_Pop_DR.xls', worksheet_name='2019',
                                           col_num=2, col_names=['municipality','pop'],
                                           skip=5, codes=codes_pop)
covid_pop = (covid_by_province.set_index('code')
                           .join(pop_by_province.set_index('code'))
                           .join(geodf.set_index('code')))
covid_pop['new_per_100k'] = (100000*covid_pop['new_cases']/covid_pop['pop']).round(2)
covid_pop['total_per_100k'] = (100000*covid_pop['ALL']/covid_pop['pop']).round(2)
covid_pop['active_per_100k'] = (100000*covid_pop['ACT']/covid_pop['pop']).round(2)
covid_pop = gpd.GeoDataFrame(covid_pop)

covid_pop.reset_index()[['date', 'province', 'ALL', 'new_cases']].rename(columns={'ALL':'total_cases'}).to_csv('./dash_data/ts_data.csv', header=True)

covid_pop_sorted = covid_pop.sort_values(by=['date', 'ALL'])
# animation frame parameter should be string or int
covid_pop_sorted['day'] = covid_pop_sorted.date.apply(lambda x: (x - min(covid_pop_sorted.date)).days + 1)

covid_yesterday = gpd.GeoDataFrame(covid_pop.loc[covid_pop.date == max(covid_pop.date)])
#from plotly.offline import download_plotlyjs, init_notebook_mode, plot
import plotly.express as px
#init_notebook_mode()

fig_yesterday_map_total = px.choropleth_mapbox(
    covid_yesterday,
    geojson=covid_yesterday.geometry,
    locations=covid_yesterday.index,
    color='total_per_100k',
    color_continuous_scale='Burgyl',
    hover_name='province',
    labels={'total_per_100k':'total infections<br>per 100k pop'},
    title='Total confirmed cases per 100,000 population by province',
    center={'lat': 42.734189, 'lon': 25.1635087},
    mapbox_style='carto-darkmatter',
    opacity=1,
    zoom=6
)
fig_yesterday_map_total.update_layout(margin={"r":0,"t":40,"l":0,"b":0}, template='plotly_dark')
#fig_yesterday_map_total.show()
fig_yesterday_map_total = fig_yesterday_map_total.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')


fig_yesterday_map_new = px.choropleth_mapbox(
    covid_yesterday,
    geojson=covid_yesterday.geometry,
    locations=covid_yesterday.index,
    color='new_per_100k',
    color_continuous_scale='Burgyl',
    hover_name='province',
    labels={'new_per_100k':'infections<br>per 100k pop'},
    title=f"New cases by province for {covid_yesterday.date.max().strftime('%d %b %Y')}",
    center={'lat': 42.734189, 'lon': 25.1635087},
    mapbox_style='carto-darkmatter',
    opacity=1,
    zoom=6
)
fig_yesterday_map_new.update_layout(margin={"r":0,"t":40,"l":0,"b":0}, template='plotly_dark')
#fig_yesterday_map_new.show()
fig_yesterday_map_new = fig_yesterday_map_new.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')


logger.info('Charting age bands')
covid_by_age_band = (pd.read_csv('./data/Разпределение по дата и по възрастови групи.csv', parse_dates=['Дата'])
                     .rename(columns={'Дата':'date'}))
covid_by_age_band.head()

age_band_colors = ['green', 'cyan', 'magenta', 'ghostwhite', 'coral', 'royalblue', 'darkred', 'orange', 'brown']

fig_age = go.Figure()
i=0
for col in covid_by_age_band.columns[1:]:
    fig_age.add_trace(go.Scatter(x=covid_by_age_band['date'], y=covid_by_age_band[col], mode='lines',
                                 line=dict(width=3, color=age_band_colors[i]), name=col))
    i+=1
fig_age.update_layout(title='Total cases over time by age band', template='plotly_dark')

fig_age.add_trace(go.Scatter(x=[pd.Timestamp(2020,6,15), pd.Timestamp(2020,6,15)], y=[-1500,5000],
                             mode='lines', line=dict(dash='dash'), name='Borders reopening', marker_color='white'))
fig_age.add_trace(go.Scatter(x=[pd.Timestamp(2020,7,1), pd.Timestamp(2020,7,1)], y=[-1500,5000],
                             mode='lines', line=dict(dash='dash'), name='Football Cup final', marker_color='white'))
fig_age.add_trace(go.Scatter(x=[pd.Timestamp(2020,7,11), pd.Timestamp(2020,7,11)], y=[-1500,5000],
                             mode='lines', line=dict(dash='dash'), name='Start of protests', marker_color='white'))
fig_age.add_trace(go.Scatter(x=[pd.Timestamp(2020,9,2), pd.Timestamp(2020,9,2)], y=[-1500,5000],
                             mode='lines', line=dict(dash='dash'), name='1st mass protest', marker_color='white'))
fig_age.add_trace(go.Scatter(x=[pd.Timestamp(2020,9,15), pd.Timestamp(2020,9,15)], y=[-1500,5000],
                             mode='lines', line=dict(dash='dash'), name='Schools opening', marker_color='white'))
fig_age = fig_age.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')


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
provinces.head()

bg_total_new = provinces.groupby('date').sum()
bg_total_new.head()

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


orig_bg, smoothed_bg = prepare_cases(bg_total_new)
orig_bg.to_csv('./dash_data/r0_bg_original.csv', header=True)
smoothed_bg.to_csv('./dash_data/r0_bg_smoothed.csv', header=True)

fig_new_bg = go.Figure()
fig_new_bg.add_trace(go.Scatter(x=orig_bg.reset_index()['date'], y=orig_bg.reset_index()['ALL'],
                                      mode='lines', line=dict(dash='dot'), name='Actual'))
fig_new_bg.add_trace(go.Scatter(x=smoothed_bg.reset_index()['date'], y=smoothed_bg.reset_index()['ALL'],
                                      mode='lines', line=dict(width=3), name='Smoothed', marker_color='steelblue'))
fig_new_bg.update_layout(title='Daily new cases in Bulgaria')
fig_new_bg = fig_new_bg.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')


# smoothed new recoveries
bg_recovered = covid_general[['date', 'total_recoveries']].groupby('date')['total_recoveries'].sum()
orig_recovered_bg, smoothed_recovered_bg = prepare_cases(bg_recovered)
orig_recovered_bg.to_csv('./dash_data/r0_bg_original_recovered.csv', header=True)
smoothed_recovered_bg.to_csv('./dash_data/r0_bg_smoothed_recovered.csv', header=True)

fig_recovered_bg = go.Figure()
fig_recovered_bg.add_trace(go.Scatter(x=orig_recovered_bg.reset_index()['date'], y=orig_recovered_bg.reset_index()['total_recoveries'],
                                      mode='lines', line=dict(dash='dot'), name='Actual'))
fig_recovered_bg.add_trace(go.Scatter(x=smoothed_recovered_bg.reset_index()['date'], y=smoothed_recovered_bg.reset_index()['total_recoveries'],
                                      mode='lines', line=dict(width=3), name='Smoothed', marker_color='lime'))
fig_recovered_bg.update_layout(title='Daily new recoveries in Bulgaria')
fig_recovered_bg = fig_recovered_bg.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')


# smoothed deaths
bg_deaths = covid_general[['date', 'total_deaths']].groupby('date')['total_deaths'].sum()
orig_deaths_bg, smoothed_deaths_bg = prepare_cases(bg_deaths)
orig_deaths_bg.to_csv('./dash_data/r0_bg_original_deaths.csv', header=True)
smoothed_deaths_bg.to_csv('./dash_data/r0_bg_smoothed_deaths.csv', header=True)

fig_deaths_bg = go.Figure()
fig_deaths_bg.add_trace(go.Scatter(x=orig_deaths_bg.reset_index()['date'], y=orig_deaths_bg.reset_index()['total_deaths'],
                                      mode='lines', line=dict(dash='dot'), name='Actual'))
fig_deaths_bg.add_trace(go.Scatter(x=smoothed_deaths_bg.reset_index()['date'], y=smoothed_deaths_bg.reset_index()['total_deaths'],
                                      mode='lines', line=dict(width=3), name='Smoothed'))
fig_deaths_bg.update_layout(title='Daily new deaths in Bulgaria')
fig_deaths_bg = fig_deaths_bg.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

# smoothed new cases by age band
covid_by_age_band_diff_smoothed = covid_by_age_band.set_index('date').diff().rolling(9,
        win_type='gaussian',
        min_periods=1,
        center=True).mean(std=3).round()

fig_age_diff = go.Figure()
i=0
for col in covid_by_age_band_diff_smoothed.columns:
    fig_age_diff.add_trace(go.Scatter(x=covid_by_age_band_diff_smoothed.index, y=covid_by_age_band_diff_smoothed[col], mode='lines', line_shape='spline',
                                 line=dict(color=age_band_colors[i]), name=col))
    i+=1
fig_age_diff.update_layout(title='New confirmed cases by age band (smoothed figures)')


# age band new cases per 100k pop
pop_by_age_band = read_nsi_age_bands('./data/Pop_6.1.2_Pop_DR.xls', worksheet_name='2019', col_num=2, col_names=['age_band', 'pop'], skip=5, rows_needed=22)
covid_by_age_band_diff_smoothed_per100k = covid_by_age_band_diff_smoothed.copy()
for col in covid_by_age_band_diff_smoothed_per100k.columns:
    covid_by_age_band_diff_smoothed_per100k[col] = (100000*covid_by_age_band_diff_smoothed_per100k[col]/pop_by_age_band.loc[pop_by_age_band.covid_age_band == col, 'pop'].values).round(0)
    
fig_age_per100k = go.Figure()
i=0
for col in covid_by_age_band_diff_smoothed_per100k.columns:
    fig_age_per100k.add_trace(go.Scatter(x=covid_by_age_band_diff_smoothed_per100k.index, y=covid_by_age_band_diff_smoothed_per100k[col], mode='lines', line_shape='spline',
                                 line=dict(width=2, color=age_band_colors[i]), name=col))
    i+=1
fig_age_per100k.update_layout(title='New confirmed daily cases by age band per 100,000 population (smoothed figures)')



# provinces

provinces_list = covid_pop[['province', 'pop']].drop_duplicates().sort_values(by='pop', ascending=False).province.values

# create subplots structure
fig_new_by_province = make_subplots(
                            rows=int(len(provinces_list)/2),
                            cols=2,
                            subplot_titles = [f"{province}" for province in provinces_list]
                        )
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
    

for i, province in list(enumerate(provinces_list)):
    i += 1
    import math
    row_num = math.ceil(i/2)
    if i % 2 != 0:
        col_num = 1
    else:
        col_num = 2

    fig_new_by_province.add_trace(go.Scatter(x=r0_provinces_original.reset_index()['date'], y=r0_provinces_original['new_cases'],
                                      mode='lines', line=dict(dash='dot'), name='Actual'),
                                  row=row_num, col=col_num)
    fig_new_by_province.add_trace(go.Scatter(x=r0_provinces_smoothed.reset_index()['date'], y=r0_provinces_smoothed['new_cases'],
                                      mode='lines', line=dict(width=3), name='Smoothed'),
                                  row=row_num, col=col_num)
    fig_new_by_province.update_layout(title='Daily new cases by province', height=3200, showlegend=False)

#fig_new_by_province.show()
fig_new_by_province = fig_new_by_province.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')


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


sigmas = np.linspace(1/100, 1, 100)
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

result_bg.tail()

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

fig_rt = go.Figure()
fig_rt.add_trace(go.Scatter(x=index_bg, y=lowfn_bg(date2num(extended_bg)),
                            fill='none', mode='lines', line_color="rgba(38,38,38,0.9)", line_shape='spline',
                            name="Low density interval"))
fig_rt.add_trace(go.Scatter(x=index_bg, y=highfn_bg(date2num(extended_bg)),
                            fill='tonexty', mode='none', fillcolor="rgba(65,65,65,1)", line_shape='spline',
                            name="High density interval"))
fig_rt.add_trace(go.Scatter(x=index_bg, y=values_bg, mode='markers+lines', line=dict(width=0.3, dash='dot'), line_shape='spline',
                            marker_color=values_bg, marker_colorscale='RdYlBu_r', marker_line_width=1.2,
                            marker_cmin=0.5, marker_cmax=1.4, name='R<sub>t'))
fig_rt.update_layout(yaxis=dict(range=[0,4]), title="Real-time R<sub>t</sub> for Bulgaria", showlegend=False)
fig_rt.show()
fig_rt = fig_rt.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')


logger.info('Calculating Rt for provinces')

sigmas = np.linspace(1/20, 1, 20)

provinces_to_process = provinces

results = {}

for province_name, cases in provinces_to_process.groupby(level='province'):
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
    #clear_output(wait=True)
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


fig_rt_province_yesterday = go.Figure()
fig_rt_province_yesterday.add_trace(go.Bar(x=mr.index, y=mr.Estimated, marker_color=mr.colors, 
                          error_y=dict(type='data', array=mr.diff_up, arrayminus=mr.diff_down)))
fig_rt_province_yesterday.update_layout(title='R<sub>t</sub> by province for the last daily update')
fig_rt_province_yesterday.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

logger.info('Charting Rt for provinces')

def generate_rt_by_province(provinces, final_results):
    provinces_list = covid_pop[['province', 'pop']].drop_duplicates().sort_values(by='pop', ascending=False).province.values
    data = final_results.reset_index()
    # create base subplot
    fig_rt_province = make_subplots(
                                rows=int(len(provinces_list)/2),
                                cols=2,
                                subplot_titles=[province for province in provinces_list],
                            )
    # calculate figures for provinces
    for i, province in list(enumerate(provinces_list)):
        subset = data.loc[data.province==province]

    # add charts for provinces
        i += 1
        row_num = math.ceil(i/2)
        if i % 2 != 0:
            col_num = 1
        else:
            col_num = 2

        fig_rt_province.add_trace(go.Scatter(x=subset.date, y=subset.Low_90,
                                fill='none', mode='lines', line_color="rgba(38,38,38,0.9)", line_shape='spline',
                                name="Low density interval"), row=row_num, col=col_num)
        fig_rt_province.add_trace(go.Scatter(x=subset.date, y=subset.High_90,
                                fill='tonexty', mode='none', fillcolor="rgba(65,65,65,1)", line_shape='spline',
                                name="High density interval"), row=row_num, col=col_num)
        fig_rt_province.add_trace(go.Scatter(x=subset.date, y=subset.Estimated, mode='markers+lines',
                                line=dict(width=0.3, dash='dot'), line_shape='spline',
                                marker_color=subset.Estimated, marker_colorscale='RdYlBu_r', marker_line_width=1.2,
                                marker_cmin=0.5, marker_cmax=1.4, name='R<sub>t'), row=row_num, col=col_num)

    fig_rt_province.update_layout(yaxis=dict(range=[0,4]), title="Real-time R<sub>t</sub> by province",
                                      height=4000, showlegend=False)
    return fig_rt_province


fig_rt_province_actual = generate_rt_by_province(provinces, final_results)

logger.info('Starting ARIMA')

import itertools
import pandas as pd
import seaborn as sns
import math
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
#%matplotlib inline
import warnings
warnings.simplefilter('ignore')


ts_data = pd.read_csv('./dash_data/ts_data.csv')
#ts_data.head()

#Function for making a time series on a province and plotting the rolled mean and standard deviation
def roll(ts_data, region, column='total_cases'):
    test_s=ts_data.loc[(ts_data['province']==region)]  
    test_s=test_s[['date',column]]
    test_s=test_s.set_index('date')
    test_s.astype('int64')
    a=len(test_s.loc[(test_s[column]>=10)])
    test_s=test_s[-a:]
    return (test_s.rolling(window=7,center=False).mean().dropna())

def rollPlot(ts_data, region, column='total_cases'):
    test_s=ts_data.loc[(ts_data['province']==region)]  
    test_s=test_s[['date',column]]
    test_s=test_s.set_index('date')
    test_s.astype('int64')
    a=len(test_s.loc[(test_s[column]>=10)])
    test_s=test_s[-a:]
    plt.figure(figsize=(16,6))
    plt.plot(test_s.rolling(window=7,center=False).mean().dropna(),label='Rolling Mean')
    plt.plot(test_s[column], label = 'Total cases')
    plt.plot(test_s.rolling(window=7,center=False).std(),label='Rolling std')
    plt.legend()
    plt.title('Cases distribution in %s with rolling mean and standard' %region)
    plt.xticks([])

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
            result = model.fit()
            if (result.aic<=a) :
                a=result.aic
                param=var
        except:
            continue
        
    #Modeling
    model = ARIMA(train, order=param)
    result = model.fit()
    
    #import matplotlib as mpl
    #with mpl.rc_context():
        #mpl.rc("figure", figsize=(12,5))
        #result.plot_predict(start=int(len(train) * 0.7), end=int(len(train) * 1.2))
        #plt.savefig(f"assets/arima_forecast_{province}.png", transparent=True)
    
    pred=result.forecast(steps=len(test))[0]
    #Printing the error metrics
    #print(result.summary())
    model_error = mape(test,pred)
    #print('\nMean absolute percentage error: %f'%model_error)
    #Plotting results
    fig = go.Figure()
    #fig.add_trace(go.Scatter(x=rolling[:len(train)].index, y=rolling[column][:len(train)], name='History', mode='lines'))
    #fig.add_trace(go.Scatter(x=rolling[-len(test):].index, y=rolling[column][-len(test):], name='Actual', mode='lines'))
    #fig.add_trace(go.Scatter(x=rolling[-len(test):].index, y=pred, name='Forecast', mode='lines'))
    #fig.update_layout(title = f'True vs Predicted values for {"new cases" if column=="new_cases" else "total cases"} (7 days rolling mean) in {province} for {forecast_days} days')
    #fig.show()

    return (pred, result, fig, model_error, rolling.index, rolling[column])

logger.info('Getting ARIMA predictions for provinces')

from datetime import datetime
start = datetime.now()
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

logger.info(f'Provinces ARIMA runtime: {datetime.now() - start}')

arima_provinces_df.to_csv('./dash_data/arima_provinces.csv', header=True)


df = result_bg[['Estimated']].rename(columns={'Estimated':'data'})

import matplotlib.dates as mdates


train = df.iloc[:-10, :]
test = df.iloc[-10:, :]
pred = test.copy()
df.plot(figsize=(12,3));
plt.title('Rt');

df['z_data'] = (df['data'] - df.data.rolling(window=12).mean()) / df.data.rolling(window=12).std()
df['zp_data'] = df['z_data'] - df['z_data'].shift(12)

def plot_rolling(df):
    fig, ax = plt.subplots(3,figsize=(12, 9))
    ax[0].plot(df.index, df.data, label='raw data')
    ax[0].plot(df.data.rolling(window=12).mean(), label="rolling mean");
    ax[0].plot(df.data.rolling(window=12).std(), label="rolling std (x10)");
    ax[0].legend()

    ax[1].plot(df.index, df.z_data, label="de-trended data")
    ax[1].plot(df.z_data.rolling(window=12).mean(), label="rolling mean");
    ax[1].plot(df.z_data.rolling(window=12).std(), label="rolling std (x10)");
    ax[1].legend()

    ax[2].plot(df.index, df.zp_data, label="12 lag differenced de-trended data")
    ax[2].plot(df.zp_data.rolling(window=12).mean(), label="rolling mean");
    ax[2].plot(df.zp_data.rolling(window=12).std(), label="rolling std (x10)");
    ax[2].legend()

    plt.tight_layout()
    fig.autofmt_xdate()
    
plot_rolling(df)

logger.info("Holt's model")

from statsmodels.tsa.holtwinters import Holt

df = df.resample("D").sum()
train = df.iloc[:-15]
test = df.iloc[-15:]
train.index = pd.to_datetime(train.index)
test.index = pd.to_datetime(test.index)
pred = test.copy()

    
model_double = Holt(np.asarray(train['data']))
model_double._index = pd.to_datetime(train.index)

fit1_double = model_double.fit(smoothing_level=.3, smoothing_trend=.05)
pred1_double = fit1_double.forecast(15)
fit2_double = model_double.fit(optimized=True)
pred2_double = fit2_double.forecast(15)
fit3_double = model_double.fit(smoothing_level=.3, smoothing_trend=.2)
pred3_double = fit3_double.forecast(15)

fig_exp_smoothing_double = go.Figure()
fig_exp_smoothing_double.add_trace(go.Scatter(x=df.index, y=df.data, name='actual data'))

for p, f, c in zip((pred1_double, pred2_double, pred3_double),(fit1_double, fit2_double, fit3_double),('coral','yellow','cyan')):
    fig_exp_smoothing_double.add_trace(go.Scatter(x=train.index, y=f.fittedvalues, marker_color=c, mode='lines',
                            name=f"alpha={str(f.params['smoothing_level'])[:4]}, beta={str(f.params['smoothing_trend'])[:4]}")
    )
    fig_exp_smoothing_double.add_trace(go.Scatter(
        x=pd.date_range(start=test.index.min(), periods=len(test) + len(p)),
        y=p, marker_color=c, mode='lines', showlegend=False)
    )
    print(f"\nMean absolute percentage error: {mape(test['data'].values,p).round(2)} (alpha={str(f.params['smoothing_level'])[:4]}, beta={str(f.params['smoothing_trend'])[:4]})")

fig_exp_smoothing_double.update_layout(title="Holt's (double) exponential smoothing for R<sub>t</sub> in Bulgaria")
fig_exp_smoothing_double.show()
fig_exp_smoothing_double = fig_exp_smoothing_double.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')


def double_exp_smoothing(ts_data, province, column='total_cases', forecast_days=15):
    df = ts_data.set_index('date')
    df = df.loc[df['province'] == province]
    df = df.resample("D").sum()
    
    train = df.iloc[:-forecast_days]
    test = df.iloc[-forecast_days:]
    pred = test.copy()
    
    model = Holt(np.asarray(train[column].values))
    model._index = pd.to_datetime(train.index)

    fit1 = model.fit(smoothing_level=.3, smoothing_trend=.05)
    pred1 = fit1.forecast(15)
    fit2 = model.fit(optimized=True)
    pred2 = fit2.forecast(15)
    fit3 = model.fit(smoothing_level=.3, smoothing_trend=.2)
    pred3 = fit3.forecast(15)

    fig_exp_smoothing_double = go.Figure()
    fig_exp_smoothing_double.add_trace(go.Scatter(x=train.index, y=train[column], name='Training data', mode='lines'))
    fig_exp_smoothing_double.add_trace(go.Scatter(x=test.index, y=test[column], name='Testing data', mode='lines', marker_color='coral'))

    for p, f, c in zip((pred1, pred2, pred3),(fit1, fit2, fit3),('darkcyan','gold','cyan')):
        fig_exp_smoothing_double.add_trace(go.Scatter(x=train.index, y=f.fittedvalues, marker_color=c, mode='lines',
                                name=f"alpha={str(f.params['smoothing_level'])[:4]}, beta={str(f.params['smoothing_trend'])[:4]}")
        )
        fig_exp_smoothing_double.add_trace(go.Scatter(
            x=pd.date_range(start=test.index.min(), periods=len(test) + len(p)),
            y=p, marker_color=c, mode='lines', showlegend=False)
        )
        print(f"\nMean absolute percentage error: {mape(test[column].values,p).round(2)} (alpha={str(f.params['smoothing_level'])[:4]}, beta={str(f.params['smoothing_trend'])[:4]})")

    fig_exp_smoothing_double.update_layout(title=f"Holt (double) exponential smoothing for {'new cases' if column == 'new_cases' else 'total cases'} in {province}")
    return fig_exp_smoothing_double


logger.info("Holt-Winter's model")

from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES


#build and train the model on the training data
model_triple = HWES(train.data, seasonal_periods=7, trend='add', seasonal='mul')
fitted_triple = model_triple.fit(optimized=True, use_brute=True)

#print out the training summary
print(fitted_triple.summary())

#create an out of sample forcast for the next steps beyond the final data point in the training data set
pred_triple = fitted_triple.forecast(steps=15)

print(f"\nMean absolute percentage error: {mape(test['data'].values,pred_triple).round(2)}")

#plot the training data, the test data and the forecast on the same plot
fig_exp_smoothing_triple = go.Figure()
fig_exp_smoothing_triple.add_trace(go.Scatter(x=train.index[30:], y=train.data[30:], name='Training data', mode='lines'))
fig_exp_smoothing_triple.add_trace(go.Scatter(x=test.index, y=test.data, name='Testing data', mode='lines'))
fig_exp_smoothing_triple.add_trace(go.Scatter(x=train.index[30:], y=fitted_triple.fittedvalues[30:], name='Model fit', mode='lines', marker_color='lime'))
fig_exp_smoothing_triple.add_trace(go.Scatter(
    x=pd.date_range(start=test.index.min(), periods=len(test) + len(pred_triple)),
    y=pred_triple, name='Forecast', marker_color='gold', mode='lines')
)
fig_exp_smoothing_triple.update_layout(title='Holt-Winters (triple) exponential smoothing for R<sub>t</sub> in Bulgaria')
fig_exp_smoothing_triple.show()
fig_exp_smoothing_triple = fig_exp_smoothing_triple.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')


def triple_exp_smoothing(ts_data, province, column='total_cases', forecast_days=15):
    df = ts_data.set_index('date')
    # replace zeros with 0.1 as the multiplicative seasonal element o HWES requires strictly positive values
    df = df.loc[df['province'] == province].replace(0,0.1)
    df = df.resample("D").sum()
    
    train = df.iloc[:-forecast_days]
    test = df.iloc[-forecast_days:]
    pred = test.copy()
    
    model_triple = HWES(train[column], seasonal_periods=7, trend='add', seasonal='mul')
    fitted_triple = model_triple.fit(optimized=True, use_brute=True)
    pred_triple = fitted_triple.forecast(steps=forecast_days)
    pred_triple_error = mape(test[column].values,pred_triple).round(2)
    print(f"\nMean absolute percentage error: {pred_triple_error}")

    #plot the training data, the test data and the forecast on the same plot
    fig_exp_smoothing_triple = go.Figure()
    fig_exp_smoothing_triple.add_trace(go.Scatter(x=train.index[30:], y=train[column][30:], name='Training data', mode='lines'))
    fig_exp_smoothing_triple.add_trace(go.Scatter(x=train.index[30:], y=fitted_triple.fittedvalues[30:], name='Model fit', mode='lines', marker_color='lime'))
    fig_exp_smoothing_triple.add_trace(go.Scatter(x=test.index, y=test[column], name='Testing data', mode='lines', marker_color='coral'))
    fig_exp_smoothing_triple.add_trace(go.Scatter(
        x=pd.date_range(start=test.index.min(), periods=len(test) + len(pred_triple)),
        y=pred_triple, name='Forecast', marker_color='gold', mode='lines')
    )
    fig_exp_smoothing_triple.update_layout(title=f'Holt-Winters (triple) exponential smoothing for {"new cases" if column == "new_cases" else "total cases"} in {province} for {forecast_days} days')
    return fig_exp_smoothing_triple, pred_triple_error


logger.info('Starting git push')
from func_git import *
git_push_result = git_push_automation()
logger.info(git_push_result)


logger.info('FINISHED! Starting dash...')

import app as appscript
app.run_server(debug=True)



