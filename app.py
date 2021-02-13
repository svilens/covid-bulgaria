from func_logging import *
import geopandas as gpd
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta

# from external modules
from func_read_spatial import *
from func_read_covid import *
from func_read_nsi import *

logger = get_logger_app('app')
logger.info('Starting the process')


####### GENERAL STATS #######

logger.info('Reading general stats')
covid_general = read_covid_general('./data/COVID_general.csv', 'Дата')

import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "plotly_dark"
from plotly.offline import plot


logger.info('Creating chart 1: Cumulative cases over time')
fig_gen_stats = go.Figure()
fig_gen_stats.add_trace(go.Scatter(x=covid_general.date, y=covid_general.total_cases, name='Confirmed'))
fig_gen_stats.add_trace(go.Scatter(x=covid_general.date, y=covid_general.active_cases, line=dict(color='yellow'), name='Active'))
fig_gen_stats.add_trace(go.Scatter(x=covid_general.date, y=covid_general.total_recoveries, line=dict(color='green'), name='Recovered'))
fig_gen_stats.add_trace(go.Scatter(x=covid_general.date, y=covid_general.total_deaths, line=dict(color='red'), name='Deaths'))
fig_gen_stats.update_layout(title='Number of cases over time (cumulative)')
fig_gen_stats.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')


logger.info('Creating chart 2: Cases per week')

#covid_general['week'] = covid_general['date'].dt.isocalendar().week
epoch = pd.Timestamp("2019-12-30") # continue the week number count when the year changes
covid_general['day_name'] = covid_general['date'].dt.day_name()
covid_general['week'] = (np.where(covid_general.date.astype("datetime64").le(epoch),
                               covid_general.date.dt.isocalendar().week,
                               covid_general.date.sub(epoch).dt.days//7+1)
)
covid_general_weekly = covid_general.groupby('week')[['new_cases', 'new_deaths', 'new_recoveries']].sum()
covid_general_weekly['new_cases_pct_change'] = covid_general_weekly['new_cases'].pct_change()

if covid_general_weekly.iloc[-1,0] == 53:
    pass
else:
    # getting the current day to calculate projected values for the current week
    last_day_num = covid_general['date'].dt.isocalendar().day.values[-1]
    # projected current week confirmed cases
    covid_general_weekly.iloc[-1,0] = round(covid_general_weekly.iloc[-1,0] * [24.9 if x==1 else 5.4 if x==2 else 2.6 if x==3 else 1.8 if x==4 else 1.4 if x==5 else 1.1 if x==6 else 1 for x in [last_day_num]][0],0)
    # projected current week death cases
    covid_general_weekly.iloc[-1,1] = round(covid_general_weekly.iloc[-1,1] * [14.7 if x==1 else 3.6 if x==2 else 2.2 if x==3 else 1.6 if x==4 else 1.3 if x==5 else 1.1 if x==6 else 1 for x in [last_day_num]][0],0)
    # projected current week recovered cases
    covid_general_weekly.iloc[-1,2] = round(covid_general_weekly.iloc[-1,2] * [12.7 if x==1 else 4.3 if x==2 else 2.5 if x==3 else 1.8 if x==4 else 1.3 if x==5 else 1.1 if x==6 else 1 for x in [last_day_num]][0],0)

# removing the first week as it starts on Saturday
covid_general_weekly = covid_general_weekly[1:]

from plotly.subplots import make_subplots

fig_gen_stats_weekly = make_subplots(specs=[[{"secondary_y": True}]])
fig_gen_stats_weekly.add_trace(go.Scatter(x=covid_general_weekly.index[1:], y=covid_general_weekly.new_cases[1:], name='New confirmed cases', line_shape='spline'), secondary_y=True)
fig_gen_stats_weekly.add_trace(go.Bar(x=covid_general_weekly.index[1:], y=covid_general_weekly.new_deaths[1:], name='New death cases'), secondary_y=False)
fig_gen_stats_weekly.add_trace(go.Scatter(x=covid_general_weekly.index[1:], y=covid_general_weekly.new_recoveries[1:], name='New recoveries', line_shape='spline'), secondary_y=True)
fig_gen_stats_weekly.add_annotation(
    x=53, y=530, text="New Year",
    showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
    arrowcolor='orange', font=dict(color='orange', size=15),
    ax=0, ay=-100)
fig_gen_stats_weekly.update_layout(title = 'New cases per week (projected estimations for the current week)')
fig_gen_stats_weekly.update_xaxes(title_text="week number")
fig_gen_stats_weekly.update_yaxes(title_text="Confirmed/recovered", secondary_y=True)
fig_gen_stats_weekly.update_yaxes(title_text="Deaths", secondary_y=False)
fig_gen_stats_weekly.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')


logger.info('Creating chart 3: New cases weekly % change')
fig_gen_stats_weekly_new_pct = go.Figure()
fig_gen_stats_weekly_new_pct.add_trace(go.Scatter(x=covid_general_weekly.index[1:], y=covid_general_weekly.new_cases_pct_change[1:], line=dict(color='orange'), line_shape='spline', name='Confirmed % change'))
fig_gen_stats_weekly_new_pct.add_annotation(
    x=25, y=0.08, text="Borders reopening",
    showarrow=True, arrowhead=1,
    arrowcolor='ivory', font=dict(color='ivory'),
    ax=20, ay=50)
fig_gen_stats_weekly_new_pct.add_annotation(
    x=27, y=0.41, text="Football cup final",
    showarrow=True, arrowhead=1,
    arrowcolor='cornflowerblue', font=dict(color='cornflowerblue'),
    ax=-30)
fig_gen_stats_weekly_new_pct.add_annotation(
    x=28, y=0.43, text="Start of protests",
    showarrow=True, arrowhead=1,
    arrowcolor='cyan', font=dict(color='cyan'),
    ax=20, ay=-50)
fig_gen_stats_weekly_new_pct.add_annotation(
    x=36, y=-0.04, text="First mass protest",
    showarrow=True, arrowhead=1,
    arrowcolor='lime', font=dict(color='lime'),
    ay=-50)
fig_gen_stats_weekly_new_pct.add_annotation(
    x=38, y=0.12, text="Schools opening",
    showarrow=True, arrowhead=1,
    arrowcolor='red', font=dict(color='red'),
    ay=-70)
fig_gen_stats_weekly_new_pct.add_annotation(
    x=48, y=-0.08, text="Second lockdown",
    showarrow=True, arrowhead=1,
    arrowcolor='yellow', font=dict(color='yellow'),
    ax=30, ay=-50)
fig_gen_stats_weekly_new_pct.add_annotation(
    x=52, y=-0.49   , text="Antigen tests",
    showarrow=True, arrowhead=1,
    arrowcolor='orange', font=dict(color='orange'),
    ay=20)
fig_gen_stats_weekly_new_pct.add_annotation(
    x=57, y=0.33, text="Vaccines",
    showarrow=True, arrowhead=1,
    arrowcolor='green', font=dict(color='green'))
fig_gen_stats_weekly_new_pct.add_annotation(
    x=58, y=0.26, text="Schools partial reopening",
    showarrow=True, arrowhead=1,
    arrowcolor='red', font=dict(color='red'),
    ax=-20, ay=30)
fig_gen_stats_weekly_new_pct.update_layout(title='New cases over time - weekly % change', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
        yaxis=dict(tickformat=',.0%', hoverformat=',.2%'))


logger.info('Creating chart 4: New cases per week + summer events')
fig_gen_stats_weekly_events = go.Figure()
fig_gen_stats_weekly_events.add_trace(go.Scatter(x=covid_general_weekly.index[1:], 
                                                 y=covid_general_weekly.new_cases[1:],
                                                 name='New confirmed cases'))
fig_gen_stats_weekly_events.add_annotation(
    x=25, y=626, text="Borders reopening",
    showarrow=True, arrowhead=1,
    arrowcolor='ivory', font=dict(color='ivory'),
    ax=10)
fig_gen_stats_weekly_events.add_annotation(
    x=27, y=1072, text="Football cup final",
    showarrow=True, arrowhead=1,
    arrowcolor='cornflowerblue', font=dict(color='cornflowerblue'))
fig_gen_stats_weekly_events.add_annotation(
    x=28, y=1518, text="Start of protests",
    showarrow=True, arrowhead=1,
    arrowcolor='cyan', font=dict(color='cyan'),
    ax=10)
fig_gen_stats_weekly_events.add_annotation(
    x=36, y=906, text="First mass protest",
    showarrow=True, arrowhead=1,
    arrowcolor='lime', font=dict(color='lime'))
fig_gen_stats_weekly_events.add_annotation(
    x=38, y=948, text="Schools opening",
    showarrow=True, arrowhead=1,
    arrowcolor='red', font=dict(color='red'),
    ax=10, ay=-30)
fig_gen_stats_weekly_events.update_layout(title='New confirmed cases per week + summer events', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
fig_gen_stats_weekly_events.update_xaxes(range=[24, 43])
fig_gen_stats_weekly_events.update_yaxes(range=[0, 6000])


logger.info('Creating chart 5: New cases per week, events and second lockdown')
fig_gen_stats_weekly_events_2 = go.Figure()
fig_gen_stats_weekly_events_2.add_trace(go.Scatter(x=covid_general_weekly.index[1:], 
                                                 y=covid_general_weekly.new_cases[1:],
                                                 name='New confirmed cases'))
fig_gen_stats_weekly_events_2.add_annotation(
    x=25, y=706, text="Borders reopening",
    showarrow=True, arrowhead=1,
    arrowcolor='ivory', font=dict(color='ivory'),
    ax=10)
fig_gen_stats_weekly_events_2.add_annotation(
    x=27, y=1152, text="Football cup final",
    showarrow=True, arrowhead=1,
    arrowcolor='cornflowerblue', font=dict(color='cornflowerblue'),
    ax=5, ay=-50)
fig_gen_stats_weekly_events_2.add_annotation(
    x=28, y=1598, text="Start of protests",
    showarrow=True, arrowhead=1,
    arrowcolor='cyan', font=dict(color='cyan'),
    ax=50, ay=-30)
fig_gen_stats_weekly_events_2.add_annotation(
    x=36, y=986, text="First mass protest",
    showarrow=True, arrowhead=1,
    arrowcolor='lime', font=dict(color='lime'))
fig_gen_stats_weekly_events_2.add_annotation(
    x=38, y=1028, text="Schools opening",
    showarrow=True, arrowhead=1,
    arrowcolor='red', font=dict(color='red'),
    ax=10, ay=-50)
fig_gen_stats_weekly_events_2.add_annotation(
    x=48, y=21150, text="Second lockdown",
    showarrow=True, arrowhead=1,
    arrowcolor='yellow', font=dict(color='yellow'),
    ax=30, ay=-30)
fig_gen_stats_weekly_events_2.add_annotation(
    x=52, y=6255, text="Antigen tests",
    showarrow=True, arrowhead=1,
    arrowcolor='orange', font=dict(color='orange'),
    ay=50)
fig_gen_stats_weekly_events_2.add_annotation(
    x=57, y=3922, text="Vaccines",
    showarrow=True, arrowhead=1,
    arrowcolor='green', font=dict(color='green'),
    ax=-20)
fig_gen_stats_weekly_events_2.add_annotation(
    x=58, y=4934, text="Schools partial reopening",
    showarrow=True, arrowhead=1,
    arrowcolor='red', font=dict(color='red'),
    ax=-20, ay=30)
fig_gen_stats_weekly_events_2.update_layout(title='New confirmed cases per week + summer events + second lockdown', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
fig_gen_stats_weekly_events_2.update_xaxes(range=[24, covid_general_weekly.index[-1]])


### Rates

logger.info('Calculating rates')
covid_general['total_cases_7days_ago'] = covid_general['total_cases'].shift(periods=7)
covid_general['total_cases_14days_ago'] = covid_general['total_cases'].shift(periods=14)
covid_general['death_rate'] = (covid_general['total_deaths'] / covid_general['total_cases_14days_ago']).round(4)
covid_general['recovery_rate'] = (covid_general['total_recoveries'] / covid_general['total_cases_14days_ago']).round(4)
covid_general['hospitalized_rate'] = (covid_general['hospitalized'] / covid_general['active_cases']).round(4)
covid_general['intensive_care_rate'] = (covid_general['intensive_care'] / covid_general['hospitalized']).round(4)
covid_general['tests_positive_rate'] = (covid_general['new_cases'] / covid_general['daily_tests']).round(4)
covid_general['death_rate_v2'] = (covid_general['total_deaths'] / (covid_general['total_deaths'] + covid_general['total_recoveries'])).round(4)
covid_general['recovery_rate_v2'] = (covid_general['total_recoveries'] / (covid_general['total_deaths'] + covid_general['total_recoveries'])).round(4)


logger.info('Creating chart 6: Rates')
fig_rates_mort_rec = go.Figure()
fig_rates_mort_rec.add_trace(go.Scatter(x=covid_general.date, y=covid_general.death_rate,
                               line_shape='spline', line=dict(color='red'), name='Mortality rate'))
fig_rates_mort_rec.add_trace(go.Scatter(x=covid_general.date, y=covid_general.recovery_rate,
                               line_shape='spline', line=dict(color='green'), name='Recovery rate', visible='legendonly'))
fig_rates_mort_rec.update_layout(title='COVID-19 mortality and recovery rates (based on confirmed cases by 14 days ago)',
        yaxis=dict(tickformat=',.0%', hoverformat=',.2%'))

fig_rates_mort_rec_v2 = go.Figure()
fig_rates_mort_rec_v2.add_trace(go.Scatter(x=covid_general.date, y=covid_general.death_rate_v2,
                               line_shape='spline', line=dict(color='red'), name='Mortality rate'))
fig_rates_mort_rec_v2.add_trace(go.Scatter(x=covid_general.date, y=covid_general.recovery_rate_v2,
                               line_shape='spline', line=dict(color='green'), name='Recovery rate', visible='legendonly'))
fig_rates_mort_rec_v2.update_layout(title='COVID-19 mortality and recovery rates (based on closed cases)',
        yaxis=dict(tickformat=',.0%', hoverformat=',.2%'))


fig_rates_hospitalized = go.Figure()
fig_rates_hospitalized.add_trace(go.Scatter(x=covid_general.date, y=covid_general.hospitalized_rate,
                               line_shape='spline', line=dict(color='yellow'), name='Hospitalized rate'))
fig_rates_hospitalized.add_trace(go.Scatter(x=covid_general.date, y=covid_general.intensive_care_rate,
                               line_shape='spline', line=dict(color='orange'), name='Intensive care rate'))
fig_rates_hospitalized.update_layout(title="COVID-19 hospitalized and intensive care rates over time",
        yaxis=dict(tickformat=',.0%', hoverformat=',.2%'))

fig_rates_positive_tests = go.Figure()
fig_rates_positive_tests.add_trace(go.Scatter(x=covid_general.date, y=covid_general.tests_positive_rate,
                               line_shape='spline', line=dict(color='cyan'), name='Tests positive rate'))
fig_rates_positive_tests.update_layout(title="COVID-19 positive tests rate",
        yaxis=dict(tickformat=',.0%', hoverformat=',.2%'))

# for the dashboard
for f in [fig_rates_mort_rec, fig_rates_mort_rec_v2, fig_rates_hospitalized, fig_rates_positive_tests]:
    f.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')


### Map provinces

logger.info('Reading spatial data')
geodf = read_spatial_data('./shape/BGR_adm1.shp', codes_spatial)
covid_by_province = read_covid_by_province('./data/COVID_provinces.csv', date_col='Дата')
pop_by_province = read_population_data('./data/Pop_6.1.1_Pop_DR.xls', worksheet_name='2019',
                                           col_num=2, col_names=['municipality','pop'],
                                           skip=5, codes=codes_pop)

covid_pop = (covid_by_province.set_index('code')
                           .join(pop_by_province.set_index('code'))
                           .join(geodf[['code','province']].set_index('code'))
            )
covid_pop['new_per_100k'] = (100000*covid_pop['new_cases']/covid_pop['pop']).round(0)
covid_pop['total_per_100k'] = (100000*covid_pop['ALL']/covid_pop['pop']).round(0)
covid_pop['active_per_100k'] = (100000*covid_pop['ACT']/covid_pop['pop']).round(0)
#covid_pop = gpd.GeoDataFrame(covid_pop)

covid_pop_sorted = covid_pop.sort_values(by=['date', 'ALL'])
# animation frame parameter should be string or int
#covid_pop_sorted['day'] = covid_pop_sorted.date.apply(lambda x: (x - min(covid_pop_sorted.date)).days + 1)

geodf['geometry'] = geodf['geometry'].simplify(tolerance=0.00001, preserve_topology=True)

covid_yesterday = gpd.GeoDataFrame(
        covid_pop.loc[covid_pop.date == max(covid_pop.date)]
        .rename(columns={'ALL':'total cases', 'ACT':'active cases', 'new_cases':'new cases'})
        .join(geodf[['code','geometry']].set_index('code'))
        )


logger.info('Creating chart 7: Provinces map - total cases per 100k pop')
import plotly.express as px

fig_yesterday_map_total = px.choropleth_mapbox(
    covid_yesterday,
    geojson=covid_yesterday.geometry,
    locations=covid_yesterday.index,
    color='total_per_100k',
    color_continuous_scale='Burgyl',
    hover_name='province',
    hover_data=['total cases'],
    labels={'total_per_100k':'total infections<br>per 100k pop'},
    title='Total confirmed cases per 100,000 population by province',
    center={'lat': 42.734189, 'lon': 25.1635087},
    mapbox_style='carto-darkmatter',
    opacity=0.85,
    zoom=6
)
fig_yesterday_map_total.update_layout(margin={"r":0,"t":40,"l":0,"b":0}, template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')


logger.info('Creating chart 8: Provinces map - new cases per 100k pop')
fig_yesterday_map_new = px.choropleth_mapbox(
    covid_yesterday,
    geojson=covid_yesterday.geometry,
    locations=covid_yesterday.index,
    color='new_per_100k',
    color_continuous_scale='Burgyl',
    hover_name='province',
    hover_data=['new cases'],
    labels={'new_per_100k':'new infections<br>per 100k pop'},
    title=f"New daily cases per 100,000 population by province",
    center={'lat': 42.734189, 'lon': 25.1635087},
    mapbox_style='carto-darkmatter',
    opacity=1,
    zoom=6
)
fig_yesterday_map_new.update_layout(margin={"r":0,"t":40,"l":0,"b":0}, template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')


logger.info('Creating chart 9: Provinces map - active cases per 100k pop')
fig_yesterday_map_active = px.choropleth_mapbox(
    covid_yesterday,
    geojson=covid_yesterday.geometry,
    locations=covid_yesterday.index,
    color='active_per_100k',
    color_continuous_scale='Burgyl',
    hover_name='province',
    hover_data=['active cases'],
    labels={'active_per_100k':'active infections<br>per 100k pop'},
    title=f"Currently active cases per 100,000 population by province",
    center={'lat': 42.734189, 'lon': 25.1635087},
    mapbox_style='carto-darkmatter',
    opacity=1,
    zoom=6
)
fig_yesterday_map_active.update_layout(margin={"r":0,"t":40,"l":0,"b":0}, template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')


### Age bands

logger.info('Reading age bands data')
covid_by_age_band = (pd.read_csv('./data/COVID_age_bands.csv', parse_dates=['Дата']).rename(columns={'Дата':'date'}))


logger.info('Creating chart 10: Cumulative cases by age band')
age_band_colors = ['green', 'cyan', 'magenta', 'ghostwhite', 'coral', 'royalblue', 'darkred', 'orange', 'brown']

fig_age = go.Figure()
i=0
for col in covid_by_age_band.columns[1:]:
    fig_age.add_trace(go.Scatter(x=covid_by_age_band['date'], y=covid_by_age_band[col], mode='lines',
                                 line=dict(width=3, color=age_band_colors[i]), name=col))
    i+=1
fig_age.update_layout(title='Total cases over time by age band', template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

fig_age.add_annotation(
    x=pd.Timestamp(2020,6,15), y=0, text="Borders reopening",
    showarrow=True, arrowhead=1,
    arrowcolor='ivory', font=dict(color='ivory'),
    ay=20)
fig_age.add_annotation(
    x=pd.Timestamp(2020,7,1), y=0, text="Football cup final",
    showarrow=True, arrowhead=1,
    arrowcolor='cornflowerblue', font=dict(color='cornflowerblue'),
    ay=40)
fig_age.add_annotation(
    x=pd.Timestamp(2020,7,11), y=0, text="Start of protests",
    showarrow=True, arrowhead=1,
    arrowcolor='cyan', font=dict(color='cyan'),
    ax=20, ay=20)
fig_age.add_annotation(
    x=pd.Timestamp(2020,9,2), y=0, text="First mass protest",
    showarrow=True, arrowhead=1,
    arrowcolor='lime', font=dict(color='lime'),
    ay=20)
fig_age.add_annotation(
    x=pd.Timestamp(2020,9,15), y=0, text="Schools opening",
    showarrow=True, arrowhead=1,
    arrowcolor='red', font=dict(color='red'),
    ax=10, ay=40)
fig_age.add_annotation(
    x=pd.Timestamp(2020,11,28), y=0, text="Second lockdown",
    showarrow=True, arrowhead=1,
    arrowcolor='yellow', font=dict(color='yellow'),
    ay=20)
fig_age.add_annotation(
    x=pd.Timestamp(2020,12,24), y=0, text="Antigen tests",
    showarrow=True, arrowhead=1,
    arrowcolor='orange', font=dict(color='orange'),
    ay=20)
fig_age.add_annotation(
    x=pd.Timestamp(2021,1,25), y=0, text="Vaccines",
    showarrow=True, arrowhead=1,
    arrowcolor='green', font=dict(color='green'),
    ay=20)
fig_age.add_annotation(
    x=pd.Timestamp(2021,2,4), y=0, text="Schools partial reopening",
    showarrow=True, arrowhead=1,
    arrowcolor='red', font=dict(color='red'),
    ax=-10, ay=40)


###### TESTS ######
logger.info('Test types')
tests = read_covid_tests('./data/COVID_test_type.csv', 'Дата')

logger.info('Creating chart 11: Daily tests by test type')

fig_tests_daily = go.Figure()
fig_tests_daily.add_trace(go.Scatter(x=tests.date, y=tests.new_pcr, name='Daily PCR tests', mode='lines', line_shape='spline', marker_color='coral'))
fig_tests_daily.add_trace(go.Scatter(x=tests.date, y=tests.new_antigen, name='Daily Antigen tests', mode='lines', line_shape='spline', marker_color='darkcyan'))
fig_tests_daily.update_layout(title='New COVID-19 tests per day by test type', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')


logger.info('Creating chart 12: Positive tests % by test type')

fig_tests_daily_positive = go.Figure()
fig_tests_daily_positive.add_trace(go.Scatter(x=tests.date, y=tests.pos_rate_pcr, name='Positive PCR %', mode='lines', line_shape='spline', marker_color='coral'))
fig_tests_daily_positive.add_trace(go.Scatter(x=tests.date, y=tests.pos_rate_antigen, name='Positive Antigen %', mode='lines', line_shape='spline', marker_color='darkcyan'))
fig_tests_daily_positive.update_layout(title='Positive tests % by test type', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(tickformat=',.0%', hoverformat=',.2%'))


logger.info('Creating chart 13: Tests Sankey')

tests_sankey_data_since_antigen = pd.DataFrame()
tests_sankey_data_since_antigen['total_tests'] = [tests['new_tests'].sum()]
tests_sankey_data_since_antigen['total_tests_pcr'] = [tests['new_pcr'].sum()]
tests_sankey_data_since_antigen['total_tests_antigen'] = [tests['new_antigen'].sum()]
tests_sankey_data_since_antigen['total_positive'] = [tests['new_positive'].sum()]
tests_sankey_data_since_antigen['total_positive_pcr'] = [tests['new_positive_pcr'].sum()]
tests_sankey_data_since_antigen['total_positive_antigen'] = [tests['new_positive_antigen'].sum()]
tests_sankey_data_since_antigen['new_positive_pcr'] = [tests['new_positive_pcr'].values[-1]]
tests_sankey_data_since_antigen['new_positive_antigen'] = [tests['new_positive_antigen'].values[-1]]

colors_list = [
        u'rgba(5, 116, 187, 0.8)',
        u'rgba(225, 127, 14, 0.8)',
        u'rgba(44, 160, 44, 0.8)',
        u'rgba(214, 39, 40, 0.8)',
        u'rgba(140, 86, 75, 0.8)',
        u'rgba(238, 43, 148, 0.8)',
        u'rgba(83, 128, 94, 0.8)',
        u'rgba(188, 189, 34, 0.8)',
        u'rgba(23, 190, 207, 0.8)',
        u'rgba(74, 64, 48, 0.8)',
        u'rgba(170, 139, 180, 0.8)'
]

fig_tests_sankey = go.Figure(
    data = [
        go.Sankey(
            valueformat = ",", # integer
            node = dict(
                pad = 15,
                thickness = 15,
                line = dict(color = 'black', width = 0.5),
                #label = sankey_dummy.groupby('col1').count().index.tolist(),
                label = ['Total tests', 'PCR tests', 'Antigen tests', 'Positive PCR', 'Positive Antigen', 'Positive for yesterday', 'Positive for yesterday'],
                x = [0.1, 0.4, 0.4, 0.55, 0.55, 0.9, 0.9],
                y = [0.5, 0.1, 0.6, 0.05, 0.3, 0.05, 0.3],
                color = colors_list[:7]
            ),
            link = dict(
                #source = sankey_dummy['src_ID'].tolist(),
                #target = sankey_dummy['tgt_ID'].tolist(),
                #value = sankey_dummy['count'].tolist(),
                #color = colors_list[:len(sankey_dummy)]
                source = [0, 0, 1, 2, 3, 4],
                target = [1, 2, 3, 4, 5, 6],
                value = [
                    tests_sankey_data_since_antigen['total_tests_pcr'].values[-1],
                    tests_sankey_data_since_antigen['total_tests_antigen'].values[-1],
                    tests_sankey_data_since_antigen['total_positive_pcr'].values[-1],
                    tests_sankey_data_since_antigen['total_positive_antigen'].values[-1],
                    tests_sankey_data_since_antigen['new_positive_pcr'].values[-1],
                    tests_sankey_data_since_antigen['new_positive_antigen'].values[-1]
                ],
                color = colors_list[-6:],
                hovertemplate = "from %{source.label} <br />" + "were %{target.label}"
            )
        )    
    ]    
).update_layout(title_text='Tests by type since 24th Dec 2020 - Sankey diagram', font_size=12, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')


logger.info('Creating chart 14: Hospitalized patients')

fig_hospitalized = make_subplots(specs=[[{"secondary_y": True}]])
fig_hospitalized.add_trace(go.Scatter(x=covid_general.date, y=covid_general.hospitalized, name='Hospitalized'), secondary_y=False)
fig_hospitalized.add_trace(go.Scatter(x=covid_general.date, y=covid_general.intensive_care, name='Intensive care units'), secondary_y=True)
fig_hospitalized.update_yaxes(title_text="Hospitalized patients", secondary_y=False)
fig_hospitalized.update_yaxes(title_text="Patients in intensive care units", secondary_y=True)
fig_hospitalized.update_layout(title="Currently hospitalized patients by date", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
####### REPRODUCTION NUMBER #######

logger.info('Starting Rt processing')

provinces = covid_pop_sorted[['province', 'date', 'ALL']].groupby(['province','date']).ALL.sum()

orig_bg = pd.read_csv('./dash_data/r0_bg_original.csv')
smoothed_bg = pd.read_csv('./dash_data/r0_bg_smoothed.csv')


logger.info('Creating chart 15: Smoothed new cases - BG')
fig_new_bg = go.Figure()
fig_new_bg.add_trace(go.Scatter(x=orig_bg.reset_index()['date'], y=orig_bg.reset_index()['ALL'],
                                      mode='lines', line=dict(dash='dot'), name='Actual'))
fig_new_bg.add_trace(go.Scatter(x=smoothed_bg.reset_index()['date'], y=smoothed_bg.reset_index()['ALL'],
                                      mode='lines', line=dict(width=3), name='Smoothed', marker_color='royalblue'))
fig_new_bg.update_layout(title='Daily new cases in Bulgaria', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')


logger.info('Creating chart 16: Smoothed recoveries - BG')
orig_recovered_bg = pd.read_csv('./dash_data/r0_bg_original_recovered.csv')
smoothed_recovered_bg = pd.read_csv('./dash_data/r0_bg_smoothed_recovered.csv')
fig_recovered_bg = go.Figure()
fig_recovered_bg.add_trace(go.Scatter(x=orig_recovered_bg.reset_index()['date'], y=orig_recovered_bg.reset_index()['total_recoveries'],
                                      mode='lines', line=dict(dash='dot'), name='Actual'))
fig_recovered_bg.add_trace(go.Scatter(x=smoothed_recovered_bg.reset_index()['date'], y=smoothed_recovered_bg.reset_index()['total_recoveries'],
                                      mode='lines', line=dict(width=3), name='Smoothed', marker_color='green'))
fig_recovered_bg.update_layout(title='Daily new recoveries in Bulgaria', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')


logger.info('Creating chart 17: Smoothed deaths - BG')
orig_deaths_bg = pd.read_csv('./dash_data/r0_bg_original_deaths.csv')
smoothed_deaths_bg = pd.read_csv('./dash_data/r0_bg_smoothed_deaths.csv')
fig_deaths_bg = go.Figure()
fig_deaths_bg.add_trace(go.Scatter(x=orig_deaths_bg.reset_index()['date'], y=orig_deaths_bg.reset_index()['total_deaths'],
                                      mode='lines', line=dict(dash='dot'), name='Actual'))
fig_deaths_bg.add_trace(go.Scatter(x=smoothed_deaths_bg.reset_index()['date'], y=smoothed_deaths_bg.reset_index()['total_deaths'],
                                      mode='lines', line=dict(width=3), name='Smoothed'))
fig_deaths_bg.update_layout(title='Daily new deaths in Bulgaria', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')


logger.info('Creating chart 18: Smoothed new cases - BG - age bands')
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
fig_age_diff.update_layout(title='New confirmed cases per day by age band (smoothed figures)', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')


logger.info("Creating chart 19: Smoothed new cases - BG - age bands per 100,000 pop")
pop_by_age_band = read_nsi_age_bands('./data/Pop_6.1.2_Pop_DR.xls', worksheet_name='2019', col_num=2, col_names=['age_band', 'pop'], skip=5, rows_needed=22)
covid_by_age_band_diff_smoothed_per100k = covid_by_age_band_diff_smoothed.copy()
for col in covid_by_age_band_diff_smoothed_per100k.columns:
    covid_by_age_band_diff_smoothed_per100k[col] = (100000*covid_by_age_band_diff_smoothed_per100k[col]/pop_by_age_band.loc[pop_by_age_band.covid_age_band == col, 'pop'].values).round(0)
    
fig_age_per100k = go.Figure()
i=0
for col in covid_by_age_band_diff_smoothed_per100k.columns:
    fig_age_per100k.add_trace(go.Scatter(x=covid_by_age_band_diff_smoothed_per100k.index, y=covid_by_age_band_diff_smoothed_per100k[col], mode='lines', line_shape='spline',
                                 line=dict(color=age_band_colors[i]), name=col))
    i+=1
fig_age_per100k.update_layout(title='New confirmed daily cases by age band per 100,000 population (smoothed figures)', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')


logger.info('Creating chart 20: Smoothed new cases - provinces')
provinces_list = covid_pop[['province', 'pop']].drop_duplicates().sort_values(by='pop', ascending=False).province.values

# create subplots structure
fig_new_by_province = make_subplots(
                            rows=int(len(provinces_list)/2),
                            cols=2,
                            subplot_titles = [f"{province}" for province in provinces_list]
                        )

original_provinces = pd.read_csv('./dash_data/r0_provinces_original.csv')
smoothed_provinces = pd.read_csv('./dash_data/r0_provinces_smoothed.csv')

# add charts for provinces
for i, province in list(enumerate(provinces_list)):
    original = original_provinces.loc[original_provinces.province == province]
    smoothed = smoothed_provinces.loc[smoothed_provinces.province == province]

    i += 1
    import math
    row_num = math.ceil(i/2)
    if i % 2 != 0:
        col_num = 1
    else:
        col_num = 2

    fig_new_by_province.add_trace(go.Scatter(x=original['date'], y=original['new_cases'],
                                      mode='lines', line=dict(dash='dot'), name='Actual'),
                                  row=row_num, col=col_num)
    fig_new_by_province.add_trace(go.Scatter(x=smoothed['date'], y=smoothed['new_cases'],
                                      mode='lines', line=dict(width=3), name='Smoothed'),
                                  row=row_num, col=col_num)

fig_new_by_province.update_layout(title='Daily new cases by province (smoothed figures)', height=3200, showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
fig_new_by_province.update_xaxes(range=[date.today() - timedelta(days=5*30), date.today()])


### Rt for BG

logger.info('Creating chart 21: Rt for BG')
result_bg = pd.read_csv('./dash_data/r0_bg_r0.csv')

index_bg = result_bg['date']
values_bg = result_bg['Estimated']

from scipy import stats as sps
from scipy.interpolate import interp1d
from matplotlib.dates import date2num, num2date

lowfn_bg = interp1d(date2num(index_bg),
                     result_bg['Low_90'].values,
                     bounds_error=False,
                     fill_value='extrapolate')
    
highfn_bg = interp1d(date2num(index_bg),
                      result_bg['High_90'].values,
                      bounds_error=False,
                      fill_value='extrapolate')

extended_bg = pd.date_range(start=index_bg.values[0],
                             end=index_bg.values[-1])

fig_rt = go.Figure()
fig_rt.add_trace(go.Scatter(x=index_bg, y=lowfn_bg(date2num(extended_bg)),
                            fill='none', mode='lines', line_color="rgba(38,38,38,0.9)", line_shape='spline',
                            name="Low density interval"))
fig_rt.add_trace(go.Scatter(x=index_bg, y=highfn_bg(date2num(extended_bg)),
                            fill='tonexty', mode='none', fillcolor="rgba(65,65,65,1)", line_shape='spline',
                            name="High density interval"))
fig_rt.add_trace(go.Scatter(x=index_bg, y=values_bg, mode='markers+lines', line=dict(width=0.3, dash='dot'), line_shape='spline',
                            marker_color=values_bg, marker_colorscale='RdYlGn_r', marker_line_width=1.2,
                            marker_cmin=0.5, marker_cmax=1.4, name='R<sub>t'))
fig_rt.update_layout(yaxis=dict(range=[0,4]), title="Real-time R<sub>t</sub> for Bulgaria", showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')


### Rt by province ###

logger.info('Reading Rt for provinces')
final_results = pd.read_csv('./dash_data/r0_provinces_r0.csv')

mr = final_results.loc[final_results['date'] == final_results['date'].max(), ['province', 'Estimated', 'High_90', 'Low_90']]

mr['diff_up'] = mr['High_90'] - mr['Estimated']
mr['diff_down'] = mr['Estimated'] - mr['Low_90']
# create status column
mr_conditions = [(mr.High_90 <= 1), (mr.Low_90 >= 1)]
mr_values = ['Likely under control', 'Likely not under control']
mr['status'] = np.select(mr_conditions, mr_values)
mr_colors = ['green', 'crimson']
mr['colors'] = np.select(mr_conditions, mr_colors)
mr.loc[mr.status=="0", 'colors'] = 'rgb(189,166,17)' #dark yellow
#append population and sort 
mr = mr.merge(covid_pop[['province', 'pop']].drop_duplicates(), on='province').sort_values(by='pop', ascending=False)


fig_rt_province_yesterday = go.Figure()
fig_rt_province_yesterday.add_trace(go.Bar(x=mr.province, y=mr.Estimated, marker_color=mr.colors, 
                          error_y=dict(type='data', array=mr.diff_up, arrayminus=mr.diff_down)))
fig_rt_province_yesterday.update_layout(title='R<sub>t</sub> by province for the last daily update', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')


logger.info('Creating chart 22: Rt by province')

def generate_rt_by_province(provinces, final_results):
    provinces_list = covid_pop[['province', 'pop']].drop_duplicates().sort_values(by='pop', ascending=False).province.values
    data = final_results#.reset_index()
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

        fig_rt_province.add_trace(go.Scatter(x=subset.date[3:], y=subset.Low_90[3:],
                                fill='none', mode='lines', line_color="rgba(38,38,38,0.9)", line_shape='spline',
                                name="Low density interval"), row=row_num, col=col_num)
        fig_rt_province.add_trace(go.Scatter(x=subset.date[3:], y=subset.High_90[3:],
                                fill='tonexty', mode='none', fillcolor="rgba(65,65,65,1)", line_shape='spline',
                                name="High density interval"), row=row_num, col=col_num)
        fig_rt_province.add_trace(go.Scatter(x=subset.date[3:], y=subset.Estimated[3:], mode='markers+lines',
                                line=dict(width=0.3, dash='dot'), line_shape='spline',
                                marker_color=subset.Estimated, marker_colorscale='RdYlGn_r', marker_line_width=1.2,
                                marker_cmin=0.5, marker_cmax=1.4, name='R<sub>t'), row=row_num, col=col_num)

    fig_rt_province.update_layout(title="Real-time R<sub>t</sub> by province", height=4000, showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig_rt_province.update_yaxes(range=[0, 2.5])
    return fig_rt_province


fig_rt_province_actual = generate_rt_by_province(provinces, final_results)


####### ARIMA #######

logger.info('Reading ARIMA')

ts_data = covid_pop.reset_index()[['date', 'province', 'ALL', 'new_cases']].rename(columns={'ALL':'total_cases'})

ts_data_r0 = final_results[['province', 'date', 'Estimated']].rename(columns={'Estimated':'r0'})
ts_data_r0['date'] = pd.to_datetime(ts_data_r0.date)
ts_data = pd.merge(ts_data, ts_data_r0, how='left', on=['date','province'])

def split(ts, forecast_days=15):
    #size = int(len(ts) * math.log(0.80))
    size=-forecast_days
    train= ts[:size]
    test = ts[size:]
    return(train,test)

def mape(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


arima_provinces_df = pd.read_csv('./dash_data/arima_provinces.csv')

def arima_chart(province):
    arima_filtered = arima_provinces_df.loc[arima_provinces_df.province == province]
    fig_arima = go.Figure()
    fig_arima.add_trace(go.Scatter(x=arima_filtered['date'][:-15], y=arima_filtered['value'][:-15], name='Historical data', mode='lines'))
    fig_arima.add_trace(go.Scatter(x=arima_filtered['date'][-15:], y=arima_filtered['value'][-15:], name='Validation data', mode='lines'))
    fig_arima.add_trace(go.Scatter(x=arima_filtered['date'][-15:], y=arima_filtered['pred'][-15:], name='Forecast', mode='lines'))
    fig_arima.update_layout(title = f'True vs Predicted values for total cases (7 days rolling mean) in {province} for 15 days', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    arima_error = arima_filtered['error'].values[0]
    return fig_arima, arima_error


######## EXPONENTIAL SMOOTHING ##########

df = result_bg.set_index('date')[['Estimated']].rename(columns={'Estimated':'data'})
df['z_data'] = (df['data'] - df['data'].rolling(window=12).mean()) / df['data'].rolling(window=12).std()
df['zp_data'] = df['z_data'] - df['z_data'].shift(12)

logger.info('Starting Double exponential smoothing (Holt)')

from statsmodels.tsa.holtwinters import Holt

df.index = pd.to_datetime(df.index)
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

logger.info('Creating chart 23: Double exp smoothing')
fig_exp_smoothing_double = go.Figure()
fig_exp_smoothing_double.add_trace(go.Scatter(x=df.index[30:], y=df.data[30:], name='Historical data'))

for p, f, c in zip((pred1_double, pred2_double, pred3_double),(fit1_double, fit2_double, fit3_double),('coral','yellow','cyan')):
    fig_exp_smoothing_double.add_trace(go.Scatter(x=train.index[30:], y=f.fittedvalues[30:], marker_color=c, mode='lines',
                            name=f"alpha={str(f.params['smoothing_level'])[:4]}, beta={str(f.params['smoothing_trend'])[:4]}")
    )
    fig_exp_smoothing_double.add_trace(go.Scatter(
        x=pd.date_range(start=test.index.min(), periods=len(test) + len(p)),
        y=p, marker_color=c, mode='lines', showlegend=False)
    )
    print(f"\nMean absolute percentage error: {mape(test['data'].values,p).round(2)} (alpha={str(f.params['smoothing_level'])[:4]}, beta={str(f.params['smoothing_trend'])[:4]})")

fig_exp_smoothing_double.update_layout(title="Holt's (double) exponential smoothing for R<sub>t</sub> in Bulgaria", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')


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
    fig_exp_smoothing_double.add_trace(go.Scatter(x=train.index, y=train[column], name='Historical data', mode='lines'))
    fig_exp_smoothing_double.add_trace(go.Scatter(x=test.index, y=test[column], name='Validation data', mode='lines', marker_color='coral'))

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


logger.info('Starting Triple exponential smoothing (Holt-Winters)')

from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES

#build and train the model on the training data
model_triple = HWES(train.data, seasonal_periods=7, trend='add', seasonal='mul')
fitted_triple = model_triple.fit(optimized=True, use_brute=True)

#print out the training summary
#print(fitted_triple.summary())

#create an out of sample forcast for the next steps beyond the final data point in the training data set
pred_triple = fitted_triple.forecast(steps=15)

#print(f"\nMean absolute percentage error: {mape(test['data'].values,pred_triple).round(2)}")

logger.info('Creating chart 24: Triple exp smoothing')
#plot the training data, the test data and the forecast on the same plot
fig_exp_smoothing_triple = go.Figure()
fig_exp_smoothing_triple.add_trace(go.Scatter(x=train.index[30:], y=train.data[30:], name='Historical data', mode='lines'))
fig_exp_smoothing_triple.add_trace(go.Scatter(x=train.index[30:], y=fitted_triple.fittedvalues[30:], name='Model fit', mode='lines', marker_color='lime'))
fig_exp_smoothing_triple.add_trace(go.Scatter(x=test.index, y=test.data, name='Validation data', mode='lines', marker_color='coral'))
fig_exp_smoothing_triple.add_trace(go.Scatter(
    x=pd.date_range(start=test.index.min(), periods=len(test) + len(pred_triple)),
    y=pred_triple, name='Forecast', marker_color='gold', mode='lines')
)
fig_exp_smoothing_triple.update_layout(title="Holt-Winters' (triple) exponential smoothing for R<sub>t</sub> in Bulgaria", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')


def triple_exp_smoothing(ts_data, province, column='total_cases', forecast_days=15):
    df = ts_data.set_index('date')
    # replace zeros with 0.1 as the multiplicative seasonal element o HWES requires strictly positive values
    df = df.loc[((df['province'] == province) & (df[column].notnull()))].replace(0,0.1)
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
    fig_exp_smoothing_triple.add_trace(go.Scatter(x=train.index[30:], y=train[column][30:], name='Historical data', mode='lines'))
    fig_exp_smoothing_triple.add_trace(go.Scatter(x=train.index[30:], y=fitted_triple.fittedvalues[30:], name='Model fit', mode='lines', marker_color='lime'))
    fig_exp_smoothing_triple.add_trace(go.Scatter(x=test.index, y=test[column], name='Validation data', mode='lines', marker_color='coral'))
    fig_exp_smoothing_triple.add_trace(go.Scatter(
        x=pd.date_range(start=test.index.min(), periods=len(test) + len(pred_triple)),
        y=pred_triple, name='Forecast', marker_color='gold', mode='lines')
    )
    fig_exp_smoothing_triple.update_layout(title=f'Holt-Winters (triple) exponential smoothing for {"new cases" if column == "new_cases" else "total cases" if column == "total_cases" else "reproduction number"} in {province} for {forecast_days} days')
    return fig_exp_smoothing_triple, pred_triple_error


logger.info('Creating dash structure')

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc


app = dash.Dash(
    name='COVID-19 in Bulgaria',
    #external_stylesheets=[dbc.themes.DARKLY],
    external_stylesheets=['./assets/bootstrap_adjusted.css'],
    meta_tags = [
        {
            "name": "description",
            "content": "Live coronavirus statistics for Bulgaria on both national and province level. Daily updated visualizations showing confirmed COVID-19 cases, recovered, death and currently active cases, COVID-19 spread by province, by age band, real-time reproduction number by province, recovery and death rates (potentially caused by SARS-CoV-2), hospitalized patients, patients in intensive care units. Also allows the user to create predictions for the future cases on a province level, using ARIMA model or triple exponential smoothing."
        },
        {
            "name": "desc",
            "content": "Актуална статистика за случаите на коронавирус в България - на национално и областно ниво. Визуализациите се обновяват ежедневно и включват нови случаи, излекувани, хоспитализирани, смъртни случаи."
        }
        #{
        #    "name": "viewport", "content": "width=device-width, initial-scale=1.0"
        #}
    ]
)
app.title = 'COVID-19 in Bulgaria'

# real domain G-XW4L84LCB7
# heroku domain G-2FCPJC5BDW

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        <!-- Global site tag (gtag.js) - Google Analytics -->
        <script async src="https://www.googletagmanager.com/gtag/js?id=G-XW4L84LCB7"></script>
        <script>
          window.dataLayer = window.dataLayer || [];
          function gtag(){dataLayer.push(arguments);}
          gtag('js', new Date());

          gtag('config', 'G-XW4L84LCB7');
        </script>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''


logger.info('Creating dash helper vars')
# daily figures for the cards
new_t = covid_general.new_cases.tail(1).values[0]
new_t_minus_1 = covid_general.new_cases.tail(2).head(1).values[0]
deaths_t = covid_general.new_deaths.tail(1).values[0]
deaths_t_minus_1 = covid_general.new_deaths.tail(2).head(1).values[0]
recoveries_t = covid_general.new_recoveries.tail(1).values[0]
recoveries_t_minus_1 = covid_general.new_recoveries.tail(2).head(1).values[0]
active_t = covid_general.active_cases.tail(1).values[0]
active_t_minus_1 = covid_general.active_cases.tail(2).head(1).values[0]
active_t_minus_2 = covid_general.active_cases.tail(3).head(1).values[0]
active_change_t = covid_general.active_cases.tail(1).values[0] - covid_general.active_cases.tail(2).head(1).values[0]
active_change_t_minus_1 = covid_general.active_cases.tail(2).head(1).values[0] - covid_general.active_cases.tail(3).head(1).values[0]
active_change_t_perc_dod = abs((active_change_t - active_change_t_minus_1)/(active_change_t_minus_1+0.1)) * (-1 if active_change_t < active_change_t_minus_1 else 1)
hospitalized_t = covid_general.hospitalized.tail(1).values[0]
hospitalized_t_minus_1 = covid_general.hospitalized.tail(2).head(1).values[0]
hospitalized_t_minus_2 = covid_general.hospitalized.tail(3).head(1).values[0]
hospitalized_change_t = covid_general.hospitalized.tail(1).values[0] - covid_general.hospitalized.tail(2).head(1).values[0]
hospitalized_change_t_minus_1 = covid_general.hospitalized.tail(2).head(1).values[0] - covid_general.hospitalized.tail(3).head(1).values[0]
hospitalized_change_t_perc_dod = abs((hospitalized_change_t - hospitalized_change_t_minus_1)/(hospitalized_change_t_minus_1+0.1)) * (-1 if hospitalized_change_t < hospitalized_change_t_minus_1 else 1)
intensive_care_t = covid_general.intensive_care.tail(1).values[0]
intensive_care_t_minus_1 = covid_general.intensive_care.tail(2).head(1).values[0]
intensive_care_t_minus_2 = covid_general.intensive_care.tail(3).head(1).values[0]
intensive_care_change_t = covid_general.intensive_care.tail(1).values[0] - covid_general.intensive_care.tail(2).head(1).values[0]
intensive_care_change_t_minus_1 = covid_general.intensive_care.tail(2).head(1).values[0] - covid_general.intensive_care.tail(3).head(1).values[0]
intensive_care_change_t_perc_dod = abs((intensive_care_change_t - intensive_care_change_t_minus_1)/(intensive_care_change_t_minus_1+0.1)) * (-1 if intensive_care_change_t < intensive_care_change_t_minus_1 else 1)


logger.info('Creating dash cards')
cards = dbc.CardDeck(className="carddeck", children=[
            dbc.Card([
                dbc.CardBody(
                    [
                        html.H4("Confirmed", className="card-title"),
                        html.P(
                            f"{covid_general.total_cases.tail(1).values[0]:,}",
                            className="card-value",
                        ),
                        html.P(
                            f"Today: {covid_general.new_cases.tail(1).values[0]:,}",
                            className="card-target",
                        ),
                        html.Span(
                            f"{(covid_general.new_cases.tail(1).values[0] - covid_general.new_cases.tail(2).head(1).values[0])/covid_general.new_cases.tail(2).head(1).values[0].round(2):.0%} day-over-day",
                            className="card-diff-up" if covid_general.new_cases.tail(1).values[0] < covid_general.new_cases.tail(2).head(1).values[0] else "card-diff-down",
                        )
                    ]
                ),
            ]),
            dbc.Card([
                dbc.CardBody(
                    [
                        html.H4("Deaths", className="card-title"),
                        html.P(
                            f"{covid_general.total_deaths.tail(1).values[0]:,}",
                            className="card-value",
                        ),
                        html.P(
                            f"Today: {covid_general.new_deaths.tail(1).values[0]:,}",
                            className="card-target",
                        ),
                        html.Span(
                            f"{(covid_general.new_deaths.tail(1).values[0] - covid_general.new_deaths.tail(2).head(1).values[0])/covid_general.new_deaths.tail(2).head(1).values[0].round(2):.0%} day-over-day",
                            className="card-diff-up" if covid_general.new_deaths.tail(1).values[0] < covid_general.new_deaths.tail(2).head(1).values[0] else "card-diff-down",
                        )
                    ]
                ),
            ]),
            dbc.Card([
                dbc.CardBody(
                    [
                        html.H4("Recoveries", className="card-title"),
                        html.P(
                            f"{covid_general.total_recoveries.tail(1).values[0]:,}",
                            className="card-value",
                        ),
                        html.P(
                            f"Today: {covid_general.new_recoveries.tail(1).values[0]:,}",
                            className="card-target",
                        ),
                        html.Span(
                            f"{(covid_general.new_recoveries.tail(1).values[0] - covid_general.new_recoveries.tail(2).head(1).values[0])/covid_general.new_recoveries.tail(2).head(1).values[0].round(2):.0%} day-over-day",
                            className="card-diff-up" if covid_general.new_recoveries.tail(1).values[0] > covid_general.new_recoveries.tail(2).head(1).values[0] else "card-diff-down",
                        )
                    ]
                ),
            ]),
            dbc.Card([
                dbc.CardBody(
                    [
                        html.H4("Active cases", className="card-title"),
                        html.P(
                            f"{active_t:,}",
                            className="card-value",
                        ),
                        html.P(
                            f"Today: {active_change_t:,}",
                            className="card-target",
                        ),
                        html.Span(
                            f"{active_change_t_perc_dod:.0%} day-over-day",
                            className="card-diff-up" if active_change_t_perc_dod < 0 else "card-diff-down",
                        )
                    ]
                ),
            ]),
            dbc.Card([
                dbc.CardBody(
                    [
                        html.H4("Hospitalized", className="card-title"),
                        html.P(
                            f"{hospitalized_t:,}",
                            className="card-value",
                        ),
                        html.P(
                            f"Today: {hospitalized_change_t:,}",
                            className="card-target",
                        ),
                        html.Span(
                            f"{hospitalized_change_t_perc_dod:.0%} day-over-day",
                            className="card-diff-up" if hospitalized_change_t_perc_dod < 0 else "card-diff-down",
                        )
                    ]
                ),
            ]),
            dbc.Card([
                dbc.CardBody(
                    [
                        html.H4("Intensive care", className="card-title"),
                        html.P(
                            f"{intensive_care_t:,}",
                            className="card-value",
                        ),
                        html.P(
                            f"Today: {intensive_care_change_t:,}",
                            className="card-target",
                        ),
                        html.Span(
                            f"{intensive_care_change_t_perc_dod:.0%} day-over-day",
                            className="card-diff-up" if intensive_care_change_t_perc_dod < 0 else "card-diff-down",
                        )
                    ]
                ),
            ])
        ])

daily_summary = html.Div(
    [
    dcc.Markdown(f"The number of **new confirmed cases** from the last daily update is **{new_t:,}** and the **recovered cases** are **{recoveries_t:,}**. Yesterday **{deaths_t:,}** infected people have lost their lives, which adds up to {covid_general.total_deaths.tail(1).values[0]:,} total death cases since the start of the pandemic. The total number of people with **currently active infections has {'increased' if new_t > recoveries_t else 'decreased'} with {abs(new_t - recoveries_t - deaths_t):,}** and is now {active_t:,}, which is {active_t / pop_by_province['pop'].sum():.2%} of the total population in Bulgaria, or {int((100000*active_t / pop_by_province['pop'].sum()).round(0)):,} infected people per 100,000 population."),
    dcc.Markdown(f"The number of **hospitalized patients has {'increased' if hospitalized_change_t > 0 else 'decreased'} with {abs(hospitalized_change_t):,}** and now stands at {hospitalized_t:,}, which is {hospitalized_t / active_t:.2%} of the currently active cases. The number of **patients in intensive care units has {'increased' if intensive_care_change_t > 0 else 'decreased'} with {abs(intensive_care_change_t):,} and is now {intensive_care_t:,}**, which is {intensive_care_t / hospitalized_t:.2%} of the currently hospitalized patients, or {intensive_care_t / active_t:.2%} of the total active cases."),
    dcc.Markdown(f"The provinces with the highest number of new cases are {covid_yesterday.sort_values(by='new cases',ascending=False).head(3)['province'].values[0]} ({int(covid_yesterday.sort_values(by='new cases',ascending=False).head(3)['new cases'].values[0]):,}), {covid_yesterday.sort_values(by='new cases',ascending=False).head(3)['province'].values[1]} ({int(covid_yesterday.sort_values(by='new cases',ascending=False).head(3)['new cases'].values[1]):,}) and {covid_yesterday.sort_values(by='new cases',ascending=False).head(3)['province'].values[2]} ({int(covid_yesterday.sort_values(by='new cases',ascending=False).head(3)['new cases'].values[2]):,}). In terms of new daily cases per 100,000 population, the leading provinces for the last day are **{covid_yesterday.sort_values(by='new_per_100k',ascending=False).head(3)['province'].values[0]} ({int(covid_yesterday.sort_values(by='new_per_100k',ascending=False).head(3)['new_per_100k'].values[0]):,}), {covid_yesterday.sort_values(by='new_per_100k',ascending=False).head(3)['province'].values[1]} ({int(covid_yesterday.sort_values(by='new_per_100k',ascending=False).head(3)['new_per_100k'].values[1]):,})** and **{covid_yesterday.sort_values(by='new_per_100k',ascending=False).head(3)['province'].values[2]} ({int(covid_yesterday.sort_values(by='new_per_100k',ascending=False).head(3)['new_per_100k'].values[2]):,})**. Still, the top 3 provinces that need special attention with highest number of active infections per 100,000 population are **{covid_yesterday.sort_values(by='active_per_100k',ascending=False).head(3)['province'].values[0]} ({int(covid_yesterday.sort_values(by='active_per_100k',ascending=False).head(3)['active_per_100k'].values[0]):,}), {covid_yesterday.sort_values(by='active_per_100k',ascending=False).head(3)['province'].values[1]} ({int(covid_yesterday.sort_values(by='active_per_100k',ascending=False).head(3)['active_per_100k'].values[1]):,})** and **{covid_yesterday.sort_values(by='active_per_100k',ascending=False).head(3)['province'].values[2]} ({int(covid_yesterday.sort_values(by='active_per_100k',ascending=False).head(3)['active_per_100k'].values[2]):,})**."),
    dcc.Markdown(f"The **reproduction number** from the last day on a national level is **{round(values_bg.values[-1],2)}**, whereas the average for the past 7 days it is **{round(values_bg.values[-7:].mean(),2)}**, which means that the overall spread of the disease is **{'decreasing decisively' if values_bg.values[-7:].mean() < 0.5 else 'well under control' if values_bg.values[-7:].mean() < 0.75 else 'under control' if values_bg.values[-7:].mean() < 0.95 else 'continuing to grow almost linearly' if values_bg.values[-7:].mean() < 1.05 else 'slightly not under control' if values_bg.values[-7:].mean() < 1.25 else 'continuing to increase and is not under control' if values_bg.values[-7:].mean() < 1.5 else 'growing up at a very fast pace'}**, i.e. 100 infectious people are directly spreading the disease to {int(round(100*values_bg.values[-7:].mean(),0))} other people."),
    dcc.Markdown(f"The provinces with **the highest reproduction number** as of yesterday are **{mr.sort_values(by='Estimated').province.values[-1]} ({mr.sort_values(by='Estimated').Estimated.values[-1]:.2f}), {mr.sort_values(by='Estimated').province.values[-2]} ({mr.sort_values(by='Estimated').Estimated.values[-2]:.2f})** and **{mr.sort_values(by='Estimated').province.values[-3]} ({mr.sort_values(by='Estimated').Estimated.values[-3]:.2f})**. On the other hand, the provinces where the disease is spreading at the slowest pace are **{mr.sort_values(by='Estimated').province.values[0]} ({mr.sort_values(by='Estimated').Estimated.values[0]:.2f}), {mr.sort_values(by='Estimated').province.values[1]} ({mr.sort_values(by='Estimated').Estimated.values[1]:.2f})** and **{mr.sort_values(by='Estimated').province.values[2]} ({mr.sort_values(by='Estimated').Estimated.values[2]:.2f})**.")
    ]
)


logger.info('Creating dash tabs')
tabs = html.Div([
    dcc.Tabs(
        id = 'tabs-css',
        parent_className = 'custom-tabs',
        className = 'custom-tabs-container',
        children = [
            dcc.Tab(
                label = "General",
                className = "custom-tab",
                selected_className = "custom-tab--selected",
                children = [
                    html.Br(),
                    html.Br(),
                    html.H4("Daily summary"),
                    html.Br(),
                    daily_summary,
                    html.Br(),
                    html.Br(),
                    html.H4("Cases over time (cumulative figures)"),
                    html.Br(),
                    dcc.Graph(figure=fig_gen_stats),
                    dcc.Graph(figure=fig_hospitalized),
                    html.Br(),
                    html.H4("Smoothed figures on a daily basis"),
                    html.Br(),
                    dcc.Graph(figure=fig_new_bg),
                    dcc.Graph(figure=fig_recovered_bg),
                    dcc.Graph(figure=fig_deaths_bg),
                    html.Br(),
                    html.H4("Smoothed number of cases by age bands"),
                    html.Br(),
                    dcc.Graph(figure=fig_age_diff),
                    html.P("The figure for the age bands above is no respecter of the proportion of each age band from the total population. Therefore, we can calculate the new cases by each age band per 100,000 population of that age band, which can be a better indicatior for the infectivity rate across different age bands. It clearly shows the higher infection risk for middle-aged and older people, compared to the younger generation:"),
                    dcc.Graph(figure=fig_age_per100k),
                    html.Br(),
                    html.H4("Cases on a weekly basis"),
                    html.Br(),
                    dcc.Graph(figure=fig_gen_stats_weekly),
                    html.P("The weekly cases distribution chart above shows that the number of new confirmed cases per week was relatively stable during the summer (week 25 - week 40), but drastically started to increase in the early October (week 40). The number of new cases per week reached its peak about mid November (week 47), but then started to decrease. About that time the number of new recoveries have also started to increase greatly and although they lost some momentum, the recoveries surpassed the new cases per week before mid December (week 50). On the bad side, between mid October and early December (weeks 42-49), the number of new death cases were breaking the all-time record for 8 consecutive weeks, but that tendency ended in mid-December (week 52)."),
                    html.Br(),
                    html.H4("COVID-19 tests - PCR vs Antigen"),
                    html.Br(),
                    html.P("Since 24th December 2020, the antigen tests are also included in the official statistics."),
                    dcc.Graph(figure=fig_tests_daily),
                    html.Br(),
                    dcc.Graph(figure=fig_tests_sankey)
                ]
            ),
            dcc.Tab(
                label = "By province",
                className = "custom-tab",
                selected_className = "custom-tab--selected",
                children = [
                    html.Br(),
                    html.Br(),
                    html.P(f"The provinces with the highest number of new cases are {covid_yesterday.sort_values(by='new cases',ascending=False).head(3)['province'].values[0]} ({int(covid_yesterday.sort_values(by='new cases',ascending=False).head(3)['new cases'].values[0]):,}), {covid_yesterday.sort_values(by='new cases',ascending=False).head(3)['province'].values[1]} ({int(covid_yesterday.sort_values(by='new cases',ascending=False).head(3)['new cases'].values[1]):,}) and {covid_yesterday.sort_values(by='new cases',ascending=False).head(3)['province'].values[2]} ({int(covid_yesterday.sort_values(by='new cases',ascending=False).head(3)['new cases'].values[2]):,}). In terms of new daily cases per 100,000 population, the leading provinces for the last day are {covid_yesterday.sort_values(by='new_per_100k',ascending=False).head(3)['province'].values[0]} ({int(covid_yesterday.sort_values(by='new_per_100k',ascending=False).head(3)['new_per_100k'].values[0]):,}), {covid_yesterday.sort_values(by='new_per_100k',ascending=False).head(3)['province'].values[1]} ({int(covid_yesterday.sort_values(by='new_per_100k',ascending=False).head(3)['new_per_100k'].values[1]):,}) and {covid_yesterday.sort_values(by='new_per_100k',ascending=False).head(3)['province'].values[2]} ({int(covid_yesterday.sort_values(by='new_per_100k',ascending=False).head(3)['new_per_100k'].values[2]):,})."),
                    html.Br(),
                    dcc.Graph(figure=fig_yesterday_map_new),
                    html.Br(),
                    html.Br(),
                    html.P(f"Still, the top 3 provinces that need special attention with highest number of active infections per 100,000 population are {covid_yesterday.sort_values(by='active_per_100k',ascending=False).head(3)['province'].values[0]} ({int(covid_yesterday.sort_values(by='active_per_100k',ascending=False).head(3)['active_per_100k'].values[0]):,}), {covid_yesterday.sort_values(by='active_per_100k',ascending=False).head(3)['province'].values[1]} ({int(covid_yesterday.sort_values(by='active_per_100k',ascending=False).head(3)['active_per_100k'].values[1]):,}) and {covid_yesterday.sort_values(by='active_per_100k',ascending=False).head(3)['province'].values[2]} ({int(covid_yesterday.sort_values(by='active_per_100k',ascending=False).head(3)['active_per_100k'].values[2]):,})."),
                    html.Br(),
                    dcc.Graph(figure=fig_yesterday_map_active),
                    html.Br(),
                    html.Br(),
                    html.P("Below the provinces are color-coded according to the number of total confirmed cases per 100,000 population. This map isn't as important as the other two above for the spread of the disease, because part of the historical confirmed cases are already 'closed'."),
                    dcc.Graph(figure=fig_yesterday_map_total),
                    html.Br(),
                    html.Br(),
                    dcc.Graph(figure=fig_new_by_province)
                ]
            ),
            dcc.Tab(
                label = "Rates",
                className = "custom-tab",
                selected_className = "custom-tab--selected",
                children = [
                    html.Br(),
                    html.Br(),
                    html.H4("Mortality and Recovery rate"),
                    html.Br(),
                    html.H6("The figure below represents the mortality rate and recovery rate over time. Recovery rate is turned off by default for better visibility."),
                    html.H6("The mortality and recovery rates are calculated as the total number of deaths/recoveries for the respective date divided by the total number of confirmed cases 14 days before that date. This is needed in order to take into account the long incubation period and the time needed for the infection end with a fatal result, or for the immune system to overcome the disease."),
                    html.Br(),
                    dcc.Graph(figure=fig_rates_mort_rec),
                    html.Br(),
                    html.Br(),
                    html.H6("Another approach for mortality/recovery rates is to divide them by the number of 'closed' cases (the sum of total deaths and recoveries)."),
                    html.Br(),
                    dcc.Graph(figure=fig_rates_mort_rec_v2),
                    html.Br(),
                    html.Br(),
                    html.H4("Hospitalized rate and intensive care unit rate"),
                    html.Br(),
                    html.H6("Below are the hospitalized and intensive care rates. Hospitalized rate is the number of patients with confirmed COVID-19 currently in hospitals, as a percentage of the total currently active cases. The intensive care rate, on the other hand, is the percentage of patients currently in intensive care units as part of the total people with COVID-19 in hospitals."),
                    html.Br(),
                    dcc.Graph(figure=fig_rates_hospitalized),
                    html.Br(),
                    html.Br(),
                    html.H4("Positive tests rate"),
                    html.Br(),
                    html.H6("The chart below shows the percentage of positive PCR tests over time as part of the total PCR tests."),
                    html.Br(),
                    dcc.Graph(figure=fig_rates_positive_tests),
                    html.Br(),
                    html.H6("Since 24th December 2020, the antigen tests are also included in the official statistics."),
                    html.Br(),
                    dcc.Graph(figure=fig_tests_daily_positive),
                    html.Br()
                ]
            ),
            dcc.Tab(
                label = "Reproduction",
                className = "custom-tab",
                selected_className = "custom-tab--selected",
                children = [
                    html.Br(),
                    html.Br(),
                    html.P("The estimated daily reproduction number represents how many people are directly infected by 1 infectious person per each day. Ideally, we want this number to be lower than 1. Otherwise, the disease is spreading linearly (=1) or exponentially (>1)."),
                    html.Div([
                    	html.Small("The calculation of Rt is done using the algorithm proposed by Kevin Systrom, "),
                    	html.Small(html.A('which can be found here.', href='https://github.com/k-sys/covid-19/blob/master/Realtime%20R0.ipynb', target='_blank'))
                    ], style={'font-size':'19px'}),
                    html.Br(),
                    dcc.Markdown(f"The reproduction number from the last day on a national level is **{round(values_bg.values[-1],2)}**, whereas the average for the past 7 days it is **{round(values_bg.values[-7:].mean(),2)}**, which means that the overall spread of the disease is **{'decreasing decisively' if values_bg.values[-7:].mean() < 0.5 else 'well under control' if values_bg.values[-7:].mean() < 0.75 else 'under control' if values_bg.values[-7:].mean() < 0.95 else 'continuing to grow almost linearly' if values_bg.values[-7:].mean() < 1.05 else 'slightly not under control' if values_bg.values[-7:].mean() < 1.25 else 'continuing to increase and is not under control' if values_bg.values[-7:].mean() < 1.5 else 'growing up at a very fast pace'}**, i.e. 100 infectious people are directly spreading the disease to {int(round(100*values_bg.values[-7:].mean(),0))} other people."),
                    html.Br(),
                    dcc.Graph(figure=fig_rt),
                    html.Br(),
                    html.Br(),
                    dcc.Markdown(f"The provinces with the highest reproduction number as of yesterday are **{mr.sort_values(by='Estimated').province.values[-1]} ({mr.sort_values(by='Estimated').Estimated.values[-1]:.2f}), {mr.sort_values(by='Estimated').province.values[-2]} ({mr.sort_values(by='Estimated').Estimated.values[-2]:.2f}) and {mr.sort_values(by='Estimated').province.values[-3]} ({mr.sort_values(by='Estimated').Estimated.values[-3]:.2f})**. On the other hand, the provinces where the disease is spreading at the slowest pace are **{mr.sort_values(by='Estimated').province.values[0]} ({mr.sort_values(by='Estimated').Estimated.values[0]:.2f}), {mr.sort_values(by='Estimated').province.values[1]} ({mr.sort_values(by='Estimated').Estimated.values[1]:.2f}) and {mr.sort_values(by='Estimated').province.values[2]} ({mr.sort_values(by='Estimated').Estimated.values[2]:.2f})**."),
                    dcc.Graph(figure=fig_rt_province_yesterday),
                    html.Br(),
                    html.Br(),
                    html.P('Below is a chart showing the daily reproduction number by province. Note that each chart is covering a different period of time, becase for each province we pick the start date as the moment when we have started to constantly see 10+ daily confirmed cases.'),
                    dcc.Graph(figure=fig_rt_province_actual)
                ]
            ),
            dcc.Tab(
                label = "Predictive models",
                className = "custom-tab",
                selected_className = "custom-tab--selected",
                children = [
                    html.Br(),
                    html.Br(),
                    html.H3('ARIMA'),
                    html.Br(),
                    html.P("The ARIMA model was used to predict the 7-days rolling mean number of the total confirmed cases."),
                    html.Br(),
                    dcc.Markdown("You can use the dropdown below to select your desired **province** and create live predictions using ARIMA model."),
                    html.Div([
                        html.Div(className='dropdown', children=[
                            dcc.Dropdown(
                                id='province-dropdown-arima',
                                className='dropdown',
                                options=[{'label':p, 'value':p} for p in ts_data['province'].unique()],
                                placeholder='Select a province',
                                value='Sofiya-Grad',
                                style=dict(width='70%')
                            )
                        ]),
                        html.Div([
                            dcc.Loading(
                                id='custom-arima-output-loader', type='default',
                                children=[
                                    html.Div(dcc.Graph(id="custom-arima-output"))
                                ]
                            ),
                            html.H6(id="custom-arima-output-error"),
                            html.P(id="custom-arima-output-summary")
                        ])
                    ])
        #        ]
        #    ),
        #    dcc.Tab(
        #        label = "Exponential smoothing",
        #        className = "custom-tab",
        #        selected_className = "custom-tab--selected",
        #        children = [
                    ,html.Br(), html.Br(), html.H3('Exponential smoothing'),
        #           html.Br(),
                    html.Br(),
                    html.P("The exponential smoothing models can be used to predict the value of the daily total cases, new cases or the reproduction number Rt by province."),
                    html.Br(),
                    dcc.Markdown("You can use the options below to select your desired **province**, **variable** and **forecast period** and create live predictions using triple exponential smoothing model."),
                    html.P("Note that the reproduction number might not be a good choice for some provinces, for which we don't have long enough history with 10+ confirmed cases per day."),
                    html.Div([
                        html.Div(className='dropdown', children=[
                            dcc.Dropdown(
                                id='province-dropdown',
                                className='dropdown',
                                options=[{'label':p, 'value':p} for p in ts_data['province'].unique()],
                                placeholder='Select a province',
                                value='Sofiya-Grad',
                                style=dict(width='140%')
                            ),
                            dcc.Dropdown(
                                id='variable-dropdown',
                                className='dropdown',
                                options=[
                                    {'label': 'Total cases', 'value': 'total_cases'},
                                    {'label': 'New cases', 'value': 'new_cases'},
                                    {'label': 'Reproduction number', 'value': 'r0'}
                                ],
                                placeholder='Select a variable to be predicted',
                                value='total_cases',
                                style=dict(width='160%')
                            ),
                            dcc.Input(
                                id='forecast-length-input',
                                className='dropdown',
                                type='number',
                                placeholder='Forecast period (days)',
                                style=dict(width='20%'),
                                value=15, min=1, max=100, step=1,
                                debounce=True # press Enter to send the input
                            )
                        ]),
                        html.Div([
                            dcc.Loading(
                                id='custom-triple-output-loader', type='default',
                                children=[
                                    html.Div(dcc.Graph(id="custom-triple-output"))
                                ]
                            ),
                            html.H6(id="custom-triple-output-error")
                        ])
                    ]),
                    html.Br(),
                    html.Br(),
                    html.P("Below are some generic examples with the results of double and triple exponential smoothing models predicting the reproduction number Rt at a national level."),
                    html.Br(),
                    html.Br(),
                    html.H4("Double exponential smoothing"),
                    html.Br(),
                    dcc.Graph(figure=fig_exp_smoothing_double),
                    html.P(f"\nMean absolute percentage error: {mape(test['data'].values,pred1_double).round(2)} (alpha={str(fit1_double.params['smoothing_level'])[:4]}, beta={str(fit1_double.params['smoothing_trend'])[:4]})"),
                    html.P(f"\nMean absolute percentage error: {mape(test['data'].values,pred2_double).round(2)} (alpha={str(fit2_double.params['smoothing_level'])[:4]}, beta={str(fit2_double.params['smoothing_trend'])[:4]})"),
                    html.P(f"\nMean absolute percentage error: {mape(test['data'].values,pred3_double).round(2)} (alpha={str(fit3_double.params['smoothing_level'])[:4]}, beta={str(fit3_double.params['smoothing_trend'])[:4]})"),
                    html.Br(),
                    html.Br(),
                    html.H4("Triple exponential smoothing"),
                    html.Br(),
                    dcc.Graph(figure=fig_exp_smoothing_triple),
                    html.P(f"\nMean absolute percentage error: {mape(test['data'].values,pred_triple).round(2)}"),
                    html.Br()
                ]
            ),
            dcc.Tab(
                label = "Events",
                className = "custom-tab",
                selected_className = "custom-tab--selected",
                children = [
                    html.Br(),
                    html.Br(),
                    html.H4("Did the crowded events  during the summer had any effect on the number of new cases?"),
                    html.Br(),
                    html.P("Five major dates were selected: the end of the first lockdown, when the borders with Greece were re-opened (15th June); the football cup final with 20,000 spectators on the stands (1st July); the beginning of the anti-government protests (11th July); the first mass anti-government protest (2nd September); the school opening (15th September). The second lockdown date was also added (28th November 2020), as well as the date since the Antigen tests started to get into the official statistics (24th December 2020)."),
                    html.Br(),
                    html.P("The first chart shows that despite the high activity during the summer (summer holidays, mass events, daily protests), there was very little or no immediate effect on the number of new confirmed cases."),
                    html.P("It is expected to have a few (2-4) weeks lag between the event and the result, but still the only major increase of the confirmed cases began almost a month after the school opening, and there was no other change that can be associated with the other mass events."),
                    html.Br(),
                    dcc.Graph(figure=fig_age),
                    html.Br(),
                    html.Br(),
                    html.P("The second chart shows week-over-week percentage change of new confirmed cases. Again, there is no evidence about immediate increase of the new cases after mass outdoor events."),
                    html.Br(),
                    dcc.Graph(figure=fig_gen_stats_weekly_new_pct),
                    html.Br(),
                    html.P("The fact that the outdoor events during the summer didn't have any significant effect on the number of new cases per week can also be seen on the chart below:"),
                    html.Br(),
                    dcc.Graph(figure=fig_gen_stats_weekly_events),
                    html.Br(),
                    html.P("Above we can see some significant increase of the new cases in the beginning of the period, after opening the borders with Greece. However, the reason for the increase in the next few weeks could probably be due to the requirement for a negative PCR test result for the people who were travelling abroad - then, many people had to undergo test because they were asked to, part of them have received a positive result, but otherwise they wouldn't have tested themselves, and therefore they would have been left out of the statistics. The conclusion here is that in the beginning of the period, the growth of the new cases might have been caused by the bureaucratic requirement for a negative PCR test."),
                    html.P("There isn't any significant change in the new cases after the other public events, except the schools opening. It might have caused persistent spread of the disease over the students, and therefore to their parents and relatives. The schools opening is not a one-time event and, unlike the other events, it is carried out indoors, which could potentially increase the infectivity. Also, the younger people might often be asymptomatic, which can make them unaware that they might be spreading the virus. Figure 2 also shows that in the second half of November, when many schools turned to a home-based education, the spread of the disease has lost its momentum. Therefore, the schools opening is a potential root cause for the increased number of cases, that could be investigated further."),
                    html.P("Another potential reason that could have played part in the increased cases in October is the change in the weather conditions. However, it will be difficult to explain why the virus infectivity has lost its momentum at the end of November. This could probably be due to various government measures and restrictions - it is also obvious how much has decreased the weekly confirmed cases just two weeks after the second lockdown:"),
                    dcc.Graph(figure=fig_gen_stats_weekly_events_2)
                ]
            ),
        ]
    )
])

footer = html.Div(
    [
        html.Small("Data sources: COVID-19 data from "),
        html.Small(html.A("Open Data Portal", href="https://data.egov.bg/covid-19?section=8&subsection=16&item=36", target="_blank")),
        html.Small(" • Spatial data from "),
        html.Small(html.A("DIVA GIS", href="https://www.diva-gis.org/gdata", target="_blank")),
        html.Small(" • Demographic data from "),
        html.Small(html.A("NSI", href="https://www.nsi.bg/bg/content/2974/население", target="_blank")),
        html.Br(),
        html.Small("Designed by "),
        html.Small(html.A("Svilen Stefanov", href="https://www.linkedin.com/in/svilen-stefanov/", target="_blank")),
        html.Small(" and "),
        html.Small(html.A("Ivaylo Stoyanov", href="https://www.linkedin.com/in/ivaylo-stoyanov-0124b2119/", target="_blank")),
        html.Small(", inspired by "),
        html.Small(html.A("Martin Boyanov", href="https://www.linkedin.com/in/martin-boyanov-1ab2124a/", target="_blank")),
        html.Br(),
        html.Small(html.A("Source code", href="https://github.com/svilens/covid-bulgaria/", target="_blank")),
    ], style={'font-style':'italic', 'padding-left':'10px', 'textAlign':'center'}
)


logger.info('Creating dash layout')
app.layout = html.Div([
    html.H1(children='COVID-19 in Bulgaria', style={'padding-left':'5px'}),
    html.P(f"Last update: {covid_general.date.tail(1).dt.date.values[0].strftime('%d-%b-%Y')}", style={'textAlign':'left', 'padding-left':'15px', 'color':'gold', 'font-style':'italic'}),
    cards,
    tabs,
    footer
])


#Callbacks
logger.info('Creating dash callbacks')
@app.callback(
    [
        dash.dependencies.Output('custom-arima-output', 'figure'),
        dash.dependencies.Output('custom-arima-output-error', 'children')
        #dash.dependencies.Output('custom-arima-output-summary', 'children')
    ],
    [
        dash.dependencies.Input('province-dropdown-arima', 'value')
    ])
def update_arima_output_province(province):
    arima_result = arima_chart(province)
    return (
        arima_result[0],
        [f"Mean absolute percentage error: {arima_result[1].round(2)}"]
        #[pprint.pprint(str(arima_provinces.get(province+'_summary')), width=100)]
    )

@app.callback(
    [
        dash.dependencies.Output('custom-triple-output', 'figure'),
        dash.dependencies.Output('custom-triple-output-error', 'children')
    ],
    [
        dash.dependencies.Input('province-dropdown', 'value'),
        dash.dependencies.Input('variable-dropdown', 'value'),
        dash.dependencies.Input('forecast-length-input', 'value')
    ])
def update_triple_output_province(province, variable, forecast_length):
    custom_triple = triple_exp_smoothing(ts_data, province, variable, forecast_length)
    return (
        custom_triple[0].update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'),
        f"Mean absolute percentage error: {custom_triple[1]}"
    )


logger.info('Running dash server')

server = app.server

if __name__ == '__main__':
    app.run_server(debug=True)

