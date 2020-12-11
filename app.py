from logging_func import *
import geopandas as gpd
import pandas as pd
import numpy as np
from datetime import datetime

# from external modules
from read_spatial import *
from read_covid import *
from read_nsi import *

logger = get_logger_app('app')
logger.info('Starting the process')


####### GENERAL STATS #######

logger.info('Reading general stats')
covid_general = read_covid_general('./data/Обща статистика за разпространението.csv', 'Дата')
#covid_general.head()

import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "plotly_dark"


logger.info('Creating chart 1: Cumulative cases over time')
fig_gen_stats = go.Figure()
fig_gen_stats.add_trace(go.Scatter(x=covid_general.date, y=covid_general.total_cases, name='Confirmed'))
fig_gen_stats.add_trace(go.Scatter(x=covid_general.date, y=covid_general.active_cases, line=dict(color='yellow'), name='Active'))
fig_gen_stats.add_trace(go.Scatter(x=covid_general.date, y=covid_general.total_recoveries, line=dict(color='green'), name='Recovered'))
fig_gen_stats.add_trace(go.Scatter(x=covid_general.date, y=covid_general.total_deaths, line=dict(color='red'), name='Deaths'))
fig_gen_stats.update_layout(title='Number of cases over time (cumulative)')
fig_gen_stats.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

covid_general['week'] = covid_general.date.dt.isocalendar().week
covid_general_weekly = covid_general.groupby('week')[['new_cases', 'new_deaths', 'new_recoveries']].sum()
covid_general_weekly['new_cases_pct_change'] = covid_general_weekly['new_cases'].pct_change()
# removing the first week as it starts on Saturday, and the last week, as it would be incomplete in most cases
covid_general_weekly = covid_general_weekly[1:-1]


logger.info('Creating chart 2: Cases per week')
from plotly.subplots import make_subplots

fig_gen_stats_weekly = make_subplots(specs=[[{"secondary_y": True}]])
fig_gen_stats_weekly.add_trace(go.Scatter(x=covid_general_weekly.index[1:], y=covid_general_weekly.new_cases[1:], name='New confirmed cases'), secondary_y=False)
fig_gen_stats_weekly.add_trace(go.Bar(x=covid_general_weekly.index[1:], y=covid_general_weekly.new_deaths[1:], name='New death cases'), secondary_y=True)
fig_gen_stats_weekly.add_trace(go.Scatter(x=covid_general_weekly.index[1:], y=covid_general_weekly.new_recoveries[1:], name='New recoveriers'), secondary_y=True)
fig_gen_stats_weekly.update_layout(title = 'New cases per week')
fig_gen_stats_weekly.update_xaxes(title_text="week number")
fig_gen_stats_weekly.update_yaxes(title_text="Confirmed cases", secondary_y=False)
fig_gen_stats_weekly.update_yaxes(title_text="Deaths / recoveries", secondary_y=True)
fig_gen_stats_weekly.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

logger.info('Creating chart 3: New cases weekly % change')
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
fig_gen_stats_weekly_new_pct.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')


logger.info('Creating chart 4: New cases per week + summer events')
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
fig_gen_stats_weekly_events.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')


### Rates

logger.info('Calculating rates')
covid_general['total_cases_7days_ago'] = covid_general['total_cases'].shift(periods=7)
covid_general['total_cases_14days_ago'] = covid_general['total_cases'].shift(periods=14)
covid_general['death_rate'] = (covid_general['total_deaths'] / covid_general['total_cases_14days_ago']).round(4)
covid_general['recovery_rate'] = (covid_general['total_recoveries'] / covid_general['total_cases_14days_ago']).round(4)
covid_general['hospitalized_rate'] = (covid_general['hospitalized'] / covid_general['active_cases']).round(4)
covid_general['intensive_care_rate'] = (covid_general['intensive_care'] / covid_general['hospitalized']).round(4)
covid_general['tests_positive_rate'] = (covid_general['new_cases'] / covid_general['daily_tests']).round(4)


logger.info('Creating chart 5: Rates')
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


### Map provinces

logger.info('Reading spatial data')
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

covid_pop_sorted = covid_pop.sort_values(by=['date', 'ALL'])
# animation frame parameter should be string or int
covid_pop_sorted['day'] = covid_pop_sorted.date.apply(lambda x: (x - min(covid_pop_sorted.date)).days + 1)

covid_yesterday = gpd.GeoDataFrame(covid_pop.loc[covid_pop.date == max(covid_pop.date)])
import plotly.express as px


logger.info('Creating chart 6: Provinces map - total cases per 100k pop')
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
fig_yesterday_map_total.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')


logger.info('Creating chart 7: Provinces map - new cases per 100k pop')
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
fig_yesterday_map_new.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')


logger.info('Creating chart 8: Provinces map - active cases per 100k pop')
fig_yesterday_map_active = px.choropleth_mapbox(
    covid_yesterday,
    geojson=covid_yesterday.geometry,
    locations=covid_yesterday.index,
    color='active_per_100k',
    color_continuous_scale='Burgyl',
    hover_name='province',
    labels={'active_per_100k':'active cases<br>per 100k pop'},
    title=f"Currently active cases by province for {covid_yesterday.date.max().strftime('%d %b %Y')}",
    center={'lat': 42.734189, 'lon': 25.1635087},
    mapbox_style='carto-darkmatter',
    opacity=1,
    zoom=6
)
fig_yesterday_map_active.update_layout(margin={"r":0,"t":40,"l":0,"b":0}, template='plotly_dark')
fig_yesterday_map_active.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')


### Age bands

logger.info('Reading age bands data')
covid_by_age_band = (pd.read_csv('./data/Разпределение по дата и по възрастови групи.csv', parse_dates=['Дата']).rename(columns={'Дата':'date'}))


logger.info('Creating chart 9: Cumulative cases by age band')
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
fig_age.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')


####### REPRODUCTION NUMBER #######

logger.info('Starting Rt processing')

provinces = covid_pop_sorted[['province', 'date', 'ALL']].groupby(['province','date']).ALL.sum()

orig_bg = pd.read_csv('./dash_data/r0_bg_original.csv')
smoothed_bg = pd.read_csv('./dash_data/r0_bg_smoothed.csv')

logger.info('Creating chart 10: Smoothed new cases - BG')
fig_new_bg = go.Figure()
fig_new_bg.add_trace(go.Scatter(x=orig_bg.reset_index()['date'], y=orig_bg.reset_index()['ALL'],
                                      mode='lines', line=dict(dash='dot'), name='Actual'))
fig_new_bg.add_trace(go.Scatter(x=smoothed_bg.reset_index()['date'], y=smoothed_bg.reset_index()['ALL'],
                                      mode='lines', line=dict(width=3), name='Smoothed', marker_color='steelblue'))
fig_new_bg.update_layout(title='Daily new cases in Bulgaria')
fig_new_bg.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')


logger.info('Creating chart 11: Smoothed recoveries - BG')
orig_recovered_bg = pd.read_csv('./dash_data/r0_bg_original_recovered.csv')
smoothed_recovered_bg = pd.read_csv('./dash_data/r0_bg_smoothed_recovered.csv')
fig_recovered_bg = go.Figure()
fig_recovered_bg.add_trace(go.Scatter(x=orig_recovered_bg.reset_index()['date'], y=orig_recovered_bg.reset_index()['total_recoveries'],
                                      mode='lines', line=dict(dash='dot'), name='Actual'))
fig_recovered_bg.add_trace(go.Scatter(x=smoothed_recovered_bg.reset_index()['date'], y=smoothed_recovered_bg.reset_index()['total_recoveries'],
                                      mode='lines', line=dict(width=3), name='Smoothed', marker_color='green'))
fig_recovered_bg.update_layout(title='Daily new recoveries in Bulgaria')
fig_recovered_bg.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')


logger.info('Creating chart 12: Smoothed deaths - BG')
orig_deaths_bg = pd.read_csv('./dash_data/r0_bg_original_deaths.csv')
smoothed_deaths_bg = pd.read_csv('./dash_data/r0_bg_smoothed_deaths.csv')
fig_deaths_bg = go.Figure()
fig_deaths_bg.add_trace(go.Scatter(x=orig_deaths_bg.reset_index()['date'], y=orig_deaths_bg.reset_index()['total_deaths'],
                                      mode='lines', line=dict(dash='dot'), name='Actual'))
fig_deaths_bg.add_trace(go.Scatter(x=smoothed_deaths_bg.reset_index()['date'], y=smoothed_deaths_bg.reset_index()['total_deaths'],
                                      mode='lines', line=dict(width=3), name='Smoothed'))
fig_deaths_bg.update_layout(title='Daily new deaths in Bulgaria')
fig_deaths_bg.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')


logger.info('Creating chart 13: Smoothed new cases - BG - age bands')
covid_by_age_band_diff_smoothed = covid_by_age_band.set_index('date').diff().rolling(9,
        win_type='gaussian',
        min_periods=1,
        center=True).mean(std=3).round()

fig_age_diff = go.Figure()
i=0
for col in covid_by_age_band_diff_smoothed.columns:
    fig_age_diff.add_trace(go.Scatter(x=covid_by_age_band_diff_smoothed.index, y=covid_by_age_band_diff_smoothed[col], mode='lines', line_shape='spline',
                                 line=dict(width=3, color=age_band_colors[i]), name=col))
    i+=1
fig_age_diff.update_layout(title='New confirmed cases by age band (smoothed figures)')
fig_age_diff.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')



logger.info('Creating chart 14: Smoothed new cases - provinces')
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
    fig_new_by_province.update_layout(title='Daily new cases by province', height=3200, showlegend=False)

fig_new_by_province.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')


### Rt for BG

logger.info('Creating chart 15: Rt for BG')
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
                            marker_color=values_bg, marker_colorscale='RdYlBu_r', marker_line_width=1.2,
                            marker_cmin=0.5, marker_cmax=1.4, name='R<sub>t'))
fig_rt.update_layout(yaxis=dict(range=[0,4]), title="Real-time R<sub>t</sub> for Bulgaria", showlegend=False)
fig_rt.show()
fig_rt.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')


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
mr.loc[mr.status=="0", 'colors'] = 'grey'


fig_rt_province_yesterday = go.Figure()
fig_rt_province_yesterday.add_trace(go.Bar(x=mr.province, y=mr.Estimated, marker_color=mr.colors, 
                          error_y=dict(type='data', array=mr.diff_up, arrayminus=mr.diff_down)))
fig_rt_province_yesterday.update_layout(title='R<sub>t</sub> by province for the last daily update')
#fig_rt_province_yesterday.show()
fig_rt_province_yesterday.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')


logger.info('Creating chart 16: Rt by province')

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

    fig_rt_province.update_layout(yaxis=dict(range=[0,4]), title="Real-time R<sub>t</sub> by province", height=4000, showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig_rt_province


fig_rt_province_actual = generate_rt_by_province(provinces, final_results)


####### ARIMA #######

logger.info('Reading ARIMA')

ts_data = covid_pop.reset_index()[['date', 'province', 'ALL', 'new_cases']].rename(columns={'ALL':'total_cases'})

ts_data_r0 = final_results[['province', 'date', 'Estimated']].rename(columns={'Estimated':'r0'})
ts_data_r0.date = pd.to_datetime(ts_data_r0.date)
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

train = df.iloc[:-10, :]
test = df.iloc[-10:, :]
pred = test.copy()


df['z_data'] = (df['data'] - df.data.rolling(window=12).mean()) / df.data.rolling(window=12).std()
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

logger.info('Creating chart 17: Double exp smoothing')
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

fig_exp_smoothing_double.update_layout(title="Holt's (double) exponential smoothing for R<sub>t</sub> in Bulgaria")
fig_exp_smoothing_double.show()
fig_exp_smoothing_double.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')


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

logger.info('Creating chart 18: Triple exp smoothing')
#plot the training data, the test data and the forecast on the same plot
fig_exp_smoothing_triple = go.Figure()
fig_exp_smoothing_triple.add_trace(go.Scatter(x=train.index[30:], y=train.data[30:], name='Historical data', mode='lines'))
fig_exp_smoothing_triple.add_trace(go.Scatter(x=train.index[30:], y=fitted_triple.fittedvalues[30:], name='Model fit', mode='lines', marker_color='lime'))
fig_exp_smoothing_triple.add_trace(go.Scatter(x=test.index, y=test.data, name='Validation data', mode='lines', marker_color='coral'))
fig_exp_smoothing_triple.add_trace(go.Scatter(
    x=pd.date_range(start=test.index.min(), periods=len(test) + len(pred_triple)),
    y=pred_triple, name='Forecast', marker_color='gold', mode='lines')
)
fig_exp_smoothing_triple.update_layout(title="Holt-Winters' (triple) exponential smoothing for R<sub>t</sub> in Bulgaria")
fig_exp_smoothing_triple.show()
fig_exp_smoothing_triple.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')


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


app = dash.Dash(name='COVID-19 in Bulgaria', external_stylesheets=[dbc.themes.DARKLY])
app.title = 'COVID-19 in Bulgaria'

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        <!-- Global site tag (gtag.js) - Google Analytics -->
        <script async src="https://www.googletagmanager.com/gtag/js?id=G-2FCPJC5BDW"></script>
        <script>
          window.dataLayer = window.dataLayer || [];
          function gtag(){dataLayer.push(arguments);}
          gtag('js', new Date());

          gtag('config', 'G-2FCPJC5BDW');
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
                    html.H4("Cases over time (cumulative figures)"),
                    html.Br(),
                    dcc.Graph(figure=fig_gen_stats),
                    html.Br(),
                    html.Br(),
                    html.H4("Smoothed figures on a daily basis"),
                    html.Br(),
                    dcc.Graph(figure=fig_new_bg),
                    html.Br(),
                    dcc.Graph(figure=fig_recovered_bg),
                    html.Br(),
                    dcc.Graph(figure=fig_deaths_bg),
                    html.Br(),
                    dcc.Graph(figure=fig_age_diff),
                    html.Br(),
                    html.Br(),
                    html.H4("Cases on a weekly basis"),
                    html.Br(),
                    dcc.Graph(figure=fig_gen_stats_weekly),
                    html.P("The weekly cases distribution chart above shows that the number of new confirmed cases per week was relatively stable during the summer (week 25 - week 40), but drastically started to increase in the early October. Although the number of cases is currently at its highest level ever, the exponential growth seems to be ended by the end of November. The other good news is that the number of new recoveries have also increased greatly and are still following their exponential trend. Unfortunately, the number of new death cases are also making a new record in each of the past 8 weeks.")
                ]
            ),
            dcc.Tab(
                label = "By province",
                className = "custom-tab",
                selected_className = "custom-tab--selected",
                children = [
                    html.Br(),
                    html.Br(),
                    html.P("Provinces, color-coded according to the number of total confirmed cases per 100,000 population:"),
                    html.Br(),
                    dcc.Graph(figure=fig_yesterday_map_total),
                    html.Br(),
                    html.Br(),
                    html.P(f"Provinces, color-coded according to the number of the currently active cases per 100,000 population:"),
                    html.Br(),
                    dcc.Graph(figure=fig_yesterday_map_active),
                    html.Br(),
                    html.Br(),
                    html.P(f"Provinces, color-coded according to the number of the new confirmed cases for the last daily update per 100,000 population:"),
                    html.Br(),
                    dcc.Graph(figure=fig_yesterday_map_new),
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
                    html.Br()
                ]
            ),
            dcc.Tab(
                label = "Rt",
                className = "custom-tab",
                selected_className = "custom-tab--selected",
                children = [
                    html.Br(),
                    html.Br(),
                    html.P("The estimated daily reproduction number represents how many people are directly infected by 1 infectious person per each day. Ideally, we want this number to be lower than 1. Otherwise, the disease is spreading linearly (=1) or exponentially (>1)."),
                    html.Br(),
                    dcc.Graph(figure=fig_rt),
                    html.Br(),
                    html.Br(),
                    dcc.Graph(figure=fig_rt_province_yesterday),
                    html.Br(),
                    html.Br(),
                    html.P('Below is a chart showing the daily reproduction number by province. Note that each chart is covering a different period of time, becase for each province we pick the start date as the moment when we have started to constantly see 10+ daily confirmed cases.'),
                    dcc.Graph(figure=fig_rt_province_actual)
                ]
            ),
            dcc.Tab(
                label = "ARIMA",
                className = "custom-tab",
                selected_className = "custom-tab--selected",
                children = [
                    html.Br(),
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
                            html.H5(id="custom-arima-output-error"),
                            html.P(id="custom-arima-output-summary")
                        ])
                    ])
                ]
            ),
            dcc.Tab(
                label = "Exponential smoothing",
                className = "custom-tab",
                selected_className = "custom-tab--selected",
                children = [
                    html.Br(),
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
                            html.H5(id="custom-triple-output-error")
                        ])
                    ]),
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
                    html.P("Five major dates were selected: the end of the first lockdown, when the borders with Greece were re-opened (15th June); the football cup final with 20,000 spectators on the stands (1st July); the beginning of the anti-government protests (11th July); the first mass anti-government protest (2nd September); the school opening (15th September)."),
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
                    html.P("Another potential reason that could have played part in the increased cases in October is the change in the weather conditions. However, it will be difficult to explain why the virus infectivity has lost its momentum at the end of November - unless it is caused by government measures and restrictions.")
                ]
            ),
        ]
    )
])


logger.info('Creating dash layout')
app.layout = html.Div([
    html.H1(children='COVID-19 in Bulgaria'),
    html.P(f"Data version: {covid_general.date.tail(1).dt.date.values[0].strftime('%d-%b-%Y')}"),
    cards,
    tabs
])


logger.info('Creating dash callbacks')
#Callbacks
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



