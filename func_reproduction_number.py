import pandas as pd
import numpy as np
from matplotlib.dates import date2num
from scipy import stats as sps
from scipy.interpolate import interp1d

def prepare_cases(cases):
    new_cases = cases.diff()

    smoothed = new_cases.rolling(7,
        win_type='gaussian',
        min_periods=1,
        center=True).mean(std=3).round()
    
    idx_start = np.searchsorted(smoothed, 7)
    smoothed = smoothed.iloc[idx_start:]
    original = new_cases.loc[smoothed.index]
    
    return original, smoothed


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

        fig_rt_province.add_trace(go.Scatter(
                x=subset.date[3:],
                y=subset.Low_90[3:],
                fill='none',
                mode='lines', line_color="rgba(38,38,38,0.9)", line_shape='spline',
                name="Low density interval"),
            row=row_num, col=col_num)
        fig_rt_province.add_trace(go.Scatter(
                x=subset.date[3:],
                y=subset.High_90[3:],
                fill='tonexty', mode='none',
                fillcolor="rgba(65,65,65,1)",
                line_shape='spline',
                name="High density interval"),
            row=row_num, col=col_num)
        fig_rt_province.add_trace(go.Scatter(
                x=subset.date[3:],
                y=subset.Estimated[3:],
                mode='markers+lines',
                line=dict(width=0.3, dash='dot'), line_shape='spline',
                marker_color=subset.Estimated, marker_colorscale='RdYlGn_r', marker_line_width=1.2,
                marker_cmin=0.5, marker_cmax=1.4, name='R<sub>t'),
            row=row_num, col=col_num)

    fig_rt_province.update_layout(
        title="Real-time R<sub>t</sub> by province",
        height=4000, showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig_rt_province.update_yaxes(range=[0, 2.5])
    return fig_rt_province
