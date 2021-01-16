import pandas as pd

def read_covid_by_province(path, date_col):
    covid_by_province = pd.read_csv(path, parse_dates=[date_col])
    covid_by_province.rename(columns={date_col:'date'}, inplace=True)
    
    covid_by_province_transposed = (
        covid_by_province
        .melt(id_vars='date', var_name = 'reg_variable', value_name='value')
    )
    
    covid_by_province_transposed['code'] = covid_by_province_transposed['reg_variable'].apply(lambda x: x[:3])
    covid_by_province_transposed['variable'] = covid_by_province_transposed['reg_variable'].apply(lambda x: x[-3:])

    covid_pivot = (
        pd.pivot_table(
            covid_by_province_transposed,
            index=['date','code'],
            columns='variable', values='value', aggfunc='first'
        ).reset_index()
    )

    covid_pivot.sort_values(by=['code','date'], inplace=True)
    covid_pivot['new_cases'] = covid_pivot['ALL'].diff()
    covid_pivot = covid_pivot.loc[covid_pivot.date != min(covid_pivot.date)]

    return covid_pivot


def read_covid_general(path, date_col):
    covid_general_data = pd.read_csv(path, parse_dates=[date_col])
    covid_general_data.rename(columns={date_col:'date', 'Тестове за денонощие':'daily_tests', 'Активни случаи':'active_cases',
                                       'Потвърдени случаи':'total_cases', 'Нови случаи за денонощие':'new_cases',
                                       'Хоспитализирани':'hospitalized', 'В интензивно отделение':'intensive_care',
                                       'Излекувани':'total_recoveries','Излекувани за денонощие':'new_recoveries',
                                       'Починали':'total_deaths', 'Починали за денонощие':'new_deaths'}, inplace=True)
    covid_selected = covid_general_data[['date', 'daily_tests', 'active_cases', 'total_cases', 'new_cases', 'hospitalized', 'intensive_care',
                                         'total_recoveries', 'new_recoveries', 'total_deaths', 'new_deaths']]
    return covid_selected


def read_covid_tests(path, date_col):
    tests = pd.read_csv(path, parse_dates=[date_col]).rename(columns={
        date_col:'date',
        'Общо тестове':'total_tests',
        'PCR тестове':'total_tests_pcr',
        'Антигени тестове':'total_tests_antigen',
        'Общо тестове за денонощие':'new_tests',
        'PCR тестове за денонощие':'new_pcr',
        'Антигени тестове за денонощие':'new_antigen',
        'Установени случаи общо':'total_positive',
        'Установени случаи чрез PCR':'total_positive_pcr',
        'Установени случаи чрез антиген':'total_positive_antigen',
        'Установени случаи за денонощие':'new_positive',
        'Установени случаи за денонощие чрез PCR':'new_positive_pcr',
        'Установени случаи за денонощие чрез антиген':'new_positive_antigen'
        }
    )

    tests['pos_rate_pcr'] = (100*tests['new_positive_pcr'] / tests['new_pcr']).round(2)
    tests['pos_rate_antigen'] = (100*tests['new_positive_antigen'] / tests['new_antigen']).round(2)
    return tests


