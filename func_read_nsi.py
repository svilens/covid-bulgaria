import pandas as pd

# province codes in the correct order to append to the NSI population data
codes_pop = ['SOF','PDV','VAR','BGS','SZR','BLG','PAZ',
             'PVN','VTR','SFO','HKV','RSE','SLV','SHU',
             'DOB','VRC','KRZ','MON','LOV','PER','JAM',
             'KNL','TGV','RAZ','SLS','GAB','SML','VID']


# population data is provided by NSI on multiple levels (municipality and province) within the same table
# only the figures on province level are needed, the rest will be discarded

def read_population_data(path, worksheet_name, col_num, col_names, skip, codes):
	pop_by_municipality = pd.read_excel(
		path, sheet_name=worksheet_name, index_col=None, usecols=range(col_num), names=col_names, skiprows=skip)
	
	# group on municipality to get the max value, then sort by population column (descending)
	# drop the first record (as it's duplicated) and take the next 28 records - they are the province capitals
	
	pop_by_province = (
		pop_by_municipality.groupby('municipality')[col_names[1]].max().reset_index()
		.sort_values(by=col_names[1],ascending=False)[1:29].rename(columns={col_names[0]:'province'}))

	pop_by_province_code = (
		pd.concat(
			[
	        	pd.DataFrame(data=codes, columns=['code']),
	        	pop_by_province.reset_index().drop(['index','province'], axis=1)
	    	],
	    	axis=1
	    )
	)

	return pop_by_province_code


def read_nsi_age_bands(path, worksheet_name, col_num, col_names, skip, rows_needed):
    nsi_age_bands = pd.read_excel(
		path, sheet_name=worksheet_name, index_col=None, usecols=range(col_num),
        names=col_names, skiprows=skip, nrows=rows_needed
	)
    nsi_age_bands = nsi_age_bands.replace(0, '0 - 0').replace('100 +', '100 - 150')
    for row in range(len(nsi_age_bands)):
        if int(nsi_age_bands.loc[nsi_age_bands.index==row,'age_band'].str.split('- ').values[0][1]) <= 19:
            nsi_age_bands.loc[nsi_age_bands.index==row,'covid_age_band'] = '0 - 19'
        elif int(nsi_age_bands.loc[nsi_age_bands.index==row,'age_band'].str.split('- ').values[0][1]) <= 29:
            nsi_age_bands.loc[nsi_age_bands.index==row,'covid_age_band'] = '20 - 29'
        elif int(nsi_age_bands.loc[nsi_age_bands.index==row,'age_band'].str.split('- ').values[0][1]) <= 39:
            nsi_age_bands.loc[nsi_age_bands.index==row,'covid_age_band'] = '30 - 39'
        elif int(nsi_age_bands.loc[nsi_age_bands.index==row,'age_band'].str.split('- ').values[0][1]) <= 49:
            nsi_age_bands.loc[nsi_age_bands.index==row,'covid_age_band'] = '40 - 49'
        elif int(nsi_age_bands.loc[nsi_age_bands.index==row,'age_band'].str.split('- ').values[0][1]) <= 59:
            nsi_age_bands.loc[nsi_age_bands.index==row,'covid_age_band'] = '50 - 59'
        elif int(nsi_age_bands.loc[nsi_age_bands.index==row,'age_band'].str.split('- ').values[0][1]) <= 69:
            nsi_age_bands.loc[nsi_age_bands.index==row,'covid_age_band'] = '60 - 69'
        elif int(nsi_age_bands.loc[nsi_age_bands.index==row,'age_band'].str.split('- ').values[0][1]) <= 79:
            nsi_age_bands.loc[nsi_age_bands.index==row,'covid_age_band'] = '70 - 79'
        elif int(nsi_age_bands.loc[nsi_age_bands.index==row,'age_band'].str.split('- ').values[0][1]) <= 89:
            nsi_age_bands.loc[nsi_age_bands.index==row,'covid_age_band'] = '80 - 89'
        else:
            nsi_age_bands.loc[nsi_age_bands.index==row,'covid_age_band'] = '90+'
    nsi_age_bands = nsi_age_bands.groupby('covid_age_band')[col_names[-1]].sum().reset_index()
    nsi_age_bands['band_prop'] = (
        nsi_age_bands[col_names[-1]] / nsi_age_bands[col_names[-1]].sum()
    ).round(4)
    return nsi_age_bands
