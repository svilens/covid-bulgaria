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

