import geopandas as gpd
import pandas as pd

# province codes appended to the spatial data as they are set in COVID-19 data by province
codes_spatial = ['BLG','BGS','DOB','GAB','SOF','HKV','KRZ',
         'KNL','LOV','MON','PAZ','PER','PVN','PDV',
         'RAZ','RSE','SHU','SLS','SLV','SML','SFO',
         'SZR','TGV','VAR','VTR','VID','VRC','JAM']

def read_spatial_data(path, codes):
	geo_df = gpd.read_file(path, encoding='utf-8')
	geo_df = geo_df[["NAME_1", "geometry"]].rename(columns={"NAME_1":"province"})
	geo_df_w_codes = (
	    pd.concat(
	    	[
	    		gpd.GeoDataFrame(data=codes, columns=['code']),
	            geo_df.replace('Grad Sofiya','Sofiya-Grad').replace('Sofia','Sofiya-Oblast')
	        ],
	        axis=1
	    )
	    .sort_values(by='province')
	)
	return geo_df_w_codes