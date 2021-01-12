import os
import sys
import requests
import json
import pandas as pd
from pandas import json_normalize
from datetime import datetime
import pytz


# backup files
def backup_existing_file(filename):
    # if the file already exists
    if os.path.isfile(os.path.join(os.getcwd() + '\\data\\' + filename)):
        # if a backup_ of the file exists, delete it
        if os.path.isfile(os.path.join(os.getcwd() + '\\data\\backup_' + filename)):
            os.remove(os.path.join(os.getcwd() + '\\data\\backup_' + filename))
            print(f'The previous backup of {filename} is not needed and was deleted.')
        # rename the current file, adding a 'backup_' prefix
        os.rename(
            os.path.join(os.getcwd() + '\\data\\' + filename), 
            os.path.join(os.getcwd() + '\\data\\backup_' + filename)
        )
        print(f'Backed up existing file: {filename}')


# download newest data
api_url = 'https://data.egov.bg/api/getResourceData'

resources = {
    '(1) general stats': ['e59f95dd-afde-43af-83c8-ea2916badd19', 'COVID_general.csv'],
    '(2) by province': ['cb5d7df0-3066-4d7a-b4a1-ac26525e0f0c', 'COVID_provinces.csv'],
    '(3) by age band': ['8f62cfcf-a979-46d4-8317-4e1ab9cbd6a8', 'COVID_age_bands.csv'],
    '(4) by test type': ['0ce4e9c3-5dfc-46e2-b4ab-42d840caab92', 'COVID_test_type.csv']
}


def get_covid_data():
    for resource_key in resources.keys():
        api_token = resources[resource_key][0]
        print(f'Downloading COVID data: {resource_key}')
        
        api_headers = {
            'content-type': 'application/json',
            'api_key': api_token,
            'resource_uri': api_token
        }
        
        # request the data from the API
        response = requests.post(url=api_url, data=api_headers)
        
        # if the response is 'success'
        if response.status_code == 200:
            data_json = json.loads(response.content.decode('utf-8'))
            # convert to pandas df
            data_json_normalized = json_normalize(data_json, 'data')
            # set the first line as column headers
            df = data_json_normalized.iloc[1:,:]
            df.columns = data_json_normalized.iloc[0,:].values
            
            # check if the source data has been updated today
            if df.iloc[-1,0] == datetime.now(pytz.timezone('Europe/Sofia')).strftime(format='%Y/%m/%d'):
                # backup the existing file and save the newest data
                backup_existing_file(resources[resource_key][1])
                df.to_csv(os.getcwd() + '\\data\\' + resources[resource_key][1], header=True, encoding='utf-8', index=False)
                print(f'Saved newest data file: {resources[resource_key][1]}')
            else:
                return 'old'
        else:
            return 'error'
