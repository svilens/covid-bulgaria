# COVID-19 in Bulgaria

## How to run
Run app_processing.py. It downloads the latest data, does some processing (90 mins) and creates some files in dash_data folder.
Then, run app.py to read the dash_data files, draw charts and create a dashboard.
```
python app_processing.py
python app.py
```

This dash was previously available at covid-bulgaria.com, but since mid 2022 it was abandoned and can now run only locally.

## Business understanding
The SARS-CoV-2 pandemic is making a huge impact on the worldwide industry, society and everyday life. The consequences of the coronavirus disease (COVID-19) outbreak are unprecedented and felt around the world. The world of work is being profoundly affected by the pandemic. In addition to the threat to public health, the economic and social disruption threatens the long-term livelihoods and wellbeing of millions. The pandemic is heavily affecting labour markets, economies and enterprises, including global supply chains, leading to widespread business and social disruptions.

The COVID-19 pandemic can affect almost all areas of life - including healthcare capacity and efficiency, business relationships, turnovers and market shares, supply chains, labour markets, education, tourism, social activities and so on.

That's why it is important to analyze the available data and make informed decisions that could reduce the spread of the disease. The data analysis and modelling stands in the core of getting the right insights and making right decisions at the right moment. Spreading the infections is of a local nature, and therefore, all decisions not only need to be prompt, but also to be made according to the specific regional characteristics.

## Data understanding
To reach the goals set in the business understanding section, we are using several datasets:

- SARS-COVID19 data - Open Data Portal (egov.bg)
We have three data feeds - (1) general COVID-19 statistics (new cases, deaths, recoveries, hospitalized, etc.) on a national level, (2) number of confirmed cases by age bands, and (3) number of confirmed cases by province. The data covers the period since 6th July 2020 and is being updated on a daily basis on the Open Data Portal.
- Population data - Население по области, общини, местоживеене и пол | Национален статистически институт (nsi.bg)
- To calculate the daily number of infections per 100,000 population for each province, we are using demographic statistics for 2019 published by NSI.
Shape files - Download data by country | DIVA-GIS (diva-gis.org)
To plot the stats by province and color-code the provinces by the number of new cases in each, we are using shape files with spatial data on a province level.
