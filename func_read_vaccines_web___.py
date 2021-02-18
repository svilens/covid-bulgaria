import os 
from selenium import webdriver

chromedriver_dir = './tools/chromedriver.exe'
vaccines_url = 'https://coronavirus.bg/bg/statistika'

codes_vaccines = ['BLG','BGS','VAR','VTR','VID','VRC','GAB',
                  'DOB','KRZ','KNL','LOV','MON','PAZ','PER',
                  'PVN','PDV','RAZ','RSE','SLS','SLV','SML',
                  'SFO','SOF','SZR','TGV','HKV','SHU','JAM']

def get_vaccines_data_web(url, chromedriver_dir):
    options = webdriver.ChromeOptions()
    #options.binary_location=str(os.environ.get('GOOGLE_CHROME_BIN')) # REQUIRED FOR HEROKU
    #options.add_argument('--disable-gpu') # REQUIRED FOR HEROKU
    #options.add_argument('--no-sandbox') # REQUIRED FOR HEROKU
    options.add_argument("browser.download.folderList=2");
    options.add_argument("browser.helperApps.alwaysAsk.force=False");
    options.add_argument("browser.download.manager.showWhenStarting=False");
    options.add_argument("browser.download.manager.showAlertOnComplete=False");
    options.add_argument("browser.helperApps.neverAsk.saveToDisk=True");
    #options.add_argument(f"browser.download.dir={download_dir}");
    options.add_argument('--no-proxy-server');
    options.add_argument("--proxy-server='direct://'");
    options.add_argument("--proxy-bypass-list=*");
    options.headless = False
    
    driver = webdriver.Chrome(chrome_options=options, executable_path=chromedriver_dir)
    
    driver.get(url)
    table_vac = driver.find_element_by_xpath("//div[@class='col stats']")
    
    # strip headers and footer
    vaccines_raw = table_vac.text.split('\n')[5:-4]
    
    import pandas as pd
    vaccines_df = pd.DataFrame()
        
    for i in vaccines_raw:
        line = i.replace('-','0').split(' ')
        
        # adjust the names of provinces containing a space
        if len(line) == 7:
            line = [line[0] + ' ' + line[1], line[2], line[3], line[4], line[5], line[6]]
        
        # convert to int
        line_int = [
            line[0],
            int(line[1]),
            int(line[2]),
            int(line[3]),
            int(line[4]),
            int(line[5])
        ]
    
        # add each province data to a dataframe
        vacc_line = pd.DataFrame([line_int])
        vaccines_df = pd.concat([vaccines_df, vacc_line])
    
    # rename columns
    vaccines_df.columns=['province', 'total', 'new_pfizer', 'new_astrazeneca', 'new_moderna', 'second_dose']
    vaccines_df = vaccines_df.reset_index().drop(['index', 'province'], axis=1)
    vaccines_df = pd.concat([pd.DataFrame(codes_vaccines, columns=['code']),
               vaccines_df], axis=1)
    
    return vaccines_df