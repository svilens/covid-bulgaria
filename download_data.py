import os
from selenium import webdriver
from selenium.webdriver.firefox.options import Options

GOOGLE_CHROME_BIN = '/app/.apt/usr/bin/google_chrome' # REQUIRED FOR HEROKU
CHROMEDRIVER_PATH = '/app/.chromedriver/bin/chromedriver' # REQUIRED FOR HEROKU

def backup_existing_file(filename):
    # if the file already exists
    if os.path.isfile(os.path.join(os.getcwd() + '\\data\\' + filename)):
        # if a backup_ of the file exists, delete it
        if os.path.isfile(os.path.join(os.getcwd() + '\\data\\backup_' + filename)):
            os.remove(os.path.join(os.getcwd() + '\\data\\backup_' + filename))
            print(f'The previous backup of "{filename}" is not needed and was deleted.')
        # rename the current file, adding a 'backup_' prefix
        os.rename(
            os.path.join(os.getcwd() + '\\data\\' + filename), 
            os.path.join(os.getcwd() + '\\data\\backup_' + filename)
        )
        print(f'The existing file "{filename}" was backed up.')


def firefox_driver(download_dir, driver_dir):
    fp = webdriver.FirefoxProfile()
    fp.set_preference("browser.download.folderList", 2)
    fp.set_preference("browser.helperApps.alwaysAsk.force", False);
    fp.set_preference("browser.download.manager.showWhenStarting", False)
    fp.set_preference("browser.download.manager.showAlertOnComplete", False)
    fp.set_preference('browser.helperApps.neverAsk.saveToDisk','application/zip,application/octet-stream,application/x-zip-compressed,multipart/x-zip,application/x-rar-compressed, application/octet-stream,application/msword,application/vnd.ms-word.document.macroEnabled.12,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/vnd.ms-excel,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,application/rtf,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,application/vnd.ms-excel,application/vnd.ms-word.document.macroEnabled.12,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/xls,application/msword,text/csv,application/vnd.ms-excel.sheet.binary.macroEnabled.12,text/plain,text/csv/xls/xlsb,application/csv,application/download,application/vnd.openxmlformats-officedocument.presentationml.presentation,application/octet-stream')
    fp.set_preference("browser.download.dir", download_dir)
    options = Options()
    options.headless = True
    return webdriver.Firefox(firefox_profile=fp, executable_path=driver_dir, options=options)


def chrome_driver(download_dir, driver_dir):
    options = webdriver.ChromeOptions()
    #options.binary_location=str(os.environ.get('GOOGLE_CHROME_BIN')) # REQUIRED FOR HEROKU
    #options.add_argument('--disable-gpu') # REQUIRED FOR HEROKU
    #options.add_argument('--no-sandbox') # REQUIRED FOR HEROKU
    options.add_argument("browser.download.folderList=2");
    options.add_argument("browser.helperApps.alwaysAsk.force=False");
    options.add_argument("browser.download.manager.showWhenStarting=False");
    options.add_argument("browser.download.manager.showAlertOnComplete=False");
    options.add_argument("browser.helperApps.neverAsk.saveToDisk=True");
    options.add_argument(f"browser.download.dir={download_dir}");
    options.add_argument('--no-proxy-server');
    options.add_argument("--proxy-server='direct://'");
    options.add_argument("--proxy-bypass-list=*");
    options.headless = True
    return webdriver.Chrome(chrome_options=options, executable_path=driver_dir)
    #return webdriver.Chrome(chrome_options=options, executable_path=CHROMEDRIVER_PATH) # REQUIRED FOR HEROKU


download_dir = os.getcwd() + '\\data\\'
geckodriver_dir = './tools/geckodriver.exe'
chromedriver_dir = './tools/chromedriver.exe'


def download_files(url, driver_engine, download_dir=download_dir):
	# driver_engine could be "firefox" or "chrome"
    import time

    if driver_engine == "firefox":
        driver_dir = geckodriver_dir
        driver = firefox_driver(download_dir, driver_dir)
    else:
        driver_dir = chromedriver_dir
        driver = chrome_driver(download_dir, driver_dir)
        # the lines below are specific for Chrome - otherwise it fails to download a file when ran in headless mode
        driver.command_executor._commands["send_command"] = ("POST", '/session/$sessionId/chromium/send_command')
        params = {'cmd': 'Page.setDownloadBehavior', 'params': {'behavior': 'allow', 'downloadPath': download_dir}}
        command_result = driver.execute("send_command", params)

    driver.get(url)
    time.sleep(2)
    download_button = driver.find_element_by_xpath('//button[@name="download"]')
    download_button.click()
    time.sleep(2)
    file_description = driver.find_element_by_xpath("//div[@class='col-xs-12 p-l-r-none']//h2[1]").text
    print(f'The most recent file "{file_description}" was downloaded.')
    time.sleep(1)
    driver.quit()