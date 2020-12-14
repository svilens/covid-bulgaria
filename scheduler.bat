@ECHO OFF 
TITLE Execute Python script on Anaconda environment
ECHO Please Wait...
:: Section 1: Activate the environment.
ECHO ============================
ECHO Activate Conda
ECHO ============================
@CALL "F:\Anaconda\Scripts\activate.bat" TestEnvironment
:: Section 2: Execute python script.
ECHO ============================
ECHO Run Python script
ECHO ============================
python D:\Git\github\smule-heroku\app_processing.py

ECHO ============================
ECHO End
ECHO ============================

PAUSE