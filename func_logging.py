import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from logging import FileHandler

LOG_FILE = "app.log"
FORMATTER = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")

def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler

def get_file_handler():
    #file_handler = TimedRotatingFileHandler(LOG_FILE, when='midnight')
    file_handler = FileHandler(LOG_FILE, mode='a')
    file_handler.setFormatter(FORMATTER)
    return file_handler

def get_logger(logger_name):
    logger = logging.getLogger(logger_name)
    #if not getattr(logger, 'handler_set', None):
    logger.setLevel(logging.DEBUG) # better to have too much log than not enough
    if not logger.hasHandlers():
        logger.addHandler(get_console_handler())
        logger.addHandler(get_file_handler())
        # with this pattern, it's rarely necessary to propagate the error up to parent
    logger.propagate = False
 #   logger.handler_set = True
    return logger



def get_logger_app(self):
    loglevel = logging.INFO
    l = logging.getLogger('app.log')
    if not l.hasHandlers():
        l.setLevel(loglevel)
        h = logging.StreamHandler()
        f = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        h.setFormatter(f)
        l.addHandler(h)
        l.addHandler((get_file_handler()))
        l.setLevel(loglevel)
        l.handler_set = True
    return l