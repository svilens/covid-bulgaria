import logging

def create_logger(filename, loglevel=20, log_to_file=True, log_printout=True):
    """
    Logger that prints out messages with timestamps to a file and/or to the console.

    Parameters
    ----------
    filename : str
        Name of the output file.
    loglevel : int, optional
        Logging level (DEBUG=10, INFO=20, WARN=30, ERROR=40). The default is 20.
    log_to_file : boolean, optional
        An indication whether to record the message to a file on the disk. The default is True.
    log_printout : TYPE, optional
        An indication whether to print out the message to the console. The default is True.

    Returns
    -------
    logger : logging.Logger
        A logger object

    Example
    -------
    my_logger = create_logger('filename.log')
    my_logger.info('My message')

    """
    logger = logging.getLogger(filename)
    formatter = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")

    if log_to_file:
        file_handler = logging.FileHandler(filename, mode='a')
        file_handler.setFormatter(formatter)
    if log_printout:
        printout_handler = logging.StreamHandler()
        printout_handler.setFormatter(formatter)

    if not logger.hasHandlers():
        logger.setLevel(loglevel)
        if log_to_file: logger.addHandler(file_handler)
        if log_printout: logger.addHandler(printout_handler)
        logger.handler_set = True

    return logger
