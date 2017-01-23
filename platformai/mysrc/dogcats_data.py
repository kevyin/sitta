#!/usr/bin/python

__author__ = 'kevin'

from optparse import OptionParser
import os, sys
import logging
DATA_HOME_DIR = os.getcwd()
DATA_HOME_DIR = os.path.join(os.getcwd(), 'output')

#####################################################################################
# MAIN
def main():
    """Script Entry Point"""

    usage =  "\n\n%prog [options] DATA_HOME OUTPUT_DIR"
    parser = OptionParser(usage=usage)
    parser.add_option("", "--run",
                      dest="run", action='store_true', default=False,
                      help="Really run the changes, not a dry run")
    parser.add_option("-d", "--loglevel",
                      dest="loglevel", default="Info",
                      help="Logging level Debug|Info|Warning|Error")
    parser.add_option("-l", "--log",
                      dest="logfile", default="",
                      help="Logfile")
    (opts, args) = parser.parse_args()

    log_level = opts.loglevel
    if log_level == 'Debug':
        debug_mode = logging.DEBUG
    if log_level == 'Info':
        debug_mode = logging.INFO
    if log_level == 'Warning':
        debug_mode = logging.WARNING
    if log_level == 'Error':
        debug_mode = logging.ERROR

    logger = logging.getLogger()
    logger.setLevel(debug_mode)

    if opts.logfile is not "":
        log_file = opts.logfile
        hdlr = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr)

    logging.info(" ".join(sys.argv))

    global DATA_HOME_DIR
    DATA_HOME_DIR = args[0]
    global OUTPUT_DIR
    OUTPUT_DIR = args[1]

    from utils import *
    from vgg16 import Vgg16





###############################################################################
if __name__ == "__main__":
    main()
