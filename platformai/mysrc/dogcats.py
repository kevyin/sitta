#!/usr/bin/python

__author__ = 'kevin'

from optparse import OptionParser
import os, sys
import logging, datetime
from utils import *
from vgg16 import Vgg16
DATA_HOME_DIR = os.getcwd()
OUTPUT_PREFIX = os.path.join(os.getcwd(), 'output')

DATE_STR = datetime.datetime.today().strftime("%Y_%m%d_%H%M%S")

#####################################################################################
# MAIN
def main():
    """Script Entry Point"""

    usage =  "\n\n%prog [options] DATA_HOME OUTPUT_DIR"
    parser = OptionParser(usage=usage)
    parser.add_option("", "--run",
                      dest="run", action='store_true', default=False,
                      help="Really run the changes, not a dry run")
    parser.add_option("-b", "--batch_size",
                      dest="batch_size", default=64,
                      help="batch_size [64]")
    parser.add_option("-e", "--epochs",
                      dest="epochs", default=1,
                      help="epochs [1]")
    parser.add_option("-r", "--learning_rate",
                      dest="learning_rate", default=0.01,
                      help="learning rate [0.01]")
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
    global OUTPUT_PREFIX
    OUTPUT_PREFIX = args[1]


    def vgg(data_home, batch_size):
        test_path = os.path.join(DATA_HOME_DIR, 'test')
        results_path = os.path.join(DATA_HOME_DIR, 'results')
        train_path = os.path.join(DATA_HOME_DIR, 'train')
        valid_path = os.path.join(DATA_HOME_DIR, 'valid')

        vgg = Vgg16()
        batches = vgg.get_batches(train_path, batch_size=batch_size)
        val_batches = vgg.get_batches(valid_path, batch_size=batch_size)

        return (vgg, batches, val_batches)

    (vgg, batches, val_batches) = vgg(DATA_HOME_DIR, opts.batch_size)
    vgg.finetune(batches)
    vgg.model.optimizer.lr = opts.learning_rate

    # run epochs and save weights at each
    for epoch in range(opts.epochs):
        logging.info('Running epoch %d' % epoch)
        vgg.fit(batches, val_batches, nb_epoch=1)
        latest_weights_file = OUTPUT_PREFIX + DATA_HOME_DIR + str(epoch) + '.h5'
        vgg.model.save_weights(latest_weights_file)
    logging.info('Complete %s fit operations' % opts.epochs)








###############################################################################
if __name__ == "__main__":
    main()
