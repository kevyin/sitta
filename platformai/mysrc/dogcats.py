#!/usr/bin/python

__author__ = 'kevin'

from optparse import OptionParser
import os, sys
import logging, datetime
from utils import *
from vgg16 import Vgg16
from keras.callbacks import CSVLogger
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
    parser.add_option("-k", "--kaggle_submission",
                      dest="kaggle_submission", default=None,
                      help="kaggle submission file")
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

        return (vgg, batches, val_batches, test_path, results_path, train_path, valid_path, batch_size)

    (vgg, batches, val_batches,
     test_path, results_path,
     train_path, valid_path, batch_size) = vgg(DATA_HOME_DIR, opts.batch_size)
    vgg.finetune(batches)
    vgg.model.optimizer.lr = opts.learning_rate


    # run epochs and save weights at each
    for epoch in range(int(opts.epochs)):
        prefix = OUTPUT_PREFIX + DATE_STR + '-' + str(epoch)
        logging.info('Running epoch %s' % prefix)

        training_log = prefix + '.log.csv'
        csv_logger = CSVLogger(training_log, separator=',', append=True)
        callbacks = [csv_logger]

        vgg.fit(batches, val_batches, nb_epoch=1, callbacks=callbacks)
        latest_weights_file = prefix + '.h5'
        vgg.model.save_weights(latest_weights_file)
    logging.info('Complete %s fit operations' % opts.epochs)


    ## validate predictions
    # vgg.model.load_weights(latest_weights_file)
    # val_batches, probs = vgg.test(valid_path, batch_size = batch_size)
    # expected_labels = val_batches.classes #0 or 1
    #
    # # Round our predictions to 0/1 to generate labels
    # our_predictions = probs[:,0]
    # our_labels = np.round(1-our_predictions)
    #
    # from sklearn.metrics import confusion_matrix
    # cm = confusion_matrix(expected_labels, our_labels)
    # plot_confusion_matrix(cm, val_batches.class_indices)

    ## Format to Kaggle

    if opts.kaggle_submission:
        ## Generate predictions
        batches, preds = vgg.test(test_path, batch_size=batch_size*2)
        filenames = batches.filenames[:5]
        #Save our test results arrays so we can use them again later
        save_array(results_path + 'test_preds.dat', preds)
        save_array(results_path + 'filenames.dat', filenames)

        isdog = preds[:,1]
        isdog = isdog.clip(min=0.05, max=0.95)

        ids = np.array([int(f[8:f.find('.')]) for f in filenames])
        subm = np.stack([ids,isdog], axis=1)

        np.savetxt(opts.kaggle_submission, subm, fmt='%d,%.5f', header='id,label', comments='')







###############################################################################
if __name__ == "__main__":
    main()
