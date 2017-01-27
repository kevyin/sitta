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


def get_ll_layers():
    return [
        BatchNormalization(input_shape=(4096,)),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ]


def get_conv_model(model):
    layers = model.layers
    last_conv_idx = [index for index,layer in enumerate(layers)
                     if type(layer) is Convolution2D][-1]

    conv_layers = layers[:last_conv_idx+1]
    conv_model = Sequential(conv_layers)
    fc_layers = layers[last_conv_idx+1:]
    return conv_model, fc_layers, last_conv_idx


def get_fc_layers(p, in_shape):
    return [
        MaxPooling2D(input_shape=in_shape),
        Flatten(),
        Dense(4096, activation='relu'),
        BatchNormalization(),
        Dropout(p),
        Dense(4096, activation='relu'),
        BatchNormalization(),
        Dropout(p),
        Dense(2, activation='softmax')
    ]


# def my_vgg(data_home, batch_size):

    # return (vgg_, batches, val_batches, test_path, results_path, train_path, valid_path, batch_size,
    #         trn_features, val_features)

def train_last_layer(i, ll_trn_feat, ll_val_feat, trn_labels, val_labels, model_path, callbacks):
    ll_layers = get_ll_layers()
    ll_model = Sequential(ll_layers)
    ll_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    ll_model.optimizer.lr=1e-5
    ll_model.fit(ll_trn_feat, trn_labels, validation_data=(ll_val_feat, val_labels), nb_epoch=12, callbacks=callbacks)
    ll_model.optimizer.lr=1e-7
    ll_model.fit(ll_trn_feat, trn_labels, validation_data=(ll_val_feat, val_labels), nb_epoch=1, callbacks=callbacks)
    ll_model.save_weights(model_path+'ll_bn' + i + '.h5')

    vgg = Vgg16()
    model = vgg.model
    model.pop(); model.pop(); model.pop()
    for layer in model.layers: layer.trainable=False
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    ll_layers = get_ll_layers()
    for layer in ll_layers: model.add(layer)
    for l1,l2 in zip(ll_model.layers, model.layers[-3:]):
        l2.set_weights(l1.get_weights())
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    model.save_weights(model_path+'bn' + i + '.h5')
    return model


def train_dense_layers(i, model, trn, val, trn_features, val_features, trn_labels, val_labels, batch_size, model_path, callbacks):
    conv_model, fc_layers, last_conv_idx = get_conv_model(model)
    conv_shape = conv_model.output_shape[1:]
    fc_model = Sequential(get_fc_layers(0.5, conv_shape))
    for l1,l2 in zip(fc_model.layers, fc_layers):
        weights = l2.get_weights()
        l1.set_weights(weights)
    fc_model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy',
                     metrics=['accuracy'])
    fc_model.fit(trn_features, trn_labels, nb_epoch=2,
                 batch_size=batch_size, validation_data=(val_features, val_labels), callbacks=callbacks)

    gen = image.ImageDataGenerator(rotation_range=10, width_shift_range=0.05,
                                   width_zoom_range=0.05, zoom_range=0.05,
                                   channel_shift_range=10, height_shift_range=0.05, shear_range=0.05, horizontal_flip=True)
    batches = gen.flow(trn, trn_labels, batch_size=batch_size)
    val_batches = image.ImageDataGenerator().flow(val, val_labels,
                                                  shuffle=False, batch_size=batch_size)

    for layer in conv_model.layers: layer.trainable = False
    for layer in get_fc_layers(0.5, conv_shape): conv_model.add(layer)
    for l1,l2 in zip(conv_model.layers[last_conv_idx+1:], fc_model.layers):
        l1.set_weights(l2.get_weights())

    conv_model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy',
                       metrics=['accuracy'])
    conv_model.save_weights(model_path+'no_dropout_bn' + i + '.h5')
    conv_model.fit_generator(batches, samples_per_epoch=batches.N, nb_epoch=1,
                             validation_data=val_batches, nb_val_samples=val_batches.N, callbacks=callbacks)
    for layer in conv_model.layers[16:]: layer.trainable = True
    conv_model.fit_generator(batches, samples_per_epoch=batches.N, nb_epoch=8,
                             validation_data=val_batches, nb_val_samples=val_batches.N, callbacks=callbacks)

    conv_model.optimizer.lr = 1e-7
    conv_model.fit_generator(batches, samples_per_epoch=batches.N, nb_epoch=10,
                             validation_data=val_batches, nb_val_samples=val_batches.N, callbacks=callbacks)
    conv_model.save_weights(model_path + 'aug' + i + '.h5')




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
    parser.add_option("-w", "--weights",
                      dest="weights", default=None,
                      help="Load previous weights")
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

    data_home = DATA_HOME_DIR
    batch_size = opts.batch_size

    test_path = os.path.join(data_home, 'test/')
    results_path = os.path.join(data_home, 'results/')
    train_path = os.path.join(data_home, 'train/')
    valid_path = os.path.join(data_home, 'valid/')
    model_path = os.path.join(data_home, 'models/')
    if not os.path.exists(model_path): os.mkdir(model_path)

    batches = get_batches(train_path, shuffle=False, batch_size=batch_size)
    val_batches = get_batches(valid_path, shuffle=False, batch_size=batch_size)

    (val_classes, trn_classes, val_labels, trn_labels,
     val_filenames, filenames, test_filenames) = get_classes(data_home +'/')

    vgg_ = Vgg16()
    model = vgg_.model
    print "Initial model"
    print model.summary()
    conv_layers,fc_layers = split_at(model, Convolution2D)
    conv_model = Sequential(conv_layers)

    train_convlayer_features_bc= os.path.join(model_path, 'train_convlayer_features.bc')
    valid_convlayer_features_bc= os.path.join(model_path, 'valid_convlayer_features.bc')

    # precompute conv layers
    if os.path.exists(train_convlayer_features_bc) and os.path.exists(valid_convlayer_features_bc):
        trn_features = load_array(train_convlayer_features_bc)
        val_features = load_array(valid_convlayer_features_bc)
    else:
        val_features = conv_model.predict_generator(val_batches, val_batches.nb_sample)
        trn_features = conv_model.predict_generator(batches, batches.nb_sample)
        save_array(train_convlayer_features_bc, trn_features)
        save_array(valid_convlayer_features_bc, val_features)

    # precompute training and validation layers with image decoding and resizing already done

    train_data_bc = os.path.join(model_path, 'train_data.bc')
    valid_data_bc = os.path.join(model_path, 'valid_data.bc')

    if os.path.exists(train_data_bc) and os.path.exists(valid_data_bc):
        trn = load_array(train_data_bc)
        val = load_array(valid_data_bc)
    else:
        trn = get_data(train_path)
        val = get_data(valid_path)
        save_array(train_data_bc, trn)
        save_array(valid_data_bc, val)

    # precompute the output of all but the last dropout
    model.pop()
    model.pop()

    train_ll_feat_bc = os.path.join(model_path, 'train_ll_feat.bc')
    valid_ll_feat_bc = os.path.join(model_path, 'valid_ll_feat.bc')

    if os.path.exists(train_ll_feat_bc) and os.path.exists(valid_ll_feat_bc):
        ll_trn_feat = load_array(train_ll_feat_bc)
        ll_val_feat = load_array(valid_ll_feat_bc)
    else:
        ll_trn_feat = model.predict_generator(batches, batches.nb_sample)
        ll_val_feat = model.predict_generator(val_batches, val_batches.nb_sample)
        save_array(train_ll_feat_bc, ll_trn_feat)
        save_array(valid_ll_feat_bc, ll_val_feat)

    # precompute test data
    test_data_bc = os.path.join(model_path, 'test_data.bc')
    if os.path.exists(test_data_bc):
        test = load_array(test_data_bc)
    else:
        test = get_data(test_path)
        save_array(test_data_bc, test)

    ## Format to Kaggle
    if not opts.kaggle_submission:
        for i in range(5):
            prefix = OUTPUT_PREFIX + DATE_STR + '-' + str(i)
            logging.info('Running epoch %s' % prefix)

            training_log = prefix + '.log.csv'
            csv_logger = CSVLogger(training_log, separator=',', append=True)
            callbacks = [csv_logger]
            i = str(i)
            model = train_last_layer(i, ll_trn_feat, ll_val_feat, trn_labels, val_labels, model_path, callbacks)
            train_dense_layers(i, model, trn, val, trn_features, val_features, trn_labels, val_labels, batch_size, model_path, callbacks)
        # if opts.weights:
        #     vgg.model.load_weights(opts.weights)
        #
        # vgg.finetune(batches)
        # vgg.model.optimizer.lr = opts.learning_rate
        #
        #
        # # run epochs and save weights at each
        # for epoch in range(int(opts.epochs)):
        #     prefix = OUTPUT_PREFIX + DATE_STR + '-' + str(epoch)
        #     logging.info('Running epoch %s' % prefix)
        #
        #     training_log = prefix + '.log.csv'
        #     csv_logger = CSVLogger(training_log, separator=',', append=True)
        #     callbacks = [csv_logger]
        #
        #     vgg.fit(batches, val_batches, nb_epoch=1, callbacks=callbacks)
        #     latest_weights_file = prefix + '.h5'
        #     vgg.model.save_weights(latest_weights_file)
        # logging.info('Complete %s fit operations' % opts.epochs)


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

    else:
        ## Generate predictions
        prefix = opts.kaggle_submission

        # pred_dat = prefix + '.test_preds.dat'
        # filenames_dat = prefix + '.filenames.dat'
        #
        # if os.path.exists(pred_dat) and os.path.exists(filenames_dat):
        #     preds = load_array(pred_dat)
        #     filenames = load_array(filenames_dat)
        # else:
        #     vgg.model.load_weights(opts.kaggle_submission)
        #     batches, preds = vgg.test(test_path, batch_size=batch_size)
        #     filenames = batches.filenames
        #     #Save our test results arrays so we can use them again later
        #     save_array(pred_dat, preds)
        #     save_array(filenames_dat, filenames)
        #
        # isdog = preds[:,1]
        # isdog = isdog.clip(min=0.05, max=0.95)
        #
        # ids = np.array([int(f[8:f.find('.')]) for f in filenames])
        # subm = np.stack([ids,isdog], axis=1)
        #
        # subm_file = prefix + '.submission.csv'
        # np.savetxt(subm_file, subm, fmt='%d,%.5f', header='id,label', comments='')







###############################################################################
if __name__ == "__main__":
    main()
