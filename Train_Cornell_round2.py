import sys
sys.path.append('../nmt-keras')
sys.path.append('../nmt-keras/nmt_keras')

import utils 
from config import load_parameters
from data_engine.prepare_data import keep_n_captions
from keras_wrapper.cnn_model import loadModel
from keras_wrapper.dataset import loadDataset
from keras_wrapper.utils import decode_predictions_beam_search
from model_zoo import TranslationModel


Cornell_Rd2 = loadDataset('query_to_reply/Dataset_Cornell_Rd2.pkl')
model = loadModel('trained_models/Dec_26', 1)

training_params = {'n_epochs': 1, 'batch_size': 80,'maxlen': 30, 'epochs_for_save': 5, 'verbose': 1, 'eval_on_sets': [], 'reload_epoch': 0, 'epoch_offset': 1}

nmt_model.trainNet(ds, training_params)
