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


Cornell_Rd2 = loadDataset('query_to_reply/Dataset_Cornell.pkl')

params = load_parameters()
params['INPUT_VOCABULARY_SIZE'] = Cornell_Rd2.vocabulary_len[params['INPUTS_IDS_DATASET'][0]]
params['OUTPUT_VOCABULARY_SIZE'] = Cornell_Rd2.vocabulary_len[params['OUTPUTS_IDS_DATASET'][0]]

params['SOURCE_TEXT_EMBEDDING_SIZE'] = 300
params['TARGET_TEXT_EMBEDDING_SIZE'] = 300

params['SRC_PRETRAINED_VECTORS'] = '../Google_w2v.npy'
params['TRG_PRETRAINED_VECTORS'] = '../Google_w2v.npy'


#Model parameters
params['ENCODER_RNN_TYPE'] = 'GRU'
params['DECODER_RNN_TYPE'] = 'GRU'

#params['N_LAYERS_ENCODER'] = 2
#params['N_LAYERS_DECODER'] = 2
params['ENCODER_HIDDEN_SIZE'] = 512
params['DECODER_HIDDEN_SIZE'] = 512
params['MODEL_SIZE'] = 512
params['SRC_PRETRAINED_VECTORS_TRAINABLE'] = False
params['TRG_PRETRAINED_VECTORS_TRAINABLE'] = False 

#params['SKIP_VECTORS_HIDDEN_SIZE'] = 512
params['ATTENTION_SIZE'] = 512

params['RELOAD'] = 1
params['RELOAD_EPOCH'] = True
params['REBUILD_DATASET'] = True
params['DATA_ROOT_PATH'] = 'query_to_reply/Dataset_Cornell.pkl'

nmt_model = TranslationModel(params, 
	model_type='GroundHogModel',
    weights_path='trained_models/Dec_27/epoch_1_init.h5',
	model_name='Dec_27',
	vocabularies=Cornell_Rd2.vocabulary,
	store_path='trained_models/Dec_27_v1/',
	verbose=True)


training_params = {'n_epochs': 1, 'batch_size': 80,'maxlen': 30, 'epochs_for_save': 5, 'verbose': 1, 'eval_on_sets': [], 'reload_epoch': 1, 'epoch_offset': 1}

model.trainNet(Cornell_Rd2, training_params)
