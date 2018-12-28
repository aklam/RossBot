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

params = load_parameters()
params['INPUT_VOCABULARY_SIZE'] = 30000
params['OUTPUT_VOCABULARY_SIZE'] = 30000

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
params['REBUILD_DATASET'] = False

nmt_model = TranslationModel(params, 
	model_type='GroundHogModel',
    weights_path='trained_models/Dec_27/epoch_1_init.h5',
	model_name='Dec_27',
	vocabularies=Cornell_Rd2.vocabulary,
	store_path='trained_models/Dec_27_v1/',
	verbose=True)

inputMapping = dict()
for i, id_in in enumerate(params['INPUTS_IDS_DATASET']):
    pos_source = Cornell_Rd2.ids_inputs.index(id_in)
    id_dest = nmt_model.ids_inputs[i]
    inputMapping[id_dest] = pos_source

nmt_model.setInputsMapping(inputMapping)
outputMapping = dict()
for i, id_out in enumerate(params['OUTPUTS_IDS_DATASET']):
    pos_target = Cornell_Rd2.ids_outputs.index(id_out)
    id_dest = nmt_model.ids_outputs[i]
    outputMapping[id_dest] = pos_target
nmt_model.setOutputsMapping(outputMapping)


training_params = {'n_epochs': 10, 'batch_size': 20,'maxlen': 30, 'epochs_for_save': 5, 'verbose': 1, 'eval_on_sets': [], 'reload_epoch': 1, 'epoch_offset': 1}

nmt_model.trainNet(Cornell_Rd2, training_params)
