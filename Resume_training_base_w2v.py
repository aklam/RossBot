import sys
sys.path.append('../nmt-keras')
sys.path.append('../nmt-keras/nmt_keras')

import utils 
from config import load_parameters
from data_engine.prepare_data import keep_n_captions, update_dataset_from_file
from keras_wrapper.cnn_model import loadModel
from keras_wrapper.dataset import loadDataset
from keras_wrapper.utils import decode_predictions_beam_search
from model_zoo import TranslationModel


#Dataset_Cornell_base is the large dataset as a keras_wrapper.dataset Dataset instance
ds = loadDataset('query_to_reply/Dataset_Cornell_base.pkl')

params = load_parameters()

params['SOURCE_TEXT_EMBEDDING_SIZE'] = 300
params['TARGET_TEXT_EMBEDDING_SIZE'] = 300

params['SRC_PRETRAINED_VECTORS'] = '../Google_w2v.npy'
params['TRG_PRETRAINED_VECTORS'] = '../Google_w2v.npy'


#Model parameters
params['ENCODER_RNN_TYPE'] = 'GRU'
params['DECODER_RNN_TYPE'] = 'GRU'

params['N_LAYERS_ENCODER'] = 2
params['N_LAYERS_DECODER'] = 2
params['ENCODER_HIDDEN_SIZE'] = 512
params['DECODER_HIDDEN_SIZE'] = 512
params['MODEL_SIZE'] = 512
params['SRC_PRETRAINED_VECTORS_TRAINABLE'] = True
params['TRG_PRETRAINED_VECTORS_TRAINABLE'] = True 

#params['SKIP_VECTORS_HIDDEN_SIZE'] = 512
params['ATTENTION_SIZE'] = 512

params['RELOAD'] = 10
params['RELOAD_EPOCH'] = True
params['REBUILD_DATASET'] = False
params['DATA_ROOT_PATH'] = 'data/'

# This is the new data that I want to train on. Do I need to make a new keras_wrapper.dataset Dataset instance?

params['TOKENIZATION_METHOD'] = 'tokenize_basic'

params['INPUT_VOCABULARY_SIZE'] = ds.vocabulary_len[params['INPUTS_IDS_DATASET'][0]]
params['OUTPUT_VOCABULARY_SIZE'] = ds.vocabulary_len[params['OUTPUTS_IDS_DATASET'][0]]

print(ds)

nmt_model = TranslationModel(params, 
    model_type='GroundHogModel',
    weights_path='trained_models/512/512_Base/epoch_10_init.h5',
    model_name='512_Trained_w2v_Base',
    vocabularies=ds.vocabulary,
    store_path='trained_models/512_Trained_w2v_Base/',
    verbose=True)

inputMapping = dict()
for i, id_in in enumerate(params['INPUTS_IDS_DATASET']):
    pos_source = ds.ids_inputs.index(id_in)
    id_dest = nmt_model.ids_inputs[i]
    inputMapping[id_dest] = pos_source

nmt_model.setInputsMapping(inputMapping)
outputMapping = dict()
for i, id_out in enumerate(params['OUTPUTS_IDS_DATASET']):
    pos_target = ds.ids_outputs.index(id_out)
    id_dest = nmt_model.ids_outputs[i]
    outputMapping[id_dest] = pos_target
nmt_model.setOutputsMapping(outputMapping)

training_params = {'n_epochs': 12, 'batch_size': 20,'maxlen': 30, 'epochs_for_save': 1, 'verbose': 1, 'eval_on_sets': [], 'reload_epoch': 10, 'epoch_offset': 10}

print(ds)

nmt_model.trainNet(ds, training_params)
