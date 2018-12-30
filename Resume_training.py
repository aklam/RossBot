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

# For the model params, INPUT_VOCABULARY_SIZE and OUTPUT_VOCABULARY_SIZE should I set that equal to the size of the new or the old dataset?

# Should the vocabularies argument in TranslationModel be set to the new dataset vocabulary or the old dataset vocabulary?

#params['SOURCE_TEXT_EMBEDDING_SIZE'] = 300
#params['TARGET_TEXT_EMBEDDING_SIZE'] = 300
#
#params['SRC_PRETRAINED_VECTORS'] = '../Google_w2v.npy'
#params['TRG_PRETRAINED_VECTORS'] = '../Google_w2v.npy'
#
#
##Model parameters
#params['ENCODER_RNN_TYPE'] = 'GRU'
#params['DECODER_RNN_TYPE'] = 'GRU'
#
#params['N_LAYERS_ENCODER'] = 2
#params['N_LAYERS_DECODER'] = 2
#params['ENCODER_HIDDEN_SIZE'] = 512
#params['DECODER_HIDDEN_SIZE'] = 512
#params['MODEL_SIZE'] = 512
#params['SRC_PRETRAINED_VECTORS_TRAINABLE'] = False
#params['TRG_PRETRAINED_VECTORS_TRAINABLE'] = False 
#
##params['SKIP_VECTORS_HIDDEN_SIZE'] = 512
#params['ATTENTION_SIZE'] = 512

params['RELOAD'] = 1
params['RELOAD_EPOCH'] = True
params['REBUILD_DATASET'] = False
params['DATA_ROOT_PATH'] = 'data/'

# This is the new data that I want to train on. Do I need to make a new keras_wrapper.dataset Dataset instance?
params['SRC_LAN'] = 'query'
params['TRG_LAN'] = 'reply'
params['TEXT_FILES'] = {'train': 'Cornell_train_2.', 'val': 'Cornell_valid_2.'}
params['TOKENIZATION_METHOD'] = 'tokenize_basic'
params['INPUT_VOCABULARY_SIZE'] = ds.vocabulary_len[params['INPUTS_IDS_DATASET'][0]]
params['OUTPUT_VOCABULARY_SIZE'] = ds.vocabulary_len[params['OUTPUTS_IDS_DATASET'][0]]


ds_2 = update_dataset_from_file(ds=ds, 
    input_text_filename='data/Cornell_train_2.query',
    params=params,
    splits=['train'],
    output_text_filename='data/Cornell_train_2.reply',
    compute_state_below=True,
    recompute_references=False)


nmt_model = TranslationModel(params, 
    model_type='GroundHogModel',
    weights_path='trained_models/Vanilla_model_resume_test/epoch_1_init.h5',
    model_name='Vanilla_model_resumed',
    vocabularies=ds_2.vocabulary,
    store_path='trained_models/Vanilla_model_resumed/',
    verbose=True)

inputMapping = dict()
for i, id_in in enumerate(params['INPUTS_IDS_DATASET']):
    pos_source = ds_2.ids_inputs.index(id_in)
    id_dest = nmt_model.ids_inputs[i]
    inputMapping[id_dest] = pos_source

nmt_model.setInputsMapping(inputMapping)
outputMapping = dict()
for i, id_out in enumerate(params['OUTPUTS_IDS_DATASET']):
    pos_target = ds_2.ids_outputs.index(id_out)
    id_dest = nmt_model.ids_outputs[i]
    outputMapping[id_dest] = pos_target
nmt_model.setOutputsMapping(outputMapping)

training_params = {'n_epochs': 3, 'batch_size': 20,'maxlen': 30, 'epochs_for_save': 1, 'verbose': 1, 'eval_on_sets': [], 'reload_epoch': 1, 'epoch_offset': 1}

print(ds_2)

nmt_model.trainNet(ds_2, training_params)
