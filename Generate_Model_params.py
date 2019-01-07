import sys
import os
sys.path.append('../nmt-keras')
sys.path.append('../nmt-keras/nmt_keras')

import utils 
from config import load_parameters
from keras_wrapper.extra.read_write import dict2pkl

if not os.path.exists("model_params"):
    os.makedirs("model_params")

def make_params(model_name, model_size):
	params = load_parameters()
	params['INPUT_VOCABULARY_SIZE'] = 30000
	params['OUTPUT_VOCABULARY_SIZE'] = 30000

	params['SOURCE_TEXT_EMBEDDING_SIZE'] = 300
	params['TARGET_TEXT_EMBEDDING_SIZE'] = 300

	params['SRC_PRETRAINED_VECTORS'] = '../Google_w2v.npy'
	params['TRG_PRETRAINED_VECTORS'] = '../Google_w2v.npy'

	params['ENCODER_RNN_TYPE'] = 'GRU'
	params['DECODER_RNN_TYPE'] = 'GRU'

	params['N_LAYERS_ENCODER'] = 2
	params['N_LAYERS_DECODER'] = 2
	params['ENCODER_HIDDEN_SIZE'] = model_size
	params['DECODER_HIDDEN_SIZE'] = model_size
	params['MODEL_SIZE'] = model_size
	params['ATTENTION_SIZE'] = model_size

	params['SRC_PRETRAINED_VECTORS_TRAINABLE'] = True
	params['TRG_PRETRAINED_VECTORS_TRAINABLE'] = True 
	dict2pkl(params, "model_params/"+model_name)

make_params("Ross_M3", 512)