import sys
sys.path.append('../nmt-keras')
sys.path.append('../nmt-keras/nmt_keras')

import utils 
from config import load_parameters
from data_engine.prepare_data import keep_n_captions

from keras_wrapper.cnn_model import loadModel
from keras_wrapper.dataset import Dataset, saveDataset, loadDataset
from keras_wrapper.utils import decode_predictions_beam_search

from model_zoo import TranslationModel

def make_Cornell_round2():
    ds = Dataset('Cornell', '/data')

    ds.setInput('data/Cornell_train_2_query.en',
        'train',
        type='text',
        id='source_text',
        pad_on_batch=True,
        tokenization='tokenize_basic',
        build_vocabulary=True,
        fill='end',
        max_text_len=30,
        max_words=30000,
        min_occ=0)

    ds.setInput('data/Cornell_valid_2_query.en',
        'val',
        type='text',
        id='source_text',
        pad_on_batch=True,
        tokenization='tokenize_basic',
        fill='end',
        max_text_len=30,
        max_words=0)

    ds.setInput('data/Cornell_train_2_reply.en',
        'train',
        type='text',
        id='state_below',
        required=False,
        tokenization='tokenize_basic',
        pad_on_batch=True,
        build_vocabulary='target_text',
        offset=1,
        fill='end',
        max_text_len=30,
        max_words=30000)

    ds.setInput(None, 
        'val',
        type='ghost',
        id='state_below',
        pad_on_batch=True,
        required=False)

    ds.setOutput('data/Cornell_train_2_reply.en',
        'train',
        type='text',
        id='target_text',
        tokenization='tokenize_basic',
        build_vocabulary=True,
        pad_on_batch=True,
        sample_weights=True,
        max_text_len=30,
        max_words=30000,
        min_occ=0)
    
    ds.setOutput('data/Cornell_valid_2_reply.en',
        'val',
        type='text',
        id='target_text',
        pad_on_batch=True,
        tokenization='tokenize_basic',
        sample_weights=True,
        max_text_len=30,
        max_words=0)

    keep_n_captions(ds, repeat=1, n=1, set_names=['val'])

    saveDataset(ds, 'query_to_reply')


def make_Ross_round2():
    ds = Dataset("Ross", '/data')

    ds.setInput('data/Ross_train_query.en',
        'train',
        type='text',
        id='source_text',
        pad_on_batch=True,
        tokenization='tokenize_basic',
        build_vocabulary=True,
        fill='end',
        max_text_len=30,
        max_words=30000,
        min_occ=0)

    ds.setInput('data/Ross_valid_query.en',
        'val',
        type='text',
        id='source_text',
        pad_on_batch=True,
        tokenization='tokenize_basic',
        fill='end',
        max_text_len=30,
        max_words=0)

    ds.setInput('data/Ross_train_reply.en',
        'train',
        type='text',
        id='state_below',
        required=False,
        tokenization='tokenize_basic',
        pad_on_batch=True,
        build_vocabulary='target_text',
        offset=1,
        fill='end',
        max_text_len=30,
        max_words=30000)

    ds.setInput(None, 
        'val',
        type='ghost',
        id='state_below',
        pad_on_batch=True,
        required=False)

    ds.setOutput('data/Ross_train_reply.en',
        'train',
        type='text',
        id='target_text',
        tokenization='tokenize_basic',
        build_vocabulary=True,
        pad_on_batch=True,
        sample_weights=True,
        max_text_len=30,
        max_words=30000,
        min_occ=0)
    
    ds.setOutput('data/Ross_valid_reply.en',
        'val',
        type='text',
        id='target_text',
        pad_on_batch=True,
        tokenization='tokenize_basic',
        sample_weights=True,
        max_text_len=30,
        max_words=0)

    keep_n_captions(ds, repeat=1, n=1, set_names=['val'])

    ds.merge_vocabularies(['round_1_source','round_2_source'])

    saveDataset(ds, 'query_to_reply')

make_Cornell_round2()
make_Ross_round2()