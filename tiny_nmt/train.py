import collections
import os
import random
import string
import numpy as np

import tensorflow as tf
from tensorflow.python.ops import lookup_ops 

from model import NmtModel
from data_utils import get_iterator, get_infer_iterator, create_src_tgt_lists

TRAIN_SIZE = 200000
BATCH_SIZE = 2000
INFER_DATA_SIZE = 5000
N_EPOCH = 200
OUTDIR = '/tmp/tf/log/nmt/test/'

class TrainModel(
    collections.namedtuple("TrainModel", ("graph", "model", "iterator"))):
  pass


class InferModel(
    collections.namedtuple(
        "InferModel",
        ("graph", "model", "iterator", "src_placeholder", "batch_size_placeholder"))):
  pass


def create_train_model(hparams):
    src_voc, tgt_voc, src_data, tgt_data = create_src_tgt_lists(TRAIN_SIZE)
    print 'Training set created. Size = {}'.format(TRAIN_SIZE)
    g = tf.Graph()
    with g.as_default(), tf.container("train"):
        iterator = get_iterator(
            src_dataset=tf.data.Dataset.from_tensor_slices(tf.constant(src_data)),
            tgt_dataset=tf.data.Dataset.from_tensor_slices(tf.constant(tgt_data)),           
            src_vocab_table=lookup_ops.index_table_from_tensor(tf.constant(src_voc)),
            tgt_vocab_table=lookup_ops.index_table_from_tensor(tf.constant(tgt_voc)),
            batch_size=BATCH_SIZE,
            eos="eos",
            sos="sos"
        )        
        model = NmtModel(
            hparams, "train", iterator,
            src_vocab_table=lookup_ops.index_table_from_tensor(src_voc),
            tgt_vocab_table=lookup_ops.index_table_from_tensor(tgt_voc))
    return TrainModel(graph=g, model=model, iterator=iterator)


def create_infer_model(hparams):
    # Only need the vocab tables here. Sample size doesn't matter.
    src_voc, tgt_voc, _, _ = create_src_tgt_lists(100)
    
    g = tf.Graph()
    with g.as_default(), tf.container("infer"):
        src_vocab_table = lookup_ops.index_table_from_tensor(tf.constant(src_voc))
        tgt_vocab_table = lookup_ops.index_table_from_tensor(tf.constant(tgt_voc))
        reverse_tgt_vocab_table = lookup_ops.index_to_string_table_from_tensor(tf.constant(tgt_voc))
        
        src_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
        batch_size_placeholder = tf.placeholder(shape=[], dtype=tf.int64)
        src_dataset = tf.data.Dataset.from_tensor_slices(src_placeholder)
        iterator = get_infer_iterator(
            src_dataset,
            src_vocab_table,
            batch_size_placeholder, "eos")
        
        model = NmtModel(hparams, "infer", iterator,
                         src_vocab_table=src_vocab_table,
                         tgt_vocab_table=tgt_vocab_table,
                         reverse_tgt_vocab_table=reverse_tgt_vocab_table)
        
    return InferModel(graph=g, model=model, iterator=iterator,
                      src_placeholder=src_placeholder,
                      batch_size_placeholder=batch_size_placeholder)
 
       
def restore_model(model, sess):
    latest_ckpt = tf.train.latest_checkpoint(OUTDIR)
    model.saver.restore(sess, latest_ckpt)


def run_sample_decode(infer_model, infer_sess, infer_src_dataset, infer_tgt_dataset, sample_size):
    def _decode(infer_model, src, tgt, verbose=True):
        iter_feed_dict = {
            infer_model.src_placeholder: [src],
            infer_model.batch_size_placeholder: 1
        }
        infer_sess.run(infer_model.iterator.initializer, feed_dict=iter_feed_dict)
        sample_words = infer_model.model.decode(infer_sess)
        output = sample_words[0, :].tolist()
        if "eos" in output:
            output = output[:output.index("eos")]
        if verbose:
            print "src: ", src
            print "tgt: ", tgt
            print "nmt: ", " ".join(output)

        return " ".join(output)
    
    restore_model(infer_model.model, infer_sess)
    
    for i in range(sample_size):
        id = random.randint(0, len(infer_src_dataset) - 1)
        _decode(infer_model, infer_src_dataset[id], infer_tgt_dataset[id])
    
    outputs = []
    for i in range(26):
        outputs.append(_decode(infer_model, str(i+1), string.ascii_lowercase[i], verbose=False))
            
    print 'char prediction: {}'.format(', '.join(outputs))
    

def train(hparams):
    train_model = create_train_model(hparams)
    train_sess = tf.Session(graph=train_model.graph)
    
    infer_model = create_infer_model(hparams)
    infer_sess = tf.Session(graph=infer_model.graph)
    _, _, infer_src_dataset, infer_tgt_dataset = create_src_tgt_lists(INFER_DATA_SIZE)
    
    summary_writer = tf.summary.FileWriter(os.path.join(OUTDIR, 'train_log'), train_model.graph)
    
    print 'Init training and inference models...'
    with train_model.graph.as_default():
        train_sess.run(tf.global_variables_initializer())
        train_sess.run(tf.tables_initializer())        
        train_sess.run(train_model.iterator.initializer)
        global_step = train_sess.run(train_model.model.global_step)
    
    with infer_model.graph.as_default():
        infer_sess.run(tf.tables_initializer())
            
    print 'Start training.'    
    epoch = 0
    while epoch < N_EPOCH:
        try:
            _, train_summary, _, global_step_v = train_model.model.train(train_sess)
            summary_writer.add_summary(train_summary, global_step_v)
        except tf.errors.OutOfRangeError:
            epoch += 1
            print "\n", "* " * 10
            print "Done with epoch {}. Global step = {}".format(epoch, global_step_v)
            train_model.model.saver.save(
                train_sess, os.path.join(OUTDIR, 'translate.ckpt'), global_step=global_step)            
            run_sample_decode(
                infer_model, infer_sess, infer_src_dataset, infer_tgt_dataset, sample_size=2)
            train_sess.run(train_model.iterator.initializer)
            continue
        global_step = global_step_v
        
    train_model.model.saver.save(
        train_sess, os.path.join(OUTDIR, 'translate.ckpt'), global_step=global_step)
    summary_writer.close()
    
    
if __name__ == "__main__":
    np.random.seed(1)
    train(tf.contrib.training.HParams(
        src_voc_size=28,
        tgt_voc_size=28,
        src_emb_size=5,
        tgt_emb_size=5,
        num_units=28,
        num_layers=2,
        forget_bias=0.2,
        learning_rate=0.1,
        time_major=True
    ))
