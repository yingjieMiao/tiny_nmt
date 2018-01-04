import collections
import string
import numpy as np

import tensorflow as tf

class BatchedInput(
    collections.namedtuple("BatchedInput",
                           ("initializer", "source", "target_input",
                            "target_output", "source_sequence_length",
                            "target_sequence_length"))):
  pass


def get_infer_iterator(src_dataset, src_vocab_table, batch_size, eos):
    src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(eos)), tf.int32)
    src_dataset = (src_dataset
                   .map(lambda src: tf.string_split([src]).values)
                   .map(lambda src: tf.cast(src_vocab_table.lookup(src), tf.int32))
                   .map(lambda src: (src, tf.size(src))))
    batched = src_dataset.padded_batch(
        batch_size,
        padded_shapes=(tf.TensorShape([None]), tf.TensorShape([])),
        padding_values=(src_eos_id, 0)
    )
    batched_iter = batched.make_initializable_iterator()
    (src_ids, src_seq_len) = batched_iter.get_next()
    return BatchedInput(
        initializer=batched_iter.initializer,
        source=src_ids,
        target_input=None,
        target_output=None,
        source_sequence_length=src_seq_len,
        target_sequence_length=None
    )
    

def get_iterator(src_dataset,
                 tgt_dataset,
                 src_vocab_table,
                 tgt_vocab_table,
                 batch_size,
                 sos,
                 eos,
                 random_seed=1,
                 num_parallel_calls=4,
                 output_buffer_size=None,
                 reshuffle_each_iteration=True):
  if not output_buffer_size:
    output_buffer_size = batch_size * 1000
  src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(eos)), tf.int32)
  tgt_sos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(sos)), tf.int32)
  tgt_eos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(eos)), tf.int32)

  src_tgt_dataset = (tf.data.Dataset
                     .zip((src_dataset, tgt_dataset))
                     .shuffle(output_buffer_size, random_seed, reshuffle_each_iteration))

  src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt: (
          tf.string_split([src]).values, tf.string_split([tgt]).values),
      num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

  src_tgt_dataset = src_tgt_dataset.filter(
      lambda src, tgt: tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0))

  src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt: (tf.cast(src_vocab_table.lookup(src), tf.int32),
                        tf.cast(tgt_vocab_table.lookup(tgt), tf.int32)),
      num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

  src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt: (src,
                        tf.concat(([tgt_sos_id], tgt), 0),
                        tf.concat((tgt, [tgt_eos_id]), 0)),
      num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

  src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt_in, tgt_out: (
          src, tgt_in, tgt_out, tf.size(src), tf.size(tgt_in)),
      num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

  def batching_func(x):
    return x.padded_batch(
        batch_size,
        padded_shapes=(
            tf.TensorShape([None]),  # src
            tf.TensorShape([None]),  # tgt_input
            tf.TensorShape([None]),  # tgt_output
            tf.TensorShape([]),  # src_len
            tf.TensorShape([])),  # tgt_len
        padding_values=(
            src_eos_id,  # src
            tgt_eos_id,  # tgt_input
            tgt_eos_id,  # tgt_output
            0,  # src_len -- unused
            0))  # tgt_len -- unused

  batched_dataset = batching_func(src_tgt_dataset)
  batched_iter = batched_dataset.make_initializable_iterator()
  (src_ids, tgt_input_ids, tgt_output_ids, src_seq_len,
   tgt_seq_len) = (batched_iter.get_next())
  return BatchedInput(
      initializer=batched_iter.initializer,
      source=src_ids,
      target_input=tgt_input_ids,
      target_output=tgt_output_ids,
      source_sequence_length=src_seq_len,
      target_sequence_length=tgt_seq_len)
  
# Create training or inference dataset.
def create_src_tgt_lists(n_samples, min_seq_len=1, max_seq_len=10):
    src_voc = [str(i+1) for i in range(26)] + ["eos", "sos"]
    tgt_voc = [string.ascii_lowercase[i] for i in range(26)] + ["eos", "sos"]
    
    seq_lens = np.random.randint(min_seq_len, max_seq_len+1, n_samples)
    src_matrix = np.random.randint(0, 26, (n_samples, max_seq_len))
    
    src_dataset = [''] * n_samples
    tgt_dataset = [''] * n_samples
    for i in range(n_samples):
        seq = src_matrix[i, 0:seq_lens[i]].tolist()
        src_dataset[i] = ' '.join(src_voc[x] for x in seq)
        tgt_dataset[i] = ' '.join(tgt_voc[x] for x in seq)
    return src_voc, tgt_voc, src_dataset, tgt_dataset
