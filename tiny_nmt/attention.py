# Based on:
#  https://github.com/tensorflow/nmt/blob/master/nmt/attention_model.py    
#  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/seq2seq/python/ops/attention_wrapper.py

import collections

import tensorflow as tf

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import core as layers_core
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.util import nest

_zero_state_tensors = rnn_cell_impl._zero_state_tensors  # pylint: disable=protected-access

# `memory` has shape [B, T, units], assuming it is from encoder outputs.
def _prepare_memory(memory, memory_sequence_length):
    memory = ops.convert_to_tensor(memory, name="memory")
    memory_sequence_length = ops.convert_to_tensor(
        memory_sequence_length, name="memory_sequence_length")
    assert memory.shape[0].value == memory_sequence_length.shape[0].value
    
    seq_len_mask = array_ops.sequence_mask(
        memory_sequence_length,
        maxlen=array_ops.shape(nest.flatten(memory)[0])[1],
        dtype=tf.float32)
    seq_len_mask = array_ops.reshape(
                    seq_len_mask,
                    array_ops.concat((
                        array_ops.shape(seq_len_mask),
                        array_ops.ones(1, dtype=dtypes.int32)), 0))
    return memory * seq_len_mask

class LuongAttentionMechanism(object):
    def __init__(self, num_units, memory, memory_sequence_length):
        self.values = _prepare_memory(memory, memory_sequence_length)
        # The memory layer serves as a square matrix (which is a bilinear map
        # works on a pair of encoder output and decoder cell output).
        self.memory_layer = layers_core.Dense(
                num_units, name="memory_layer", use_bias=False, dtype=tf.float32)
        self.keys = self.memory_layer(self.values)
    
    # `query` is just cell output.
    def __call__(self, query):
        with tf.variable_scope("luong_attention"):
            score = self._luong_score(query, self.keys)
        alignments = nn_ops.softmax(score)
        return alignments

    def _luong_score(self, query, keys):
        # `query` is [B, units], `keys` is [B, T, units]
        assert query.get_shape()[-1] == keys.get_shape()[-1]
        query = array_ops.expand_dims(query, 1)
        # [B, 1, units] * [B, units, T] gives [B, 1, T].
        score = math_ops.matmul(query, tf.transpose(keys, [0, 2, 1]))
        score = array_ops.squeeze(score, [1])
        return score
    
    
class AttentionWrapperState(
    collections.namedtuple("AttentionWrapperState",
                           ("cell_state", "attention", "time"))):
    def clone(self, **kwargs):
        return super(AttentionWrapperState, self)._replace(**kwargs)
    
    
class AttentionWrapper(rnn_cell_impl.RNNCell):
    def __init__(self, cell, attention_mechanism, attention_layer_size,
                 output_attention, name):
        super(AttentionWrapper, self).__init__(name=name)
        self.cell = cell
        self.attention_layer_size = attention_layer_size
        self.attention_layer = layers_core.Dense(
            attention_layer_size, name="attention_layer", use_bias=False, dtype=tf.float32)        
        self.output_attention = output_attention
        self.attention_mechanism = attention_mechanism
    
    def call(self, inputs, state):
        # Mix current input with attention from pervious step
        cell_inputs = array_ops.concat([inputs, state.attention], -1)
        cell_output, updated_cell_state = self.cell(cell_inputs, state.cell_state)
        attention, alignments = self._compute_attention(
            self.attention_mechanism, cell_output, self.attention_layer)
        next_state = AttentionWrapperState(
            time=state.time + 1,
            cell_state=updated_cell_state,
            attention=attention
        )
        if self.output_attention:
            return attention, next_state
        else:
            return cell_output, next_state

    def _compute_attention(self, attention_mechanism, cell_output, attention_layer):
        alignments = attention_mechanism(cell_output)
        # Expands to [B, 1, T]
        expanded_alignments = array_ops.expand_dims(alignments, 1)
        # [B, 1, T] * [B, T, units] gives [B, 1, units]
        context = math_ops.matmul(expanded_alignments, attention_mechanism.values)
        context = array_ops.squeeze(context, [1])
        attention = attention_layer(array_ops.concat([cell_output, context], 1))
        return attention, alignments
    
    @property
    def output_size(self):
        if self.output_attention:
            return self.attention_layer_size
        else:
            return self.cell.output_size    
    
    @property
    def state_size(self):
        return AttentionWrapperState(
            cell_state=self.cell.state_size,
            time=tensor_shape.TensorShape([]),
            attention=self.attention_layer_size
        )
        
    def zero_state(self, batch_size, dtype):
        return AttentionWrapperState(
            cell_state=self.cell.zero_state(batch_size, dtype),
            time=array_ops.zeros([], dtype=dtypes.int32),
            attention=_zero_state_tensors(self.attention_layer_size, batch_size, dtype))
