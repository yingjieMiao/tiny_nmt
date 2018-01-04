import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.python.layers import core as layers_core

class NmtModel:
    """Basic encoder-decoder model.
    
       Trainig and inference share most part of the graph construction logic.
    """    
    def __init__(self, hparams, mode, iterator, 
                 src_vocab_table, tgt_vocab_table,
                 reverse_tgt_vocab_table=None):
        self.mode = mode
        self.iterator = iterator
        self.src_vocab_table = src_vocab_table
        self.tgt_vocab_table = tgt_vocab_table
        self.time_major = hparams.time_major
        self.batch_size = tf.size(self.iterator.source_sequence_length)
        
        # Embeddings and projections are shared
        tf.get_variable_scope().set_initializer(
                tf.random_uniform_initializer(-1., 1.))
        self._init_embeddings(hparams)
        self._projection_layer(hparams)
        
        # Construct encoder-decoder. Training and inference have different
        # Decoder logic.
        res = self._build_encoder_decoder(hparams)
        
        if self.mode == "train":
            self.logits, self.train_loss, _ = res
        if self.mode == "infer":
            self.infer_logits, _, self.sample_id = res
            self.sample_words = reverse_tgt_vocab_table.lookup(tf.to_int64(self.sample_id))
        
        self.global_step = tf.Variable(0, trainable=False)
        if self.mode == "train":
            self._update_op(hparams)
        
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
        
        if self.mode == "train":
            self.train_summary = tf.summary.merge([
                    tf.summary.scalar("train_loss", self.train_loss)
                ] + self.grad_summary
            )
    
    def _init_embeddings(self, hparams):
        with tf.variable_scope("embeddings"):
            with tf.variable_scope("encoder"):
                self.embedding_encoder = (
                    tf.get_variable("embedding_encoder", [hparams.src_voc_size, hparams.src_emb_size]))
            with tf.variable_scope("decoder"):
                self.embedding_decoder = (
                    tf.get_variable("embedding_decoder", [hparams.tgt_voc_size, hparams.tgt_emb_size]))
                
    def _projection_layer(self, hparams):
        with tf.variable_scope("decoder/output_projection"):
            self.output_layer = layers_core.Dense(
                hparams.tgt_voc_size, use_bias=False, name="output_projection")
    
    def _build_encoder_decoder(self, hparams):
        with tf.variable_scope("dynamic_seq2seq"):
            encoder_outputs, encoder_state = self._build_encoder(hparams)
            logits, sample_id = self._build_decoder(encoder_state, hparams)
            if self.mode != "infer":
                loss = self._compute_loss(logits)
            else:
                loss = None
        return logits, loss, sample_id
    
    def _build_encoder(self, hparams):
        iterator = self.iterator
        if self.time_major:
            source = tf.transpose(iterator.source)
        with tf.variable_scope("encoder"):
            encoder_emb_inp = tf.nn.embedding_lookup(self.embedding_encoder, source)
            cell = self._build_encoder_cell(hparams)
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                cell, encoder_emb_inp,
                dtype=tf.float32,
                sequence_length=iterator.source_sequence_length,
                time_major=self.time_major,
                swap_memory=True)
        return encoder_outputs, encoder_state
    
    def _build_encoder_cell(self, hparams):
        return self._build_rnn_cell(hparams.num_units, hparams.num_layers, hparams.forget_bias)
    
    def _build_decoder(self, encoder_state, hparams):
        tgt_sos_id = tf.cast(self.tgt_vocab_table.lookup(tf.constant("sos")),
                             tf.int32)
        tgt_eos_id = tf.cast(self.tgt_vocab_table.lookup(tf.constant("eos")),
                             tf.int32)        
        iterator = self.iterator
        maximum_iterations = self._get_infer_maximum_iterations(iterator.source_sequence_length)
        with tf.variable_scope("decoder") as decoder_scope:
            cell = self._build_decoder_cell(hparams)
            decoder_init_state = encoder_state
            
            if self.mode != "infer":
                target_input = iterator.target_input
                if self.time_major:
                    target_input = tf.transpose(target_input)                
                decoder_emb_inp = tf.nn.embedding_lookup(self.embedding_decoder, target_input)
                helper = seq2seq.TrainingHelper(decoder_emb_inp, iterator.target_sequence_length, time_major=self.time_major)
                decoder = seq2seq.BasicDecoder(cell, helper, decoder_init_state)

                outputs, _, _ = seq2seq.dynamic_decode(decoder, output_time_major=self.time_major, swap_memory=True, scope=decoder_scope)
                sample_id = outputs.sample_id
                logits = self.output_layer(outputs.rnn_output)
            else:
                start_tokens = tf.fill([self.batch_size], tgt_sos_id)
                end_token = tgt_eos_id
                helper = seq2seq.GreedyEmbeddingHelper(self.embedding_decoder, start_tokens, end_token)
                decoder = seq2seq.BasicDecoder(cell, helper, decoder_init_state, output_layer=self.output_layer)
                outputs, _, _ = seq2seq.dynamic_decode(decoder, maximum_iterations=maximum_iterations, output_time_major=self.time_major, swap_memory=True, scope=decoder_scope)                
                logits = outputs.rnn_output
                sample_id = outputs.sample_id
            
        return logits, sample_id
            
    def _build_decoder_cell(self, hparams):
        return self._build_rnn_cell(hparams.num_units, hparams.num_layers, hparams.forget_bias)  

    def _build_rnn_cell(self, num_units, num_layers, forget_bias):
        cell_list = []
        for i in range(num_layers):
            cell_list.append(rnn.BasicLSTMCell(num_units, forget_bias=forget_bias))
        if len(cell_list) == 1:
            return cell_list[0]
        else:
            return rnn.MultiRNNCell(cell_list)
        
    def _update_op(self, hparams):
        params = tf.trainable_variables()
        self.learning_rate = hparams.learning_rate
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        gradients = tf.gradients(self.train_loss, params)
        clipped_grads, unclipped_grads_norm = self._gradient_clip(gradients)
        self.update = optimizer.apply_gradients(zip(clipped_grads, params), global_step=self.global_step)
        self.grad_summary = [tf.summary.scalar("grad_norm", unclipped_grads_norm),
                             tf.summary.scalar("clipped_grad_norm", tf.global_norm(clipped_grads))]
        
    def _gradient_clip(self, gradients, max_gradient_norm = 5.0):
        clipped_gradients, gradient_norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
        return clipped_gradients, gradient_norm
        
    def get_max_time(self, tensor):
        time_axis = 0 if self.time_major else 1
        return tensor.shape[time_axis].value or tf.shape(tensor)[time_axis]        
        
    def _compute_loss(self, logits):
        target_output = self.iterator.target_output
        if self.time_major:
          target_output = tf.transpose(target_output)
        max_time = self.get_max_time(target_output)
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=target_output, logits=logits)
        target_weights = tf.sequence_mask(
            self.iterator.target_sequence_length, max_time, dtype=logits.dtype)
        if self.time_major:
          target_weights = tf.transpose(target_weights)

        loss = tf.reduce_sum(
            crossent * target_weights) / tf.to_float(self.batch_size)
        return loss
    
    def _get_infer_maximum_iterations(self, source_sequence_length):
        decoding_length_factor = 2.0
        max_encoder_length = tf.reduce_max(source_sequence_length)
        maximum_iterations = tf.to_int32(tf.round(
          tf.to_float(max_encoder_length) * decoding_length_factor))
        return maximum_iterations    
    
    def train(self, sess):
        assert self.mode == 'train'
        return sess.run([
            self.update,
            self.train_summary,
            self.train_loss,
            self.global_step])
    
    def _infer(self, sess):
        assert self.mode == 'infer'
        return sess.run([self.infer_logits, self.sample_id, self.sample_words])
    
    def decode(self, sess):
        _, _, sample_words = self._infer(sess)
        if self.time_major:
            sample_words = sample_words.transpose()
        return sample_words