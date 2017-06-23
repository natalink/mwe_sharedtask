#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

import datetime
import numpy as np
import sys
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
import tensorflow.contrib.losses as tf_losses
import tensorflow.contrib.metrics as tf_metrics

import mwe_dataset

def highway_layer(x, batch_size, activation=tf.nn.relu, carry_bias=-1.0):
    original_shape = x.get_shape().as_list()
    w_shape = [original_shape[-1], original_shape[-1]]
    b_shape = [original_shape[-1]]

    original_shape[0] = batch_size
    original_shape[1] = -1
    x = tf.reshape(x, [-1, original_shape[-1]])

    W_H = tf.Variable(tf.random_normal(w_shape, stddev=0.1), trainable=True, name="weight")
    b_H = tf.Variable(tf.constant(carry_bias, shape=b_shape), name="bias")

    W_T = tf.Variable(tf.random_normal(w_shape, stddev=0.1), trainable=True, name="weight_transform")
    b_T = tf.Variable(tf.constant(0.1, shape=b_shape), name="bias_transform")

    T = tf.sigmoid(tf.matmul(x, W_T) + b_T, name="transform_gate")
    H = activation(tf.matmul(x, W_H) + b_H, name="activation")
    C = tf.sub(1.0, T, name="carry_gate")

    y = tf.add(tf.mul(H, T), tf.mul(x, C), "y")
    y = tf.reshape(y, tf.stack(original_shape))
    return y


class Network:
    EMB_INITIALIZER=tf.random_normal_initializer(stddev=0.01)

    def __init__(self, rnn_cell, rnn_cell_dim, method, data_train, logdir, expname, threads, restore_path ,seed=42):
        n_words = len(data_train.factors[data_train.FORMS]['words'])
        n_tags = len(data_train.factors[data_train.TAGS]['words'])
        n_lemmas = len(data_train.factors[data_train.LEMMAS]['words'])
        n_mwe = len(data_train.factors[data_train.MWE]['words'])
        n_alphabet = len(data_train._alphabet)

        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.summary_writer = tf.train.SummaryWriter("{}/{}-{}".format(logdir, timestamp, expname), flush_secs=10)



        # Construct the graph
        with self.session.graph.as_default():
            if rnn_cell == "LSTM":
                rnn_cell = tf.nn.rnn_cell.LSTMCell(rnn_cell_dim)
            elif rnn_cell == "GRU":
                rnn_cell = tf.nn.rnn_cell.GRUCell(rnn_cell_dim)
            else:
                raise ValueError("Unknown rnn_cell {}".format(rnn_cell))

            self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name="global_step")
            self.sentence_lens = tf.placeholder(tf.int32, [None], name="sentence_lens")
            self.forms = tf.placeholder(tf.int32, [None, None], name="forms")
            self.tags = tf.placeholder(tf.int32, [None, None], name="tags")
            self.lemmas = tf.placeholder(tf.int32, [None, None], name="lemmas")
            self.mwe = tf.placeholder(tf.int32, [None, None], name="mwe")

            self.charseqs_ids = tf.placeholder(tf.int32, [None, None], name="charseqs_ids")
            self.charseqs = tf.placeholder(tf.int32, [None, None], name="charseqs")
            self.charseqs_lens = tf.placeholder(tf.int32, [None, None], name="charseqs_lens")

            self.keep_dropout = tf.placeholder_with_default(1.0, shape=None, name="keep_dropout")
            self.charseqs_size = tf.placeholder(tf.int32)

            # Create embeddings
            self.form_emb_matrix = tf.get_variable("word_embedding_matrix",
                                                   [n_words, rnn_cell_dim],
                                                   initializer=self.EMB_INITIALIZER,
                                                   dtype=tf.float32)
            self.lemma_emb_matrix  = tf.get_variable("lemma_embedding_matrix",
                                                     [n_lemmas, rnn_cell_dim],
                                                     initializer=self.EMB_INITIALIZER,
                                                     dtype=tf.float32)
            self.tag_emb_matrix = tf.get_variable("tag_embedding_matrix",
                                                  [n_tags, rnn_cell_dim],
                                                  initializer=self.EMB_INITIALIZER,
                                                  dtype=tf.float32)

            self.forms_embed = tf.nn.embedding_lookup(self.form_emb_matrix, self.forms)
            self.lemmas_embed = tf.nn.embedding_lookup(self.lemma_emb_matrix, self.lemmas)
            self.tags_embed =  tf.nn.embedding_lookup(self.tag_emb_matrix, self.tags)


            if "char_" in method:
                self.char_emb_matrix =  tf.get_variable("char_embedding_matrix",
                                                        [n_tags, rnn_cell_dim],
                                                        initializer=self.EMB_INITIALIZER,
                                                        dtype=tf.float32)
                self.char_embeddings = tf.nn.embedding_lookup(self.char_emb_matrix, self.charseqs)
                self.char_embeddings = tf.nn.dropout(self.char_embeddings, self.keep_dropout)

                if "rnn_" in method:
                    ch_rnn_cell = tf.nn.rnn_cell.GRUCell(int(embedding_size / 2))
                    _, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=ch_rnn_cell,
                                                                cell_bw=ch_rnn_cell,
                                                                inputs=self.char_embeddings,
                                                                sequence_length=self.charseqs_lens,
                                                                dtype=tf.float32)
                    self.form_emb_matrix = tf.concat(1, states)

                elif "conv_" in method:
                    self.char_embeddings = tf.expand_dims(self.char_embeddings, -1)
                    pooled_layer = []
                    for i in range(5):
                        for j in range(num_of_filters):
                            filt = tf.get_variable("conv_filter_{}_{}".format(i+1, j),
                                                    initializer=tf.random_uniform([i+2, embedding_size, 1, 1], -1.0, 1.0),
                                                    trainable=True)
                            conv_layer = tf.nn.conv2d(self.char_embeddings,
                                                      filter=filt,
                                                      strides=[1,1,embedding_size,1],
                                                      padding="VALID",
                                                      name="conv_{}_{}".format(i+1, j))
                            # This should be a better approach for pooling from variable length sequence
                            maxpool_layer = tf.reduce_max(conv_layer, 1, keep_dims=True)
                            pooled_layer.append(maxpool_layer)
                    pooled_concatenated = tf.squeeze(tf.concat(3, pooled_layer), [2])
                    h_layer = pooled_concatenated
                    # Add at least one highway layer for good measure
                    for i in range(1):
                        h_layer = highway_layer(h_layer, self.charseqs_size)
                    self.form_emb_matrix = h_layer

                self.forms_embed = tf.nn.embedding_lookup(self.form_emb_matrix, self.charseqs_ids)

            # Combine the embeddings
            self.inputs_embed = tf.nn.dropout(tf.concat(2, [self.forms_embed, self.lemmas_embed, self.tags_embed]))
            # TODO: linearne transformovat?


            hidden_states, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=rnn_cell,
                                                               cell_bw=rnn_cell,
                                                               inputs=self.inputs_embed,
                                                               sequence_length=self.sentence_lens,
                                                               dtype=tf.float32) # concatenate embedding
            outputs = tf.concat(2, hidden_states)
            mask_tensor = tf.sequence_mask(self.sentence_lens)
            output_layer = tf_layers.fully_connected(output_layer, n_mwe, activation_fn=None)
            self.predictions = tf.argmax(output_layer, 2)            

            masked_mwe = tf.boolean_mask(self.mwe, mask_tensor)
            masked_output = tf.boolean_mask(output_layer, mask_tensor)
            masked_predictions = tf.boolean_mask(self.predictions, mask_tensor)

            loss = tf_losses.sparse_softmax_cross_entropy(masked_output, masked_mwe, scope="loss")
            self.training = tf.train.AdamOptimizer().minimize(loss, global_step=self.global_step)
            self.accuracy = tf_metrics.accuracy(masked_predictions, masked_mwe) #vysledni scotre bude jiny !!! cely score na tech MWE


            self.summary = {}
            for dataset_name in ["train", "dev"]:
                self.summary[dataset_name] = tf.summary.merge([tf.summary.scalar(dataset_name+"/loss", loss),
                                                               tf.summary.scalar(dataset_name+"/accuracy", self.accuracy)])

            self.saver = tf.train.Saver(max_to_keep=None)
            # Initialize variables
            self.session.run(tf.initialize_all_variables())
            if self.summary_writer:
                self.summary_writer.add_graph(self.session.graph)

            if restore_path is not None:
                self.saver.restore(self.session, restore_path)

    def save(self, saved_path):
        self.saver.save(self.session, saved_path)

    @property
    def training_step(self):
        return self.session.run(self.global_step)

    def train(self, sentence_lens, dropout, forms, lemmas, tags, mwe, charseqs_ids, charseqs, charseqs_lens):
        _, summary = self.session.run([self.training, self.summary],
                                      {self.sentence_lens: sentence_lens,
                                       self.forms: forms, #pridej lemmas, mwe
			                           self.lemmas: lemmas,
                        	           self.tags: tags,
                                       self.mwe: mwe,
                                       self.charseqs_ids: charseqs_ids,
                                       self.charseqs: charseqs,
                                       self.charseqs_lens: charseqs_lens,
                                       self.charseqs_size: len(charseqs),
                                       self.keep_dropout: dropout})
        self.summary_writer.add_summary(summary["train"], self.training_step)
#TODO3eval, pred: + lemmas, mwe
    def evaluate(self, sentence_lens, forms, lemmas, tags, mwe, charseqs_ids, charseqs, charseqs_lens):
        accuracy, summary = self.session.run([self.accuracy, self.summary],
                                   {self.sentence_lens: sentence_lens,
                                    self.forms: forms,
                                    self.lemmas: lemmas,
                                    self.tags: tags,
                                    self.mwe: mwe,
                                    self.charseqs_ids: charseqs_ids,
                                    self.charseqs: charseqs,
                                    self.charseqs_lens: charseqs_lens,
                                    self.charseqs_size: len(charseqs),
                                    self.keep_dropout: 1.0})
        self.summary_writer.add_summary(summary["dev"], self.training_step)
        return accuracy

    def predict(self, sentence_lens, forms, lemmas, tags, charseqs_ids, charseqs, charseqs_lens):
        return self.session.run(self.predictions,
                                {self.sentence_lens: sentence_lens,
                                 self.forms: forms,
                                 self.lemmas: lemmas,
                                 self.tags: tags,
                                 self.charseqs_ids: charseqs_ids,
                                 self.charseqs: charseqs,
                                 self.charseqs_lens: charseqs_lens,
                                 self.charseqs_size: len(charseqs),
                                 self.keep_dropout: 1.0})


if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
    parser.add_argument("--data_train", default="50000cs-train.txt", type=str, help="Training data file.")
    parser.add_argument("--data_dev", default="cs-dev.txt", type=str, help="Development data file.")
    parser.add_argument("--data_test", default="cs-test.txt", type=str, help="Testing data file.")
    parser.add_argument("--data_test_blind", default="cs-test.txt", type=str, help="Blind test parsemetsv")
    parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
    parser.add_argument("--method", default="learned_we", type=str, help="Which method of word embeddings to use.")
    parser.add_argument("--logdir", default="logs", type=str, help="Logdir name.")
    parser.add_argument("--rnn_cell", default="GRU", type=str, help="RNN cell type.")
    parser.add_argument("--rnn_cell_dim", default=100, type=int, help="RNN cell dimension.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--restore", default=None, type=str, help="Restore session with a model")
    args = parser.parse_args()
    lang = args.data_test.split("/")[1]

    # Load the data
    print("Loading the data.", file=sys.stderr)
    data_train = mwe_dataset.MorphoDataset(args.data_train, add_bow_eow=True)
    data_dev = mwe_dataset.MorphoDataset(args.data_dev, train=data_train, add_bow_eow=True) #
    data_test = mwe_dataset.MorphoDataset(args.data_test, train=data_train, add_bow_eow=True) #
    data_test_blind = mwe_dataset.MorphoDataset(args.data_test_blind, train=data_train, add_bow_eow=True)  #
    # Construct the network
    print("Constructing the network for $lang", file=sys.stderr)
    expname = "{}-tagger-{}{}-m{}-bs{}-epochs{}".format(lang, args.rnn_cell, args.rnn_cell_dim, args.method,
                                                        args.batch_size, args.epochs)
    network = Network(rnn_cell=args.rnn_cell,
                      rnn_cell_dim=args.rnn_cell_dim,
                      method=args.method,
                      data_train=data_train,
                      logdir=args.logdir,
                      expname=expname,
                      threads=args.threads,
                      restore_path=args.restore)

    # Train
    best_dev_accuracy = 0
    test_predictions = None
    if args.restore is not None:
        test_sentence_lens, test_word_ids, test_charseqs_ids, test_charseqs, test_charseqs_lens = \
            data_test.whole_data_as_batch(including_charseqs=True)
        test_predictions = network.predict(test_sentence_lens,
                                           test_ids[data_train.FORMS],
                                           test_ids[data_train.LEMMAS],
                                           test_ids[data_train.TAGS],
                                           test_word_ids[data_train.MWE],
                                           test_charseqs_ids[data_train.FORMS],
                                           test_charseqs[data_train.FORMS],
                                           test_charseqs_lens[data_train.FORMS])  # ALSO< lemmas and tags
    else:
        for epoch in range(args.epochs):
            print("Training epoch {}".format(epoch + 1), file=sys.stderr)
            while not data_train.epoch_finished():
                sentence_lens, word_ids, charseqs_ids, charseqs, charseqs_lens = \
                    data_train.next_batch(args.batch_size, including_charseqs=True)
                saved_model = "saved_model_{}".format(expname)


                network.train(sentence_lens,
                              word_ids[data_train.FORMS],
                              word_ids[data_train.LEMMAS],
                              word_ids[data_train.TAGS],
                              word_ids[data_train.MWE],
                              charseqs_ids[data_train.FORMS],
                              charseqs[data_train.FORMS],
                              charseqs_lens[data_train.FORMS]) # lemma_ids deprel_ids
            # To use character-level embeddings, pass including_charseqs=True to next_batch
            # and instead of word_ids[data_train.FORMS] use charseq_ids[data_train.FORMS],
            # charseqs[data_train.FORMS] and charseq_lens[data_train.FORMS]
                network.save(saved_model)
            dev_sentence_lens, dev_word_ids, dev_charseqs_ids, dev_charseqs, dev_charseqs_lens = \
                data_dev.whole_data_as_batch(including_charseqs=True)
            dev_accuracy = network.evaluate(dev_sentence_lens,
                                            dev_word_ids[data_train.FORMS],
                                            dev_word_ids[data_train.LEMMAS],
                                            dev_word_ids[data_train.TAGS],
                                            dev_word_ids[data_train.MWE],
                                            dev_charseqs_ids[data_train.FORMS],
                                            dev_charseqs[data_train.FORMS],
                                            dev_charseqs_lens[data_train.FORMS])
            print("Development accuracy after epoch {} is {:.2f}.".format(epoch + 1, 100. * dev_accuracy), file=sys.stderr)

            if dev_accuracy > best_dev_accuracy:
                best_dev_accuracy = dev_accuracy
                test_sentence_lens, test_word_ids, test_charseqs_ids, test_charseqs, test_charseqs_lens = \
                    data_test.whole_data_as_batch(including_charseqs=True)
                test_predictions = network.predict(test_sentence_lens,
                                                   test_word_ids[data_train.FORMS],
                                                   test_word_ids[data_train.LEMMAS],
                                                   test_word_ids[data_train.TAGS],
                                                   test_charseqs_ids[data_train.FORMS],
                                                   test_charseqs[data_train.FORMS],
                                                   test_charseqs_lens[data_train.FORMS]) #ALSO< lemmas and tags
                print ("test_predictions: ", test_predictions)
                blind_sentence_lens, blind_word_ids, blind_charseqs_ids, blind_charseqs, blind_charseqs_lens = \
                    data_test_blind.whole_data_as_batch(including_charseqs=True)
                blind_predictions = network.predict(blind_sentence_lens,
                                                    blind_word_ids[data_train.FORMS],
                                                    blind_word_ids[data_train.LEMMAS],
                                                    blind_word_ids[data_train.TAGS],
                                                    blind_charseqs_ids[data_train.FORMS],
                                                    blind_charseqs[data_train.FORMS],
                                                    blind_charseqs_lens[data_train.FORMS])  # ALSO< lemmas and tags
                print("blind_test_predictions: ", blind_predictions)
    # Print test predictions
    test_forms = data_test.factors[data_test.FORMS]['strings'] # We use strings instead of words, because words can be <unk>
    test_lemmas = data_test.factors[data_test.LEMMAS]['words']
    test_tags = data_test.factors[data_test.TAGS]['words']
    test_mwe = data_test.factors[data_test.MWE]['words']
    blind_forms = data_test_blind.factors[data_test_blind.FORMS]['strings'] # We use strings instead of words, because words can be <unk>
    blind_lemmas = data_test_blind.factors[data_test_blind.LEMMAS]['words']
    blind_tags = data_test_blind.factors[data_test_blind.TAGS]['words']
    blind_mwe = data_test_blind.factors[data_test_blind.MWE]['words']


    import time
    timestr = time.strftime("%Y%m%d-%H%M%S")
    print (timestr)
    testfilename = '{}_{}_{}'.format(args.data_train, expname, timestr)
    blindfilename = '{}_{}_{}_blindtest'.format(args.data_train, expname, timestr)
    f1=open(testfilename, 'w+',  encoding="utf-8")
    for i in range(len(data_test.sentence_lens)):
        for j in range(data_test.sentence_lens[i]):
            print("{}\t_\t{}".format(test_forms[i][j], test_mwe[test_predictions[i, j]]), file=f1)
            #print("{}\t_\t{}".format(test_forms[i][j], test_mwe[test_predictions[i, j]]))
        print("", file=f1)
    f1.close()

    f2 = open(blindfilename, 'w+', encoding="utf-8")
    for i in range(len(data_test_blind.sentence_lens)):
        for j in range(data_test_blind.sentence_lens[i]):
            print("{}\t_\t{}".format(blind_forms[i][j], blind_mwe[blind_predictions[i, j]]), file=f2)
            #print("{}\t_\t{}".format(test_forms[i][j], test_mwe[test_predictions[i, j]]))
        print("", file=f2)
    f2.close()
