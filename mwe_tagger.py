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

class Network:
    def __init__(self, rnn_cell, rnn_cell_dim, method, data_train, logdir, expname, threads=1, seed=42):
        n_words = len(data_train.factors[data_train.FORMS]['words'])
        n_tags = len(data_train.factors[data_train.TAGS]['words'])
        n_lemmas = len(data_train.factors[data_train.LEMMAS]['words'])
        n_mwe = len(data_train.factors[data_train.MWE]['words'])

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
            self.sentence_lens = tf.placeholder(tf.int32, [None])
            self.forms = tf.placeholder(tf.int32, [None, None])
            self.tags = tf.placeholder(tf.int32, [None, None])
            self.lemmas = tf.placeholder(tf.int32, [None, None])
            self.mwe = tf.placeholder(tf.int32, [None, None])
            embeddings_forms = tf.get_variable("word_embedding_matrix", [n_words,rnn_cell_dim], dtype = tf.float32)
            embeddings_lemmas = tf.get_variable("lemma_embedding_matrix", [n_lemmas,rnn_cell_dim], dtype = tf.float32)
#            embeddings_tags = tf.get_variable("tag_embedding_matrix", [n_tags,rnn_cell_dim], dtype = tf.float32)
            embedding_in_forms = tf.nn.embedding_lookup(embeddings_forms, self.forms)
            embedding_in_lemmas = tf.nn.embedding_lookup(embeddings_lemmas, self.lemmas)
#            embedding_in_tags =  tf.nn.embedding_lookup(embeddings_tags, self.tags)
           #TODO concatenata tf.cvoncat emb_form emb_lemma emb_tags (zkusit udelat one-hot na tag - nebo zapomenout na to)
            embedding_in = tf.concat(2, [embedding_in_forms, embedding_in_lemmas]) #, embedding_in_tags]) #mail Milanovi - je to tak???
            bi_out, bi_out_states = tf.nn.bidirectional_dynamic_rnn(rnn_cell, rnn_cell, embedding_in, self.sentence_lens, dtype=tf.float32) # concatenate embedding
            layer1 = tf.concat(2, bi_out)
            layer = tf_layers.fully_connected(layer1, n_mwe,activation_fn=None, scope="output_layer" )
            

            mask = tf.sequence_mask(self.sentence_lens)
            masked_mwe = tf.boolean_mask(self.mwe, mask)

            self.predictions = tf.cast(tf.argmax(layer, 2, name="predictions"), dtype=tf.int32)
            masked_predictions = tf.boolean_mask(self.predictions, mask)
            loss = tf_losses.sparse_softmax_cross_entropy(tf.boolean_mask(layer,mask), masked_mwe, scope="loss")
            self.training = tf.train.AdamOptimizer().minimize(loss, global_step=self.global_step)
            self.accuracy = tf_metrics.accuracy(masked_predictions, masked_mwe) #vysledni scotre bude jiny !!! cely score na tech MWE


            self.dataset_name = tf.placeholder(tf.string, [])
            self.summary = tf.merge_summary([tf.scalar_summary(self.dataset_name+"/loss", loss),
                                             tf.scalar_summary(self.dataset_name+"/accuracy", self.accuracy)])

            # Initialize variables
            self.session.run(tf.initialize_all_variables())

    @property
    def training_step(self):
        return self.session.run(self.global_step)

    def train(self, sentence_lens, forms, lemmas, tags, mwe):
        _, summary = self.session.run([self.training, self.summary],
                                      {self.sentence_lens: sentence_lens, self.forms: forms, #pridej lemmas, mwe
			                           self.lemmas: lemmas,
                        	           self.tags: tags,
                                       self.mwe: mwe, self.dataset_name: "train"})
        self.summary_writer.add_summary(summary, self.training_step)
#TODO3eval, pred: + lemmas, mwe
    def evaluate(self, sentence_lens, forms, lemmas, tags, mwe):
        accuracy, summary = self.session.run([self.accuracy, self.summary],
                                   {self.sentence_lens: sentence_lens, self.forms: forms,
                                    self.lemmas: lemmas,
                                    self.tags: tags,
                                   self.mwe: mwe, self.dataset_name: "dev"})
        self.summary_writer.add_summary(summary, self.training_step)
        return accuracy

    def predict(self, sentence_lens, forms, lemmas, tags):
        return self.session.run(self.predictions,
                                {self.sentence_lens: sentence_lens, self.forms: forms,
                                 self.lemmas: lemmas,
                                 self.tags: tags},
)

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
    parser.add_argument("--epochs", default=2, type=int, help="Number of epochs.")
    parser.add_argument("--method", default="learned_we", type=str, help="Which method of word embeddings to use.")
    parser.add_argument("--logdir", default="logs", type=str, help="Logdir name.")
    parser.add_argument("--rnn_cell", default="GRU", type=str, help="RNN cell type.")
    parser.add_argument("--rnn_cell_dim", default=100, type=int, help="RNN cell dimension.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Load the data
    print("Loading the data.", file=sys.stderr)
    data_train = mwe_dataset.MorphoDataset(args.data_train, add_bow_eow=True)
    data_dev = mwe_dataset.MorphoDataset(args.data_dev, train=data_train, add_bow_eow=True) #
    data_test = mwe_dataset.MorphoDataset(args.data_test, train=data_train, add_bow_eow=True) #

    # Construct the network
    print("Constructing the network.", file=sys.stderr)
    expname = "tagger-{}{}-m{}-bs{}-epochs{}".format(args.rnn_cell, args.rnn_cell_dim, args.method, args.batch_size, args.epochs)
    network = Network(rnn_cell=args.rnn_cell, rnn_cell_dim=args.rnn_cell_dim, method=args.method,
                      data_train = data_train, logdir=args.logdir, expname=expname, threads=args.threads)

    # Train
    best_dev_accuracy = 0
    test_predictions = None

    for epoch in range(args.epochs):
        print("Training epoch {}".format(epoch + 1), file=sys.stderr)
        while not data_train.epoch_finished():
            sentence_lens, word_ids = data_train.next_batch(args.batch_size)
            network.train(sentence_lens, word_ids[data_train.FORMS], word_ids[data_train.LEMMAS], word_ids[data_train.TAGS], word_ids[data_train.MWE]) # lemma_ids deprel_ids
            # To use character-level embeddings, pass including_charseqs=True to next_batch
            # and instead of word_ids[data_train.FORMS] use charseq_ids[data_train.FORMS],
            # charseqs[data_train.FORMS] and charseq_lens[data_train.FORMS]

        dev_sentence_lens, dev_word_ids = data_dev.whole_data_as_batch()
        dev_accuracy = network.evaluate(dev_sentence_lens, dev_word_ids[data_dev.FORMS],dev_word_ids[data_dev.LEMMAS], dev_word_ids[data_dev.TAGS], dev_word_ids[data_dev.MWE])
        print("Development accuracy after epoch {} is {:.2f}.".format(epoch + 1, 100. * dev_accuracy), file=sys.stderr)

        if dev_accuracy > best_dev_accuracy:
            best_dev_accuracy = dev_accuracy
            test_sentence_lens, test_word_ids = data_test.whole_data_as_batch()
            test_predictions = network.predict(test_sentence_lens, test_word_ids[data_test.FORMS], test_word_ids[data_test.LEMMAS], test_word_ids[data_test.TAGS] ) #ALSO< lemmas and tags

    # Print test predictions
    test_forms = data_test.factors[data_test.FORMS]['strings'] # We use strings instead of words, because words can be <unk>
    test_lemmas = data_test.factors[data_test.LEMMAS]['words']
    test_tags = data_test.factors[data_test.TAGS]['words']
    test_mwe = data_test.factors[data_test.MWE]['words']

    f1=open('testfile', 'w+',  encoding="utf-8")
    for i in range(len(data_test.sentence_lens)):
        for j in range(data_test.sentence_lens[i]):
            print("{}\t_\t{}".format(test_forms[i][j], test_mwe[test_predictions[i, j]]))#, file=f1)
        print((), file=f1)
