'''
Created on Sep 21, 2016

@author: jerrik
'''

import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import argparse
import logging

import helper
from model import NewsModel
from Config import Config

parser = argparse.ArgumentParser(description="training options")

parser.add_argument('--load-config', action='store_true', dest='load_config', default=False)
parser.add_argument('--gpu-num', action='store', dest='gpu_num', default=0, type=int)
parser.add_argument('--train-test', action='store', dest='train_test', default='train', choices=['train', 'test'])
parser.add_argument('--weight-path', action='store', dest='weight_path', required=True)

parser.add_argument('--debug-enable', action='store_true', dest='debug_enable', default=False)

args = parser.parse_args()


file_names = {'val_data':'en_senti_dev',
              'test_data':'en_senti_test',
              'train_data':'en_senti_train'}

class News_class_Model(NewsModel):
    """Abstracts a Tensorflow graph for a learning task.

    We use various Model classes as usual abstractions to encapsulate tensorflow
    computational graphs. Each algorithm you will construct in this homework will
    inherit from a Model object.
    """
    def __init__(self, data_path='./all_data/'):
        """options in this function"""
        self.config = Config()
        
        self.weight_Path = args.weight_path
        if args.load_config == False:
            self.config.saveConfig(self.weight_Path+'/config')
            print 'config file generated, please specify --load-config and run again'
            sys.exit()
        else:
            self.config.loadConfig(self.weight_Path+'/config')

        self.load_data(data_path)
            
        self.add_placeholders()
        inputs, self.embed_normalize_op = self.add_embedding()
        self.logits = self.add_model(inputs)
        
        self.predict_prob = tf.nn.sigmoid(self.logits, 'predict_probability')
        self.loss = self.add_loss_op(self.logits, tf.to_float(self.label_placeholder))
        self.train_op = self.add_train_op(self.loss)
        
        MyVars = [v for v in tf.trainable_variables()]
        print MyVars
        for var in MyVars:
            if var.name == 'Embedding:0':
                self.embed_matrix = var
            if var.name == 'rnnLayer0/RNN/BasicLSTMCell/Linear/Matrix:0':
                self.rnn_matrix = var
            if var.name == 'rnnLayer0/RNN/BasicLSTMCell/Linear/Bias:0':
                self.rnn_bias = var
            if var.name == 'fully_connected/weights:0':
                self.fully_weight=var
            if var.name == 'fully_connected/biases:0':
                self.fully_bias = var

    def load_data(self, data_path):
        self.vocab = helper.Vocab()
        tag2id, id2tag = helper.load_tag(data_path+'class.txt')
        self.id2tag = id2tag
        
        val_data = helper.load_data(filePath=data_path+file_names['val_data'])
        test_data = helper.load_data(filePath=data_path+file_names['test_data'])
        train_data = helper.load_data(filePath=data_path+file_names['train_data'])
        
        self.val_data_y, val_data = helper.mkDataSet(val_data, tag2id)
        self.test_data_y, test_data = helper.mkDataSet(test_data, tag2id)
        self.train_data_y, train_data = helper.mkDataSet(train_data, tag2id)
        
        if os.path.exists(data_path + 'vocab.txt'):
            self.vocab.load_vocab_from_file(data_path + 'vocab.txt')
        else:
            words = helper.flatten([val_data, test_data, train_data])
            self.vocab.construct(words)
            self.vocab.limit_vocab_length(self.config.vocab_size)
            self.vocab.save_vocab(data_path+'.vocab.txt')

        self.val_data_len, self.val_data_x = helper.encodeNpad(val_data, self.vocab, self.config.num_steps)
        self.test_data_len, self.test_data_x = helper.encodeNpad(test_data, self.vocab, self.config.num_steps)
        self.train_data_len, self.train_data_x = helper.encodeNpad(train_data, self.vocab, self.config.num_steps)
        if self.config.pre_trained:
            embed = helper.readEmbedding(data_path + 'embed/H'+str(self.config.embed_size)+'.utf8')
            self.embed_matrix = helper.mkEmbedMatrix(embed, self.vocab.word_to_index)
        else:
            pass
        
    def add_placeholders(self):

        self.input_placeholder = tf.placeholder(tf.int32, (None, self.config.num_steps))
        self.label_placeholder = tf.placeholder(tf.int32, (None, self.config.class_num))
        self.seqLen_placeholder = tf.placeholder(tf.int32, (None))
        self.dropout_placeholder = tf.placeholder(tf.float32)

    def create_feed_dict(self, input_batch, seqLen_batch, label_batch=None):

        feed_dict = {
          self.input_placeholder   : input_batch,
          self.seqLen_placeholder  : seqLen_batch,
          self.dropout_placeholder : self.config.dropout,
        }

        if np.any(label_batch):
            feed_dict[self.label_placeholder] = label_batch

        return feed_dict

    def add_model(self, inputs):
        print "Loading basic lstm model.."
        inputs = [tf.nn.dropout(x, self.dropout_placeholder) for x in inputs]
        for i in range(self.config.rnn_numLayers):
            with tf.variable_scope('rnnLayer'+str(i)):
                lstm_cell = rnn_cell.BasicLSTMCell(self.config.hidden_size)
                outputs, _ = rnn.rnn(lstm_cell, inputs, dtype=tf.float32, sequence_length=self.seqLen_placeholder)
                inputs = outputs
        final_state = tf.add_n(outputs)

        logits = tf.contrib.layers.fully_connected(
          final_state, self.config.class_num, activation_fn=None)

        return logits

    def add_loss_op(self, logits, labels):
        print 'Sigmoid activation'
        logging.info('Sigmoid activation')
        labels_float = tf.to_float(labels)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits, labels_float)

        loss = tf.reduce_mean(loss)
        reg_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if v != self.embedding])
        return loss + self.config.reg * reg_loss

    def add_train_op(self, loss):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.learning_rate = tf.train.exponential_decay(self.config.lr, global_step,
                                                   self.config.decay_steps, self.config.decay_rate, staircase=True)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op

    def run_epoch(self, sess, data_x, data_y, len_list, verbose=10):
        """Runs an epoch of training.

        Trains the model for one-epoch.

        Args:
            sess: tf.Session() object
            data_x: input data, have shape of (data_num, num_steps)
            data_y: label, have shape of (data_num, class_num)
            len_list: length list correspond to data_x, have shape of (data_num)
        Returns:
            average_loss: scalar. Average minibatch loss of model on epoch.
        """
        total_steps = sum(1 for x in helper.data_iter(data_x, data_y, len_list, self.config.batch_size))
        total_loss = []
        for step, (data_train, label_train, data_len) in enumerate(
                helper.data_iter(data_x, data_y, len_list, self.config.batch_size)):
            feed_dict = self.create_feed_dict(data_train, data_len, label_train)
            _, loss, lr = sess.run([self.train_op, self.loss, self.learning_rate], feed_dict=feed_dict)
            #sess.run(self.embed_normalize_op)
            total_loss.append(loss)
            if verbose and step % verbose == 0:
                sys.stdout.write('\r{} / {} : loss = {}, lr = {}'.format(
                  step, total_steps, np.mean(total_loss[-verbose:]), lr))
                sys.stdout.flush()
        return np.mean(total_loss)

    def predict(self, sess, data_x, len_list):
        """Make predictions from the provided model.
        Args:
            sess: tf.Session()
            data_x: input data matrix have the shape of (data_num, num_steps)
            len_list: input data_length have the shape of (data_num)
        Returns:
          ret_pred_prob: Probability of the prediction with respect to each class
        """
        ret_pred_prob = []
        for (input_data_batch, input_seqLen_batch) in helper.pred_data_iter(
            data_x, len_list, self.config.batch_size):
            feed_dict = self.create_feed_dict(input_data_batch, input_seqLen_batch)
            pred_prob = sess.run(self.predict_prob, feed_dict=feed_dict)
            ret_pred_prob.append(pred_prob)
        ret_pred_prob = np.concatenate(ret_pred_prob, axis=0)
        return ret_pred_prob
    
    def add_embedding(self):
        """Add embedding layer. that maps from vocabulary to vectors.
        inputs: a list of tensors each of which have a size of [batch_size, embed_size]
        """
        if self.config.pre_trained:
            self.embedding = tf.Variable(self.embed_matrix, 'Embedding')
        else:
            self.embedding = tf.get_variable(
              'Embedding',
              [len(self.vocab), self.config.embed_size], trainable=True)
        
        normalized_embed = tf.nn.l2_normalize(self.embedding, dim=1, name='normalize_embedding')
        normalize_embed_op = self.embedding.assign(normalized_embed, use_locking=True)
        
        inputs = tf.nn.embedding_lookup(self.embedding, self.input_placeholder) ## (batch_size, num_steps, embed_size)
#         
#         ###########################
#         input = tf.reshape(inputs, [-1, self.config.num_steps, 1, self.config.embed_size])
#         
#         filter_shape = [4, 1, self.config.embed_size, self.config.embed_size]
#         W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
#         b = tf.Variable(tf.constant(0.1, shape=[self.config.embed_size]), name="b")
#         conv = tf.nn.conv2d(                # size (batch_size, num_step, 1, out_channel)
#           input,
#           W,
#           strides=[1, 1, 1, 1],
#           padding="SAME",
#         name="conv")
#         
#         inputs = tf.squeeze(input, [2])
#         ###########################
        
        inputs = [tf.squeeze(x, [1]) for x in tf.split(1, self.config.num_steps, inputs)] ##[(batch_size, embed_size), ...]
        return inputs, normalize_embed_op

###################################################################################################

def train_run():
    logging.info('Training start')
    with tf.Graph().as_default():
        with tf.device("/gpu:" + str(args.gpu_num)):
            classifier = News_class_Model()
        saver = tf.train.Saver({'embed_matrix': classifier.embed_matrix, 'rnn_matrix': classifier.rnn_matrix, 
                                'rnn_bias': classifier.rnn_bias, 'fully_weight': classifier.fully_weight, 'fully_bias': classifier.fully_bias})
        
        config=tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        with tf.Session(config=config) as sess:

            best_accuracy = 0
            best_val_epoch = 0
            sess.run(tf.initialize_all_variables())
            
            for epoch in range(classifier.config.max_epochs):
                print "="*20+"Epoch ", epoch, "="*20
                loss = classifier.run_epoch(sess, classifier.train_data_x, 
                        classifier.train_data_y, classifier.train_data_len)
                print
                print "Mean loss in this epoch is: ", loss
                logging.info("Mean loss in {}th epoch is: {}".format(epoch, loss) )
                print '='*50
                
                if args.debug_enable:  
                    ################################
                    print '#'*20, 'ON TRAINING SET START ', '#'*20
                    pred_prob = classifier.predict(sess, classifier.train_data_x, classifier.train_data_len)

                    pred_matrix = helper.pred_from_probability_sig(pred_prob, threshold=0.5)
                        
                    val_accuracy = helper.calculate_accuracy(pred_matrix, classifier.train_data_y)
                    
    #                 val_accuracy = helper.calculate_accuracy_from_prob(pred_prob, classifier.val_data_y)
                    print "Overall training accuracy is: {}".format(val_accuracy)
                    logging.info("Overall training accuracy is: {}".format(val_accuracy))
    
                    print '#'*20, 'ON TRAINING SET END ', '#'*20
                    ################################
                
                pred_prob = classifier.predict(sess, classifier.val_data_x, classifier.val_data_len)
                
                pred_matrix = helper.pred_from_probability_sig(pred_prob, threshold=0.5)
                    
                val_accuracy = helper.calculate_accuracy(pred_matrix, classifier.val_data_y)

                print "Overall test accuracy is: {}".format(val_accuracy)
                logging.info("Overall test accuracy is: {}".format(val_accuracy))
                
                if best_accuracy < val_accuracy:
                    best_accuracy = val_accuracy
                    best_val_epoch = epoch
                    if not os.path.exists(classifier.weight_Path):
                        os.makedirs(classifier.weight_Path)

                    saver.save(sess, classifier.weight_Path+'/news.weights')
                if epoch - best_val_epoch > classifier.config.early_stopping:
                    logging.info("Normal Early stop")
                    break
            ret_embed = sess.run(classifier.embedding)
            with open(classifier.weight_Path +'/embed_classi', 'w') as fd:
                for i, word_embed in enumerate(ret_embed):
                    fd.write('{}\t{}\n'.format(classifier.vocab.index_to_word[i], ' '.join(word_embed)))
    logging.info("Training complete")

def test_run():

    with tf.Graph().as_default():
        with tf.device("/gpu:" + str(args.gpu_num)):   #gpu_num options
            classifier = News_class_Model()
        saver = tf.train.Saver({'embed_matrix': classifier.embed_matrix, 'rnn_matrix': classifier.rnn_matrix, 
                                'rnn_bias': classifier.rnn_bias, 'fully_weight': classifier.fully_weight, 'fully_bias': classifier.fully_bias})

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        with tf.Session(config=config) as sess:
            sess.run(tf.initialize_all_variables())
            saver.restore(sess, classifier.weight_Path+'/news.weights')
            
            pred_prob = classifier.predict(sess, classifier.test_data_x, classifier.test_data_len)
            pred_matrix = helper.pred_from_probability_sig(pred_prob, threshold=0.5)
            
            test_accuracy = helper.calculate_accuracy(pred_matrix, classifier.test_data_y)
#             test_accuracy = helper.calculate_accuracy_from_prob(pred_prob, classifier.test_data_y)
            logging.info('='*30 + 'TEST RUN START' + '='*30)
            print "Overall test accuracy is: {}".format(test_accuracy)
            logging.info("Overall test accuracy is: {}".format(test_accuracy))
            logging.info('='*30 + 'TEST RUN END' + '='*30)

def main(_):
    logging.basicConfig(filename=args.weight_path+'/run.log', format='%(levelname)s %(asctime)s %(message)s', level=logging.INFO)
    if args.train_test == "train":
        train_run()
    else:
        test_run()

if __name__ == '__main__':
    tf.app.run()
