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

class Generate_Model(NewsModel):

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
        self.embed_normalize_op = self.add_embedding()
        
        self.input_emb, self.update_op, self.assign_input_op, inputs = self.add_input()
        
        self.logits = self.add_model(inputs)
        
        self.predict_prob = tf.nn.sigmoid(self.logits, 'predict_probability')
        self.loss = self.add_loss_op(self.logits, tf.to_float(self.label_placeholder))
        
        self.train_op = self.take_gradient(self.loss, self.input_ids, self.input_grad)
        
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

        self.input_placeholder = tf.placeholder(tf.int64, (None, self.config.num_steps), name='input_placeholder')
        self.label_placeholder = tf.placeholder(tf.int32, (None, self.config.class_num), name='label_placeholder')
        self.seqLen_placeholder = tf.placeholder(tf.int32, (None), name='seqlen_placeholder')

    def create_feed_dict(self, input_batch, seqLen_batch, label_batch):

        feed_dict = {
          self.input_placeholder   : input_batch,
          self.seqLen_placeholder  : seqLen_batch,
          self.label_placeholder   : label_batch
        }

        return feed_dict

    def add_model(self, inputs):
        print "Loading basic lstm model.."
        
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
                
        labels_float = tf.to_float(labels)
        inference = tf.sigmoid(logits)
        loss = tf.sqrt((inference - labels_float)**2)
        loss = tf.reduce_mean(loss)
        return loss  #now maximize loss by tuning inpu
    
    def run_epoch(self, sess, data_x, data_y, length, verbose=10):
        filePath = self.weight_Path+'/output.out'
        
        data_x = np.array(data_x, dtype=np.int32)
        data_y = np.array(data_y, dtype=np.int32)
        length = np.array(length, dtype=np.int32)
        if len(data_x.shape) <2:
            data_x = data_x.reshape(1, -1)
        if len(data_y.shape) <2:
            data_y = data_y.reshape(1, -1)
        if not length.shape:
            length = np.array([length])
        
        feed_dict = self.create_feed_dict(data_x, length, data_y)
        sess.run(self.assign_input_op, feed_dict=feed_dict)
        sentence = []
        for i, idx in enumerate(data_x.flatten()):
            if i< length[0]:
                sentence.append(self.vocab.decode(idx))
    
        prev_sentence=' '.join(sentence)
        open(filePath, 'a+').write('='*50 + '\n')
        open(filePath, 'a+').write('{}, {}\n'.format(data_y[0, 0], prev_sentence))
        print prev_sentence, data_y[0, 0]
        loss = 0.0
        cnt=0
        
        while loss < 0.8:
            cnt +=1
            _ = sess.run([self.train_op], feed_dict=feed_dict)
            sess.run(self.update_op)
            sentence_ids, loss = sess.run([self.input_ids, self.loss], feed_dict)
#             if cnt%100 == 1:
#                 sys.stdout.write('\rstep = {} : loss = {}'.format(cnt, loss))
#                 sys.stdout.flush()
            sentence=[]
            for i, idx in enumerate(sentence_ids.flatten()):
                if i< length[0]:
                    sentence.append(self.vocab.decode(idx))
            if prev_sentence != ' '.join(sentence):
                prev_sentence = ' '.join(sentence)
                print prev_sentence, data_y[0, 0], loss
                open(filePath, 'a+').write('{} {}, {}\n'.format(data_y[0, 0], loss, prev_sentence))
            
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
        
        return normalize_embed_op

    def add_input(self):
        self.input_ids = tf.Variable(np.zeros([1, self.config.num_steps]), name='Input_Ids', dtype=tf.int64, trainable=False) #(1, num_steps)
        input_emb = tf.nn.embedding_lookup(self.embedding, self.input_ids) #(1, num_steps, embed_size)
        
        self.input_grad = tf.Variable(np.zeros([self.config.num_steps, 
            self.config.embed_size]), name='Input_grad', dtype=tf.float32, trainable=False) #(num_steps, embed_size)
        
        assign_grad_op = self.input_grad.assign(np.zeros([self.config.num_steps, self.config.embed_size]))
        assign_input_op = self.input_ids.assign(self.input_placeholder)
        assign_input_op = tf.group(assign_grad_op, assign_input_op)
        
        input_emb_squz = tf.squeeze(input_emb, [0]) #(num_steps, embed_size)
        
        compare_matrix = input_emb_squz + self.input_grad  #(num_steps, embed_size)
#         distance = tf.sqrt(tf.reduce_sum((tf.expand_dims(compare_matrix, 1) - 
#                                           tf.expand_dims(self.embedding, 0))**2, 2)) #(num_steps, vocab_size)
        
        distance = tf.matmul(tf.nn.l2_normalize(compare_matrix, 1), tf.nn.l2_normalize(self.embedding, 1), 
                             transpose_a=False, transpose_b=True) #(num_steps, vocab_size)
        
        nearest_words = tf.argmax(distance, dimension=1) #(num_steps)
        
#         mask = tf.equal( nearest_words, tf.squeeze(self.input_ids, [0])) #(num_steps)
#         mask = tf.expand_dims(mask, 1)      
#         grad_update_op = self.input_grad.assign(tf.to_float(mask) * self.input_grad)
        grad_update_op = self.input_grad.assign(tf.zeros_like(self.input_grad))
        input_update_op = self.input_ids.assign(tf.expand_dims(nearest_words, 0))
        update_op = tf.group(grad_update_op, input_update_op)
        
        
        inputs = [tf.squeeze(x, [1]) for x in tf.split(1, self.config.num_steps, input_emb)]
        return input_emb, update_op, assign_input_op, inputs
    
    def take_gradient(self, loss, input_ids, grad_accu):
        lr = self.config.lr
        grad = tf.gradients(loss, self.input_emb) #(1, num_steps, embed_size)
        grad=grad[0]
        grad_rescaled = tf.nn.l2_normalize(grad, [1, 2]) * lr
        print grad_rescaled
        print tf.squeeze(grad_rescaled, [0])
        get_grad_op = grad_accu.assign_add(tf.squeeze(grad_rescaled, [0]))
        return get_grad_op
###################################################################################################

def train_run():
    logging.info('Training start')
    with tf.Graph().as_default():
        with tf.device("/gpu:" + str(args.gpu_num)):
            classifier = Generate_Model()
        saver = tf.train.Saver({'embed_matrix': classifier.embed_matrix, 'rnn_matrix': classifier.rnn_matrix, 
                                'rnn_bias': classifier.rnn_bias, 'fully_weight': classifier.fully_weight, 'fully_bias': classifier.fully_bias})
        
        config=tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        with tf.Session(config=config) as sess:
            sess.run(tf.initialize_all_variables())
            saver.restore(sess, classifier.weight_Path+'/news.weights')
            
            ret_embed = sess.run(classifier.embedding)
            with open(classifier.weight_Path +'/embed_gen', 'w') as fd:
                for i, word_embed in enumerate(ret_embed):
                    fd.write('{}\t{}\n'.format(classifier.vocab.index_to_word[i], ' '.join([str(j) for j in word_embed])))
                    
            for i in range(len(classifier.test_data_len)):
                print '='*30
                data_x = classifier.test_data_x[i]
                data_y = classifier.test_data_y[i]
                data_len = classifier.test_data_len[i]
                classifier.run_epoch(sess, data_x, data_y, data_len)
                print '*'*30
                
            ret_embed = sess.run(classifier.embedding)
            with open(classifier.weight_Path +'/embed_gen_2', 'w') as fd:
                for i, word_embed in enumerate(ret_embed):
                    fd.write('{}\t{}\n'.format(classifier.vocab.index_to_word[i], ' '.join(' '.join([str(j) for j in word_embed]))))
                
    logging.info("Training complete")

def main(_):
    logging.basicConfig(filename=args.weight_path+'/run.log', format='%(levelname)s %(asctime)s %(message)s', level=logging.INFO)
    if args.train_test == "train":
        train_run()

if __name__ == '__main__':
    tf.app.run()
