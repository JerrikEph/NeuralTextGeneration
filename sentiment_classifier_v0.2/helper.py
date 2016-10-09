import numpy as np
import operator
from collections import defaultdict
import logging

class Vocab(object):
    def __init__(self, unk='<unk>'):
        self.word_to_index = {}
        self.index_to_word = {}
        self.word_freq = defaultdict(int)
        self.total_words = 0
        self.unknown = unk
        self.add_word(self.unknown, count=0)

        
    def add_word(self, word, count=1):
        if word not in self.word_to_index:
            index = len(self.word_to_index)
            self.word_to_index[word] = index
            self.index_to_word[index] = word
        self.word_freq[word] += count

        
    def construct(self, words):
        for word in words:
            self.add_word(word)
        self.total_words = float(sum(self.word_freq.values()))
        print '{} total words with {} uniques'.format(self.total_words, len(self.word_freq))
 

    def limit_vocab_length(self, length):
        """
        Truncate vocabulary to keep most frequent words
        
        Args:
            None
            
        Returns:
            None 
        """
        if length > self.__len__():
            return
        new_word_to_index = {self.unknown:0}
        new_index_to_word = {0:self.unknown}
        self.word_freq.pop(self.unknown)          #pop unk word
        sorted_tup = sorted(self.word_freq.items(), key=operator.itemgetter(1))
        sorted_tup.reverse()
        vocab_tup = sorted_tup[:length]
        self.word_freq = dict(vocab_tup)
        for word in self.word_freq:
            index = len(new_word_to_index)
            new_word_to_index[word] = index
            new_index_to_word[index] = word
        self.word_to_index = new_word_to_index
        self.index_to_word = new_index_to_word
        self.word_freq[self.unknown]=0
        
        
    def save_vocab(self, filePath):
        """
        Save vocabulary a offline file
        
        Args:
            filePath: where you want to save your vocabulary, every line in the 
            file represents a word with a tab seperating word and it's frequency
            
        Returns:
            None 
        """
        self.word_freq.pop(self.unknown)
        sorted_tup = sorted(self.word_freq.items(), key=operator.itemgetter(1))
        sorted_tup.reverse()
        with open(filePath, 'w') as fd:
            for (word, freq) in sorted_tup:
                fd.write('{}\t{}\n'.format(word, freq))
            

    def load_vocab_from_file(self, filePath, sep='\t'):
        """
        Truncate vocabulary to keep most frequent words
        
        Args:
            filePath: vocabulary file path, every line in the file represents 
                a word with a tab seperating word and it's frequency
            
        Returns:
            None 
        """
        with open(filePath, 'r') as fd:
            for line in fd:
                word, freq = line.split(sep)
                index = len(self.word_to_index)
                self.word_to_index[word] = index
                self.index_to_word[index] = word
                self.word_freq[word] = freq
            print 'load from <'+filePath+'>, there are {} words in dictionary'.format(len(self.word_freq))
 

    def encode(self, word):
        if word not in self.word_to_index:
            word = self.unknown
        return self.word_to_index[word]

    
    def decode(self, index):
        return self.index_to_word[index]

    
    def __len__(self):
        return len(self.word_to_index)

def load_data(filePath, multi_label=False, tagSep=','):
    """
    Load dataset from file,
    each line in the file should be structured as "label1<tagSep>label2<sep>dataItem"

    Args:
        filePath: where the data resides
        multi_label: if set to True then should process multi 
                    tag return dict with tuple of tag as its key
        tagSep: label separator

    Returns:
        tagTupItem: a list of tuples structured as ((label1, label2, ...), dataItem)
    """
    tagTupItem = []
    with open(filePath, 'r') as fd:
        for line in fd:
            receive = line.split()
            if multi_label:
                tagTupItem.append((tuple(receive[0].strip().split(tagSep)), receive[1:])) #store tag as tuple
            else:
                tagTupItem.append((receive[0].strip(), receive[1:]))   #store tag as string
    return tagTupItem

def mkDataSet(tagTupItem, tag2id):
    label_matrix = []
    data_matrix = []
    for (tags, dataItem) in tagTupItem:
        label = [tag2id[tags]]
        label_matrix.append(label)
        data_matrix.append(dataItem)
    return label_matrix, data_matrix

def encodeNpad(dataList, vocab, trunLen=0):
    sentLen = []
    data_matrix = []
    for wordList in dataList:
        length = len(wordList)
        if trunLen !=0:
            length=min(length, trunLen)
        sentEnc = []
        if trunLen == 0:
            for word in wordList:
                sentEnc.append(vocab.encode(word))
        else:
            for i in range(trunLen):
                if i < length:
                    sentEnc.append(vocab.encode(wordList[i]))
                else:
                    sentEnc.append(vocab.encode(vocab.unknown))
        sentLen.append(length)
        data_matrix.append(sentEnc)
    return sentLen, data_matrix
            
def load_tag(tagPath):
    """
    Load tag from file

    Args:
        tagPath: path of file which store tags

    Returns:
        tag2id: a dict map tag to id
        id2tag: a dict map id to tag
    """
    tag2id={}
    id2tag={}
    with open(tagPath, 'r') as fd:
        for line in fd:
            tag = line.strip()
            index = len(tag2id)
            tag2id[tag]=index
            id2tag[index]=tag
    return tag2id, id2tag
    
def flatten(li):
    ret = []
    for item in li:
        if isinstance(item, list) or isinstance(item, tuple):
            ret += flatten(item)
        else:
            ret.append(item)
    return ret

def pred_from_probability(prob_matrix, label_num):
    """
    Load tag from file

    Args:
        prob_matrix: probability matrix have the shape of (data_num, class_num), 
            type of float. Generated from softmax activation
        label_num: specify how much positive class to pick, have the shape of (data_num), type of int

    Returns:
        ret: for each case, set all positive class to 1, shape of(data_num, class_num)
    """
    order = np.argsort(prob_matrix)
    ret = np.zeros_like(prob_matrix, np.int32)
    
    for i in range(len(label_num)):
        ret[i][order[i][-label_num[i]:]]=1
    return ret

def pred_from_probability_sig(prob_matrix, threshold=0.5):
    """
    Load tag from file

    Args:
        prob_matrix: probability matrix have the shape of (data_num, class_num), 
            type of float. Generated from sigmoid activation
        threshold: when larger than threshold, consider it as true or else false
    Returns:
        ret: for each case, set all positive class to 1, shape of(data_num, class_num)
    """
    np_matrix = np.array(prob_matrix)
    ret = (np_matrix > threshold)*1
    return ret


def calculate_accuracy(pred_matrix, label_matrix):
    """
    Load tag from file

    Args:
        pred_matrix: prediction matrix shape of (data_num, class_num), type of int
        label_matrix: true label matrix, same shape and type as pred_matrix

    Returns:
        accuracy: accuracy of the prediction, type float
    """
    def ListMatch(a, b):
        for i in range(len(a)):
            if a[i] != b[i]:
                return False
        return True
    data_num = len(label_matrix)
    match = 0
    for i in range(len(label_matrix)):
        if ListMatch(pred_matrix[i], label_matrix[i]):
            match+=1
    return float(match)/data_num

def calculate_accuracy_from_prob(prob_matrix, label_matrix):
    label_num = np.sum(label_matrix, axis=1)
    pred_matrix = pred_from_probability(prob_matrix, label_num)
    return calculate_accuracy(pred_matrix, label_matrix)

def readEmbedding(fileName):
    """
    Read Embedding Function
    
    Args:
        fileName : file which stores the embedding
    Returns:
        embeddings_index : a dictionary contains the mapping from word to vector
    """
    embeddings_index = {}
    with open(fileName, 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

def mkEmbedMatrix(embed_dic, vocab_dic):
    """
    Construct embedding matrix
    
    Args:
        embed_dic : word-embedding dictionary
        vocab_dic : word-index dictionary
    Returns:
        embedding_matrix: return embedding matrix
    """
    if type(embed_dic) is not dict or type(vocab_dic) is not dict:
        raise TypeError('Inputs are not dictionary')
    if len(embed_dic) < 1 or len(vocab_dic) <1:
        raise ValueError('Input dimension less than 1')
    
    EMBEDDING_DIM = len(embed_dic.items()[0][1])
    embedding_matrix = np.zeros((len(vocab_dic) + 1, EMBEDDING_DIM), dtype=np.float32)
    for word, i in vocab_dic.items():
        embedding_vector = embed_dic.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
    
def makeMask(steps, lengths):
    """
    Make a embedding mask, meant to mask out those paddings
    
    Args:
        steps: step size
        lengths: lengths
    Returns:
        ret: mask matrix, type ndarray
    
    """
    ret = np.zeros([len(lengths), steps])
    for i in range(len(lengths)):
        for j in range(steps):
            if(j<lengths[i]):
                ret[i, j]=1
    return ret

def data_iter(data_x, data_y, len_list, batch_size):
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    len_list = np.array(len_list)
    
    data_len = len(data_x)
    epoch_size = data_len // batch_size
    
    idx = np.arange(data_len)
    np.random.shuffle(idx)
    
    for i in xrange(epoch_size):
        indices = range(i*batch_size, (i+1)*batch_size)
        indices = idx[indices]
        ret_x, ret_y, ret_len = data_x[indices], data_y[indices], len_list[indices]
        yield (ret_x, ret_y, ret_len)

def pred_data_iter(data_x, len_list, batch_size):
    data_x = np.array(data_x)
    len_list = np.array(len_list)
    
    data_len = len(data_x)
    epoch_size = data_len // batch_size

    
    for i in xrange(epoch_size):
        ret_x, ret_len = data_x[i*batch_size: (i+1)*batch_size], len_list[i*batch_size: (i+1)*batch_size]
        yield (ret_x, ret_len)
    if epoch_size*batch_size < data_len:
        ret_x, ret_len = data_x[epoch_size*batch_size:], len_list[epoch_size*batch_size:]
        yield (ret_x, ret_len) 

def print_confusion(confusion, num_to_tag):
    """Helper method that prints confusion matrix."""
    # Summing top to bottom gets the total number of tags guessed as T
    total_guessed_tags = confusion.sum(axis=0)
    # Summing left to right gets the total number of true tags
    total_true_tags = confusion.sum(axis=1)
    total_tags = confusion.sum()
    def log_confuse_matrix(confusion, num_to_tag):
        logstr = '\n'
        logstr+='\t'
        for i, outer in enumerate(confusion):
            logstr+= '\t{:.6}'.format(num_to_tag[i])
        logstr+="\n"
        for i, outer in enumerate(confusion):
            logstr+='{:.6}\t'.format(num_to_tag[i])
            for inner in outer:
                logstr += '{:6}\t'.format(inner)
            logstr +='\n'
            
        logstr += "total precision: {}\n".format(np.trace(confusion) / total_tags.astype(float))
        logstr += '{:15}\t{:7}\t{:7}\n'.format('TAG', 'Prec', 'Recall')
        for i, tag in sorted(num_to_tag.items()):
            prec = confusion[i, i] / float(total_guessed_tags[i])
            recall = confusion[i, i] / float(total_true_tags[i])
            logstr += '{:15}\t{:2.4f}\t{:2.4f}\n'.format(tag, prec, recall)
        logging.info(logstr)
        
    def print_confuse_matrix(confusion, num_to_tag, truncate=10):
        logstr = '\n'
        logstr+=''
        for i, outer in enumerate(confusion[:truncate, :truncate]):
            logstr+= '\t{:.6}'.format(num_to_tag[i])
        logstr+="\n"
        for i, outer in enumerate(confusion[:truncate, :truncate]):
            logstr+='{:.6}\t'.format(num_to_tag[i])
            for inner in outer:
                logstr += '{:6}\t'.format(inner)
            logstr +='\n'
        
        logstr += "total precision: {}\n".format(np.trace(confusion) / total_tags.astype(float))
        logstr += '{:15}\t{:7}\t{:7}\n'.format('TAG', 'Prec', 'Recall')
        for i, tag in sorted(num_to_tag.items()):
            prec = confusion[i, i] / float(total_guessed_tags[i])
            recall = confusion[i, i] / float(total_true_tags[i])
            logstr += '{:15}\t{:2.4f}\t{:2.4f}\n'.format(tag, prec, recall)
        print logstr
    
    
    log_confuse_matrix(confusion, num_to_tag)
    print_confuse_matrix(confusion, num_to_tag)

def calculate_confusion(pred_matrix, label_matrix):
    """Helper method that calculates confusion matrix."""
    label_size = len(pred_matrix[0])
    confusion = np.zeros((label_size, label_size), dtype=np.int32)
    for i in xrange(len(label_matrix)):
        correct_label = [k for k, j in enumerate(label_matrix[i]) if j==1]
        guessed_label = [k for k, j in enumerate(pred_matrix[i]) if j==1]
        confusion[correct_label, guessed_label] += 1
    return confusion

