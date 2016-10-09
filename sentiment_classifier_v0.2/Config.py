from statsmodels.discrete.tests.test_sandwich_cov import filepath
import ConfigParser
import json
class Config(object):
    """Holds model hyperparams and data information.
    
    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    """General"""
    pre_trained=False
    vocab_size=50000
    batch_size = 256
    embed_size = 200
    max_epochs = 50
    early_stopping = 5
    dropout = 0.9
    lr = 0.01
    decay_steps = 500
    decay_rate = 0.9
    class_num=1
    reg=0.2
    num_steps = 40
    
    """lstm"""
    hidden_size = 300
    rnn_numLayers=1
    
    def saveConfig(self, filePath):
        cfg = ConfigParser.ConfigParser()
        cfg.add_section('General')
        cfg.add_section('lstm')
        
        cfg.set('General', 'pre_trained', self.pre_trained)
        cfg.set('General', 'vocab_size', self.vocab_size)
        cfg.set('General', 'batch_size', self.batch_size)
        cfg.set('General', 'embed_size', self.embed_size)
        cfg.set('General', 'max_epochs', self.max_epochs)
        cfg.set('General', 'early_stopping', self.early_stopping)
        cfg.set('General', 'dropout', self.dropout)
        cfg.set('General', 'lr', self.lr)
        cfg.set('General', 'decay_steps', self.decay_steps)
        cfg.set('General', 'decay_rate',self.decay_rate)
        cfg.set('General', 'class_num', self.class_num)
        cfg.set('General', 'reg', self.reg)
        cfg.set('General', 'num_steps', self.num_steps)
        
        cfg.set('lstm', 'hidden_size', self.hidden_size)
        cfg.set('lstm', 'rnn_numLayers', self.rnn_numLayers)
        
        with open(filePath, 'w') as fd:
            cfg.write(fd)
        
    def loadConfig(self, filePath):
        cfg = ConfigParser.ConfigParser()
        cfg.read(filePath)
        
        self.pre_trained = cfg.getboolean('General', 'pre_trained')
        self.vocab_size = cfg.getint('General', 'vocab_size')
        self.batch_size = cfg.getint('General', 'batch_size')
        self.embed_size = cfg.getint('General', 'embed_size')
        self.max_epochs = cfg.getint('General', 'max_epochs')
        self.early_stopping = cfg.getint('General', 'early_stopping')
        self.dropout = cfg.getfloat('General', 'dropout')
        self.lr = cfg.getfloat('General', 'lr')
        self.decay_steps = cfg.getint('General', 'decay_steps')
        self.decay_rate = cfg.getfloat('General', 'decay_rate')
        self.class_num = cfg.getint('General', 'class_num')
        self.reg = cfg.getfloat('General', 'reg')
        self.num_steps = cfg.getint('General', 'num_steps')
        
        self.hidden_size = cfg.getint('lstm', 'hidden_size')
        self.rnn_numLayers = cfg.getint('lstm', 'rnn_numLayers')
                