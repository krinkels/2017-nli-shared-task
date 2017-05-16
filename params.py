import pickle

vocabFileDir = "wordVecs/glove.6B.50d.txt"
embed_size = 50
UNK_token = 'UNK' # the unknown token
UNK_index = 0
PAD_index = 400001
UNK_embed = [100] * embed_size # the unknown embedding
PAD_embed = [0] * embed_size   # the padding embedding  
DASH_embed = [50] * embed_size

class Config:
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    #n_word_features = 2 # Number of features for every word in the input.
    #window_size = 1
    #n_features = (2 * window_size + 1) * n_word_features # Number of features for every word in the input.
    n_classes = 400002
    dropout = 0.5
    embed_size = 50
    hidden_size = 300
    batch_size = 32
    n_epochs = 10
    max_grad_norm = 10.
    lr = 0.001

    def __init__(self,maxSeqLen):
#	maxLenRes = pickle.load(open("maxLenRes.pckl", "rb")) 
#	maxLenCont = pickle.load(open("maxLenCont.pckl", "rb"))
#	self.max_length = max(maxLenRes,maxLenCont)
	self.max_length = maxSeqLen
	self.model_output = "model.weights"
	print("Config constructed!")