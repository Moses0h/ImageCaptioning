################################################################################
# CSE 253: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################

import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import string
import os

# Make sure to use the GPU. The following line is just a check to see if GPU is availables
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: ", device)

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class Config():
    """ Simple data wrapper. """
    def __init__(self, config):
        self.temp = config['generation']['temperature']
        self.max_length = config['generation']['max_length']
        self.hidden_size = config['model']['hidden_size']
        self.deterministic = config['generation']['deterministic']
        self.embedding_size = config['model']['embedding_size']
        self.model_type = config['model']['model_type']
        if self.model_type == 'Architecture2':
            self.image_embedding_size = config['model']['image_embedding_size']
        self.images_root_dir = config['dataset']['images_root_dir']

# Build and return the model here based on the configuration.
def get_model(config_data, vocab):
    model_type = config_data['model']['model_type']
    if model_type == 'LSTM':
        return Baseline(Config(config_data), vocab)
    elif model_type == 'RNN':
        return RNN(Config(config_data), vocab)
    elif model_type == 'Architecture2':
        return Architecture2(Config(config_data), vocab)
    else:
        raise NotImplementedError

def forward(model, images, captions):
    """
        Forwards images through encoder and LSTMCell or RNNCell using teacher forcing
        Outputs the one-hot vectors for each word of the captions in the batch
    """
    encoder_output = model.encoder(images)
    captions_embed = model.embed(captions)

    curr_bs = encoder_output.shape[0]
    mem_state = model.init_mem_state(curr_bs)
    outputs = to_device(torch.empty((curr_bs, captions.shape[1], len(model.vocab))), device)

    # Use encoder output
    inp = model.prepare_input(curr_bs, eout=encoder_output, init=True)
    mem_state = model.cell(inp, mem_state)
    # Convert to one-hot word vector
    fc_output = model.fc(model.get_hidden_state(mem_state))
    outputs[:,0,:] = fc_output
    for i in range(captions.shape[1]-1): # Go through each word of caption except end
        # Use the ith target captions 
        inp = model.prepare_input(curr_bs, cin=captions_embed[:, i, :], eout=encoder_output)
        mem_state = model.cell(inp, mem_state)
        # Convert to one-hot word vector
        fc_output = model.fc(model.get_hidden_state(mem_state))
        outputs[:,i+1,:] = fc_output
    return outputs

def next_word(model, output, det=None, temp=None):
    """
    Selects next word (an index) from one-hot encoded input based on whether
    it's deterministic and its temperature
    """
    if det is None:
        det = model.c.deterministic
    if temp is None:
        temp = model.c.temp
    if det:
        return torch.argmax(output, dim=1)
    return torch.multinomial(F.softmax(output/temp, dim=1), 1).squeeze(1)

def generate(model, images, det=None, temp=None):
    """
    Generates captions represented as indices into the vocabulary
    by passing in encoded image and let the network run on its own (passes
    outputs of cell into input of cell at next time step.
    """
    encoder_output = model.encoder(images)

    curr_bs = encoder_output.shape[0]
    mem_state = model.init_mem_state(curr_bs)
    captions = to_device(torch.empty((curr_bs, model.c.max_length)), device)

    reached_end = to_device(torch.zeros(curr_bs, dtype=torch.bool), device)
    
    prev_output = None
    for i in range(model.c.max_length):
        if i == 0: # First Cell's input is special (image + (for Architecture 2) padding)
            inp = model.prepare_input(curr_bs, eout=encoder_output, init=True)
            mem_state = model.cell(inp, mem_state)
        else: # Everything else use previous output + (for Architecture 2) image
            inp = model.prepare_input(curr_bs, cin=prev_output, eout=encoder_output)
            mem_state = model.cell(inp, mem_state)
        fc_output = model.fc(model.get_hidden_state(mem_state))
        output_word = next_word(model, fc_output, det=det, temp=temp) # Get indices of output word

        # Save indices - pad if end has been seen before
        captions[:,i] = reached_end*model.vocab.word2idx['<pad>'] + ~reached_end*output_word
        # Marks the image's caption as done
        reached_end+= output_word == model.vocab.word2idx['<end>']
        prev_output = model.embed(output_word) # Indices -> embedding for next cell

    return captions


def generate_caption(exp, n=10):
    """
    Takes an experiment and uses generate to generate captions like in report 11
    for n images
    """
    mod = to_device(exp._Experiment__best_model, device)
    mod.eval()
    removeList = ['<start>', '<end>', '<pad>', '<unk>']
    root_test = os.path.join(mod.c.images_root_dir, 'test')

    n = max(0, n)

    with torch.no_grad():
        for i, (images, captions, img_ids) in enumerate(exp._Experiment__test_loader):
            ids = np.random.choice(len(images), size= min(len(images), n), replace=False)
            for data_id in ids:
                img_id = img_ids[data_id]
                path = exp._Experiment__coco_test.loadImgs(img_id)[0]['file_name'];
                image = Image.open(os.path.join(root_test, path)).convert('RGB')
                plt.imshow(image)
                plt.axis('off')
                plt.show()
                print("Actual Captions:")
                for ann in exp._Experiment__coco_test.imgToAnns[img_id]:
                    print(ann['caption'].lower())
                outputs = generate(mod, images[data_id:data_id+1].cuda())
                caption = [exp._Experiment__vocab.idx2word[ind.item()].lower() for ind in outputs[0]]
                hypothesis = [word for word in caption if word not in string.punctuation and word not in removeList]
                print(f"\nPredicted caption: {' '.join(hypothesis)}")
            n-=len(ids)
            if n <= 0:
                return


# Initialize weights
def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class Encoder(nn.Module):
    """ Wraps ResNet50 with weight initialization for the added linear layer """
    def __init__(self, embed_size):
        super(Encoder, self).__init__()
        self.model = models.resnet50(pretrained=True)
        for param in self.model.parameters(): # no old weights should be trained
            param.requires_grad = False

        self.model.fc = nn.Linear(self.model.fc.in_features, embed_size, bias=True)
        self.model.fc.apply(init_weights)

    def __call__(self, images):
        return self.forward(images)

    def forward(self, images):
        return self.model(images)

class Baseline(nn.Module):
    """
    Baseline LSTM Model
    Each class needs to define init_mem_state, get_hidden_state, and
    prepare_input for forward and generate (see above) to work on them
    """
    def __init__(self, config, vocab):
        super(Baseline, self).__init__()
        self.c = config
        self.vocab = vocab
        self.encoder = Encoder(config.embedding_size)

        self.cell = nn.LSTMCell(config.embedding_size, config.hidden_size)
        self.fc = nn.Linear(config.hidden_size, len(vocab))
        self.embed = nn.Embedding(len(vocab), config.embedding_size)

    def init_mem_state(self, curr_bs):
        return (to_device(torch.zeros((curr_bs, self.c.hidden_size)), device),
                to_device(torch.zeros((curr_bs, self.c.hidden_size)), device))

    def get_hidden_state(self, mem_state):
        return mem_state[0]

    def prepare_input(self, bs, cin=None, eout=None, init=False):
        """
            cin - Cell input which is either embedded captions, previous output, or padding
            eout - Encoder output
            init - First cell or not
            Returns appropriate input based on architecture
        """
        if init:
            return eout
        return cin

    def forward(self, images, captions):
        return forward(self, images, captions)

    def generate(self, images):
        return generate(self, images)

class RNN(nn.Module):
    """
    RNN model - Uses RNNCell and just hidden units for hidden state
    """
    def __init__(self, config, vocab):
        super(RNN, self).__init__()
        self.c = config
        self.vocab = vocab
        self.encoder = Encoder(config.embedding_size)

        self.cell = nn.RNNCell(config.embedding_size, config.hidden_size)
        self.fc = nn.Linear(config.hidden_size, len(vocab))
        self.embed = nn.Embedding(len(vocab), config.embedding_size)

    def init_mem_state(self, curr_bs):
        return to_device(torch.zeros((curr_bs, self.c.hidden_size)), device)

    def get_hidden_state(self, mem_state):
        return mem_state

    def prepare_input(self, bs, cin=None, eout=None, init=False):
        """
            cin - Cell input which is either embedded captions, previous output, or padding
            eout - Encoder output
            init - First cell or not
            Returns appropriate input based on architecture
        """
        if init:
            return eout
        return cin

    def forward(self, images, captions):
        return forward(self, images, captions)

    def generate(self, images):
        return generate(self, images)

class Architecture2(nn.Module):
    """
    Architecture 2 - Same as Baseline except prepare_input needs
    to send in image every time step
    """
    def __init__(self, config, vocab):
        super(Architecture2, self).__init__()
        self.c = config
        self.vocab = vocab
        self.encoder = Encoder(config.image_embedding_size)

        self.cell = nn.LSTMCell(config.embedding_size + config.image_embedding_size, config.hidden_size)
        self.fc = nn.Linear(config.hidden_size, len(vocab))
        self.embed = nn.Embedding(len(vocab), config.embedding_size)

    def init_mem_state(self, curr_bs):
        return (to_device(torch.zeros((curr_bs, self.c.hidden_size)), device),
                to_device(torch.zeros((curr_bs, self.c.hidden_size)), device))

    def get_hidden_state(self, mem_state):
        return mem_state[0]

    def prepare_input(self, bs, cin=None, eout=None, init=False):
        """
            cin - Cell input which is either embedded captions, previous output, or padding
            eout - Encoder output
            init - First cell or not
            Returns appropriate input based on architecture
        """
        if init:
            pad = torch.ones(bs, dtype=torch.long)*self.vocab.word2idx['<pad>']
            return self.combine_embeddings(self.embed(to_device(pad, device)), eout)
        return self.combine_embeddings(cin, eout)

    def combine_embeddings(self, caption_embeddings, image_embeddings):
        return torch.cat((caption_embeddings, image_embeddings), 1)
        
    def forward(self, images, captions):
        return forward(self, images, captions)

    def generate(self, images):
        return generate(self, images)
