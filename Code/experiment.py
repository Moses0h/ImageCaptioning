################################################################################
# CSE 253: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################

import matplotlib.pyplot as plt
import numpy as np
import nltk
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
from datetime import datetime
from tqdm import tqdm, trange

from caption_utils import *
from constants import ROOT_STATS_DIR
from dataset_factory import get_datasets
from file_utils import *
from model_factory import get_model, generate
import string


# Class to encapsulate a neural experiment.
# The boilerplate code to setup the experiment, log stats, checkpoints and plotting have been provided to you.
# You only need to implement the main training logic of your experiment and implement train, val and test methods.
# You are free to modify or restructure the code as per your convenience.
class Experiment(object):
    def __init__(self, name):
        config_data = read_file_in_dir('./', name + '.json')
        if config_data is None:
            raise Exception("Configuration file doesn't exist: ", name)

        self.__name = config_data['experiment_name']
        self.__experiment_dir = os.path.join(ROOT_STATS_DIR, self.__name)

        # Load Datasets
        self.__coco_test, self.__vocab, self.__train_loader, self.__val_loader, self.__test_loader = get_datasets(
            config_data)

        # Setup Experiment
        self.__generation_config = config_data['generation']
        self.__epochs = config_data['experiment']['num_epochs']
        self.__temp = config_data['generation']['temperature']
        self.__current_epoch = 0
        self.__training_losses = []
        self.__val_losses = []
        self.__best_model = get_model(config_data, self.__vocab)  # Save your best model in this field and use this in test method.

        self.__debug = False
        self.__test_debug = False

        # Init Model
        self.__model = get_model(config_data, self.__vocab)

        # Sets the Criterion and Optimizers
        self.__criterion = nn.CrossEntropyLoss()
        self.__optimizer = optim.Adam(self.__model.parameters(), lr=config_data['experiment']['learning_rate'])

        self.__init_model()

        # Load Experiment Data if available
        self.__load_experiment()

    # Loads the experiment data if exists to resume training from last saved checkpoint.
    def __load_experiment(self):
        os.makedirs(ROOT_STATS_DIR, exist_ok=True)

        if os.path.exists(self.__experiment_dir):
            self.__training_losses = read_file_in_dir(self.__experiment_dir, 'training_losses.txt')
            self.__val_losses = read_file_in_dir(self.__experiment_dir, 'val_losses.txt')
            self.__current_epoch = len(self.__training_losses)

            state_dict = torch.load(os.path.join(self.__experiment_dir, 'latest_model.pt'))
            self.__model.load_state_dict(state_dict['model'])
            self.__optimizer.load_state_dict(state_dict['optimizer'])

            state_dict = torch.load(os.path.join(self.__experiment_dir, 'best_model.pt'))
            self.__best_model.load_state_dict(state_dict['model'])


        else:
            os.makedirs(self.__experiment_dir)

    def __init_model(self):
        if torch.cuda.is_available():
            self.__model = self.__model.cuda().float()
            self.__criterion = self.__criterion.cuda()

    # Main method to run your experiment. Should be self-explanatory.
    def run(self):
        start_epoch = self.__current_epoch
        for epoch in trange(start_epoch, self.__epochs):  # loop over the dataset multiple times
            start_time = datetime.now()
            self.__current_epoch = epoch
            train_loss = self.__train()
            val_loss = self.__val()
            self.__record_stats(train_loss, val_loss)
            self.__log_epoch_stats(start_time)
            self.__save_model()

    # Performs one training iteration on the whole dataset and returns loss value
    def __train(self):
        self.__model.train()
        training_loss = 0
        batches = 0

        for i, (images, captions, _) in enumerate(tqdm(self.__train_loader)):
            self.__optimizer.zero_grad()
            images = images.cuda()
            captions = captions.cuda()
            outputs = self.__model.forward(images, captions)
            if self.__debug:
                print(f"Train - Epoch: {self.__current_epoch} Output: {self.output_caption(outputs[0])}")
                print(f"Train - Epoch: {self.__current_epoch} Target: {self.target_caption(captions[0])}")
            loss = self.__criterion(outputs.view(-1, len(self.__vocab)), captions.view(-1))
            loss.backward()
            self.__optimizer.step()

            training_loss += loss

            batches+=1.0
        
        return training_loss.item()/batches

    # Performs one Pass on the validation set and returns loss value. Updates the best model.
    def __val(self):
        self.__model.eval()
        val_loss = 0
        batches = 0

        with torch.no_grad():
            for i, (images, captions, _) in enumerate(tqdm(self.__val_loader)):
                images = images.cuda()
                captions = captions.cuda()
                outputs = self.__model.forward(images, captions)
                if self.__debug:
                    print(f"Val - Epoch: {self.__current_epoch} Output: {self.output_caption(outputs[0])}")
                    print(f"Val - Epoch: {self.__current_epoch} Target: {self.target_caption(captions[0])}")
                val_loss += self.__criterion(outputs.view(-1, len(self.__vocab)), captions.view(-1))
                batches+=1.0

        averageLoss = val_loss.item()/batches


        if len(self.__val_losses) > 0:
            if averageLoss < min(self.__val_losses):
                self.__best_model = copy.deepcopy(self.__model)
        else:
            self.__best_model = copy.deepcopy(self.__model)

        return averageLoss

    def output_caption(self, output, deterministic=False):
        """ Deprecated: Debug helper moved to generate_caption in model_factory """
        if deterministic:
            return ' '.join([self.__vocab.idx2word[x.item()] for x in torch.argmax(output, dim=1)])
        return ' '.join([self.__vocab.idx2word[x.item()] for x in torch.multinomial(F.softmax(output/self.__temp, dim=1), 1)])

    def target_caption(self, caption):
        """ Debug helper to generate target captions from data loader """
        return ' '.join([self.__vocab.idx2word[x.item()] for x in caption])

    # TODO: Implement your test function here. Generate sample captions and evaluate loss and
    #  bleu scores using the best model. Use utility functions provided to you in caption_utils.
    #  Note than you'll need image_ids and COCO object in this case to fetch all captions to generate bleu scores.
    def test(self, det=None, temp=None):
        if self.__debug:
            print("Running test set")
        if torch.cuda.is_available():
            self.__best_model = self.__best_model.cuda().float()
        self.__best_model.eval()
        test_loss = 0
        bleu1_value = 0
        bleu4_value = 0
        perplexity = 0

        batches = 0
        test_bleu1 = []
        test_bleu4 = []

        removeList = ['<start>', '<end>', '<pad>', '<unk>']


        with torch.no_grad():
            for iter, (images, captions, img_ids) in enumerate(tqdm(self.__test_loader)):
                images = images.cuda()
                captions = captions.cuda()

                outputs = self.__best_model.forward(images, captions)
                test_loss += self.__criterion(outputs.view(-1, len(self.__vocab)), captions.view(-1))
                batches+=1.0
                
                # Generate captions to calculate bleu scores

                if det is None:
                    det = self.__best_model.c.deterministic
                if temp is None:
                    temp = self.__best_model.c.temp

                output_caps = generate(self.__best_model, images, det=det, temp=temp)
                target_caps = [[nltk.tokenize.word_tokenize(ann['caption'].lower()) for ann in self.__coco_test.imgToAnns[img_id]] for img_id in img_ids]
                target_caps = [[[word for word in sentence if word not in string.punctuation and word not in removeList] for sentence in captionsList] for captionsList in target_caps]

                if self.__test_debug:
                    print(f"Test - Output: {self.target_caption(output_caps[0])}")
                    print(f"Test - Target: {target_caps[0]}")

                # Process the target and predicted similarly
                for i, output_cap in enumerate(output_caps):
                    caption = [self.__vocab.idx2word[ind.item()].lower() for ind in output_cap]
                    hypothesis = [word for word in caption if word not in string.punctuation and word not in removeList]
                    test_bleu1.append(bleu1(target_caps[i], hypothesis))
                    test_bleu4.append(bleu4(target_caps[i], hypothesis))


        test_loss = test_loss.item()/batches

        bleu1_value = sum(test_bleu1)/len(test_bleu1)
        bleu4_value = sum(test_bleu4)/len(test_bleu4)
        
        result_str = "Test Performance: Loss: {}, Perplexity: {}, Bleu1: {}, Bleu4: {}".format(test_loss, perplexity, bleu1_value, bleu4_value)
        self.__log(result_str)

        return test_loss, bleu1, bleu4

    def __save_model(self):
        root_model_path = os.path.join(self.__experiment_dir, 'latest_model.pt')
        model_dict = self.__model.state_dict()
        state_dict = {'model': model_dict, 'optimizer': self.__optimizer.state_dict()}
        torch.save(state_dict, root_model_path)

        root_model_path = os.path.join(self.__experiment_dir, 'best_model.pt')
        best_model_dict = self.__best_model.state_dict()
        best_state_dict = {'model': best_model_dict, 'optimizer': self.__optimizer.state_dict()}
        torch.save(best_state_dict, root_model_path)

    def __record_stats(self, train_loss, val_loss):
        self.__training_losses.append(train_loss)
        self.__val_losses.append(val_loss)

        self.plot_stats()

        write_to_file_in_dir(self.__experiment_dir, 'training_losses.txt', self.__training_losses)
        write_to_file_in_dir(self.__experiment_dir, 'val_losses.txt', self.__val_losses)

    def __log(self, log_str, file_name=None):
        print(log_str)
        log_to_file_in_dir(self.__experiment_dir, 'all.log', log_str)
        if file_name is not None:
            log_to_file_in_dir(self.__experiment_dir, file_name, log_str)

    def __log_epoch_stats(self, start_time):
        time_elapsed = datetime.now() - start_time
        time_to_completion = time_elapsed * (self.__epochs - self.__current_epoch - 1)
        train_loss = self.__training_losses[self.__current_epoch]
        val_loss = self.__val_losses[self.__current_epoch]
        summary_str = "Epoch: {}, Train Loss: {}, Val Loss: {}, Took {}, ETA: {}\n"
        summary_str = summary_str.format(self.__current_epoch + 1, train_loss, val_loss, str(time_elapsed),
                                         str(time_to_completion))
        self.__log(summary_str, 'epoch.log')

    def plot_stats(self):
        e = len(self.__training_losses)
        x_axis = np.arange(1, e + 1, 1)
        plt.figure()
        plt.plot(x_axis, self.__training_losses, label="Training Loss")
        plt.plot(x_axis, self.__val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.legend(loc='best')
        plt.title(self.__name + " Stats Plot")
        plt.savefig(os.path.join(self.__experiment_dir, "stat_plot.png"))
        plt.show()
