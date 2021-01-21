# Image Captioning

To first retrieve the COCO data needed for this project, run every cell in get_datasets.ipynb until the print('done') cell right before the Inspect Trained Model section. 

To run a training and testing of a model, you must define a config JSON file with specified hyperparameters that would indicate what type of model is being trained (LSTM, RNN, or Architecture2 specified in config_data['model']['model_type']).
The config file used should follow the same format as the default.json provided (if Architecture2 model is chosen, an extra "image_embedding_size" parameter must be defined in the model section).

Once this config file is defined, simply follow the usage steps listed below to run the correct experiment. 

Following the run and training of a model, we also have code written in the get_datasets.ipynb that displays test images with their actual labeled captions along with predictions that our model makes. To see this, go to the last cell in the get_datasets.ipynb and define the experiment name the same way you do in the usage section. You can also define n as the number of images you want to test generation on. This cell is crucial for reporting good and bad caption examples for our report. 

## Usage

* Define the configuration for your experiment. See `default.json` to see the structure and available options. 
* Implemented factories to return project specific models, datasets based on config. Add more flags as per requirement in the config.
* After defining the configuration (say `my_exp.json`) - simply run `python3 main.py my_exp` to start the experiment
* The logs, stats, plots and saved models would be stored in `./experiment_data/my_exp` dir. This can be configured in `constants.py`
* To resume an ongoing experiment, simply run the same command again. It will load the latest stats and models and resume training pr evaluate performance.

## Files
- main.py: Main driver class
- experiment.py: Main experiment class. Initialized based on config - takes care of training, saving stats and plots, logging and resuming experiments.
- dataset_factory: Factory to build datasets based on config
- model_factory.py: Factory to build models based on config
- constants.py: constants used across the project
- file_utils.py: utility functions for handling files 
- caption_utils.py: utility functions to generate bleu scores
- vocab.py: A simple Vocabulary wrapper
- coco_dataset: A simple implementation of `torch.utils.data.Dataset` the Coco Dataset
- get_datasets.ipynb: A helper notebook to set up the dataset in your workspace
