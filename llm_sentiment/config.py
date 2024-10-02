# config.py


import os

class Config:
    
#     MODEL_MAX_LENGTH = 512  # maximum number of tokens for the model input. Use this if MAX_LENGTH is not working properly.
    MAX_LENGTH = 512  # maximum number of output tokens
    TASK = "sentiment-analysis"  # task performed by LLM model. (alias "text-classification")  
    DEVICE_MAP = "auto"  # select appropriate available devices when using pipeline
    TRUNCATION = True  # if True truncate the input to a maximum length specified by the max_length argument or the model_max_length if no max_length is provided 
    PADDING = True # if True pads inputs to the longest sequence in the batch.
    
    DATASET_NAME = "takala/financial_phrasebank" # dataset name to be loaded from Hugging Face Hub
    DATASET_SPLIT = 'train' # split of the dataset to be loaded
    DATASET_CONFIG = "sentences_allagree" # dataset configuration to be used
    COLUMN_TEXT = "sentence" # dataset column that contains the text to be used for the specified task
    COLUMN_LABEL = "label" # dataset column that contains the ground truth label associated with the specified task

    BATCH_SIZE = 10 # controls the batch size which is the number of examples to be processed at the same time
    DECIMAL_PLACE = 2 # number of decimal places for rounding prediction scores displayed in DataFrame
    
    MODEL_1 = {"ID": "finiteautomata/bertweet-base-sentiment-analysis",
               "NAME": "BERTweet",
               "LABEL2ID": {'negative': 0, 'neutral': 1, 'positive': 2},
               "ID2LABEL": {0: 'negative', 1: 'neutral', 2: 'positive'}}
    # a dictionary containing the necessary information related to the first model   
       
    MODEL_2 = {"ID": "cardiffnlp/twitter-roberta-base-sentiment",
               "NAME": "TwitterRoBERTa",
               "LABEL2ID": {'negative': 0, 'neutral': 1, 'positive': 2},
               "ID2LABEL": {0: 'negative', 1: 'neutral', 2: 'positive'}}
    # a dictionary containing the necessary information related to the second model

    BASE_DIR = os.getcwd()  # base directory
    OUTPUT_DIR = os.path.join(BASE_DIR, 'output')  # output directory within base directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)  # create a output directory if does not exist within data directory
    OUTPUT_DATASET_PATH =  os.path.join(OUTPUT_DIR,'output_dataset.csv')  # path to the final output dataset containing the generated summaries