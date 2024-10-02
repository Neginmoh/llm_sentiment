# llm_sentiment/model.py


from transformers import AutoTokenizer, AutoModelForSequenceClassification 
from llm_sentiment.config import Config


class ModelManage:
    '''
    This class handles the model and tokenizer configuration.
    '''
    def __init__(self, model_choice):
        '''
        Initialize the model, model configuration, and tokenizer.

        Args:
            model_choice (dict): The configuration dictionary of the LLM model to be used from the Config class.
        '''
#         self.model_max_length = Config.MODEL_MAX_LENGTH #Use this if MAX_LENGTH is not working properly.
        self.model_choice = model_choice
        self.model_id = self.model_choice['ID']
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        
    def re_label(self):
        '''
        Customizes the ID2LABEL and LABEL2ID configuration of the chosen model.
        '''
        self.model.config.id2label = self.model_choice['ID2LABEL']
        self.model.config.label2id = self.model_choice['LABEL2ID']
        