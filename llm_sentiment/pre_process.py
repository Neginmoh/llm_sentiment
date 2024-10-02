# llm_sentiment/pre_process.py


from llm_sentiment.config import Config
from llm_sentiment.model import ModelManage

class PreProcess:
    '''
    This class preprocess the dataset by performing tokenization batch-by-batch.
    '''
    
    def __init__(self, model_choice):
        '''
        Initialize the PreProcess class

        The self.model_choice is obtained from the input provided to PreProcess instance.
        An instance of ModelManege is created using self.model_choice.
        the model is assigned from ModelManage instance.
        the tokenizer is assigned from ModelManage instance.

        Args:
            model_choice (dict): The configuration dictionary of the LLM model to be used from the Config class.
        '''
        self.model_choice = model_choice
        self.model_manage = ModelManage(self.model_choice)
        self.tokenizer = self.model_manage.tokenizer
        self.model = self.model_manage.model

    def batch_encoding(self, batch):
        '''
        Tokenizes a batch of data.
        
        Args:
            batch (datasets.Dataset): A batch of examples from a Hugging Face dataset.
        Returns:
            dict: A dictionary containing encoded tensors for the input batch with 'sentence', 'label', 'input_ids', 'attention_mask' keys.               
        '''          
        encoded_batch = self.tokenizer(batch[Config.COLUMN_TEXT],
                                        padding=Config.PADDING,
                                        truncation=Config.TRUNCATION,
                                        return_tensors="pt")
        return encoded_batch
    
    def encoding(self, dataset):
        '''
        Maps the self.batch_encoding() method to the dataset.
        
        Args:
            dataset (datasets.Dataset): a HuggingFace dataset containing data to be used.  
        Returns:
            dict: A dictionary containing encoded tensors for the dataset with 'sentence', 'label', 'input_ids', 'attention_mask' keys.               
        '''
        encoded_dataset = dataset.map(self.batch_encoding,
                                      batched=True,
                                      batch_size=Config.BATCH_SIZE)
        return encoded_dataset