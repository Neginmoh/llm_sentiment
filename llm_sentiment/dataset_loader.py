# llm_sentiment/data_loader.py


from datasets import load_dataset
from llm_sentiment.config import Config

class DataSetLoader:
    '''
    This class loads the dataset needed for sentiment analysis.
    '''
    def __init__(self, dataset_name):  
        '''
        Initializes the self.dataset_name 
        '''
        self.dataset_name = dataset_name

    def dataset_load(self):
        '''
        Returns:
            dataset (datasets.Dataset): a HuggingFace dataset containing data to be used for predictions.
        Raises:
            FileNotFoundError if self.dataset_name is not found
        '''  
        dataset = load_dataset(self.dataset_name, Config.DATASET_CONFIG, split=Config.DATASET_SPLIT)
        
        if dataset is None:
            raise FileNotFoundError(f"The dataset '{self.dataset_name}' is not found.")        
        
        return dataset
