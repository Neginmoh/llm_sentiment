# llm_sentiment/pipeline_file.py


import torch
from transformers import pipeline
from llm_sentiment.config import Config
from llm_sentiment.model import ModelManage

class PipeLine:
    '''
    Initialize and configures a pipeline to streamline the workflow and then uses the pipeline to the specified task (text classification) on the provided document.
    '''
    def __init__(self, model_choice):
        '''
        Initialize the HuggingFace pipeline

        The self.task to be performed by pipeline is set according to Config.TASK.
        The selected model is obtained from the input provided to PipeLine instance and saved in self.model_choice.
        An instance of ModelManage is created with the self.model_choice model.
        The pipeline model is assigned from the model attribute of the ModelManage instance.
        The pipeline tokenizer is assigned from the model attribute of the ModelManage instance.
        The pipeline generator self.pipe is created according to the settings.
        
        Args:
            model_choice (dict): The configuration dictionary of the LLM model to be used from the Config class.
        '''
        self.task = Config.TASK
        self.model_choice = model_choice
        self.model_manage = ModelManage(self.model_choice)
        self.model = self.model_manage.model
        self.tokenizer = self.model_manage.tokenizer

        self.pipe = pipeline(self.task,
                        model=self.model,
                        tokenizer=self.tokenizer,
                        torch_dtype=torch.bfloat16,
                        device_map=Config.DEVICE_MAP)

    def get_sentiment(self, dataset):
        '''     
        - Customizes the model configuration by applying re_label() method on the ModelManage instance.
        - Generates a list of sentiments associated with each example of the input dataset by applying self.pipe on 'sentence' column of dataset
        - The process is run batch-by-batch and results are saved to output_list list.
        
        Args:
            dataset (datasets.Dataset): a HuggingFace dataset containing data to be used.
        Returns:
            list: The list of sentiments of each example of the dataset.
        '''
        self.model_manage.re_label() 
        output_list = []
        
#         Or you could use self.pipe(KeyDataset(dataset, Config.COLUMN_TEXT),..) after executing from transformers.pipelines.pt_utils import KeyDataset:      
        for output in self.pipe(dataset[Config.COLUMN_TEXT],
                                truncation=Config.TRUNCATION,
                                padding=Config.PADDING,
                                max_length=Config.MAX_LENGTH,
                                batch_size=Config.BATCH_SIZE):
            output_list.append(output)
            
        return output_list