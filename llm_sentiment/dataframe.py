# llm_sentiment/dataframe.py


import pandas as pd
from llm_sentiment.config import Config
from llm_sentiment.prediction import Prediction

class DataFrameBuilder:
    '''
    This class creates regular and multiIndex DataFrames of the output results.
    '''
    def __init__(self, predictor_obj):
        
        self.category = predictor_obj.category        
        self.model_list = predictor_obj.model_list
        self.dataset = predictor_obj.dataset
        self.pred_list = predictor_obj.pred_both
        
    def regular(self):
        '''
        The output is a regular DataFrame with initial text data, sentiment predictions, scores using for each model.
        
        Return:
            pd.DataFrame: a regular DataFrame with initial text data, sentiment predictions, scores using for each model.
        Raises:
            ValueError, if a category other than 'pipeline_classifier' or 'classifier_classifier' is passed as input.     
        '''
        dataset_df = pd.DataFrame(self.dataset, columns = [Config.COLUMN_TEXT])
        result_df = dataset_df.copy()
        
        if self.category == 'pipeline_classifier':   
            for i, preds in enumerate(self.pred_list):
                pipe_df = pd.DataFrame(preds[0])
                pipe_df.columns=[f'{self.model_list[i]["NAME"]}_pipeline_sentiment',f'{self.model_list[i]["NAME"]}_pipeline_score']
                
                cl_df = pd.DataFrame(preds[1], columns=[f'{self.model_list[i]["NAME"]}_classifier_sentiment',f'{self.model_list[i]["NAME"]}_classifier_score'])
                
                result_df = pd.concat([result_df, pipe_df, cl_df], axis=1)
                result_df = result_df.round(Config.DECIMAL_PLACE)
            return result_df

        elif self.category == 'classifier_classifier':
            for i, preds in enumerate(self.pred_list):
                cl_df = pd.DataFrame(preds, columns=[f'{self.model_list[i]["NAME"]}_classifier_sentiment',f'{self.model_list[i]["NAME"]}_classifier_score'])
                result_df = pd.concat([result_df, cl_df], axis=1)
                
            return result_df

        else:
            raise ValueError(f"Invalid category '{self.category}'. Allowed categories are:  pipeline_classifier,  classifier_classifier.")
            
    def multi_index(self):
        '''
        The output is a multiIndex DataFrame with initial text data, sentiment predictions, scores using for each model.
        
        Return:
            pd.DataFrame: a multiIndex DataFrame with initial text data, sentiment predictions, scores using for each model.
        Raises:
            ValueError, if a category other than 'pipeline_classifier' or 'classifier_classifier' is passed as input.           
        '''
        result_df = self.regular()
        header_list = [(Config.COLUMN_TEXT, '')]
        
        if self.category == 'pipeline_classifier':
            num_models = (len(result_df.columns) - 1)//4

            for i in range(num_models): 
                header_names = [(f'{self.model_list[i]["NAME"]}_pipeline', 'sentiment'),
                                (f'{self.model_list[i]["NAME"]}_pipeline', 'score'),
                                (f'{self.model_list[i]["NAME"]}_classifier', 'sentiment'),
                                (f'{self.model_list[i]["NAME"]}_classifier', 'score')]
                header_list.extend(header_names)                 
            header_multi_index = pd.MultiIndex.from_tuples(header_list)
            result_df.columns = header_multi_index
            
            return result_df
        
        elif self.category == 'classifier_classifier':
            num_models = (len(result_df.columns) - 1)//2

            for i in range(num_models):
                header_names = [(f'{self.model_list[i]["NAME"]}_classifier', 'sentiment'),
                                (f'{self.model_list[i]["NAME"]}_classifier', 'score')]
                
                header_list.extend(header_names)            
            header_multi_index = pd.MultiIndex.from_tuples(header_list)
            result_df.columns = header_multi_index

            return result_df
        
        else:
            raise ValueError(f"Invalid category '{self.category}'. Allowed categories are:  pipeline_classifier,  classifier_classifier.")
