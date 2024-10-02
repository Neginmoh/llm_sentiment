# llm_sentiment/workflow.py


from llm_sentiment.config import Config
from llm_sentiment.prediction import Prediction
from llm_sentiment.dataframe import DataFrameBuilder
from llm_sentiment.dataset_loader import DataSetLoader


class WorkFlow:
    '''
    This class handles the workflow of loading dataset, making predictions, creating a output DataFrame, and exporting results as CSV file.
    '''

    def __init__(self, category, model_list, dataset_name, dataframe_type):
        '''
        Initializes the necessary attributes based on the inputs provided to the class object.
        '''
        self.category = category
        self.model_list = model_list
        self.dataset_name = dataset_name
        self.dataframe_type = dataframe_type

    def run(self):
        '''
        This method:
        - Creates a DataSetLoader object with self.dataset_name as input
        - Loads the dataset by calling dataset_load() on DataSetLoader object
        - Creates a Prediction object with self.category, self.model_list, and dataset passed as inputs
        - Performs predictions by calling predict() method on Prediction object
        - Passes the Prediction object as input to to DataFrameBuilder class to build an instance
        - If selected self.dataframe_type is 'MultiIndex', then multi_index() method is called on DataFrameBuilder object to create MultiIndex DataFrame of the outputs
        - If selected self.dataframe_type is 'Regular', then regular() method is called on DataFrameBuilder object to create Regular DataFrame of the outputs
        - the created DataFrame self.df is returned
        
        Ags:
            None
        Returns:
            pd.DataFrame: a multiIndex or regular DataFrame with initial text data, sentiment predictions, scores using for each model.
        '''
        dsl = DataSetLoader(self.dataset_name)
        dataset = dsl.dataset_load()
        predictor = Prediction(self.category, self.model_list, dataset)
        predictor.predict()
        dfb = DataFrameBuilder(predictor)
        if self.dataframe_type == 'MultiIndex':
            self.df = dfb.multi_index()
        else:
            self.df = dfb.regular()
            
        return self.df
    
    def save(self):
        '''
        This method exports the output DataFrame self.df to the directory at Config.OUTPUT_DATASET_PATH when called on the self.df
        '''
        self.df.to_csv(Config.OUTPUT_DATASET_PATH)
        print(f'\nOutput dataset has been saved at {Config.OUTPUT_DATASET_PATH}.\n')



