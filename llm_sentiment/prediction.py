# llm_sentiment/prediction.py


from llm_sentiment.pipeline_file import PipeLine
from llm_sentiment.classifier import Classifier

class Prediction:
    '''
    This class predicts the sentiment analysis on the dataset using the models in model_list.
    If the category is 'pipeline_classifier' it will output predictions using both pipeline and classifier for each model.
    If the category is 'classifier_classifier' it will output predictions using a classifier for each model.
    '''
    
    def __init__(self, category, model_list, dataset):
        '''
        Initializes the Prediction class.
        
        - self.category should be either: 1.'pipeline_classifier' or 2. 'classifier_classifier'.
            if is 'pipeline_classifier', then for each model in model_list a pipeline and a classifier will be used for prediction
            if is 'classifier_classifier', then for each model in model_list 
        - self.model is the list of models that will be used for prediction
        - self.dataset is the dataset that is used for prediction.            
        '''
        self.category = category
        self.model_list = model_list
        self.dataset = dataset
        
    def predict(self):
        '''
        Performs prediction using self.model_list models according to selected category on the self.dataset
        
        Args:
            None 
        Raises:
            ValueError, if a category other than 'pipeline_classifier' or 'classifier_classifier' is passed as input.
        Returns:
            list: A list containing predictions associated with each model in self.model_list using:
                - both the pipeline and classifier if 'pipeline_classifier' is selected as self.category
                - a classifier if 'classifier_classifier' is selected as self.category
        '''
        pred_both = []
   
        if self.category == 'pipeline_classifier':

            for model_i in self.model_list:
        
                pipe = PipeLine(model_i)
                pred_pipe = pipe.get_sentiment(self.dataset)

                cl = Classifier(model_i)
                pred_cl = cl.get_sentiment(self.dataset)

                pred_both.append([pred_pipe, pred_cl])
            
            self.pred_both=pred_both
            return self.pred_both

        elif self.category == 'classifier_classifier':
            
            for model_i in self.model_list:

                cl = Classifier(model_i)               
                pred_cl = cl.get_sentiment(self.dataset)
                
                pred_both.append(pred_cl)
                
            self.pred_both=pred_both
            return self.pred_both

        else:
            raise ValueError(f"Invalid category '{self.category}'. Allowed categories are:  pipeline_classifier,  classifier_classifier.")
            