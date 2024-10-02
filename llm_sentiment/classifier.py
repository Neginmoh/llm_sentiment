# llm_sentiment/classifier.py


import torch
from torch.utils.data import DataLoader
from llm_sentiment.config import Config
from llm_sentiment.pre_process import PreProcess

class Classifier:
    
    def __init__(self, model_choice):
        '''
        Initializes the Classifier class.
        
        The selected model is obtained from the input provided to Classifier instance and saved in self.model_choice.
        An instance of PreProcess is created by passing self.model_choice as input.
        The classifier model is assigned from the model attribute of the ModelManage object retrieved from the PreProcess instance.
        The classifier tokenizer is assigned from the model attribute of the ModelManage object retrieved from the PreProcess instance.
        If GPU is available the self.device is set to GPU if not, it will use CPU.
        The model is carried to self.device
        '''
        self.model_choice = model_choice
        self.preprocess_inst = PreProcess(self.model_choice)
        self.model_manage = self.preprocess_inst.model_manage
        self.tokenizer = self.model_manage.tokenizer
        self.model = self.model_manage.model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def inference(self, encoded_dataset):
        '''
        Performs model inference by passing encoded input data as input and obtaining predictions as output
        
        - Formats the encoded_dataset by converting to pytorch and keeping only specified columns, 'input_ids' and 'attention_mask'.
        - Loads a pytorch DataLoader object, enabling to run inference batch-by-batch
        - Disables gradient calculation while running inference within the loop
        - Carries the each batch of data to self.device
        - Passes 'input_ids' (containing encoded inputs) ans 'attention_mask' as inputs to the self.model
        - Obtains the logits of the model outputs
        - Applies Softmax on last column to obtain probabilities
        - Obtains the maximum probabilities saved as pred_scores and label ids with maximum probability as pred_labels
        - Detaches the tensor from computation graph and carries outputs back to CPU 
        - Repeat these steps for all batches of data, attaches predicted label ids to all_pred_labels and their scores to all_pred_scores lists
        
        Args:
            encoded_dataset (dict): A dictionary containing encoded tensors for the dataset with 'sentence', 'label', 'input_ids', 'attention_mask' keys.
        Returns:
            tuple: A tuple containing of:
                1. A list of predicted label ids for all examples
                2. A list of predicted scores associated with the predicted label ids for all examples
        '''
        encoded_dataset.set_format("torch", columns=['input_ids', 'attention_mask'])
        dataloader = DataLoader(encoded_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

        all_pred_labels = []
        all_pred_scores = []

        with torch.no_grad():
            for batch in dataloader:

                batch = {k: v.to(self.device) for k, v in batch.items()}
                output = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
                output_logits= output.logits

    
                pred_scores, pred_labels = torch.max(torch.softmax(output_logits, dim=-1), dim=-1)
                pred_labels = pred_labels.detach().cpu()
                pred_scores = pred_scores.detach().cpu()   

                all_pred_labels.extend(pred_labels)
                all_pred_scores.extend(pred_scores)

        return all_pred_labels, all_pred_scores
                            
    def get_label(self, all_preds):
        '''  
        - Customizes the model configuration by applying re_label() method on the ModelManage instance retrieved from PreProcess instance.
        - loops over the all_pred_labels and all_pred_score lists together
        - obtains the label id's and score's values from the tensor
        - obtains the appropriate class label associated with the label id
        - appends tuple of results (labels, scores) to all_preds_relabelled
        
        Args:
            all_preds (tuple): A  tuple containing:
                1. A list of predicted label ids for all examples
                2. A list of predicted scores associated with the predicted label ids for all examples
        Returns:
            list: A list containing tuples associated with each example that includes:
                1. Predicted label class
                2. Predicted score associated with the predicted label class                   
        '''
        self.model_manage.re_label()
        all_pred_labels, all_pred_scores = all_preds
        
        all_preds_relabelled = []
        for label, score in zip(all_pred_labels, all_pred_scores):
            all_preds_relabelled.append([self.model.config.id2label[label.item()], score.item()])
            
        return all_preds_relabelled

    def get_sentiment(self, dataset):
        '''
        - Tokenizes the dataset by applying encoding() method on PreProcess object and saves them to encoded_inputs
        - Passes the encoded_inputs as input to self.inference() method to perform inference and saves the predictions to all_preds
        - Obtains the label class and score for each example by self.get_label() and saves results to all_preds_relabelled
        
        Args:
            dataset (datasets.Dataset): a HuggingFace dataset containing data to be used.
        Returns:
            list: A list containing tuples associated with each example that includes:
                1. Predicted label class
                2. Predicted score associated with the predicted label class 
        '''
        encoded_inputs = self.preprocess_inst.encoding(dataset)        
        all_preds = self.inference(encoded_inputs)
        all_preds_relabelled = self.get_label(all_preds)
        
        return all_preds_relabelled
   