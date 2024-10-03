# Sentiment Analysis on Financial Data
## Project Description

This project focuses on performing sentiment analysis on financial news. By utilizing large language models (LLMs), we can classify the sentiments expressed in the dataset to determine whether they carry positive, neutral, or negative sentiments. This allows investors and analysts to gain insights into market trends and make informed decisions. We can also predict class labels by utilizing multiple LLMs and comparing their outcome scores. This could be done by utilizing a pipeline which is quick and straightforward, and/or implementing manual inference with a classifier model which allows more flexibility. With proper modifications, the project could extend to other datasets and LLMs that could be used for sentiment analysis.

## Input Data

This project loads a dataset from the Hugging Face Hub to perform a sentiment analysis task. The default dataset that we will use is [takala/financial_phrasebank](https://huggingface.co/datasets/takala/financial_phrasebank), licensed under CC BY-NC-SA 3.0. The dataset consists of sentences from English language financial news where each example is labeled either '0' ('positive'), '1' ('neutral'), or '2'('negative') sentiments. We will use the ```"sentences_allagree"``` configuration as default for this dataset.

The dataset consists of two columns:
1. "sentence": text to be analyzed 
2. "label": ground truth sentiment class label

You may choose to work with a different dataset by specifying associated variables in config.py and modifying dataset configurations as needed.

## Models

We will utilize two models available on Hugging Face Hub:
1. [**finiteautomata/bertweet-base-sentiment-analysis**](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment): A BERT-based model for English tweets, finetuned for sentiment analysis. See the model card for more information.
2. [**cardiffnlp/twitter-roberta-base-sentiment**](https://huggingface.co/finiteautomata/bertweet-base-sentiment-analysis): A roBERTa-base model trained on tweets and fine-tuned for sentiment analysis. See the model card for more information.

We have chosen to work with these two models since they are fine-tuned on sentiment analysis task and their predictions are within three classes of 'positive', 'neutral', and 'negative' sentiments. You may use your preferred models by specifying associated variables in config.py file and modifying model configurations as needed.

## Setup

**Requirements**

Install the required packages:

```python
pip install -r requirements.txt
```

## Usage

### **Run Inference:**

Run the script below and it will prompt you to enter the required inputs:

```python
python main.py
```

You will see the first prompt:
```
Please select a category:

1. pipeline_classifier
2. classifier_classifier

You can select only one of the above:
```
1. If you select ```1```, then the program outputs the results using both pipeline and classifier for each model selected.
2. If you select ```2```, then the program outputs the results using the classifier only for each model selected.

You must enter either ```1``` or ```2```. Otherwise, the prompt will repeat until a valid selection is made.

Then you will see the second prompt:
```
Please select models:

1. BERTweet
2. TwitterRoBERTa
3. Both

You can select only one of the above:
```
1. If you select ```1```, BERTweet will be used as the model.
2. If you select ```2```, TwitterRoBERTa will be used as the model.
3. If you select ```3```, both BERTweet and TwitterRoBERTa will be used as the models.

You must enter either ```1```, ```2```, or ```3```. Otherwise, the prompt will repeat until a valid selection is made.

Next, you will see the third prompt:

```
Please select a DataFrame type:

1. Regular
2. MultiIndex

You can select only one of the above:
```
1. If you select ```1```, a regular Pandas DataFrame containing results will be saved to the specified directory.
2. If you select ```2```, a MultiIndex Pandas DataFrame containing results will be saved to the specified directory.

You must enter either ```1``` or ```2```. Otherwise, the prompt will repeat until a valid selection is made.


### **Setting Variables:**

Most variables and default values could be configured in the Config.py file. Here are the variables:

- ```MAX_LENGTH``` specifies the maximum number of output tokens. The default value is set to 512.
- ```TASK``` specifies the task performed by the LLM model. Here the task we focus on is sentiment analysis, therefore set to ```"sentiment-analysis"``` (or  ```"text-classification"```)  
- ```DEVICE_MAP``` specifies the device that calculations are run on when using the pipeline.. The default value is set to ```"auto"``` which automatically selects the appropriate available devices.
- ```TRUNCATION``` specifies whether or not truncate the input to a maximum length specified by the max_length argument or the model_max_length if no max_length is provided. Default is set to ```True```.
- ```PADDING``` specifies the padding configuration to add padding tokens. The default is set to ```True``` indicating that it pads inputs to the longest sequence in the batch.
- ```DATASET_NAME``` indicates the dataset name to be loaded from Hugging Face Hub. The default dataset used here is "takala/financial_phrasebank".
- ```DATASET_SPLIT``` indicates the split of the dataset to be loaded. The default is set to ```"train"```.
- ```DATASET_CONFIG``` specifies the dataset configuration to be used. Here we use ```"sentences_allagree"``` for the default dataset.
- ```COLUMN_TEXT``` specifies the dataset column that contains the text to be used for the specified task. Here we use ```"sentence"``` column for the default dataset.
- ```COLUMN_LABEL``` specifies the dataset column that contains the ground truth label associated with the specified task. Here the ```"label"``` column contains the ground truth labels for the default dataset.
- ```BATCH_SIZE``` controls the batch size which is the number of examples to be processed at the same time. The default is set to 10.
- ```DECIMAL_PLACE``` controls the number of decimal places for rounding prediction scores displayed in output DataFrame. The default value is set to 2.
- ```MODEL_1``` a dictionary containing the necessary information related to the first model with the following keys:
    - "ID": ID of the model to be loaded from the Hugging Face Hub.
    - "NAME": Name associated with the model.
    - "LABEL2ID": a dictionary linking label classes to the label IDs.
    - "ID2LABEL": a dictionary linking label IDs to label classes.
    
    The default model#1 ```ID``` is set to ```"finiteautomata/bertweet-base-sentiment-analysis"```.

- ```MODEL_2```: similar to ```MODEL_1``` is a dictionary containing information related to the second model. The default model#2 ```ID``` is set to ```"cardiffnlp/twitter-roberta-base-sentiment"```.

- ```BASE_DIR``` sets the base directory.
- ```OUTPUT_DIR``` sets the output directory. By default, the output directory is within the base directory.
- ```OUTPUT_DATASET_PATH``` specifies the path to the output dataset. By default, the output dataset is saved as ```'output_dataset.csv'``` within ```OUTPUT_DIR```.

## Workflow

1. User is prompted to provided the following inputs:
    - Category:
        1. pipeline_classifier
        2. classifier_classifier
    - Models:
        1. BERTweet
        2. TwitterRoBERTa
        3. Both
    - DataFrame type:
        1. Regular
        2. MultiIndex

2. The specified dataset from Hugging Face Hub is loaded.
3. The models selected by the user are utilized to run inference and predict sentiment classes and scores.
4. Depending on the user selection, a classifier and/or pipeline are/is used to make predictions.
5. The output DataFrame is saved in user-specified type in a CSV file. 

## Input and Output Format

### Input format

Below are two examples from [takala/financial_phrasebank](https://huggingface.co/datasets/takala/financial_phrasebank) dataset:

Example #1:
```
{'sentence': 'According to Gran , the company has no plans to move all production to Russia , although that is where the company is growing .',
 'label': 1}
```
Example #2:
```
{'sentence': "For the last quarter of 2010 , Componenta 's net sales doubled to EUR131m from EUR76m for the same period a year earlier , while it moved to a zero pre-tax profit from a pre-tax loss of EUR7m .",
 'label': 2}
```
### Output format

Depending on the user input, a regular or MultiIndex Pandas DataFrame is saved in 'output_dataset.csv' file at ```OUTPUT_DIR```:

Here is a sample output in Pandas DataFrame if first ```1. pipeline_classifier```, then ```3. Both```, and finally ```2. MultiIndex``` are selected by user:

[![dataframe.png](https://i.postimg.cc/k4GYJsM6/dataframe.png)](https://postimg.cc/k6rvwxbq)

Note that, when a MultiIndex DataFrame is exported as CSV file, all levels of MultiIndex will be labeled to ensure correct reading of the MultiIndex DataFrame when loading the CSV file.

## Disclaimer

This repository is intended for educational purposes only and does not provide financial advice. It is provided without any warranty. Predictions made by this repository are not guaranteed. The authors are not responsible for any financial losses, results, or consequences that may arise from using this repository. Use it at your own risk. Please be aware that the code may contain bugs.

## License

Apache License 2.0. See the LICENSE file.