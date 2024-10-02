# main.py


import sys
import os
from llm_sentiment.config import Config
from llm_sentiment.workflow import WorkFlow
from llm_sentiment.menu import Menu
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    '''
    The main function which prompts the user to input required selections and based on the user selection runs inference and saves output dataset in a csv file.

    steps:
    1. Prompts user to make selections among options by creating a Menu instance and calling run() method on it.
    2. Collect the user inputs and sets the category, models and DataFrame type.
    3. Creates WorkFlow instance by passing user selections as inputs.
    4. Runs inference by calling run() method on the WorkFlow instance.
    5. Saves the output dataset by calling save() method on the WorkFlow instance.
    '''
    menu = Menu()
    menu.run()
    category = menu.category
    model_list = menu.model_list
    dataframe_type = menu.dataframe_type
    wf = WorkFlow(category, model_list, Config.DATASET_NAME, dataframe_type)
    wf.run()
    wf.save()

if __name__ == "__main__":
    main()