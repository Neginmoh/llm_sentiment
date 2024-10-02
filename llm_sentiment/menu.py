# menu.py


from llm_sentiment.config import Config

class Menu:
    '''
    This class handles the prompts printed for user and processing of the user selections.
    '''
    def __init__(self):
        '''
        Initialize the menu options for each prompt
        '''
        self.categories = {'1': 'pipeline_classifier',
                           '2': 'classifier_classifier'}
        self.model_lists = {'1': Config.MODEL_1,
                            '2': Config.MODEL_2,
                            '3': 'Both'}
        self.dataframe_type = {'1': 'Regular',
                               '2': 'MultiIndex'}

    def display_category(self):
        '''
        This method displays the first prompt description and possible options to the user
        - waits for the user to make selection
        -
        '''
        print("\nPlease select a category:\n")
        for k, v in self.categories.items():
            print(f"{k}. {v}")

    def display_models(self):
        '''
        This method displays the second prompt description and possible options to the user
        '''
        print("\nPlease select models:\n")

        for k, v in self.model_lists.items():
            if k=='3':
                print(f"{k}. {v}")
            else:
                print(f"{k}. {v['NAME']}")

    def display_dataframe(self):
        '''
        This method displays the third prompt description and possible options to the user
        '''
        print("\nPlease select a DataFrame type:\n")
        for k, v in self.dataframe_type.items():
            print(f"{k}. {v}")


    def get_user_category(self):
        '''
        This method waits for the first user input, obtains, and returns the selection
        '''
        choice_num = input('\nYou can select only one of the above:\n')
        return choice_num

    def get_user_models(self):
        '''
        This method waits for the second user input, obtains, and returns the selection
        '''
        choice_num = input('\nYou can select only one of the above:\n')
        return choice_num

    def get_user_dataframe(self):
        '''
        This method waits for the third user input, obtains, and returns the selection
        '''
        choice_num = input('\nYou can select only one of the above:\n')
        return choice_num

    def first(self):
        '''
        This method:
        - displays the first prompt message and options
        - waits for the user input, obtains and returns the selection
        - if user selection is within the provided options, prints the user selection, makes the user selection a class attribute, and returns True
        - if user selection is not within the provided options, prints a message to user and returns False
        '''
        self.display_category()
        choice_num_category = self.get_user_category()

        if choice_num_category in self.categories:

            print(f'\nSelected category: {self.categories[choice_num_category]}\n')
            self.category = self.categories[choice_num_category]
            return True
        else:
            print('\nWrong category selection. Choose from the list. \n')   
            return False

    def second(self):
        '''
        This method:
        - displays the second prompt message and options
        - waits for the user input, obtains and returns the selection
        - if user selection is within the provided options, prints the user selection, makes the user selection a class attribute, and returns True
        - if user selection is not within the provided options, prints a message to user and returns False
        '''

        self.display_models()
        choice_num_models = self.get_user_models()

        if choice_num_models in self.model_lists:
            if choice_num_models=='3':
                print(f'Selected models: {(self.model_lists["1"])["NAME"]}, {(self.model_lists["2"])["NAME"]}')
                self.model_list = [self.model_lists['1'], self.model_lists['2']]                
            else:
                print(f'Selected models: {(self.model_lists[choice_num_models])["NAME"]}')
                self.model_list = [self.model_lists[choice_num_models]]
            return True
        else:
            print('Wrong model selection. Choose from the list. \n')
            return False

    def third(self):
        '''
        This method:
        - displays the third prompt message and options
        - waits for the user input, obtains and returns the selection
        - if user selection is within the provided options, prints the user selection, makes the user selection a class attribute, and returns True
        - if user selection is not within the provided options, prints a message to user and returns False
        '''

        self.display_dataframe()
        choice_num_dataframe = self.get_user_dataframe()

        if choice_num_dataframe in self.dataframe_type:

            print(f'\nSelected DataFrame type: {self.dataframe_type[choice_num_dataframe]}\n')
            self.dataframe_type = self.dataframe_type[choice_num_dataframe]
            return True
        else:
            print('\nWrong DataFrame type selection. Choose from the list. \n')   
            return False


    def run(self):
        '''
        This method handles all three prompts in a way that:
        - As long as user inputs are among the menu options then next prompts are printed one-by-one until all correct selections are made.
        - At any step, if user input is not among the menu options, then a message is printed and the current prompt is repeated until a correct option is selected.
        '''
        selection = True
        while selection==True:
            selection = self.first()
            while selection==False:
                selection = self.first()

            selection = self.second()
            while selection==False:
                selection = self.second()

            selection = self.third()
            while selection==False:
                selection = self.third()

            break

