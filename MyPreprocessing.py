##
from scipy.io.arff import loadarff
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt


class MyPreprocessing:
    def __init__(self, filename_type='all'):
        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)
        pd.options.mode.chained_assignment = None  # default='warn'
        self.filename_type = filename_type


##
    def handleMissingValues(self, data):
        # Fill the  NaN of Fare with the mean value
        data.Fare.fillna(data.Fare.mean(), inplace=True)

        data.Embarked.fillna(data.Embarked.mode()[0], inplace=True)

        data['Title'] = data['Name'].str.extract('([A-Za-z]+)\.', expand=True)

        # Fill NaN of Age column taking into account the Title and the No of Parch
        data['Age'] = data.Age.fillna(
            data.groupby(['Title', 'Parch'])['Age'].transform('median'))
        # there are still 2 missing values because there is a case where the
        # combination of the condition includes just NaN values
        data['Age'] = data.Age.fillna(
            data.groupby(['Title'])['Age'].transform('median'))

        return data

    def replace_titles(self, x):
        title = x.Title
        # Female titles which are associated with survival=1
        if title in ['Countess', 'Mme' , 'Dona', 'Mlle', 'Lady']:
            return 'Ms'
        # Female titles that are associated with survival~0.7
        elif title in ['Mrs', 'Miss']:
            return 'Mrs'
        elif title == 'Dr' and x['Sex'] == 'female':
            # The female doctor survives so goes with survival=1
            return 'Ms'
        # The male doctor survives so goes with survival=0.5
        elif title in ['Dr', 'Master', 'Col', 'Major']:
            return 'Mr50'
        # The male doctor survives so goes with survival=0
        elif title in ['Jonkheer', 'Don', 'Rev', 'Capt']:
            return 'Mr0'
        else:
            # Sir always survives, Mr is 16% and Ms is 50% survival
            return title

##
    def fit(self, data):
        self.df_initial = self.handleMissingValues(data)

        # get label
        if self.filename_type in ['train', 'all']:
            labels = self.df_initial.Survived
            self.labels_ = labels
            self.df_initial.drop(['Survived'], axis=1, inplace=True)
        else:
            self.labels_ = pd.Series()

        df_reduced = self.df_initial.copy()

        # Remove columns
        # titanicAll dataset doesn't have PassengerId column
        if 'PassengerId' in data.columns:
            df_reduced.drop(['PassengerId'], axis=1, inplace=True)
        df_reduced.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

        # Pclass is ordinal so we convert it in object
        df_reduced['Pclass'] = df_reduced['Pclass'].astype('object')
        df_num = df_reduced.select_dtypes(exclude='object')
        df_obj = df_reduced.select_dtypes(include='object')

        df_normalized = pd.DataFrame()
        df_encoded = pd.DataFrame()
        # normalize numerical data
        if df_num.size > 0:
            min_max_scaler = preprocessing.MinMaxScaler()
            scaled = min_max_scaler.fit_transform(df_num.values.astype(float))
            df_normalized = pd.DataFrame(scaled, columns=df_num.columns)

        if df_obj.size > 0:
            '''
            # Sex is the only column that has 2 values so we can factorize it
            df_obj['Sex'] = pd.factorize(df_obj['Sex'])[0]
            sex = pd.DataFrame(df_obj.Sex, columns=['Sex'])
            df_obj.drop(['Sex'], axis=1, inplace=True)
            '''
            # replace titles
            df_obj['Title'] = df_obj.apply(self.replace_titles, axis=1)

            # we use One-Hot encoding with all the categorical values
            # Sex won't have 2 columns (1 for male and one for female)
            # because we use the drop_first option
            df_encoded = pd.get_dummies(df_obj,
                                        drop_first=True)
            # we convert them in float to avoid a warning
            df_encoded = df_encoded.astype('float')
            min_max_scaler = preprocessing.MinMaxScaler()
            scaled = min_max_scaler.fit_transform(df_encoded)
            df_encoded = pd.DataFrame(scaled, columns=df_encoded.columns)

        self.new_df = pd.concat([df_normalized, df_encoded], axis=1, sort=False)
#
#plt.interactive(False)
#plt.show(block=True)
