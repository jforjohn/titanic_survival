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
        self.new_df = pd.DataFrame()


##
    def handleMissingValues(self, data):
        # Fill the  NaN of Fare with the mean value
        data.Fare.fillna(data.Fare.median(), inplace=True)

        data.Embarked.fillna(data.Embarked.mode()[0], inplace=True)

        data['Title'] = data['Name'].str.extract('([A-Za-z]+)\.', expand=True)

        # Fill NaN of Age column taking into account the Title and the No of Parch
        data['Age'] = data.Age.fillna(
            data.groupby(['Title', 'Parch'])['Age'].transform('median'))
        # there are still 2 missing values because there is a case where the
        # combination of the condition includes just NaN values
        data['Age'] = data.Age.fillna(
            data.groupby(['Title'])['Age'].transform('median'))
        # for remaining NaNs
        data['Age'] = data.Age.fillna(data.Age.mean())
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
        elif title in ['Sir', 'Dr', 'Master', 'Col', 'Major']:
            return 'Mr50'
        # The male doctor survives so goes with survival=0
        elif title in ['Jonkheer', 'Don', 'Rev', 'Capt']:
            return 'Mr0'
        else:
            # Sir always survives, Mr is 16% and Ms is 50% survival
            return title

    def get_ticket_id(row):
        print(row)
        ticket = row.Ticket
        if ticket.isdigit():
            return 'N'
        else:
            ticket_id = ticket.replace('.', '').replace('/', '').split()[0]
            return ticket_id

    def dropDependents(self):
        self.new_df.drop(['Age', 'SibSp', 'Parch', 'Fare'], axis=1, inplace=True)

    ##
    def fit(self, data):
        self.df_initial = self.handleMissingValues(data)

        # get label
        if 'Survived' in data.columns:
            labels = self.df_initial.Survived
            self.labels_ = labels
            self.df_initial.drop(['Survived'], axis=1, inplace=True)
        else:
            self.labels_ = pd.Series()

        df = self.df_initial.copy()

        # Title
        df['Title'] = df.apply(self.replace_titles, axis=1)
        df = pd.get_dummies(df, columns=['Title'], drop_first=True)

        # Sex
        # need to give males and females numeric values
        df.loc[df["Sex"] == "male", "Sex"] = 0
        df.loc[df["Sex"] == "female", "Sex"] = 1

        # Age
        # Various ages
        df['Age_bin'] = pd.cut(df['Age'], bins=[-1, 2, 12, 17, 120],
                               labels=['Infant', 'Kid', 'Teenager', 'Adult'])
        df = pd.get_dummies(df, columns=['Age_bin'], drop_first=True)

        # Family
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        # Family size is the sum of siblings, spouses, parents, and children.
        df['Family_bin'] = pd.cut(df['FamilySize'], bins=[0, 1, 4, 7, 12],
                                  labels=['Single', 'SmallFamily', 'BigFamily', 'Team'])
        df = pd.get_dummies(df, columns=['Family_bin'], drop_first=True)

        # Fare
        df['Fare_bin'] = pd.cut(df['Fare'],
                                bins=[0, 7.91, 14.45, 31, 120],
                                labels=['Low', 'Median', 'Average', 'High'])
        df = pd.get_dummies(df, columns=['Fare_bin'], drop_first=True)

        # Embarked
        df = pd.get_dummies(df, columns=["Embarked"], prefix="Em")

        # Cabin
        # Replace the Cabin number by the type of cabin 'X' if not
        df['Cabin'] = df['Cabin'].str.extract("([a-zA-Z]+)", expand=True)
        df['Cabin'] = df['Cabin'].fillna("X")
        df = pd.get_dummies(df, columns=["Cabin"], prefix="Cabin")

        # Ticket
        # Find the length of the ticket, perhaps longer tickets are for include more amenities,
        # therefore associated with wealth.
        # df['TicketLength'] = df['Ticket'].map(lambda x: len(x))
        # df['TicketL_bin'] = pd.cut(df['TicketLength'], bins=[0,5,8,10,20], labels=['Small','Average',
        #                                                                      'Big','Huge'])
        # df = pd.get_dummies(df, columns=["TicketL_bin"])
        # df['TicketID'] = df.apply(get_ticket_id, axis=1)
        # df['TicketID'] = df.apply(get_ticket_id, axis=1)

        # Extras

        # To be a parent, you need a spouse, have at least 1 child, and be older than 18.
        # SibSp is >= 0 because you could be an adult with a sibling on board.
        parent = (df['Parch'] > 0) & (df['Age'] >= 18)

        # To be a mother, you need to be a parent and female. Fathers = Parent & Male
        df['Mother'] = ((parent == 1) & (df['Sex'] == 1)).map(lambda s: 1 if s else 0)
        df['Father'] = ((parent == 1) & (df['Sex'] == 0)).map(lambda s: 1 if s else 0)

        # Child has at least 1 parent and is 17 or younger
        child = (df['Parch'] >= 1) & (df['Age'] <= 17)

        # To be a daughter, you need to be a girl, and a child.
        # To be a son, likewise, but a boy.
        df['Daughter'] = ((df['Sex'] == 1) & (child == 1)).map(lambda s: 1 if s else 0)
        df['Son'] = ((df['Sex'] == 0) & (child == 1)).map(lambda s: 1 if s else 0)

        # Orphan if you have no parents and are 17 or younger
        df['Orphan'] = ((df['Age'] <= 17) & (df['Parch'] == 0)).map(lambda s: 1 if s else 0)

        # Combined class and gender to better organize people.
        df['RichWoman'] = ((df['Pclass'] == 1) & (df['Sex'] == 1) & (df['Age'] >= 18)).map(
            lambda s: 1 if s else 0)
        df['MiddleClassWoman'] = ((df['Pclass'] == 2) & (df['Sex'] == 1) & (df['Age'] >= 18)).map(
            lambda s: 1 if s else 0)
        df['PoorWoman'] = ((df['Pclass'] == 3) & (df['Sex'] == 1) & (df['Age'] >= 18)).map(
            lambda s: 1 if s else 0)
        df['RichMan'] = ((df['Pclass'] == 1) & (df['Sex'] == 0) & (df['Age'] >= 18)).map(
            lambda s: 1 if s else 0)
        df['MiddleClassMan'] = ((df['Pclass'] == 2) & (df['Sex'] == 0) & (df['Age'] >= 18)).map(
            lambda s: 1 if s else 0)
        df['PoorMan'] = ((df['Pclass'] == 3) & (df['Sex'] == 0) & (df['Age'] >= 18)).map(
            lambda s: 1 if s else 0)

        df['RichGirl'] = ((df['Pclass'] == 1) & (df['Age'] <= 17) & (df['Sex'] == 1)).map(
            lambda s: 1 if s else 0)
        df['MiddleClassGirl'] = ((df['Pclass'] == 2) & (df['Age'] <= 17) & (df['Sex'] == 1)).map(
            lambda s: 1 if s else 0)
        df['PoorGirl'] = ((df['Pclass'] == 3) & (df['Age'] <= 17) & (df['Sex'] == 1)).map(
            lambda s: 1 if s else 0)
        df['RichBoy'] = ((df['Pclass'] == 1) & (df['Age'] <= 17) & (df['Sex'] == 0)).map(
            lambda s: 1 if s else 0)
        df['MiddleClassBoy'] = ((df['Pclass'] == 2) & (df['Age'] <= 17) & (df['Sex'] == 0)).map(
            lambda s: 1 if s else 0)
        df['PoorBoy'] = ((df['Pclass'] == 3) & (df['Age'] <= 17) & (df['Sex'] == 0)).map(
            lambda s: 1 if s else 0)

        # Pclass
        df["Pclass"] = df["Pclass"].astype("object")
        df = pd.get_dummies(df, columns=['Pclass'], drop_first=True)

        # Drops
        # Remove columns
        # titanicAll dataset doesn't have PassengerId column
        if 'PassengerId' in data.columns:
            df.drop(['PassengerId'], axis=1, inplace=True)
        df.drop(['Name', 'Ticket', 'Fare'], axis=1, inplace=True)

        df_num = df.select_dtypes(exclude='object')
        df_obj = df.select_dtypes(include='object')

        if df_num.size > 0:
            min_max_scaler = preprocessing.MinMaxScaler()
            scaled = min_max_scaler.fit_transform(df.values.astype(float))
            df_num = pd.DataFrame(scaled, columns=df.columns)

        self.new_df = pd.concat([df_obj, df_num], axis=1, sort=False)

#plt.interactive(False)
#plt.show(block=True)
