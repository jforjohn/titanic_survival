import matplotlib.pyplot as plt
import seaborn as sns

def featureAnalysis(data):
    print('Data summary')
    print(data.describe())
    print()
    print('Categorical data summary')
    print(data.select_dtypes(['object']).describe())

    print()
    print('Pclass-Survived correlation')
    print(data[['Pclass', 'Survived']].groupby(['Pclass']).mean().
          sort_values(by='Survived', ascending=False))

    print()
    print('Sex-Survived correlation')
    print(data[['Sex', 'Survived']].groupby(['Sex']).mean().
          sort_values(by='Survived', ascending=False))

    print()
    print('SibSp-Survived correlation')
    print(data[['SibSp', 'Survived']].groupby(['SibSp']).mean().
          sort_values(by='Survived', ascending=False))

    print()
    print('Parch-Survived correlation')
    print(data[['Parch', 'Survived']].groupby(['Parch']).mean().
          sort_values(by='Survived', ascending=False))


