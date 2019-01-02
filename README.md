# titanic_survival
This is a project we will analyze and predict what sorts of passengers were likely to survive the tragedy.

The folder called titanic_survival comes with a  
- requirements.txt for downloading possible dependencies (pip install -r requirements.txt)
- titabic.cfg configuration file in which you can define the specs of the algorithm you want to run  

When you define what you want to run in the configuration file you just run the MainLauncher.py file.  

Concerning the configuration file at the titanic section:
- verbose: if it will show some tables and graphs when you run it
- path: the path with the data files 
- file_type: < all / traintest > 
	- all: is the titanicAll dataset which includes both train and test set
	- traintest: means that there is a train and test file
- tuning: the type of tuning when. Available options are gridsearch or none. IT S NOT USED NOW

### Some explanations
Right now we create just an obvious extra feature called Title which comes from the Name column. The Age has many NaNs which are filled based on the Pclass and the Title.

We discard the PassengerId, Name, Ticket, Cabin. What we can try depending on the first results is to:
- keep from the Cabin column the first letter (if it's available) and group them together based on the correlation between the new feature and Survived column of the train or titanicAll file
- combine the Parch and SibSp features