# Disaster Relief Pipeline
Repository for Udacity Data Scientist Nanodegree disaster relief pipeline project.
Partnered with Figure Eight who are providing prelabelled tweets and text messages from real natural
disasters in a number of datasets. The task is to prepare this data with an ETL pipeline and then an
ML pipeline to build a  supervised learning model. 

Following a disaster, typically millions of communications when the response organisations have the
least capacity to handle it. Typically it is only every 1 in every 1000 messages that are relevant
to a given disaster response professional. Each Disaster response organisation typically focusses on
one key area of the disaster, such as water, blocked roads, medical supplies. The data contains
\these categories and the datasets have been combined and relabelled so that categories are
consistent across the data.

Example: The keyword ‘water’ rarely maps to someone who needs fresh drinking water and will also
miss people who say ‘thirsty’ but don’t use the word water. Supervised ML will be a lot more
accurate than a keyword search for this use-case. This is a big gap in disaster response contexts.

# Project Components
1. ETL Pipeline
In a Python script, process_data.py, write a data cleaning pipeline that:
    - Loads the messages and categories datasets
    - Merges the two datasets
    - Cleans the data
    - Stores it in an SQLite database

2. ML Pipeline
    - Loads data from the SQLite database
    - Splits the dataset into training and test sets
    - Builds a text processing and machine learning pipeline
    - Trains and tunes a model using GridSearchCV
    - Outputs results on the test set
    - Exports the final model as a pickle file

3. Flask Web App
    - Presents output of search string placed into model

### Instructions:
1. Run the following commands in the project's root directory to set up a database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

# Installation 
Additional library requirements can be found in 'requirements.txt'.
Python 3.* should allow the code to run without issues

# File Descriptions
requirements.txt: Python library requirements to run the notebooks within this repository
app directory: contains html and Flask app code
jupyter directory: Contains testing code as .ipynb files
data directory: Contains python script to convert csv files to db table.
models directory: Contains pythons script to build, analyse and pickle a model object

# Licensing, Authors, Acknowledgements
Credit for this dataset is given to Udacity and Figure8.
All code within this repository has been solely developed for this project and is available for public use.



