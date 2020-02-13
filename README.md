# DisasterResponseApp: a Disaster Response Machine Learning Pipeline

### 1. Summary:
This repository contains a machine-learning-based application that can be used during disasters to filter out messages exchanged by impacted people to find whether a help is needed, and if a help is needed, the app can also categorize what kind of help is needed. This application was created as part of the requirements in Udacity's Data Scientist Nanodegree Program. It uses data from <a href=https://www.figure-eight.com/>Figure Eight</a>.

### 2. Requirements:
To install and run this application, you need following libraries:
- Python 3 or above
- Databse Libraries: SQLlite and SQLalchemy
- Data Analysis Libraries: NumPy and Pandas
- Machine Learning Libraries: Scikit-Learn
- Natural Language Processing Libraries: NLTK
- Web Application Libraries: Flask 
- Visualization Libraries: Plotly

### 3. Installation:
To install the project on your local machine, clone this repository by running:
git clone https://github.com/AlManea/DisasterResponseApp.git


### 4. Running Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### 5. Files Descriptions:
Below is a short description of the main files in this repository:
 - 'data/process_data.py': This file is a python script that loads the data from the 'messages.csv' and 'categories.csv' files, merges them, and loads them to a databse file.
 - 'models/train_classifier.py': This file is a python script that creates an AdaBoost machine learning model, tains the model and does a grid-search to find the best model parameters. It also stores the model to a .pkl file
 - 'app/run.py': This file is the main application file, built with Flask framework. It uses the data base files to visualize the message data and the trained ML model to predict an input message importance and categories.
 
 ### 6. Acknowledgments:
This project was build as part of <a href=http=https://www.udacity.com/>Udacity</a>'s Data Scientist Nanodegree Program, and it uses code snippets/files from the course. It also uses data from <a href=https://www.figure-eight.com/>Figure Eight</a>. The support from both <a href=http=https://www.udacity.com/>Udacity</a> and <a href=https://www.figure-eight.com/>Figure Eight</a> is greatly acknowledged. 
