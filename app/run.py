import json
import plotly
import pandas as pd
import numpy as np

#from nltk.stem import WordNetLemmatizer
#from nltk.tokenize import word_tokenize
import re
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from plotly.graph_objs import Pie
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    """
    Processes the input text by tokenizing it, removing stop words, and lammetizing it.
    Parameters:
     - text: The text to be processed (String)
    Returns:
     - tks2: The tokens (String list)
    
    """
    # detect urls and replace them with the 'urlplaceholder' text
    regexp_url = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(regexp_url, text)
    for u in detected_urls:
        text = text.replace(u, "urlplaceholder")

    # tokenize
    tks_0 = word_tokenize(text)
    
    # Remove stopwords
    tks_1 = []
    for tk in tks_0:
        if tk not in stopwords.words('english'):
            tks_1.append(tk)
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tks_2 = []
    for tk in tks_1:
        clean_tk = lemmatizer.lemmatize(tk).lower().strip()
        tks_2.append(clean_tk)
        
    # return the tokens
    return tks_2

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_ds', engine)

# load model
model = joblib.load("../models/DisasterResponse.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    categories  = (df.columns[4:]).values
    msg_per_cat = (df.iloc[:,4:].sum()).values
    percentages = np.float16(((df.iloc[:,4:].sum()).values/df.iloc[:,4:].sum().sum()*100))
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        
        {
            'data': [
                Bar(
                    x=categories,
                    y=msg_per_cat
                )
            ],

            'layout': {
                'title': 'Number of Messages per Disaster Category',
                'yaxis': {
                    'title': "Number of Messages"
                },
                'xaxis': {
                    'title': "Disaster Categories"
                }
            }
        },
        
         {
            'data': [
                Pie(
                    labels=categories,
                    values=msg_per_cat
                )
            ],

            'layout': {
                'title': 'Percentages of Messages in Each Disaster Category to the Total Messagescd Count'
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))
    
    for i in classification_results:
        print(i)

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
