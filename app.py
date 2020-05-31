from flask import Flask, render_template, url_for, request

# load library
import pandas as pd
import pickle 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split


app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST','GET'])
def predict():

    data = pd.read_csv("data/Youtube01-Psy.csv")

    # drop columns
    data.drop(['COMMENT_ID', 'DATE','AUTHOR'], axis=1, inplace=True)

    # divide data
    x = data['CONTENT']
    y = data['CLASS']

    # load Countvectorize
    corpus = x

    cv = CountVectorizer()

    x = cv.fit_transform(corpus)

    # TRAIN TEST SPLIT
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.33)


    # MODEL
    model = MultinomialNB()

    # fit model 
    model.fit(x_train, y_train)
    model.score(x_test, y_test)

    # save_model = pickle.dumps(model)

    # load_model = pickle.load(save_model)

    if request.method == 'POST':
        comment = request.form['comment']
        data = [comment]
        vect = cv.transform(data).toarray()
        prediction = model.predict(vect)
    
    return render_template('prediction.html', prediction=prediction)





if __name__=="__main__":
    app.run(debug=True)

