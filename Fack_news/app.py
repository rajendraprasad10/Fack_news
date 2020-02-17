from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	df = pd.read_csv('train-2.csv')
	df['Label'] = df['Label'].map({False: 0, True: 1})

	# Data cleaning and preprocessing
	import re
	import nltk

	from nltk.corpus import stopwords
	wl = nltk.WordNetLemmatizer()
	corpus = []
	for i in range(0, len(df)):
		news = re.sub('[^a-zA-Z]', ' ', df['Statement'][i])
		news = news.lower()
		news = news.split()

		news = [wl.lemmatize(word) for word in news if not word in stopwords.words('english')]
		news = ' '.join(news)
		corpus.append(news)
	
	# Extract Feature With CountVectorizer
	from sklearn.feature_extraction.text import CountVectorizer
	cv = CountVectorizer()
	x = cv.fit_transform(corpus).toarray()
	y = df['Label']

	# Fit the Data
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=0)

	#Naive Bayes Classifier
	from sklearn.naive_bayes import MultinomialNB
	clf = MultinomialNB()
	clf.fit(X_train,y_train)
	clf.score(X_test,y_test)

	#Alternative Usage of Saved Model
	joblib.dump(clf, 'NB_fake_news_model.pkl')
	NB_fake_news_model = open('NB_fake_news_model.pkl','rb')
	clf = joblib.load(NB_fake_news_model)
	if request.method == 'POST':
		news = request.form['Label']
		data = [news]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('home.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)