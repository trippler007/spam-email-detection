import streamlit as stm
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# nltk.download('punkt_tab')
# nltk.download('stopwords')

ps = PorterStemmer()

def text_transform(message):
  message = message.lower()
  message = nltk.word_tokenize(message)

  y = []
  for i in message:
    if i.isalnum():
      y.append(i)

  message = y[:]
  y.clear()

  for i in message:
    if i not in stopwords.words('english') and i not in string.punctuation:
      y.append(i)

  message = y[:]
  y.clear()

  for i in message:
      y.append(ps.stem(i))

  return " ".join(y)   # this line will convert the list in string and returns the string

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

stm.title('Spam Email Detector')

input_mail = stm.text_area("Enter your mail")


if stm.button('predict'):
  if input_mail == "":
    stm.header("mail box is empty")
  else:
    #1. preprocess
    transformed_mail = text_transform(input_mail)
    #2. vectorize
    vector_input = tfidf.transform([transformed_mail])
    #3. predict
    result = model.predict(vector_input)[0]
    #4. display
    if result == 1:
      stm.header("Spam Email")
    else:
      stm.header("Not Spam Email")