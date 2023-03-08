from flask import Flask
from flask_restful import Resource, Api, reqparse,request 
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

app = Flask(__name__)


def clean_up_sentence(sentence):
    print("clean_up_sentence called")
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words, show_details=True):
    print("bow called")
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))


def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    probability = ints[0]['probability'] 
    if float(probability) < 0.9:
        tag = "noanswer"
    print(f"tag-{tag}")
    list_of_intents = intents_json['intents']
    result = ''
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result 

def predict_class(sentence, model):
    print("predict_class called")
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    all_res = model.predict(np.array([p]))
    print(f"allres-{all_res}")
    print(f"length of allres-{len(all_res)}")
    res = model.predict(np.array([p]))[0]
    print(f"res-{res}")
    print(f"length of res-{len(res)}")
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if True ]#r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        print(f"r-{r}")
        print(f"classes-{classes[r[0]]}")
        print(f"probability-{str(r[1])}")
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res






@app.route("/chatbot", methods=["GET"])
def chatbot():
    user_input = request.args.get("input") 
      
    response = chatbot_response(user_input)  
    return response  
    

if __name__ == "__main__":
    app.run()
    

    