import io
import random
import string
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gtts import gTTS
import os
warnings.filterwarnings('ignore')
import speech_recognition as sr
import nltk
from nltk.stem import WordNetLemmatizer
#comment after first run

#nltk.download('popular',quiet=True)
#nltk.download('nps_chat',quiet=True)
#nltk.download('punkt')
#nltk.download('wordnet')


#question type classes
posts = nltk.corpus.nps_chat.xml_posts()[:10000]
#ToRecognise imput types as QUES
def dialogue_act_feature(post):
    features = {}
    for word in nltk.word_tokenize(post):
        features['contains({})'.format(word.lower())] =True
    return features
featuresets = [(dialogue_act_feature(post.text), post.get('class')) for post in posts]
size = int(len(featuresets)*0.1)
train_set, test_set = featuresets[size:],featuresets[:size]
classifier = nltk.NaiveBayesClassifier.train(train_set)

# greeting functions
# Keyword Matching
GREETING_INPUTS = ("hello","hi","greetings","sup","what's up","hey")
GREETING_RESPONSES = ["hi","hey","*nods*","hi there","Hi!, How are you today?","hello","I am glad you are talking with me"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

with open('intro_join','r',encoding='utf8',errors='ignore') as fin:
    raw = fin.read().lower()
#tokenization
sent_tokens = nltk.sent_tokenize(raw) # list of sentences
word_tokens = nltk.word_tokenize(raw) # converts list of words
#preprocessing
lemmer =  WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatizer(token) for token in tokens]
remove_punct_dict = dict((ord(punct),None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

#colour palet
def prRed(skk): print("\033[91m {}\033[00m" .format(skk)) 
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk)) 
def prYellow(skk): print("\033[93m {}\033[00m" .format(skk)) 
def prLightPurple(skk): print("\033[94m {}\033[00m" .format(skk)) 
def prPurple(skk): print("\033[95m {}\033[00m" .format(skk)) 
def prCyan(skk): print("\033[96m {}\033[00m" .format(skk)) 
def prLightGray(skk): print("\033[97m {}\033[00m" .format(skk)) 
def prBlack(skk): print("\033[98m {}\033[00m" .format(skk))

# generating response and processing
def response(user_response):
    robo_response = ''
    sent_tokens.append(user_response)
    TfidVec = TfidVectorizer(tokenizer = LemNormalize, stop_words = 'english')
    ifidf = TfidfVec.fit_transform(sent_tokens)
    vals =  cosine_similarity(tfidf[-1],tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    if (req_tfidf == 0):
        robo_response = robo_response+"I am sorry!"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        return robo_response
file = "file.mp3"
flag = True
fst = "Hello! My name is Aria.I am your personal healthcare companion. If you want to exit say Bye "
tts = gTTS(fst,'en')
tts.save(file)
os.system("mpg123" + file)
r = sr.Recognizer()
prYellow(fst)

while(flag==True):
    with sr.Microphone() as source:
        audio= r.listen(source)
    try:
        user_response = format(r.recognize(audio))
        print("\033[91m {}\033[00m" .format("YOU SAID : "+user_response))
    except sr.UnknownValueError:
        prYellow("Oops! Didn't catch that")
        pass
    
    #user_response = input()
    #user_response=user_response.lower()
    clas=classifier.classify(dialogue_act_features(user_response))
    if(clas!='Bye'):
        if(clas=='Emotion'):
            flag=False
            prYellow("Aria: You are welcome..")
        else:
            if(greeting(user_response)!=None):
                print("\033[93m {}\033[00m" .format("Aria: "+greeting(user_response)))
            else:
                print("\033[93m {}\033[00m" .format("Aria: ",end=""))
                res=(response(user_response))
                prYellow(res)
                sent_tokens.remove(user_response)
                tts = gTTS(res,lang="en",tld="co.in")
                tts.save(file)
                os.system("mpg123 " + file)
    else:
        flag=False
        prYellow("Aria: Bye! take care..")


