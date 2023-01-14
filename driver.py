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
nltk.download('popular',quiet=True)
nltk.download('nps_chat',quiet=True)
nltk.download('punkt')
nltk.download('wordnet')