from PIL import Image
import numpy as np
import pickle
import pandas as pd
import json
import tweepy
from wordcloud import WordCloud
import matplotlib.pyplot as plt
#from flasgger import Swagger
import streamlit as st
import joblib
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
st.set_option('deprecation.showPyplotGlobalUse', False)



def load_saved_artifacts():
    #print("loading saved artifacts...start")
    global __model
    __model=None
    if __model is None:
        with open('classification123.joblib', 'rb') as f:
            __model = joblib.load(f)
    #print("loading saved artifacts...done")


def authe():
    global CONSUMER_KEY
    global CONSUMER_SECRET
    global OAUTH_TOKEN
    global OAUTH_TOKEN_SECRET
    global api
    CONSUMER_KEY = '3tDVUGFlwUAfrjtTNO6k1xfqW'
    CONSUMER_SECRET = '9I3BEaSP2LqS2wfQu0qXefJXDjUsqzouhoBvbDG6onv5VfU4lL'
    OAUTH_TOKEN = '870901794452291584-zx8zAHDfvt9EdsCAAdNg9r5Se6GSiPP'
    OAUTH_TOKEN_SECRET = 'RhOPigMTtITcw1c6O4L2wGg7qcgv7lqkzQpFpbFVyHM2e'
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
    api = tweepy.API(auth, wait_on_rate_limit=True)

def catogery(text):
    text=pd.Series(data=text)
    return __model.predict(text)


def clean(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text)  # remove mentions
    text = re.sub(r'#', '', text)  # remove hashtags
    text = re.sub(r'RT[\s]+', '', text)  # remove retweets
    text = re.sub(r'https?:\/\/\S+', '', text)  # remove links
    return text


def sentiment(text):
    ps = analyzer.polarity_scores(text)
    return ps['compound']


def wordcl():  # wordcloud
    allwords = ''.join([twts for twts in df['tweets']])
    wordcloud = WordCloud(width=500, height=300, random_state=21,
                          max_font_size=119).generate(allwords)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    disp_col2.pyplot()


def getanalysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'
def sports_tweets():
    disp_col5.header('sports tweets')
    j = 1
    for i in range(0, ak_df.shape[0]):
        if(ak_df['catogery'][i] == 'sport'):
            disp_col5.write(str(j)+')'+ak_df['tweets'][i]+'\n')
            j = j+1
def business_tweets():
    disp_col5.header('business tweets')
    j = 1
    for i in range(0, ak_df.shape[0]):
        if(ak_df['catogery'][i] == 'business'):
            disp_col5.write(str(j)+')'+ak_df['tweets'][i]+'\n')
            j = j+1
def entertainment_tweets():
    disp_col5.header('entertainment tweets')
    j = 1
    for i in range(0, ak_df.shape[0]):
        if(ak_df['catogery'][i] == 'entertainment'):
            disp_col5.write(str(j)+')'+ak_df['tweets'][i]+'\n')
            j = j+1
def politic_tweets():
    disp_col5.header('political tweets')
    j = 1
    for i in range(0, ak_df.shape[0]):
        if(ak_df['catogery'][i] == 'politics'):
            disp_col5.write(str(j)+')'+ak_df['tweets'][i]+'\n')
            j = j+1
def tech_tweets():
    disp_col5.header('Tech tweets')
    j = 1
    for i in range(0, ak_df.shape[0]):
        if(ak_df['catogery'][i] == 'tech'):
            disp_col5.write(str(j)+')'+ak_df['tweets'][i]+'\n')
            j = j+1


def postive_tweets():
    dis_col4.header('postive tweets')
    j = 1
    sortedDF = analysis_df.sort_values(by=['score'])
    for i in range(0, sortedDF.shape[0]):
        if(sortedDF['analysis'][i] == 'Positive'):
            dis_col4.write(str(j)+')'+sortedDF['tweets'][i]+'\n')
            j = j+1


def negative_tweets():
    dis_col4.header('Negative tweets')
    j = 1
    sortedDF = analysis_df.sort_values(by=['score'], ascending=False)
    for i in range(0, sortedDF.shape[0]):
        if(sortedDF['analysis'][i] == 'Negative'):
            dis_col4.write(str(j)+')'+sortedDF['tweets'][i]+'\n')
            j = j+1


def neutral_tweets():
    dis_col4.header('Neutral tweets')
    j = 1
    sortedDF = analysis_df.sort_values(by=['score'], ascending=False)
    for i in range(0, sortedDF.shape[0]):
        if(sortedDF['analysis'][i] == 'Neutral'):
            dis_col4.write(str(j)+')'+sortedDF['tweets'][i]+'\n')
            j = j+1


def postive_percent():
    req_tweets = analysis_df[analysis_df.analysis == 'Positive']
    req_tweets = req_tweets['tweets']
    st.header('Postive tweets percent')
    st.write(round((req_tweets.shape[0]/df.shape[0])*100, 1))


def negative_percent():
    req_tweets = analysis_df[analysis_df.analysis == 'Negative']
    req_tweets = req_tweets['tweets']
    st.header('Negative tweets percent')
    st.write(round((req_tweets.shape[0]/df.shape[0])*100, 1))


def neutral_percent():
    req_tweets = analysis_df[analysis_df.analysis == 'Neutral']
    req_tweets = req_tweets['tweets']
    st.header('Neutral tweets percent')
    st.write(round((req_tweets.shape[0]/df.shape[0])*100, 1))


def value_coun_graph():
    analysis_df['analysis'].value_counts()
    plt.title('sentment analysis')
    plt.xlabel('sentiment')
    plt.ylabel('counts')
    analysis_df['analysis'].value_counts().plot(kind='bar')
    plt.show()
    disp_col2.pyplot()
load_saved_artifacts()
authe()
header =st.container()
dataset=st.container()
features = st.container()
images = st.container()
per= st.container()
with header:
    st.title('Welcome to Twitter Sentiment analysis')
with dataset:
    sel_col,disp_col =st.columns(2)
    username = sel_col.text_input('Select username')
    disp_col.subheader('Given Username:')
    disp_col.write(username)
    count = sel_col.number_input('no of tweets to extract',max_value=200)
    disp_col.subheader('Selected no.of Tweets')
    disp_col.write(count)
    n=sel_col.slider('Want to see sapmple no of tweets?',max_value=20)
    disp_col.subheader('No of Sample tweets showing')
    disp_col.write(n)
with features:
    posts = api.user_timeline(screen_name=username,
                              count=count, tweet_mode='extended')
    df = pd.DataFrame([tweet.full_text for tweet in posts], columns=['tweets'])
    if n>0:    
        st.subheader('Sample tweets:')
    i=1
    for tweet in posts[0:n]:
        st.write(str(i)+')' + tweet.full_text+'\n')
        i = i+1
    sel_col1,disp_col1 =st.columns(2)
    if sel_col1.button('View Raw Dataset'):
        st.write(df)
    global scores_df
    scores_df = df.copy()
    scores_df['tweets'] = scores_df['tweets'].apply(clean)
    # st.dataframe(df)
    scores_df['score'] = scores_df['tweets'].apply(sentiment)
    if disp_col1.button('Dataset wth scores'):
        st.dataframe(scores_df)
    global analysis_df
    analysis_df = scores_df.copy()
    # st.dataframe(df)
    analysis_df['analysis'] = analysis_df['score'].apply(getanalysis)
    if sel_col1.button('Dataset With Sentiment'):
        st.dataframe(analysis_df)
    ak_df = df.copy()
    ak_df['catogery']=ak_df['tweets'].apply(catogery)
    if disp_col1.button('Dataset with catogeries'):
        st.dataframe(ak_df)
with images:
    st.header('Visuvalizations')
    sel_col2,disp_col2 =st.columns(2)
    if sel_col2.button('wordcloud image of most used words'):
        disp_col2.subheader('Wordcloud image:')
        wordcl()
    if sel_col2.button('sentiment vs no.of tweets graph'):
        value_coun_graph()
with per:
    dis_col3=st.columns(1)
    sel_col4,sel_col5 = st.columns(2)
    dis_col4,disp_col5=st.columns(2)
    
    menu1 = ['None', 'positive tweets', 'negative tweets', 'neutral tweets']
    choice1 = sel_col4.selectbox('View positive, negative or neutral tweets', menu1)
    if choice1 == 'positive tweets':
        postive_tweets()
    if choice1 == 'negative tweets':
        negative_tweets()
    if choice1 == 'neutral tweets':
        neutral_tweets()
    menu1 = ['None', 'Sports related', 'Business related', 'Entertainment related','Politics related','Tech related']
    choice1 = sel_col5.selectbox('view different categories of Tweets', menu1)
    if choice1 == 'Sports related':
        sports_tweets()
    if choice1 == 'Business related':
        business_tweets()
    if choice1 == 'Entertainment related':
        entertainment_tweets()
    if choice1 == 'Politics related':
        politic_tweets()
    if choice1 == 'Tech related':
        tech_tweets()
    menu = ['None', 'positive tweet percent',
            'negative tweet percent', 'neutral tweet percent']
    choice = st.selectbox('percent of different tweets based on sentiment', menu)
    if choice == 'positive tweet percent':
        postive_percent()
    if choice == 'negative tweet percent':
        negative_percent()
    if choice == 'neutral tweet percent':
        neutral_percent()
