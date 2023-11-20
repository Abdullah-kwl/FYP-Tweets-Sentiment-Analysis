# importing required libraries
import streamlit as st 
from PIL import Image
import time


import re
import tweepy
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud,STOPWORDS
from collections import Counter
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# removing handburger & footer

# st.markdown("""
# <style>

# .css-1lsmgbg.egzxvld0 {
#   visibility: hidden;
# }
# .css-14xtw13.e8zbici0{
#   visibility: hidden;
# }

# </style>   
# """,True)



# image opem=ning
image_bird = Image.open('bird2.png')
#displaying the image on streamlit app

# API Connections
consumerKey ="Dswza7wmqAbjM6WzdDH4aGGxz"
consumerSecret  ="OgHuSWN76s3CbdeNVAydf21aVP4IfmdJ8YPOEOL276MXW6uGJ7"
accessToken  ="1297543557738356736-1pow1apnfIgVe0pKVLyiVjiwkEJOO6"
accessTokenSecret  ="j5oiC7Ky4gBGsAg0YCoqCQGTGFjSuSw1vFuWvEUSaComQ"
# Create the authentication object
authenticate = tweepy.OAuthHandler(consumerKey, consumerSecret)    
# Set the access token and access token secret
authenticate.set_access_token(accessToken, accessTokenSecret)    
# Creating the API object while passing in auth information
api = tweepy.API(authenticate, wait_on_rate_limit = True)

# Sidebar Manue
search=st.sidebar.selectbox('Main Manue',['search for People','Search for keywords'])

# search for People
if search == 'search for People' :
    
    # seeting-up image with text
    col1 , col2 = st.columns([5,1])
    with col1:
        st.title('Tweets Sentiment Analyzer')
    with col2:
        st.image(image_bird,width=50)
    
    # Taking Input of @username and limit
    name , num = st.columns(2)
    with name:
        name=st.text_input('Enter Username')
    with num:
        num=st.text_input('Enter the Limit')
    
    # making condition not work untill name and limit not entered
    if name !="" and num !="" :
        # Adding the Spinner
        with st.spinner('Wait data is collecting...'):
           # using snsscraper to extract twitter data
            try:
                limit=int(float(num))
            except:
                limit=100

            try:
                cursor=tweepy.Cursor(api.user_timeline, screen_name=name,count=200,exclude_replies=True, include_rts=False,tweet_mode="extended").items(limit)
                data=[]
                for tweet in cursor:
                    data.append([tweet.created_at,tweet.source,tweet.favorite_count,tweet.retweet_count,tweet.full_text])
    
                df = pd.DataFrame(data, columns=["Date Created", "Tweet Source", "Likes", "Retweets", "Tweets"])
            except:
                st.error('Username is Inavalid')

            

        # putting condition if @username is invalid
        try:

            if df.shape[0] ==0:
                st.error('Username is Inavalid')
            else:
                st.success('Done!')
                time.sleep(1)
                st.write('**First five tweets**')
                df_five=df['Tweets'][:5]
                st.dataframe(df_five)

                # initializing the sentiment analyzer object to check pos,neg,neu
                sia=SentimentIntensityAnalyzer()

                # Applying the Data cleaning process
                emoj = re.compile("["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                u"\U00002500-\U00002BEF"  # chinese char
                u"\U00002702-\U000027B0"
                u"\U00002702-\U000027B0"
                u"\U000024C2-\U0001F251"
                u"\U0001f926-\U0001f937"
                u"\U00010000-\U0010ffff"
                u"\u2640-\u2642" 
                u"\u2600-\u2B55"
                u"\u200d"
                u"\u23cf"
                u"\u23e9"
                u"\u231a"
                u"\ufe0f"  # dingbats
                u"\u3030"
                            "]+", re.UNICODE)

                # making all data cleaning functions
                def cleanTxt(text):
                    text = re.sub('…', '', text)
                    text = re.sub('&amp;', '', text)
                    text = re.sub('@[A-Za-z0–9]+', '', text) #Removing @mentions
                    text = re.sub('#', '', text) # Removing '#' hash tag
                    text = re.sub('|', '', text) # Removing '|' sign
                    text = re.sub('-', '', text) # Removing '-' sign
                    text = re.sub('RT[\s]+', '', text) # Removing RT
                    text = re.sub('https?:\/\/\S+', '', text) # Removing hyperlink
                    text = re.sub(emoj, '', text)
                    return text

                def get_neg(text):
                    return sia.polarity_scores(text)['neg']

                def get_neu(text):
                    return sia.polarity_scores(text)['neu']
                    
                def get_pos(text):
                    return sia.polarity_scores(text)['pos']

                def get_compound(text):
                    return sia.polarity_scores(text)['compound']

                def get_analysis(compound):
                    if compound >= 0.05:
                        return 'Positive'
                    elif compound <= -0.05 :
                        return 'Negative'
                    else:
                        return 'Neutral'

                # Applaying all the function on the data
                df['Tweets'] = df['Tweets'].apply(cleanTxt)
                df['Neg']=df['Tweets'].apply(get_neg)
                df['Neu']=df['Tweets'].apply(get_neu)
                df['Pos']=df['Tweets'].apply(get_pos)
                df['Compound']=df['Tweets'].apply(get_compound)
                df['Sentiment Analysis']=df['Compound'].apply(get_analysis)

                # Showing the labeled and cleaned data
                st.write('**Labeled & Clean Data**')
                st.dataframe(df)
                st.download_button('Download Labeled Data',df.to_csv(index=False),'Labeled_Data.csv')

                # Showuing the word cloude
                st.write('**WordCloud of user Data**')

                allwords = ' '.join([twts for twts in df['Tweets']])
                wordCloud =WordCloud(stopwords=STOPWORDS ,width = 600, height=450, random_state = 21, max_font_size = 200,background_color='#05171f').generate(allwords)
                plt.imshow(wordCloud)
                plt.axis("off")
                plt.tight_layout(pad=0, h_pad=0, w_pad=0)
                plt.savefig('WC.jpg')
                img_word= Image.open("WC.jpg") 
                st.image(img_word)

                # Showuing the Most frequently used words
                st.write('**Most Repeted Words in Tweets**')
                lower_wordz=allwords.lower()
                stop_words=set(stopwords.words('english'))
                wordz=[word for word in lower_wordz.split() if word not in stop_words]
                count=Counter(wordz)
                data=count.most_common(10)
                count_df=pd.DataFrame(data=data,columns=['words','Frequency'])
                fig_most = px.bar(count_df, x="Frequency", y="words", orientation='h',color='words')
                
                st.plotly_chart(fig_most)

                # Showing the sentiment analysis bar chart
                df_sentiment=df['Sentiment Analysis'].value_counts().rename_axis('Sentiment').reset_index(name='Frequency')
                st.write('**Bar-Chart of Sentiment Analysis**')
                fig_sentiments = px.bar(df_sentiment, x='Sentiment', y='Frequency', color="Sentiment",color_discrete_map={
                    'Positive': '#4daf4a',
                    'Neutral': '#377eb8',
                    'Negative' : '#e41a1d' })
                st.plotly_chart(fig_sentiments)

                # Showing the sentiment analysis donut chart/pie-chart
                st.write('**Donut-Chart of Sentiment Analysis**')
                fig_donut = px.pie(df_sentiment, values='Frequency', names='Sentiment', hole=.5, color="Sentiment",color_discrete_map={
                    'Positive': '#4daf4a',
                    'Neutral': '#377eb8',
                    'Negative' : '#e41a1d'})
                st.plotly_chart(fig_donut)

                # 3D Scatter Plots
                st.write('**3D Scatter Plots of Sentiment Analysis**')
                fig_3d = px.scatter_3d(df, x='Pos', y='Neu', z='Neg',color='Sentiment Analysis',color_discrete_map={
                    'Positive': '#4daf4a',
                    'Neutral': '#377eb8',
                    'Negative' : '#e41a1d'})
                st.plotly_chart(fig_3d)

        except:
            st.error('Input is Inavalid')

    else:
        st.info('Input to search data')

##################################################################################

# search for keywords
else:

    # seeting-up image with text
    col1 , col2 = st.columns([5,1])
    with col1:
        st.title('Tweets Sentiment Analyzer')
    with col2:
        st.image(image_bird,width=50)

    
    # Taking Input of @username and limit
    name , num = st.columns(2)
    with name:
        name=st.text_input('Enter Keyword')
    with num:
        num=st.text_input('Enter the Limit')
    
    # making condition not work untill name and limit not entered
    if name !="" and num !="" :
        # Adding the Spinner
        with st.spinner('Wait data is collecting...'):
           # using snsscraper to extract twitter data
            try:
                limit=int(float(num))
            except:
                limit=100
            qury=f"{name} -filter:retweets"
            try:

                cursor=tweepy.Cursor(api.search_tweets, q=qury, count=200, result_type="mixed",include_entities=True,lang="en",tweet_mode="extended").items(limit)
                data=[]
                for tweet in cursor:
                    print(tweet)
                    data.append([tweet.created_at,tweet.user.screen_name,tweet.source,tweet.favorite_count,tweet.retweet_count,tweet.full_text])
        
                df= pd.DataFrame(data, columns=["Date Created", "User Name", "Tweet Source", "Likes", "Retweets", "Tweets"])

            except:
                st.error('Input is Inavalid')

        # putting condition if @username is invalid
        # Show the first 5 rows of data
        if df.shape[0] ==0:
            st.error('Keyword is Inavalid')
        else:
            st.success('Done!')
            time.sleep(1)
            st.write('**First five tweets**')
            df_five=df['Tweets'][:5]
            st.dataframe(df_five)

            # initializing the sentiment analyzer object to check pos,neg,neu
            sia=SentimentIntensityAnalyzer()

            # Applying the Data cleaning process
            emoj = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002500-\U00002BEF"  # chinese char
            u"\U00002702-\U000027B0"
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            u"\U0001f926-\U0001f937"
            u"\U00010000-\U0010ffff"
            u"\u2640-\u2642" 
            u"\u2600-\u2B55"
            u"\u200d"
            u"\u23cf"
            u"\u23e9"
            u"\u231a"
            u"\ufe0f"  # dingbats
            u"\u3030"
                        "]+", re.UNICODE)

            # making all data cleaning functions
            def cleanTxt(text):
                text = re.sub('…', '', text)
                text = re.sub('&amp;', '', text)
                text = re.sub('@[A-Za-z0–9]+', '', text) #Removing @mentions
                text = re.sub('#', '', text) # Removing '#' hash tag
                text = re.sub('|', '', text) # Removing '|' sign
                text = re.sub('-', '', text) # Removing '-' sign
                text = re.sub('RT[\s]+', '', text) # Removing RT
                text = re.sub('https?:\/\/\S+', '', text) # Removing hyperlink
                text = re.sub(emoj, '', text)
                return text

            def get_neg(text):
                return sia.polarity_scores(text)['neg']

            def get_neu(text):
                return sia.polarity_scores(text)['neu']
                
            def get_pos(text):
                return sia.polarity_scores(text)['pos']

            def get_compound(text):
                return sia.polarity_scores(text)['compound']

            def get_analysis(compound):
                if compound >= 0.05:
                    return 'Positive'
                elif compound <= -0.05 :
                    return 'Negative'
                else:
                    return 'Neutral'

            # Applaying all the function on the data
            df['Tweets'] = df['Tweets'].apply(cleanTxt)
            df['Neg']=df['Tweets'].apply(get_neg)
            df['Neu']=df['Tweets'].apply(get_neu)
            df['Pos']=df['Tweets'].apply(get_pos)
            df['Compound']=df['Tweets'].apply(get_compound)
            df['Sentiment Analysis']=df['Compound'].apply(get_analysis)

            # Showing the labeled and cleaned data
            st.write('**Labeled & Clean Data**')
            st.dataframe(df)
            st.download_button('Download Labeled Data',df.to_csv(index=False),'Labeled_Data.csv')

            # Showuing the word cloude
            st.write('**WordCloud of user Data**')

            allwords = ' '.join([twts for twts in df['Tweets']])
            wordCloud =WordCloud(stopwords=STOPWORDS ,width = 600, height=450, random_state = 21, max_font_size = 200,background_color='#05171f').generate(allwords)
            plt.imshow(wordCloud)
            plt.axis("off")
            plt.tight_layout(pad=0, h_pad=0, w_pad=0)
            plt.savefig('WC.jpg')
            img_word= Image.open("WC.jpg") 
            st.image(img_word)

            # Showuing the Most frequently used words
            st.write('**Most Repeted Words in Tweets**')
            lower_wordz=allwords.lower()
            stop_words=set(stopwords.words('english'))
            wordz=[word for word in lower_wordz.split() if word not in stop_words]
            count=Counter(wordz)
            data=count.most_common(10)
            count_df=pd.DataFrame(data=data,columns=['words','Frequency'])
            fig_most = px.bar(count_df, x="Frequency", y="words", orientation='h',color='words')
            
            st.plotly_chart(fig_most)

            # Showing the sentiment analysis bar chart
            df_sentiment=df['Sentiment Analysis'].value_counts().rename_axis('Sentiment').reset_index(name='Frequency')
            st.write('**Bar-Chart of Sentiment Analysis**')
            fig_sentiments = px.bar(df_sentiment, x='Sentiment', y='Frequency', color="Sentiment",color_discrete_map={
                'Positive': '#4daf4a',
                'Neutral': '#377eb8',
                'Negative' : '#e41a1d' })
            st.plotly_chart(fig_sentiments)

            # Showing the sentiment analysis donut chart/pie-chart
            st.write('**Donut-Chart of Sentiment Analysis**')
            fig_donut = px.pie(df_sentiment, values='Frequency', names='Sentiment', hole=.5, color="Sentiment",color_discrete_map={
                'Positive': '#4daf4a',
                'Neutral': '#377eb8',
                'Negative' : '#e41a1d'})
            st.plotly_chart(fig_donut)

            # 3D Scatter Plots
            st.write('**3D Scatter Plots of Sentiment Analysis**')
            fig_3d = px.scatter_3d(df, x='Pos', y='Neu', z='Neg',color='Sentiment Analysis',color_discrete_map={
                'Positive': '#4daf4a',
                'Neutral': '#377eb8',
                'Negative' : '#e41a1d'})
            st.plotly_chart(fig_3d)
    else:
        st.info('Input to search data')

            

