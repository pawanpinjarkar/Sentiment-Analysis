##Twitter Sentiment Analysis for Iphone7 
#Course - FA16-BL-ILS-Z639-35041
#Author - Pawan Pinjarkar , Sunil Agrawal 
#Group - SMM Group 3 

#Import Python libaries 
import pickle as pkl
import re
from __future__ import print_function
from nltk.corpus import stopwords
from datetime import timedelta
import time
from datetime import datetime
import itertools
from operator import itemgetter
import pandas as pd
import numpy as np
import csv

##--------Input the datafile for text processing ------
IphoneSentimentAnalysisFile = input("Please enter pickle file name with processed dataset for sentiment analysis =")
print ("\nThe entered dataset filename  is   =  ", IphoneSentimentAnalysisFile)
print("Reading and processing the dataset .............................................")
#Build a dictionary whiich you want NLTK to ignore while removing stopwords. For example : not
operators = set(('no', 'nor','not','shouldn','against','aren','couldn',
                 'didn','doesn','hadn','hasn','haven','isn','mightn',
                 'mustn','needn','shouldn','wasn','weren','won','wouldn'))
stop = set(stopwords.words('english')) - operators
# The list notWithPositiveNegativeList is created to handle scenarios where a tweet says "I am not happy..." , 
#"I don't like...". As "happy" is a positive word but when it is preceeded with "not", it implies nagative sentiment.
notWithPositiveNegativeList = ['no','not','never',"don't"]
fileList = []
tweetAfterStopwords = []
finalTweetDataList=[]
finalTweetData = {}
#The sentiments are classified as 'Positve' or 'Negative' based on NLTK lexicon dictionary
pos_sent = open("positive-words.txt").read()
positive_words=pos_sent.split('\n')
neg_sent = open("negative-words.txt",encoding='utf-8', errors='ignore').read()
negative_words=neg_sent.split('\n')
sum =0
data = pkl.load(open(IphoneSentimentAnalysisFile,"rb"))
filelength = len(data)
for j in range(filelength):
    positive_counter = 0 
    negative_counter = 0
    sentiment = "Undefined"
    tweetData = data[j]    
    rawTweet = tweetData['ttext']
    #This regular expression removes hashtags(#),mentions, urls but keeps single quote -- @[A-Za-z0-9]+)|([^0-9A-Za-z' \t])|(\w+:\/\/\S+
    #(For example, don't, haven't)
    tweetCleanup = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z' \t])|(\w+:\/\/\S+)"," ",rawTweet).split()).lower()    
    stemmedWords = tweetCleanup.split(" ")
    wordCounter=0
    # The sentiments are classifed as 'Positive' or 'Negative' based on below logic -
    #If ‘number of +ve words’ >= ‘number of –ve words’ , classified as positive twee
    #If ‘number of -ve words’ > ‘number of +ve words’ , classified as negative tweet
    #If ‘number of +ve an d-ve words’ = 0 , classified as neutral tweet
    for eachword in stemmedWords:
        if eachword not in stop:
            if eachword in positive_words:
                if wordCounter is not 0 and stemmedWords[wordCounter-1]in notWithPositiveNegativeList:
                    negative_counter = negative_counter+1
                    wordCounter+=1
                    continue
                else:
                    positive_counter = positive_counter+1
            elif eachword in negative_words:
                negative_counter = negative_counter+1
        wordCounter+=1
    if positive_counter > negative_counter:
        sentiment ="Positive"
    elif positive_counter < negative_counter:
        sentiment ="Negative"        
    elif positive_counter == negative_counter and positive_counter is not 0 and negative_counter is not 0:
        # We have added a custom classifier 'ToDo' as currently we are not 
        # categorizing a sentiment based on intensity of words choosen in a tweet. 
        # We consider this is a future wotk. 
        sentiment = "ToDo" 
    else:
        sentiment ="Neutral"
    finalTweetData['tsentiment'] = sentiment
    finalTweetData['tusername'] = tweetData['tusername']
    finalTweetData['thandlename'] = tweetData['thandlename']
    finalTweetData['tgender'] = tweetData['tgender']
    finalTweetData['tlocation'] = tweetData['tlocation']
    finalTweetData['tdate'] = tweetData['tdate']
    finalTweetData['ttext'] = tweetData['ttext']
    finalTweetDataList.append(finalTweetData.copy())
    
#Total Sentiments for given dataset
PostiveCount=NegativeCount=NeutralCount=ToDoCount=0
for i in range (len(finalTweetDataList)):
    if finalTweetDataList[i]['tsentiment']=="Positive":
        PostiveCount +=1
    elif finalTweetDataList[i]['tsentiment']=="Negative":
        NegativeCount +=1
    elif finalTweetDataList[i]['tsentiment']=="Neutral":
        NeutralCount +=1
    elif finalTweetDataList[i]['tsentiment']=="ToDo":
        ToDoCount +=1
FinalClassifiedSentimentCount= len(finalTweetDataList)-ToDoCount     

print("\nTotal number of Sentimentts = " , len(finalTweetDataList), 
"\nTotal number of Positive Sentiments =" ,PostiveCount ,
"\nTotal number of Negative Sentiments = ",NegativeCount ,
"\nTotal number of Neutral Sentiments = ",NeutralCount, 
"\nTotal number of ToDo Sentiments(Noise) which is ignored = ",ToDoCount)
print(".....................................................................................")        
#Saving final PickleFile and CSV file 
pklfile="final"+IphoneSentimentAnalysisFile
pkl.dump(finalTweetDataList,open(pklfile,"wb"))
print("\nSaving the pickle file  : ",pklfile)
keys = finalTweetDataList[0].keys()
csvFile = 'final'+IphoneSentimentAnalysisFile+'.csv'
with open(csvFile, 'w',encoding='UTF-8') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(finalTweetDataList)
print("Saving the csv file : ",csvFile)
finalTweetDataList = sorted(finalTweetDataList, key=itemgetter('tlocation'))
sentimentsByStatesList=[]
sentimentsByStatesDict={}
for key, value in itertools.groupby(finalTweetDataList, key=itemgetter('tlocation')):
    count=0
    pCount=0
    nCount=0
    neutralCount=0
    gender= ""
    pMaleCount=0
    pFemaleCount=0
    nMaleCount=0
    nFemaleCount=0
    nuMaleCount=0
    nuFemaleCount=0
    for i in value:
        sentiment=i.get('tsentiment')
        if sentiment !='ToDo':
            gender = i.get('tgender')
            count +=1
            if sentiment=='Positive':
                pCount +=1
                if gender=='Male':
                    pMaleCount+=1
                elif gender=='Female':
                    pFemaleCount+=1
            elif sentiment=='Negative':
                nCount +=1
                if gender=='Male':
                    nMaleCount+=1
                elif gender=='Female':
                    nFemaleCount+=1
            elif sentiment=='Neutral':
                neutralCount +=1
                if gender=='Male':
                    nuMaleCount+=1
                elif gender=='Female':
                    nuFemaleCount+=1
    sentimentsByStatesDict['State']=key
    sentimentsByStatesDict['Total Sentiments']=count
    sentimentsByStatesDict['Total number of positive sentiments']=pCount
    sentimentsByStatesDict['Number of males with positive sentiments']=pMaleCount
    sentimentsByStatesDict['Number of females with positive sentiments']=pFemaleCount
    sentimentsByStatesDict['Total number of negative sentiments']=nCount
    sentimentsByStatesDict['Number of males with negative sentiments']=nMaleCount
    sentimentsByStatesDict['Number of females with negative sentiments']=nFemaleCount
    sentimentsByStatesDict['Total number of neutral sentiments']=neutralCount
    sentimentsByStatesDict['Number of males with neutral sentiments']=nuMaleCount
    sentimentsByStatesDict['Number of females with neutral sentiments']=nuFemaleCount
    
    sentimentsByStatesList.append(sentimentsByStatesDict.copy())

sentimentsByStatesList = sorted(sentimentsByStatesList, key=itemgetter('Total Sentiments'),reverse=True)
columnSequence =['State','Total Sentiments','Total number of positive sentiments','Number of males with positive sentiments','Number of females with positive sentiments','Total number of negative sentiments','Number of males with negative sentiments','Number of females with negative sentiments','Total number of neutral sentiments','Number of males with neutral sentiments','Number of females with neutral sentiments']
print(".....................................................................................")
print("\nFinal Number of Tweets Considered for Sentiment Analsyis(Exclused ToDo(Noise) classified Tweets) =",FinalClassifiedSentimentCount,
      "\n\nGender and location based sentiments(Positive, Negative, and Neutral) for input file ",IphoneSentimentAnalysisFile)
print(".....................................................................................")
pd.DataFrame(sentimentsByStatesList).reindex(columns=columnSequence)
