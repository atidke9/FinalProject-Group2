import nltk
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# nltk.download('stopwords')
# nltk.download('wordnet')
stopwords = nltk.corpus.stopwords.words('english')

def cleanData(list_of_strings):
    cleaned_list = []
    for string in list_of_strings:
        tokens = nltk.word_tokenize(string)
        """ remove stopwords """
        tokens = [token.lower() for token in tokens if token.lower() not in stopwords]

        """ remove punctuation"""
        tokens = [token for token in tokens if token.isalnum()]

        """ lemmatize """
        wnl = nltk.WordNetLemmatizer()
        tokens = [wnl.lemmatize(token) for token in tokens]

        cleanstring = ' '.join(tokens)
        cleaned_list.append(cleanstring)
    return cleaned_list

PATH = os.getcwd()

train_data = pd.read_csv(PATH + '/yelp_review_polarity_csv/train.csv', names = ['Label','Text'] )
trainData_list = train_data['Text'].tolist()
cleaned_trainData = cleanData(trainData_list)
train_data['cleaned_text'] = cleaned_trainData
train_data = train_data.drop(['Text'], axis = 1)
columns_titles = ["cleaned_text","Label"]
train_data = train_data.reindex(columns=columns_titles)
dict_map = {1:0, 2:1}
train_data = train_data.replace({"Label": dict_map})

""" removing rows with zero length strings """
data_list = train_data['cleaned_text'].tolist()
arr = []
for i in range(len(data_list)):
    string = data_list[i]
    tokens = nltk.word_tokenize(string)
    if len(tokens) == 0:
        arr.append(i)

print("Percentage of zero length sequences in train data, after data cleaning:", 100*len(arr)/len(train_data),"\n")
print("It is safe to discard these data points.")
train_data = train_data.drop(index = arr)

""" splitting train into train and validation dataframe """
train_data, val_data = train_test_split(train_data, train_size=0.9, random_state=random_seed)
train_data, val_data = train_data.reset_index(drop=True), val_data.reset_index(drop=True)

test_data = pd.read_csv(PATH + '/yelp_review_polarity_csv/test.csv', names = ['Label','Text'] )
testData_list = test_data['Text'].tolist()
cleaned_testData = cleanData(testData_list)
test_data['cleaned_text'] = cleaned_testData
test_data = test_data.drop(['Text'], axis = 1)
columns_titles = ["cleaned_text","Label"]
test_data = test_data.reindex(columns=columns_titles)
dict_map = {1:0, 2:1}
test_data = test_data.replace({"Label": dict_map})

""" removing rows with zero length strings """
data_list = test_data['cleaned_text'].tolist()
arr_test = []
for i in range(len(data_list)):
    string = data_list[i]
    tokens = nltk.word_tokenize(string)
    if len(tokens) == 0:
        arr_test.append(i)

print("Percentage of zero length sequences in test data, after data cleaning:", 100*len(arr_test)/len(test_data),"\n")
print("It is safe to discard these data points.")
test_data = test_data.drop(index = arr_test)

train_data.to_csv('train_cleaned.csv')
val_data.to_csv('val_cleaned.csv')
test_data.to_csv('test_cleaned.csv')






# download required file from blackboard to the above path
# upload the file to cloud
#open and save text file as a string
# glo = open(PATH + '/Train.txt')        #upload to ubuntu
# string = glo.read()

# # if downloading from url zip file
# os.system("wget https://dl.fbaipublicfiles.com/glue/data/CoLA.zip")
# os.system("unzip CoLA.zip")
# file = pd.read_csv("CoLA/test.tsv", sep='\t')
