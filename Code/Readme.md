Step 1: Download all the above files to a folder and name it Group2Project (for example)

Step 2: Download the dataset as follows:
## For Windows users
As we are downloading our dataset from Kaggle, we have to install the kaggle API first followed by few more steps.

Run the following command to access the Kaggle API using the command line (Anaconda command prompt for Pycharm):    
***pip install kaggle***

1. To use the Kaggle API, sign up for a Kaggle account at https://www.kaggle.com. 
2. Then go to the 'Account' tab of your user profile (https://www.kaggle.com/<username>/account) and select 'Create API Token'. 
3. This will trigger the download of kaggle.json, a file containing your API credentials. 
4. Place this file in the location C:\Users\<Windows-username>\.kaggle\kaggle.json

Change the directory in the command prompt to your project directory by using command:    
***cd Users/[username]/Downloads/Group2Project (in my case)***

Run the command on Anaconda command prompt:    
***kaggle datasets download ilhamfp31/yelp-review-dataset***    
This will download the dataset zip file to your project directory.

Once the above steps are followed, open DataDownload.py file. Upload the downloaded .zip file to your cloud from Pycharm. Run the code. This will unzip the dataset file and save it to a folder "yelp_review_polarity_csv"

## For Mac users

## Remaining steps
Step 3: Run the DataPrepocessing.py file after downloading the train and test dataset. Running this file will save cleaned datasets to different csv files which will be used for other codes.

Step 4: Run the project_lstm.py file to run the LSTM code on 100% training dataset.

Step 5: Run the project_lstm_25%data.py file to run the LSTM code on 25% training dataset. This way we compared the test accuracy by shrinking the data and fortunately, the accuracy didn't decrease much.
