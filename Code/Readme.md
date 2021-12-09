### Step 1: Download all the above files to a folder and name it Group2Project (for example)

### Step 2: Download the dataset as follows:
#### For Windows users
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

#### For Mac users
1. Download your Kaggle API   
  Log in to Kaggle and access your account.   
  Scroll down to the API section.     
  Click on ‘Create New API Token’ and download the kaggle.json file which contains your API token.

2. Run Download_data_shell.sh file    
  Open your terminal.   
  Type the following command:   
  ***bash Download_data_shell.sh <kaggle.json_location> <project_folder_location>***

Replace:

<kaggle.json_location> with the path of the kaggle.json file download in the previous step.

<project_folder_location> with the path of this project downloaded in your system.
 
### Step 3: Run the DataPreprocessing.py file to create the csv files of the cleaned data.    
  The cleaned data csv files will be used to run the LSTM codes (LSTM_Model.py and LSTM_Model_25%TrainData.py).
