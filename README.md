# Sentiment analysis on Yelp review data
NLP Project

The instructions for running the codes are mentioned in the Readme file in the Code folder.

**Objective**: To predict Yelp review ratings polarity (positive or negative) based on review text using Natural Language Processing (NLP) models focused on sentiment analysis. The project explores the relationship between review wording and corresponding star ratings.

**Data**: Utilizes the Yelp reviews polarity dataset from Kaggle, modified for binary classification (positive or negative ratings). The dataset comprises over 560,000 training and 38,000 test samples.

**Steps**:

**Data Preprocessing**: Involves cleaning the dataset by removing stopwords, punctuations, and lemmatizing the text to simplify and reduce unique tokens.

**Model Development**:
Recurrent Neural Networks (RNNs) with Long Short-Term Memory (LSTM): Implements an LSTM model to predict sentiment polarity, using PyTorch for model implementation.

**Transformer Models**: Utilizes pre-trained models like DistilBERT and ELECTRA for sentiment analysis, comparing their performance with LSTM.

**Training and Evaluation**: Models are trained on the preprocessed dataset, with performance evaluated based on accuracy in predicting sentiment polarity.

**Observations**: Pre-trained models showed higher accuracy compared to LSTM, even with lesser training data and without extensive data cleaning.
