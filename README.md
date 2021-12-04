# Gender Prediction


## Introduction
In this project I attempt to predict the gender (male / female) of English text authors. I built the dataset myself by using a third Python library to fetch public posts and comments from reddit. I tackled the problem with 2 different methods and compared the results. The first method is a regular Machine Learning approach using several classifiers and TfIdf values of certain features, and the second method is a deep learning approach using a bidirectional LSTM network. Additionally, I deployed the final model to the web using Flask, so it can be accessed and tested online by everyone.

Here I present a quick overview of the project. For a complete walkthrough, including the code, please head over to the [src directory](https://github.com/masalha-alaa/gender-prediction/tree/master/src) for the regular ML approach, and to my notebook [gender-recognition-keras.ipynb](https://github.com/masalha-alaa/gender-prediction/blob/master/gender_recognition_keras.ipynb) for the DL approach.

## Dataset
I used the [PSAW Python library](https://github.com/dmarx/psaw) to fetch data from reddit, and the [facebook-scraper library](https://github.com/kevinzg/facebook-scraper) to fetch data from Facebook. I ended up using the reddit data only, since the labeled FB data was very limited (since most users have their gender setting set to private).

In some subreddits such as [r/AskMen](https://www.reddit.com/r/AskMen/), [r/AskWomen](https://www.reddit.com/r/AskWomen/), [r/relationship_advice](https://www.reddit.com/r/relationship_advice/), the authors provide a label called "flair", which contains their gender. Using the PSAW library I was able to fetch 164,855 posts and comments, ~70% of which are labeled as "female", and 30% are labeled as "male". The fetching script can be found under src/fetch_data_reddit.py.

## Preprocessing
I performed the following preprocessing on the dataset before feeding it to the models:

* Lowercase: convert all the data to lowercase.
* Cleaning: drop posts with too few words, replace emojis with the token 'emj', replace URLs with the token 'url', and more.
* Split to sentences: Using the Python library [nltk](https://www.nltk.org/), I split the posts to sentences (sentences are lines that end with a stopping punctuation mark). After this step each sample is a single sentence.
* Shuffle: I shuffled the **sentences** to fade out any in-author relationships.
* Tokenization.
* Aggregate: I aggregated every 4 sentences together to make the samples a little bit longer.
* POSify: Create POS tags of the text using nltk.
* Balance classes: Cut out classes according to the smaller class.

After these steps I ended up with about 38K samples per class. That's way more than needed, so I used only 30% of it (12K per class).

## Model
### Regular Machine Learning Approach
In this approach, I built a dataframe with the following features:
* TfIdf values of the most common 1500 ngrams (1-3) in the dataset.
* TfIdf values of the most common 1500 POS trigrams in the dataset.
* Average sentence length.
* Sentiment analysis (nltk function which scores each sample of being 'positive' / 'negative' / 'neutral'.
* Custom words list: a list of custom words I thought would help the classifier (collected in various methods).

Here the results among several classifiers that I tried, including an ensemble Voting model:

|Model                   |Accuracy|  |MSE|  |Tolerance with 95% Confidence Level|
|------------------------|-----------|------|-----------------------------------|
|Logistic Regression     |66.54%     |0.335 |±0.011|
|Multinomial NB          |65.28%     |0.347 |±0.011|
|Random Forest           |63.92%     |0.361 |±0.011|
|Voting (Ensemble)       |66.44%     |0.336 |±0.011|

As can be seen, the Logistic Regression and the Voting classifier (which consists of all the other classifiers band together) are the winners.

In this approach it's interesting to see the most important words for the classifiers that influenced the classification process the most:

``
'!', '! i', '$ jj nn', "'", "'ll", "'s", ', she', '. !', '. )', '. he', '. i', '. my', '. nn vbd', '. prp vbd', '. ”', '>', 'a girl', 'a woman', 'and', 'and he', 'and i', 'and she', 'boyfriend', 'can ’', 'can ’ t', 'didn', 'didn ’', 'didn ’ t', 'don', 'don ’', 'don ’ t', 'emj', 'gf', 'girl', 'girlfriend', 'guy', 'guys', 'he', 'he was', 'he ’', 'her', 'him', 'his', 'husband', 'i', 'i had', 'i was', 'i ’', 'i ’ m', 'is', 'it ’', 'it ’ s', 'jj cc jj', 'just', 'm', 'makeup', 'man', 'me', 'mom', 'my', 'my bf', 'my gf', 'my husband', 'my partner', 'my wife', "n't", 'na', 'nn vbd dt', 'nn vbp nn', 'nnp nn nn', 'our', 'own', 'probably', 'prp $ in', 'prp $ jj', 'prp vbz jj', 's', 'she', 'she is', 'shit', 'so', 't', 'to her', 'vbd rb jj', 'vbp jj nn', 'vbz dt nn', 'was', 'when i', 'wife', 'with her', 'woman', 'wrb nn vbd', 'you', '’', '’ m', '’ re', '’ s', '’ t', '“', '”'
``

### Deep Learning Approach
In this approach I used an RNN model which consists of an Embedding layer with 100 output dimension, one Bidirectional LSTM cell with 128 units and a softmax activation layer. The loss function is categorical cross entropy, and the optimzer is Adam with 0.0001 learning rate.

What makes this approach essentially different than the regular ML approach, is that unlike the ML "BOW" approach, here we maintain the sequences. But in addition to the original text sequences, I wanted to add extra information such as POS and sentiment analysis. So I converted the text to pairs of (word, POS) and added the highest sentiment analysis category at the end of each sentence. For example, the sentence:  
_if you 're dwelling on the negative all the time you 're going to wind up ignoring or missing the positives ._  
Turned into:
_if IN you PRP 're VBP dwelling VBG on IN the DT negative JJ all PDT the DT time NN you PRP 're VBP going VBG to TO wind VB up RP ignoring VBG or CC missing VBG the DT positives NNS . . neu_  
As can be seen in this example, each token is followed by a POS, and at the end we have the sentiment analysis tag for this sentence "neu" (neutral) in this case.

I set the max epochs to 20, and set an EarlyStopping callback which monitors the validation accuracy, which made the training process stop after 12 epochs with an accuracy of 67.83%. Following is the training plot:

![Training Plot](https://user-images.githubusercontent.com/78589884/144708523-6938b1b3-c8c3-45ae-874c-7f985c1a0622.png)

## Conclusion
As can be seen from the results, the two methods are comparable but the RNN deep learning approach outperformed the regular ML approach just a litte bit (2%).

## Deployment
