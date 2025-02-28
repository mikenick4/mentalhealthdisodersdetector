
In this project I chose to create a **Mental Health Disoders Detector**.
# Steps
-Gathering data from reddit using a script that I found online. From different reasons I only kept the tittle and the post itself. 
-Getting rid of the numbers and punctuation because they dont preserve important information using RegexpTokenizer
-Joining the tokens into text format so TFIDf could accept them
-Labelling data and keeping a dictionary
-Finding the 80% and 20% percentages
-Splitting the data into train sentences and test sentences
-Using TfIdk  for preprocessing and vectorizing data - lowercasing text, setting ngram(1,2) and adding a stopwords list of English stopwords plus the words:depression,anxiety,bipolar and schiphrenia for a more accurate model
-Training three different models so I can find the one with the better scores: KNN, Nayve Bayes and SVM classifier
-Testing SVM model on three examples so I can see its accuracy.

