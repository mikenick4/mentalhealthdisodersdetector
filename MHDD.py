#Script found on the internet for gathering posts from reddit
!pip install asyncpraw 
!pip install nest_asyncio
!pip install praw
import os
import json
import asyncio
import asyncpraw
import nest_asyncio
nest_asyncio.apply()

client_id = ""
client_secret = ""
user_agent = ""

output_dir = 'reddit_posts'
os.makedirs(output_dir, exist_ok=True)

async def fetch_posts():
    reddit = asyncpraw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent
    )

    subreddit_name = 'OCD'
    subreddit = await reddit.subreddit(subreddit_name)

    posts = []
    count = 0
    async for submission in subreddit.hot(limit=None):
        if count >= 600:
            break
        if not submission.selftext.strip():
            continue

        post_data = {
            'title': submission.title,
            'content': submission.selftext,
        }
        posts.append(post_data)
        count += 1

    # Save all posts to a single JSON file
    output_file = os.path.join(output_dir, 'OCD.json')
    with open(output_file, 'w') as f:
        json.dump(posts, f, indent=2)

    print(f"Finished fetching {count} posts. Saved to {output_file}")
    await reddit.close()

# Schedule and run the coroutine
loop = asyncio.get_event_loop()
loop.run_until_complete(fetch_posts())

import json
with open("anxiety.json","r") as file:
  a=json.load(file)
with open("OCD.json","r") as file:
  o=json.load(file)
with open("depression.json","r") as file:
  d=json.load(file)
with open("bipolar.json","r") as file:
  b=json.load(file)
with open("ptsd.json","r") as file:
  p=json.load(file)
with open("schizophrenia2.json","r") as file:
  s=json.load(file)
!pip install nltk
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'[A-Za-z]+')

at=[]
dt=[]
bt=[]
st=[]
for i in range(len(a)):
  at.append(tokenizer.tokenize(a[i]["title"]))
  at.append(tokenizer.tokenize(a[i]["content"]))
for i in range(len(d)):
  dt.append(tokenizer.tokenize(d[i]["title"]))
  dt.append(tokenizer.tokenize(d[i]["content"]))
for i in range(len(b)):
  bt.append(tokenizer.tokenize(b[i]["title"]))
  bt.append(tokenizer.tokenize(b[i]["content"]))
for i in range(len(s)):
  st.append(tokenizer.tokenize(s[i]["title"]))
  st.append(tokenizer.tokenize(s[i]["content"]))

def join_sentences(sents):
  joined_sentences = [" ".join(s) for s in sents]
  return joined_sentences
atj=join_sentences(at)
dtj=join_sentences(dt)
btj=join_sentences(bt)
stj=join_sentences(st)
print(atj)

labela=[0 for i in range(len(atj))]
labeld=[2 for i in range(len(dtj))]
labelb=[3 for i in range(len(btj))]
labels=[5 for i in range(len(stj))]
labs= labela+labeld+labelb+labels
labelsname={0:"anxiety",2:"depression",3:"bipolar",5:"schizophrenia"}
print(labels)

nta=int(len(atj)*0.8)
ntd=int(len(dtj)*0.8)
ntb=int(len(btj)*0.8)
nts=int(len(stj)*0.8)
print(nta,ntd,ntb,nts)

train_sentences = atj[:nta] + dtj[:ntd] + btj[:ntb] + stj[:nts]
test_sentences = atj[nta:] + dtj[ntd:] + btj[ntb:]  + stj[nts:]
train_labels = labela[:nta]+ labeld[:ntd] + labelb[:ntb] + labels[:nts]
test_labels = labela[nta:]  + labeld[ntd:] + labelb[ntb:] + labels[nts:]
print(len(train_sentences), len(test_sentences))
print(train_labels[961])

import nltk
nltk.download('stopwords')
list=['depression', 'anxiety','bipolar','schizophrenia']+nltk.corpus.stopwords.words('english')

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(lowercase=True,ngram_range=(1,2),stop_words=list) #  I set the ngram_range from (1,3) to (1,2), as we talked at the exam
X_train_tfidf = tfidf_vectorizer.fit_transform(train_sentences)
X_train_tfidf.shape

from sklearn.neighbors import KNeighborsClassifier
clf_knn = KNeighborsClassifier(n_neighbors=45) 
clf_knn.fit(X_train_tfidf, train_labels)

from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
X_test_tfidf = tfidf_vectorizer.transform(test_sentences)
predicted = clf_knn.predict(X_test_tfidf)
print(classification_report(predicted, test_labels, target_names=['anxiety', 'depression','bipolar','schizophrenia']))

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, train_labels)
predicted_nb = clf.predict(X_test_tfidf)

print(classification_report(predicted_nb, test_labels, target_names=['anxiety','depression','bipolar','schizophrenia']))
n=tfidf_vectorizer.get_feature_names_out()
print(n[1:20])
print(len(n))

import nltk
nltk.download('stopwords')
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
text_clf_svm = Pipeline([('vect', CountVectorizer(stop_words=nltk.corpus.stopwords.words('english'))), ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, max_iter=5, random_state=42))])

text_clf_svm = text_clf_svm.fit(train_sentences, train_labels)
predicted_svm = text_clf_svm.predict(test_sentences)

print(classification_report(predicted_svm, test_labels, target_names=['anxiety','depression' ,'bipolar','schizophrenia']))

import joblib
text_clf_svm =text_clf_svm.fit(train_sentences, train_labels)
joblib.dump(text_clf_svm, 'mental_health_model.pkl')
print("Model saved as 'mental_health_model.pkl'")

text1="I can't stop overthinking about the meeting tomorrow. My chest feels tight, and my heart is racing, even though I know it’s just a small presentation. Every time I try to relax, my mind keeps jumping to 'What if I mess up?' or 'What if they judge me?' I can’t seem to breathe properly or calm down"
text2="Everything feels pointless lately. I barely have the energy to get out of bed, and even things I used to enjoy don’t make me happy anymore. It’s like this heavy cloud is over me all the time. I feel so alone, but I don’t even have the strength to reach out to anyone. Maybe it’d be better if I just disappeared."
text3="People keep telling me that the voices I hear aren’t real, but how can they say that when I hear them so clearly? Sometimes, they whisper things about me, and other times, they shout warnings. It feels like someone’s always watching me, even when I’m alone. I can’t trust anyone anymore—not even myself."
def detect_mental_health_category(text):
  model = joblib.load('mental_health_model.pkl')
  predicted_label = model.predict([text])
  if predicted_label==0:
    print("anxiety")
  elif predicted_label==2: #I edited the code so it can show the show the name of the label
    print("depression")
  elif predicted_label==3:
    print("bipolar")
  elif predicted_label==5:
    print("schizophrenia")
  return predicted_label
  
print(detect_mental_health_category(text1),detect_mental_health_category(text2),detect_mental_health_category(text3))