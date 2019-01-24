
# coding: utf-8

# In[29]:


# some necessary imports
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords 
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from matplotlib import pyplot as plt



# In[30]:


# names = ['label','comment','author','subreddit','score','ups','downs','date','created_utc','parent_comment']
# train_data = []
# train_data.append(names)

# with open('sarcasm/train-balanced-sarc.csv', newline='') as csv_file:
#     csv_reader = csv.reader(csv_file,delimiter='\t')
#     line_count = 0
#     for row in csv_reader:
#         train_data.append(row)

# with open('sarcasm/train-balanced.csv','a',newline='') as f:
#     writer=csv.writer(f)
#     for i in train_data:
#         writer.writerow(i)


# In[31]:

'''
loading data
'''
train_df = pd.read_csv('sarcasm/train-balanced.csv',error_bad_lines=False)


# In[ ]:


# train_df['comment'] += train_df['parent_comment']
# stop_words = set(stopwords.words('english')) 
# lemmatizer = WordNetLemmatizer()
# stop_words.add(' ')
# stop_words.add('')

# def stopwords(s):
#     for word in s.split():
#         word = word.strip().lower()
#         word = ''.join(filter(str.isalpha, word))
#         word = lemmatizer.lemmatize(word)
#         word = lemmatizer.lemmatize(word,'v')
#         if word not in stop_words:
#             word = word.translate(table)
#             word = lemmatizer.lemmatize(word)
#             word = lemmatizer.lemmatize(word,'v')
            
# #train_df['comment'] = train_df['commnet'].apply(lambda x: )



# In[32]:


#Some comments are missing, so we drop the corresponding rows.
train_df.dropna(subset=['comment'], inplace=True)


# In[33]:


#We notice that the dataset is indeed balanced
train_df['label'].value_counts()


# In[34]:


train_sample = train_df.sample(frac = 0.1, random_state = 1)
#We split data into training and validation parts.
train_texts, valid_texts, y_train, y_valid =         train_test_split(train_sample['comment'], train_sample['label'], random_state=17)


# In[45]:


len(valid_texts)


# In[4]:


# #We split data into training and validation parts.
# train_texts, valid_texts, y_train, y_valid = \
#         train_test_split(train_df['comment'], train_df['label'], random_state=17)


# In[ ]:


# #Tok, list of tuples:(td-idf,label)
# table = str.maketrans({key: None for key in string.punctuation})
# stop_words = set(stopwords.words('english')) 
# lemmatizer = WordNetLemmatizer()
# stop_words.add(' ')
# stop_words.add('')
# train_set = []
# V={}
# for i in range(len(train_texts)):
#     docu = {}
#     v = {}
#     words = train_texts.iloc[i].strip().split()
#     for word in words:
#         word = word.strip().lower()
#         word = ''.join(filter(str.isalpha, word))
#         word = lemmatizer.lemmatize(word)
#         word = lemmatizer.lemmatize(word,'v')
#         if word not in stop_words:
#             word = word.translate(table)
#             word = lemmatizer.lemmatize(word)
#             word = lemmatizer.lemmatize(word,'v')
#             if word not in stop_words:
#                 v[word] = 1
#                 if word in docu:
#                     docu[word]+=1
#                 else:
#                     docu[word] = 1
#     trup = (docu,y_train.iloc[i])
#     train_set.append(trup)
#     for key in v.keys():
#         if key in V:
#             V[key] +=1
#         else:
#             V[key] = 1
# len_D = len(train_set)
# for v in train_set:
#     for w in v[0]:
#         v[0][w] = v[0][w] *(np.log10(len_D/V[w]))
    
    


# In[ ]:


#Distribution of lengths for sarcastic and normal comments is almost the same.
train_df.loc[train_df['label'] == 1, 'comment'].str.len().apply(np.log1p).hist(label='sarcastic', alpha=.5)
train_df.loc[train_df['label'] == 0, 'comment'].str.len().apply(np.log1p).hist(label='normal', alpha=.5)
plt.legend();


# In[39]:


# build bigrams, put a limit on maximal number of features
# and minimal word frequency
'''
models
'''

from sklearn.svm import LinearSVC
tf_idf = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, min_df=2)
# multinomial logistic regression a.k.a softmax classifier

#naive bayes
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
tfidf_nb_pipeline = Pipeline([('tf_idf', tf_idf), 
                                 ('nb', nb)])

logit = LogisticRegression(C=1, n_jobs=4, solver='lbfgs', 
                           random_state=17, verbose=1)

svc = LinearSVC(random_state=17, tol=1e-5)


# sklearn's pipeline
tfidf_logit_pipeline = Pipeline([('tf_idf', tf_idf), 
                                 ('logit', logit)])

tfidf_svc_pipeline = Pipeline([('tf_idf', tf_idf), 
                                 ('svc', svc)])


# In[42]:


from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=5, n_iter=7, random_state=17)

svd_svc_pipeline = Pipeline([('svd', svd), 
                                 ('svc', svc)])

svd_svc_pipeline.fit(train_texts, y_train)
valid_pred_svdsvc = svd_svc_pipeline.predict(valid_texts)
acc_svd_svc = accuracy_score(y_valid, valid_pred_svdsvc)


# In[6]:


get_ipython().run_cell_magic('time', '', 'tfidf_logit_pipeline.fit(train_texts, y_train)')


# In[20]:


get_ipython().run_cell_magic('time', '', 'train_pred_logit = tfidf_logit_pipeline.predict(train_texts)')


# In[22]:


train_acc_logit = accuracy_score(y_train, train_pred_logit)
train_acc_logit


# In[9]:


get_ipython().run_cell_magic('time', '', 'valid_pred_logit = tfidf_logit_pipeline.predict(valid_texts)')


# In[10]:


acc_logit = accuracy_score(y_valid, valid_pred_logit)


# In[11]:


get_ipython().run_cell_magic('time', '', 'tfidf_svc_pipeline.fit(train_texts, y_train)')


# In[12]:


get_ipython().run_cell_magic('time', '', 'valid_pred_svc = tfidf_svc_pipeline.predict(valid_texts)')


# In[13]:


acc_svc = accuracy_score(y_valid, valid_pred_svc)


# In[23]:


train_pred_svc = tfidf_svc_pipeline.predict(train_texts)
train_acc_svc = accuracy_score(y_train, train_pred_svc)
train_acc_svc


# In[36]:



 


# In[16]:


get_ipython().run_cell_magic('time', '', 'tfidf_nb_pipeline.fit(train_texts, y_train)')


# In[17]:


get_ipython().run_cell_magic('time', '', 'valid_pred_nb = tfidf_nb_pipeline.predict(valid_texts)\nacc_nb = accuracy_score(y_valid, valid_pred_nb)')


# In[24]:


train_pred_nb = tfidf_nb_pipeline.predict(train_texts)
train_acc_nb = accuracy_score(y_train, train_pred_nb)
train_acc_nb


# In[18]:


print('svc acc', acc_svc)
print('logit acc', acc_logit)
print('naive bayes acc', acc_nb)


# In[37]:


#learning curves

import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):


    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# In[ ]:


title_svc = "Learning Curves (Linear SVC, sample rate = 0.5, C = 1)"
# SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=17)
estimator = tfidf_svc_pipeline
LC_svc = plot_learning_curve(estimator, title_svc, train_texts, y_train, (0.6, 1.01), cv=cv, n_jobs=4)

LC_svc.show()


# In[41]:


title_svd_svc = "Learning Curves (Linear SVC, sample rate = 0.1, C = 1, Using SVD)"
# SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=17)
estimator = svd_svc_pipeline
LC_svc = plot_learning_curve(estimator, title_svd_svc, train_texts, y_train, (0.6, 1.01), cv=cv, n_jobs=4)

LC_svc.show()


# In[38]:


title_logit = "Learning Curves (Logistic Regression, sample rate = 0.5, C = 1)"
# SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=17)
estimator = tfidf_logit_pipeline
LC_logit = plot_learning_curve(estimator, title_logit, train_texts, y_train, (0.6, 1.01), cv=cv, n_jobs=4)

LC_logit.show()


# In[27]:


title_nb = "Learning Curves (Naive Bayes, sample rate = 0.5)"
# SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=17)
estimator = tfidf_nb_pipeline
LC_nb = plot_learning_curve(estimator, title_nb, train_texts, y_train, (0.6, 1.01), cv=cv, n_jobs=4)

LC_nb.show()


# In[ ]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),
                         algorithm="SAMME",
                         n_estimators=200)
tfidf_bdt_pipeline = Pipeline([('tf_idf', tf_idf), 
                                 ('bdt', bdt)])


# In[ ]:


title_bdt = "Learning Curves (Adaboost, sample rate = 0.5)"
# SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
estimator = tfidf_bdt_pipeline
LC_bdt = plot_learning_curve(estimator, title_bdt, train_texts, y_train, (0.6, 1.01), cv=cv, n_jobs=4)

LC_bdt.show()


# In[ ]:


def plot_confusion_matrix(actual, predicted, classes,
                          normalize=False,
                          title='Confusion matrix', figsize=(7,7),
                          cmap=plt.cm.Blues, path_to_save_fig=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(actual, predicted).T
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Predicted label')
    plt.xlabel('True label')
    
    if path_to_save_fig:
        plt.savefig(path_to_save_fig, dpi=300, bbox_inches='tight')


# In[ ]:


#svc confusion matrix
plot_confusion_matrix(y_valid, valid_pred_svc, 
                      tfidf_svc_pipeline.named_steps['svc'].classes_, figsize=(8, 8))


# In[ ]:


# accuracy

