#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data = pd.read_csv('C:\\Users\\user\\Desktop\\Python\\Lakshmi\\severeinjury1.csv',encoding='cp1252')
                   


# In[3]:


import nltk


# In[4]:


from nltk.util import ngrams


# In[5]:


from nltk.tokenize import WhitespaceTokenizer
from nltk.stem.snowball import SnowballStemmer
stemming = SnowballStemmer("english")
data["narrative"]= data["title.new"]+data["summary.new"]


# In[6]:


def identify_tokens(row):
    summary_title = row["narrative"]
    tokens = nltk.word_tokenize(summary_title)
    token_words = ngrams(tokens,3)
    tok = [" ".join(t) for t in token_words]
    return tok


# In[7]:


data['words']= data.apply(identify_tokens,axis=1)
print(data['words'])


# In[8]:


def stem_list(row):
    my_list = row['words']
    stemmed_list = [stemming.stem(word) for word in my_list]
    return(stemmed_list)


# In[9]:


data['stemmed_words'] = data.apply(stem_list, axis=1)
data['stemmed_words']


# In[10]:


from nltk.corpus import stopwords
stops = set(stopwords.words("english"))                  


# In[11]:


def remove_stops(row):
    my_list = row['stemmed_words']
    meaningful_words = [w for w in my_list if not w in stops]
    return (meaningful_words)


# In[12]:


data['stem_meaningful'] = data.apply(remove_stops, axis=1)
data['stem_meaningful']


# In[13]:


from sklearn.feature_extraction.text import TfidfVectorizer
import math
data['stem_meaningful'] = ["  ".join(review) for review in data['stem_meaningful'].values]
data['stem_meaningful']


# In[14]:


import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
v=TfidfVectorizer()
x=v.fit_transform(data['stem_meaningful'])
first_vector_v = x[0]


# In[15]:


df = pd.DataFrame(first_vector_v.T.todense(),index = v.get_feature_names(),columns = ["tfidf"])
df.sort_values(by=["tfidf"],ascending=False)


# In[16]:


from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes,svm
from sklearn.metrics import accuracy_score


# In[17]:


import numpy as np
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(data['summary.new'],data['Tagged2'],test_size=0.2)


# In[18]:


Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)
Train_X_Tfidf = v.transform(Train_X)
Test_X_Tfidf = v.transform(Test_X)


# In[19]:


print(v.vocabulary_)
print(Train_X_Tfidf)


# In[20]:


SVM = svm.SVC(C=0.80,kernel='linear')
SVM.fit(Train_X_Tfidf,Train_Y)
predictions_SVM = SVM.predict(Test_X_Tfidf)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
target_names = ['Caught in/between objects','Collapse of object','Electrocution','Exposure to chemical substances','Exposure to extreme temperatures','Falls','Fires and explosions','Struck by moving objects','Struck by falling objects','Traffic','Others']


# In[21]:


print(confusion_matrix(Test_Y,predictions_SVM))


# In[26]:


print(classification_report(Test_Y,predictions_SVM,target_names = target_names))


# In[21]:


SVM_accuracy = accuracy_score(Test_Y, predictions_SVM)
print(accuracy_score(Test_Y, predictions_SVM))


# In[22]:


Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf,Train_Y)
predictions_NB = Naive.predict(Test_X_Tfidf)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[23]:


print(confusion_matrix(Test_Y,predictions_NB))


# In[30]:


print(classification_report(Test_Y,predictions_NB,target_names = target_names))


# In[24]:


Naive_Accuracy = accuracy_score(Test_Y, predictions_NB)
print(accuracy_score(Test_Y, predictions_NB))


# In[25]:


from sklearn.tree import DecisionTreeClassifier
clf3 = DecisionTreeClassifier()


# In[26]:


clf3.fit(Train_X_Tfidf,Train_Y)


# In[27]:


y_pred = clf3.predict((Test_X_Tfidf))
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[41]:


print(confusion_matrix(Test_Y,y_pred))
print(classification_report(Test_Y,y_pred,target_names = target_names))


# In[28]:


DecisionTree_Accuracy = accuracy_score(Test_Y, y_pred)
print(accuracy_score(Test_Y, y_pred))


# In[29]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=25)
classifier.fit(Train_X_Tfidf,Train_Y)
y_pred = classifier.predict((Test_X_Tfidf))
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[44]:


print(confusion_matrix(Test_Y,y_pred))
print(classification_report(Test_Y,y_pred,target_names = target_names))


# In[30]:


KNeighbors_Accuracy = accuracy_score(Test_Y, y_pred)
print(accuracy_score(Test_Y, y_pred))


# In[31]:


from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
import pickle
from sklearn.linear_model import LogisticRegression
X= data['summary.new']
Y = data['Tagged2']
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.2)
pipeline = Pipeline([('vect', v),
                     ('chi',  SelectKBest(chi2, k=1200)),
                     ('clf', LogisticRegression(random_state=0))])
model = pipeline.fit(X_train, y_train)
with open('LogisticRegression.pickle', 'wb') as f:
    pickle.dump(model, f)
    ytest = np.array(y_test)


# In[32]:


print(classification_report(ytest, model.predict(X_test)))
print(confusion_matrix(ytest, model.predict(X_test)))


# In[33]:


LR_Accuracy = accuracy_score(Test_Y, y_pred)
print(accuracy_score(Test_Y, y_pred))


# In[34]:


import random
best=-100000
populations = [random.randint(0.0,1.0) for x in range(6)] 
print(type(populations))
parents=[]
new_populations = []
print(populations)


# In[35]:


equation_inputs = [SVM_accuracy,DecisionTree_Accuracy,Naive_Accuracy,KNeighbors_Accuracy,LR_Accuracy]


# In[36]:


equation_inputs = [0.58,0.365,0.495,0.53,0.53]


# In[37]:


num_weights = 5


# In[38]:


import numpy
sol_per_pop = 8


# In[39]:


pop_size = (sol_per_pop,num_weights)


# In[40]:


new_population = numpy.random.uniform(low=-0.1, high=0.5, size=pop_size)
import numpy


# In[41]:


import ga


# In[42]:


sol_per_pop = 8
num_parents_mating = 4


# In[43]:


pop_size = (sol_per_pop,num_weights) 


# In[44]:


new_population = numpy.random.uniform(low=-4.0, high=4.0, size=pop_size)
print(new_population)


# In[45]:


num_generations = 5
for generation in range(num_generations):
    print("Generation : ", generation)


# In[51]:


from geneticalgorithm import geneticalgorithm as ga


# In[46]:


def cal_pop_fitness(equation_inputs, pop):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function caulcuates the sum of products between each input and its corresponding weight.
    fitness = numpy.sum(pop*equation_inputs, axis=1)
    return fitness


# In[52]:


fitness = ga.cal_pop_fitness(equation_inputs, new_population)


# In[81]:


parents = ga.select_mating_pool(new_population, fitness, 
                                     num_parents_mating)


# In[78]:


def select_mating_pool(pop, fitness, num_parents):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    parents = numpy.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = numpy.where(fitness == numpy.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999999
    return parents


# In[82]:


def crossover(parents, offspring_size):
    offspring = numpy.empty(offspring_size)
    # The point at which crossover takes place between two parents. Usually it is at the center.
    crossover_point = numpy.uint8(offspring_size[1]/2)

    for k in range(offspring_size[0]):
        # Index of the first parent to mate.
        parent1_idx = k%parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (k+1)%parents.shape[0]
        # The new offspring will have its first half of its genes taken from the first parent.
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # The new offspring will have its second half of its genes taken from the second parent.
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring


# In[83]:


def mutation(offspring_crossover):
    # Mutation changes a single gene in each offspring randomly.
    for idx in range(offspring_crossover.shape[0]):
        # The random value to be added to the gene.
        random_value = numpy.random.uniform(-1.0, 1.0, 1)
        offspring_crossover[idx, 4] = offspring_crossover[idx, 4] + random_value
    return offspring_crossover


# In[86]:


fitness = ga.cal_pop_fitness(equation_inputs, new_population)


# In[102]:


import ga
num_generations = 5

num_parents_mating = 4
for generation in range(num_generations):
    # Measuring the fitness of each chromosome in the population.
    fitness = ga.cal_pop_fitness(equation_inputs, new_population)


# In[98]:


pip install GA


# In[112]:


parents = ga.select_mating_pool(new_population, fitness, 
                                     num_parents_mating)


# In[113]:


pip install pygad


# In[125]:


import pygad


# In[129]:


import ga


# In[131]:


pip install ga


# In[126]:


def cal_pop_fitness(equation_inputs, pop):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function caulcuates the sum of products between each input and its corresponding weight.
    fitness = numpy.sum(pop*equation_inputs, axis=1)
    return fitness


# In[130]:


fitness = ga.cal_pop_fitness(equation_inputs, new_population)


# In[ ]:




