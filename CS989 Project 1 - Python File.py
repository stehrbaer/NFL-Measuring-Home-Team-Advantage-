#!/usr/bin/env python
# coding: utf-8

# In[1]:


#imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import metrics
from sklearn import cluster
from sklearn.preprocessing import scale
import xlrd


# In[2]:


import sys
print(sys.prefix)


# In[3]:


nfl_spreads = pd.read_csv("Desktop/nfl_stat/spreadspoke_scores.csv")


# In[4]:


nfl_spreads1 = nfl_spreads[nfl_spreads.schedule_season >= 1990]
#df.drop('column_name', axis=1, inplace=True)
nfl_spreads1.drop('Unnamed: 17', axis=1, inplace=True)


# In[5]:


#viewing dataset 
nfl_spreads


# In[6]:


nfl_spreads1['schedule_playoff'].unique()


# In[7]:


#data cleaning
#keeping years 1990-2020

nfl_spreads1 = nfl_spreads[nfl_spreads.schedule_season >= 1990]

nfl_regular = nfl_spreads1[nfl_spreads1.schedule_playoff == False]
nfl_regular = nfl_regular[nfl_regular.schedule_season <= 2020]

nfl_playoffs = nfl_spreads1[nfl_spreads1.schedule_playoff == True]
nfl_playoffs = nfl_playoffs[nfl_playoffs.schedule_season <= 2020]


# In[8]:


nfl_playoffs


# In[9]:


#nfl regular season - calculating the actual result metrics - score of home-team minus score of away team
nfl_regular['actual_result'] = nfl_regular['score_home'] - nfl_regular['score_away']

#playoffs - calculating the actual result metrics - score of home-team minus score of away team

nfl_playoffs['actual_result'] = nfl_playoffs['score_home'] - nfl_playoffs['score_away']


# In[10]:


nfl_regular['team_favorite_id'].unique()


# In[11]:


nfl_regular['team_home'].unique()


# In[12]:


#replacing team-favorite-id with name of the team that is favorited in order to create index 
nfl_reg = nfl_regular.replace({
    'team_favorite_id': {
        'ATL': 'Atlanta Falcons',
        'BUF': 'Buffalo Bills',
        'CHI': 'Chicago Bears',
        'CIN': 'Cincinnati Bengals',
        'CLE': 'Cleveland Browns',
        'DAL': 'Dallas Cowboys',
        'DET': 'Detroit Lions',
        'GB': 'Green Bay Packers',
        'KC' : 'Kansas City Chiefs',
        'LAC': 'Los Angeles Chargers',
        'LAR': 'Los Angeles Rams',
        'MIN': 'Minnesota Vikings',
        'DEN': 'Denver Broncos',
        'MIA': 'Miami Dolphins',
        'NYG': 'New York Giants',
        'WAS': 'Washington',
        'SF': 'San Francisco 49ers',
        'IND': 'Indianapolis Colts',
        'PHI': 'Philadelphia Eagles',
        'PIT': 'Pittsburgh Steelers',
        'LVR': 'Las Vegas Raiders',
        'TEN': 'Tennessee Titans',
        'NO': 'New Orleans Saints',
        'TB': 'Tampa Bay Buccaneers',
        'SEA': 'Seattle Seahawks',
        'NYJ': 'New York Jets',
        'ARI': 'Arizona Cardinals',
        'NE': 'New England Patriots',
        'BAL': 'Baltimore Ravens',
        'JAX': 'Jacksonville Jaguars',
        'CAR': 'Carolina Panthers',
        'HOU': 'Houston Texans',
        
    },
})

#nfl_playoffs

nfl_pf = nfl_playoffs.replace({
    'team_favorite_id': {
        'ATL': 'Atlanta Falcons',
        'BUF': 'Buffalo Bills',
        'CHI': 'Chicago Bears',
        'CIN': 'Cincinnati Bengals',
        'CLE': 'Cleveland Browns',
        'DAL': 'Dallas Cowboys',
        'DET': 'Detroit Lions',
        'GB': 'Green Bay Packers',
        'KC' : 'Kansas City Chiefs',
        'LAC': 'Los Angeles Chargers',
        'LAR': 'Los Angeles Rams',
        'MIN': 'Minnesota Vikings',
        'DEN': 'Denver Broncos',
        'MIA': 'Miami Dolphins',
        'NYG': 'New York Giants',
        'WAS': 'Washington',
        'SF': 'San Francisco 49ers',
        'IND': 'Indianapolis Colts',
        'PHI': 'Philadelphia Eagles',
        'PIT': 'Pittsburgh Steelers',
        'LVR': 'Las Vegas Raiders',
        'TEN': 'Tennessee Titans',
        'NO': 'New Orleans Saints',
        'TB': 'Tampa Bay Buccaneers',
        'SEA': 'Seattle Seahawks',
        'NYJ': 'New York Jets',
        'ARI': 'Arizona Cardinals',
        'NE': 'New England Patriots',
        'BAL': 'Baltimore Ravens',
        'JAX': 'Jacksonville Jaguars',
        'CAR': 'Carolina Panthers',
        'HOU': 'Houston Texans',
        
    },
})


# In[13]:


nfl_reg['home_team_fav'] = nfl_reg['team_favorite_id'] == nfl_reg['team_home']

#playoffs
nfl_pf['home_team_fav'] = nfl_pf['team_favorite_id'] == nfl_pf['team_home']


# In[14]:


nfl_reg['home_team_fav'].describe()
#from the 7298 counts, the home_team is favorite 4478 times, which is 56.5% of the time. 


# In[15]:


nfl_reg['spread_abs'] = abs(nfl_reg['spread_favorite'])

nfl_pf['spread_abs'] = abs(nfl_pf['spread_favorite'])


# In[16]:


#nfl_regular season - adding index for home-team-favored and covered designation

nfl_reg.loc[nfl_reg['actual_result'] > nfl_reg['spread_abs'], 'covered_spread'] = 1
nfl_reg.loc[nfl_reg['actual_result'] < nfl_reg['spread_abs'], 'covered_spread'] = 0

nfl_reg = nfl_reg.replace({
    'home_team_fav': {
        True: 1,
        False: 0, },
})

nfl_reg.loc[(nfl_reg['home_team_fav'] == 1) & (nfl_reg['covered_spread'] == 1), 'home_fav_and_covered'] = 1
nfl_reg.loc[(nfl_reg['home_team_fav'] == 0) & (nfl_reg['covered_spread'] == 0), 'home_fav_and_covered'] = 0
nfl_reg.loc[(nfl_reg['home_team_fav'] == 0) & (nfl_reg['covered_spread'] == 1), 'home_fav_and_covered'] = 0
nfl_reg.loc[(nfl_reg['home_team_fav'] == 1) & (nfl_reg['covered_spread'] == 0), 'home_fav_and_covered'] = -1

#playoffs - adding index for home-team-favored and covered designation

nfl_pf.loc[nfl_pf['actual_result'] > nfl_pf['spread_abs'], 'covered_spread'] = 1
nfl_pf.loc[nfl_pf['actual_result'] < nfl_pf['spread_abs'], 'covered_spread'] = 0

nfl_pf = nfl_pf.replace({
    'home_team_fav': {
        True: 1,
        False: 0, },
})

nfl_pf.loc[(nfl_pf['home_team_fav'] == 1) & (nfl_pf['covered_spread'] == 1), 'home_fav_and_covered'] = 1
nfl_pf.loc[(nfl_pf['home_team_fav'] == 0) & (nfl_pf['covered_spread'] == 0), 'home_fav_and_covered'] = 0
nfl_pf.loc[(nfl_pf['home_team_fav'] == 0) & (nfl_pf['covered_spread'] == 1), 'home_fav_and_covered'] = 0
nfl_pf.loc[(nfl_pf['home_team_fav'] == 1) & (nfl_pf['covered_spread'] == 0), 'home_fav_and_covered'] = -1


# In[17]:


nfl_reg


# In[18]:


hac = nfl_reg.groupby('home_fav_and_covered')
home = nfl_reg.groupby('home_team_fav')


# In[19]:


home[['actual_result', 'spread_abs']].describe()


# In[20]:


home_pf = nfl_pf.groupby('home_team_fav')
pf_home = home_pf[['actual_result', 'spread_abs']].describe()

pfhac = nfl_pf.groupby('home_fav_and_covered')
pf_homeandc = pfhac[['actual_result', 'spread_abs']].describe()


# In[21]:



pf_home['count_perc'] = pf_home.actual_result['count'] / 343
pf_home


# In[22]:


#NFL Playoffs - Mean and Std. Dev for home-team favorite designation Actual Results
pf_home.actual_result[['mean','std']].plot(kind='bar')
plt.title('NFL Playoffs - Actual Results - Mean and Standard Deviation for Home_team_favorite designation')
plt.ylabel("Value")
plt.xlabel('Home Team Favorite Status')
pf_home.spread_abs[['mean','std']].plot(kind='bar')
plt.title('NFL playoffs - Absolute Spread - Mean and Standard Deviation for Home_team_favorite designation')
plt.ylabel("Value")
plt.xlabel('Home Team Favorite Status')
plt.show()


# In[23]:


h = home[['actual_result', 'spread_abs']].describe()
print(h)

h['count_perc'] = h.actual_result['count'] / 7688
h


# In[24]:


#NFL regular season - Mean and Std. Dev for Home-Team favorite designation
h.actual_result[['mean','std']].plot(kind='bar')
plt.title('NFL Regular Season - Actual Results - Mean and Standard Deviation for Home_team_favorite designation')
plt.ylabel("Value")
plt.xlabel('Home Team Favorite Status')
h.spread_abs[['mean','std']].plot(kind='bar')
plt.title('NFL Regular Season - Absolute Spread - Mean and Standard Deviation for Home_team_favorite designation')
plt.ylabel("Value")
plt.xlabel('Home Team Favorite Status')
plt.show()


# In[25]:


resultsandspread = hac[['actual_result', 'spread_abs', 'over_under_line']].describe()
resultsandspread


# In[26]:


pf_homeandc


# In[27]:


#Summary statistics for Actual Result 
#regular season

resultsandspread.actual_result[['mean','std','min','max']].plot(kind='box')
plt.ylabel('score(home-away)')
plt.title("NFL Regular Season - Summary Statistics for Actual Result")
plt.subplot()
#playoffs 
pf_homeandc.actual_result[['mean','std','min','max']].plot(kind='box')
plt.ylabel('score(home-away)')
plt.title("NFL Playoffs - Summary Statistics for Actual Result")
plt.subplot()
plt.show()


# In[28]:


#Summary statistics for Absolute Spread
#playoffs 
pf_homeandc.spread_abs[['mean','std']].plot(kind='box')
plt.ylabel('spread')
plt.title("NFL Playoffs - Summary Statistics for Absolute Spread")
plt.subplot()
#regularseason
resultsandspread.spread_abs[['mean','std']].plot(kind='box')
plt.ylabel('spread')
plt.title("NFL Regular Season - Summary Statistics for Absolute Spread")
plt.subplot()
plt.show()


# In[29]:


resultsandspread


# In[30]:


#regularseason
#actual result variable
resultsandspread.actual_result[['mean','std','min','max']].plot(kind='bar')
plt.title('NFL Regular Season - Actual Results - Metrics for Index designation')
plt.ylabel("Value")
plt.xlabel('Home Team Favorite and Covered Index')

#absolute spread
resultsandspread.spread_abs[['mean','std','min','max']].plot(kind='bar')
plt.title('NFL Regular Season - Absolute Spread - Metrics for Index designation')
plt.ylabel("Value")
plt.xlabel('Home Team Favorite and Covered Index')
plt.show()


# In[31]:


#playoffs
#actual result variable
pf_homeandc.actual_result[['mean','std','min','max']].plot(kind='bar')
plt.title('NFL Playoffs - Actual Results - Metrics for Index designation')
plt.ylabel("Value")
plt.xlabel('Home Team Favorite and Covered Index')

#absolute spread
pf_homeandc.spread_abs[['mean','std','min','max']].plot(kind='bar')
plt.title('NFL Playoffs - Absolute Spread - Metrics for Index designation')
plt.ylabel("Value")
plt.xlabel('Home Team Favorite and Covered Index')
plt.show()


# In[32]:


#regular season
by_team = nfl_reg.groupby('team_home')

#playoffs
by_team_pf = nfl_pf.groupby("team_home")


# In[33]:


#regular season - by teams summary, shows summary statistics(counts, mean, standard dev.) of home_team_favorite status and absolute values of spread
reg_teams_summary = by_team[['home_team_fav', 'spread_abs', 'home_fav_and_covered']].agg(['count', 'mean','std'])
reg_teams_summary.columns.get_level_values(1)
reg_teams_summary.sort_values(by=[('home_team_fav', 'mean')], ascending=False)


# In[34]:


#NFL playoff by team, shows summary statistics(counts, mean, standard dev.) of home_team_favorite status and absolute values of spread
pf_teams_summary = by_team_pf[['home_team_fav', 'spread_abs', 'home_fav_and_covered']].agg(['count', 'mean','std'])
pf_teams_summary.columns.get_level_values(1)
pf_teams_summary.sort_values(by=[('home_fav_and_covered', 'count')], ascending=False)


# In[35]:


reg_season_teams_summary = by_team[['home_team_fav', 'spread_abs']].describe()


# In[36]:


by_teams = by_team['home_team_fav'].describe()


# In[37]:


#covered vs. not covered home-teams

homecovered = nfl_reg[nfl_reg.home_fav_and_covered == 1] 
didnotcover = nfl_reg[nfl_reg.home_fav_and_covered != 1] 

homecovered_pf = nfl_pf[nfl_pf.home_fav_and_covered == 1] 
didnotcover_pf = nfl_pf[nfl_pf.home_fav_and_covered != 1] 


# In[38]:


nfl_reg.columns


# In[39]:


#scatterplot for home and covered vs. did not cover - NFL Regular Season
plt.scatter(homecovered['actual_result'], homecovered['spread_abs'], c='red', label='home and covered spread', alpha=0.5)
plt.scatter(didnotcover['actual_result'], didnotcover['spread_abs'], c='blue', label= 'did not cover spread', alpha=0.5)
plt.xlabel('actual_result')
plt.ylabel('spread')
plt.legend()
plt.show()


# In[40]:


#scatterplot for home and covered vs. did not cover- NFL-playoff
plt.scatter(homecovered_pf['actual_result'], homecovered_pf['spread_abs'], c='red', label='home and covered spread', alpha=0.5)
plt.scatter(didnotcover_pf['actual_result'], didnotcover_pf['spread_abs'], c='blue', label= 'did not cover spread', alpha=0.5)
plt.xlabel('actual_result')
plt.ylabel('spread')
plt.legend()
plt.show()


# In[41]:


#correlations - NFL Regular Seasonn

corr = nfl_reg[['actual_result','spread_abs','covered_spread', 'home_fav_and_covered']].corr(method='pearson')
sns.heatmap(corr)
plt.show()
corr


# In[42]:


#Correlations - NFL Playoffs
corr1 = nfl_pf[['actual_result','spread_abs','covered_spread', 'home_fav_and_covered']].corr(method='pearson')
sns.heatmap(corr1)
plt.show()
corr1


# In[43]:


#Decision Trees - NFL Regular Season
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


data = nfl_reg[['home_fav_and_covered','covered_spread', 'actual_result', 'spread_abs','over_under_line']]

#cleaning- filling NAs
x1_data = data[['actual_result', 'spread_abs', 'over_under_line', 'covered_spread']]
y1_data = data['home_fav_and_covered']

y2_data = y1_data.fillna(0)
x_data = x1_data.fillna(0)

y_data = y2_data.array.reshape(-1,1)


data_features = ['actual_result', 'absolute_spread', 'over_under', 'covered_spread']
data_target = ['home_fav_and_covered']
y_train, y_test, x_train, x_test = sklearn.model_selection.train_test_split(y_data, x_data, test_size=0.30)
tre = tree.DecisionTreeClassifier()
tre.fit(x_train, y_train)
y_pred = tre.predict(x_test)
print(metrics.classification_report(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[44]:


#NFL playoffs - Decision Tree 
data_2 = nfl_pf[['home_fav_and_covered','covered_spread', 'actual_result', 'spread_abs','over_under_line']]

x2_data = data_2[['actual_result', 'spread_abs', 'over_under_line', 'covered_spread']]
y2_data = data_2['home_fav_and_covered']

y2_data_1 = y1_data.fillna(0)
x_data_1 = x1_data.fillna(0)

y_data_1 = y2_data_1.array.reshape(-1,1)

data_features_2 = ['actual_result', 'absolute_spread', 'over_under', 'covered_spread']
data_target_2 = ['home_fav_and_covered']
y_trainpf, y_testpf, x_trainpf, x_testpf = sklearn.model_selection.train_test_split(y_data_1, x_data_1, test_size=0.30)
tre1 = tree.DecisionTreeClassifier()
tre1.fit(x_trainpf, y_trainpf)
y_predpf = tre1.predict(x_testpf)
print(metrics.classification_report(y_testpf, y_predpf))
print(metrics.confusion_matrix(y_testpf, y_predpf))
print("Accuracy:",metrics.accuracy_score(y_testpf, y_predpf))


# In[45]:


#Decision Tree for NFL Regular Season
plt.figure(figsize=(12,12)) 
tree.plot_tree(tre, fontsize=8, max_depth = 5)
plt.show()


# In[46]:


#Linear Regression - NFL REgular Season 
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
regr = linear_model.LinearRegression()
regr.fit(y_train, x_train)
print('Coefficients: \n', regr.coef_)
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))


# In[47]:


#Linear Regression - NFL Playoffs
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
regr = linear_model.LinearRegression()
regr.fit(y_trainpf, x_trainpf)
print('Coefficients: \n', regr.coef_)
print('Mean squared error: %.2f'
      % mean_squared_error(y_testpf, y_predpf))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_testpf, y_predpf))


# In[48]:


#NFL Regular Season - Logistic Regression 
from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(x_train, y_train)
log.predict_proba(x_test)
prediction = log.predict(x_test)

print(log.score(x_test, y_test))
print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction))

cm = metrics.confusion_matrix(y_test, prediction)


# In[49]:


print(cm)


# In[50]:


#Logistic Regression using NFL Playoffs 
from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(x_trainpf, y_trainpf)
log.predict_proba(x_testpf)
predictionpf = log.predict(x_testpf)

print(log.score(x_testpf, y_testpf))
print(metrics.classification_report(y_testpf, predictionpf))
print(metrics.confusion_matrix(y_testpf, predictionpf))

cm = metrics.confusion_matrix(y_testpf, predictionpf)


# In[51]:


#Confusion Matrix for NFL playoffs 
score = log.score(x_test, y_test)
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 8);
plt.show()


# In[52]:


#Clustering - NFL REgular Season 

data2 = nfl_reg[['home_fav_and_covered','covered_spread', 'actual_result', 'spread_abs','score_home', 'over_under_line']]

outcome = data2['home_fav_and_covered']
variables_pre = data2[['actual_result', 'spread_abs', 'covered_spread', 'over_under_line']]

outcomes = outcome.dropna()
variables = variables_pre.dropna()
outcomes_rs = outcomes.array.reshape(-1,1)

variables = variables.reset_index()

scaled_var = scale(variables)
n_samples, n_features = scaled_var.shape
n_digits = 3

#Clustering - NFL playoffs
datapf = nfl_pf[['home_fav_and_covered','covered_spread', 'actual_result', 'spread_abs','score_home', 'over_under_line']]
outcomepf = datapf['home_fav_and_covered']
variables_pre_pf = datapf[['actual_result', 'spread_abs', 'covered_spread', 'over_under_line']]

outcomes_pf = outcomepf.dropna()
variables_pf = variables_pre_pf.dropna()
outcomes_rs_pf = outcomes_pf.array.reshape(-1,1)


scaled_varpf = scale(variables_pf)
n_samples, n_features = scaled_varpf.shape
n_digits = 3


# In[53]:


model = cluster.AgglomerativeClustering(n_clusters=n_digits, affinity = 'euclidean', linkage = 'complete')
model.fit(scaled_var)


# In[54]:


print('completeness', metrics.completeness_score(outcomes, model.labels_))
print('homogeneity', metrics.homogeneity_score(outcomes, model.labels_))
print('silhouette', metrics.silhouette_score(outcomes_rs, model.labels_))


# In[55]:


#NFL playoffs- Agglomerative Clustering 

model_1 = cluster.AgglomerativeClustering(n_clusters=n_digits, affinity = 'euclidean', linkage = 'complete')
model_1.fit(scaled_varpf)

print('completeness', metrics.completeness_score(outcomes_pf, model_1.labels_))
print('homogeneity', metrics.homogeneity_score(outcomes_pf, model_1.labels_))
print('silhouette', metrics.silhouette_score(outcomes_rs_pf, model_1.labels_))


# In[56]:


#NFL Regular Season - Dendrogram
from scipy.cluster.hierarchy import dendrogram, linkage
model = linkage(scaled_var, 'ward')
plt.figure()
plt.xlabel('index')
plt.ylabel('Euclidean Distance')
dendrogram(model, leaf_rotation = 90., leaf_font_size = 8., truncate_mode='lastp', p=20)
plt.show()


# In[57]:


#NFL playoffs - dendrogram 
model_1 = linkage(scaled_varpf, 'ward')
plt.figure()
plt.xlabel('index')
plt.ylabel('Euclidean Distance')
dendrogram(model_1, leaf_rotation = 90., leaf_font_size = 8., truncate_mode='lastp', p=20)
plt.show()


# In[58]:


#attempt for binary classifier - Clustering

data_binary = nfl_reg[['home_fav_and_covered','covered_spread', 'actual_result', 'spread_abs','over_under_line', 'home_team_fav']]

outcome_b = data_binary['covered_spread']
variables_pre_b = data_binary[['actual_result', 'spread_abs', 'over_under_line', 'home_fav_and_covered']]

outcomes_b = outcome_b.dropna()
variables_b = variables_pre_b.dropna()
outcomes_rs_b = outcomes_b.array.reshape(-1,1)

variables_b = variables_b.reset_index()

scaled_var_b = scale(variables_b)
n_samples, n_features = scaled_var_b.shape
n_digits_b = 2


# In[59]:


#agglomerative clustering using the binary classification 
model_b = cluster.AgglomerativeClustering(n_clusters=n_digits_b, affinity = 'euclidean', linkage = 'complete')
model_b.fit(scaled_var_b)


# In[60]:


#metrics associated with binary classification system
print('completeness', metrics.completeness_score(outcomes_b, model_b.labels_))
print('homogeneity', metrics.homogeneity_score(outcomes_b, model_b.labels_))
print('silhouette', metrics.silhouette_score(outcomes_rs_b, model_b.labels_))


# In[61]:


#Attempt at decision trees using binary classification system 
xb1_data = data_binary[['actual_result', 'spread_abs', 'over_under_line', 'home_fav_and_covered']]
yb2_data = data_binary['covered_spread']

yb1_data = yb2_data.fillna(0)
xb_data = xb1_data.fillna(0)

yb_data = yb1_data.array.reshape(-1,1)


data_features_b = ['actual_result', 'absolute_spread', 'over_under', 'covered_spread']
data_target_b = ['home_fav_and_covered']
y_trainb, y_testb, x_trainb, x_testb = sklearn.model_selection.train_test_split(yb_data, xb_data, test_size=0.30)
tre_b = tree.DecisionTreeClassifier()
tre_b.fit(x_trainb, y_trainb)
y_pred_b = tre_b.predict(x_testb)
print(metrics.classification_report(y_testb, y_pred_b))
print(metrics.confusion_matrix(y_testb, y_pred_b))
print("Accuracy:",metrics.accuracy_score(y_testb, y_pred_b))


# In[62]:


#END - Following cells were just experiment and testing. Attempted to begin analysis by factoring out home-team status and seeing how the resulting metrics would look---
#using only home-team status - and choosing target variable of covering spread or not covering spread with home-team favorite designation


# In[63]:


home_team_favs.loc[home_team_favs['actual_result'] > home_team_favs['spread_abs'], 'covered_spread'] = 'True'
home_team_favs.loc[home_team_favs['actual_result'] < home_team_favs['spread_abs'], 'covered_spread'] = 'False'


# In[ ]:


home_tf = home_team_favs.replace({
    'covered_spread': {
        'True': 1,
        'False': 0, },
})


# In[ ]:


home_tf


# In[ ]:


covered = home_team_favs[home_team_favs.covered_spread == 'True']


# In[ ]:


did_not_cover = home_team_favs[home_team_favs.covered_spread == 'False']


# In[ ]:


home_team_favs[['actual_result', 'spread_abs']].plot(kind='box')
plt.show()


# In[ ]:


plt.scatter(covered['actual_result'], covered['spread_abs'], c='red', label='covered spread', alpha=0.5)
plt.scatter(did_not_cover['actual_result'], did_not_cover['spread_abs'], c='blue', label= 'did not cover spread', alpha=0.5)
plt.xlabel('actual_result')
plt.ylabel('spread')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:


corr = home_tf[['actual_result','spread_abs','covered_spread']].corr()
sns.heatmap(corr)
plt.show()


# In[ ]:


home_tf[['actual_result','spread_abs','covered_spread']].corr(method='pearson')


# In[ ]:


covered[['actual_result','spread_abs']].corr(method='pearson')


# In[ ]:


did_not_cover[['actual_result','spread_abs']].corr(method='pearson')


# In[ ]:


home_tf['covered_spread'].fillna(0)


# In[ ]:


home_tf.columns


# In[ ]:


#clustering
outcome = home_tf['covered_spread']
variables_pre = home_tf[['actual_result', 'spread_abs','over_under_line', 'home_team_fav']]

outcome = outcome.fillna("0")
variables = variables_pre.dropna()
outcome_rs = outcome.array.reshape(-1,1)

scaled_var = scale(variables)
n_samples, n_features = scaled_var.shape
n_digits = 2


# In[ ]:


scaled_var


# In[ ]:


variables.isna().any()


# In[ ]:


outcome.isnull().any()


# In[ ]:


model = cluster.AgglomerativeClustering(n_clusters=n_digits, affinity = 'euclidean', linkage = 'complete')
model.fit(scaled_var)


# In[ ]:


print(metrics.silhouette_score(outcome_rs, model.labels_))


# In[ ]:





# In[ ]:


print('completeness', metrics.completeness_score(outcome_rs, model.labels_))
print('homogeneity', metrics.homogeneity_score(outcome_rs, model.labels_))


# In[ ]:


from scipy.cluster.hierarchy import dendrogram, linkage
model = linkage(scaled_var, 'ward')
plt.figure()
plt.xlabel('index')
plt.ylabel('Euclidean Distance')
dendrogram(model, leaf_rotation = 90., leaf_font_size = 8.,)
plt.show()


# In[ ]:





# In[ ]:


#only taking home_favorites into account
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


dataa = home_tf[['covered_spread', 'actual_result', 'spread_abs','over_under_line']]

#cleaning- filling NAs
x1_dataa = dataa[['actual_result', 'spread_abs', 'over_under_line']]
y1_dataa = dataa['covered_spread']

y2_data = y1_dataa.fillna(0)
x_datah = x1_dataa.dropna()

y_datah = y2_dataa.array.reshape(-1,1)


data_features = ['actual_result', 'absolute_spread', 'over_under']
data_target = ['covered_spread']
y_train, y_test, x_train, x_test = sklearn.model_selection.train_test_split(y_datah, x_datah, test_size=0.30)
tre = tree.DecisionTreeClassifier()
tre.fit(x_train, y_train)
y_pred = tre.predict(x_test)
print(metrics.classification_report(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[ ]:


plt.figure(figsize=(12,12)) 
tree.plot_tree(tre, fontsize=8)
plt.show()


# In[ ]:


#linear with only home_features

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)
print('Coefficients: \n', regr.coef_)
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))


# In[ ]:


x_data


# In[ ]:


y_data


# In[ ]:




