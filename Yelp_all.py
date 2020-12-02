import pandas as pd
import random
import ast # for literal evaluation of string
import datetime
import time
import numpy as np

# For Decision tree:
from sklearn.model_selection import KFold
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit



start_time = time.time_ns()
# for plots
# import matplotlib.pyplot as plt
# import seaborn as sns
# plt.style.use('fivethirtyeight')
# sns.set()
def separator():
    print('\n')
    print('-'*70)
    print('-'*70)
    print('')
    return

separator()
# # Analysing the data of business:
print(f'Loading preprocessed cleaned "yelp_academic_dataset_business.json" "business_clean.csv":')
df_business = pd.read_csv('data_clean/business_clean.csv')
print(f'Columns are : {df_business.columns}')
df_business.head(3)


# ## Categoreis of business:

# Extract all categories from str and save a Series od unique categories in a tuple:
l_cat = [] # empty list to collect all categories
for line in list(df_business.categories):
    list_add = [x.strip() for x in line.split(',')]
    l_cat.extend(list_add)
unique_cat = set(l_cat) # variable unique_cat is a set of all unique categories of business
print(f'Number of unique categories from column "categories" is: {len(unique_cat)}')

try:
    df_cat = pd.read_csv('data_clean/categories.csv', index_col ='category')
    print(df_cat.columns)
except:
    # datafarame with unique categories and their frequency:
    df_cat = pd.DataFrame()
    for cat in list(unique_cat):
        df_line = pd.DataFrame([l_cat.count(cat)], index=[cat], columns=['count'])
        df_cat = df_cat.append(df_line).sort_values(by=['count'], ascending=False)
    df_cat.head(10)

    # Add averaged star (raiting) for every category
    df_cat["stars"] = ""
    for cat in list(df_cat.index):
        raiting = float(0)
        l_index = 0
        count=0
        for line in list(df_business.categories):
            l_index += 1
            if cat in line:
                try:
                    raiting += df_business.stars[l_index]
                except:
                    pass
                count = count + 1
        raiting = raiting/count
        df_cat.stars.loc[str(cat)] = raiting
        print(  f'Number od {cat} shop: {count},\nAverage raiting: {raiting:.2f}\n')
        print('-*-'*20)
    df_cat['category'] = df_cat.index
    df_cat.to_csv('data_clean/categories.csv',index=False)

print(f'The most common categories and there counts are:')
for cat, amount in zip(df_cat.index[:15],df_cat['count'][:15]):
    print(f'{cat:27} {amount:5} items')


df_b_star_open = df_business.pivot_table(
    values='is_open', index=['stars'], aggfunc=np.mean)
df_b_star_open['closed_shops'] = (1 - df_b_star_open.is_open)
df_b_star_open = df_b_star_open.sort_values(
    by=['closed_shops'], ascending=False)
df_b_star_open *= 100
df_b_star_open  = round((df_b_star_open),0)

# print('='*80,'\n')
# print('='*80)

separator()
print(f'Percent of closed and open shops, grouped by average star raiting on Yelp \n')
print(df_b_star_open)

separator()
print('Analysis only fashion shops:')
df_fashion = pd.DataFrame()
count = 0
l_index = 0
raiting = float(0)
text = 'Fashion'
for line in list(df_business.categories):
    l_index += 1
    if text in line:
        raiting += df_business.stars[l_index]
        count = count + 1
raiting = raiting/count

# Make new column of df_business with boolean: fashion category
l_fashion = []  # empty list to collect  True and False
for line in list(df_business.categories):
    if 'Fashion' in [x.strip() for x in line.split(',')]:
        l_fashion.append(True)
    else:
        l_fashion.append(False)

df_business['fashion'] = l_fashion
df_fashion = df_business[df_business['fashion'] == True]
df_fashion = df_fashion.reset_index().drop(columns=['index'])
print(f'Number od fashion shops: {count}, \nAverage raiting: {raiting:.2f}')

df_f_star_open = df_fashion.pivot_table(
    values='is_open', index=['stars'], aggfunc=np.mean)
df_f_star_open['closed_shops'] = (1 - df_f_star_open.is_open)
df_f_star_open = df_f_star_open.sort_values(
    by=['closed_shops'], ascending=False)
df_f_star_open *= 100
df_f_star_open  = round((df_f_star_open),0)
print(df_f_star_open)

#================================================================
# Extraction of additional attributes from a column: 'attributes'
#================================================================
separator()
t = pd.to_timedelta((time.time_ns() - start_time)*10**(-9), unit='s')
print(f" Running time is:  {t}")
separator()


separator()
print('The most common keys from dictionary in attributes:\n')
l_keys = []
for dict_att in df_fashion.attributes:
    try:
        l_keys.append(list(ast.literal_eval(dict_att).keys()))
    except:
        l_keys.append('No_info')
# add new column to the datafarme
l_keys = [item for sublist in l_keys for item in sublist]
# set(l_keys)

# datafarame with unique categories and their frequency:
df_keys = pd.DataFrame()
for key in list(set(l_keys)):
    df_line = pd.DataFrame([l_keys.count(key)], index=[key], columns=['count'])
    df_keys = df_keys.append(df_line).sort_values(
        by=['count'], ascending=False)
print(df_keys.head(10))

#================================================================
separator()
print('Extract information about restaurant price around the shop:')
price = []
for dict_att in df_fashion.attributes:
    try:
        price.append(int(ast.literal_eval(dict_att)['RestaurantsPriceRange2']))
    except:
        price.append(np.nan)

df_fashion['price'] = price

plot_data = df_fashion.pivot_table(values='is_open',index= ['price'], aggfunc=np.mean)
plot_data =  round((1 - plot_data)*100,0)

print(f'price range - % of closed shops')
for price,closed in zip(plot_data.index,plot_data.is_open):
    print(f'{price:^11} - {closed:^15}')


#================================================================
separator()
print('Extract information about avaliable garage:\n')
attribute = 'garage'
garage = []
for dict_att in df_fashion.attributes:
    try:
        garage.append(int(ast.literal_eval(ast.literal_eval(dict_att)['BusinessParking'])['garage']))
    except:
        garage.append(np.nan)
# add new column to the datafarme
df_fashion['garage'] = garage

df = df_fashion
x, y, hue = attribute, 'proportion', 'is_open'
hue_order = [1, 0]
prop_df = (df[hue]
            .groupby(df[x])
            .value_counts(normalize=True)
            .rename(y)
            .reset_index())
df0 = prop_df[prop_df.is_open==0]
print(f'garage - % of closed shops')
for parking,closed in zip(df0.garage, round(df0.proportion*100,0)):
    print(f'{parking:^6} - {closed:^15}')

#================================================================
separator()
print('Extract information about avaliable lot:\n')
attribute = 'lot'
lot = []
for dict_att in df_fashion.attributes:
    try:
        lot.append(int(ast.literal_eval(ast.literal_eval(dict_att)['BusinessParking'])[attribute]))
    except:
        lot.append(np.nan)
# add new column to the datafarme
df_fashion['lot'] = lot

df = df_fashion
x, y, hue = attribute, 'proportion', 'is_open'
hue_order = [1, 0]
prop_df = (df[hue]
            .groupby(df[x])
            .value_counts(normalize=True)
            .rename(y)
            .reset_index())
df0 = prop_df[prop_df.is_open==0]
print(f'lot - % of closed shops')
for parking,closed in zip(df0.lot, round(df0.proportion*100,0)):
    print(f'{parking:^3} - {closed:^15}')

#================================================================
separator()
print('Extract information about avaliable street parking:\n')
attribute = 'street'
street = []
for dict_att in df_fashion.attributes:
    try:
        street.append(int(ast.literal_eval(ast.literal_eval(dict_att)['BusinessParking'])[attribute]))
    except:
        street.append(np.nan)
# add new column to the datafarme
df_fashion['street'] = street

df = df_fashion
x, y, hue = attribute, 'proportion', 'is_open'
hue_order = [1, 0]
prop_df = (df[hue]
            .groupby(df[x])
            .value_counts(normalize=True)
            .rename(y)
            .reset_index())
df0 = prop_df[prop_df.is_open==0]
print(f'street - % of closed shops')
for parking,closed in zip(df0.street, round(df0.proportion*100,0)):
    print(f'{parking:^6} - {closed:^15}')

#================================================================
separator()
print('Extract information about avaliable bike parking:\n')
attribute = 'BikeParking'
l_bike = []
for dict_att in df_fashion.attributes:
    try:
        l_bike.append(int(ast.literal_eval(ast.literal_eval(dict_att)[attribute])))
    except:
        l_bike.append(np.nan)
# add new column to the datafarme
df_fashion[attribute] = l_bike

df = df_fashion
x, y, hue = attribute, 'proportion', 'is_open'
hue_order = [1, 0]
prop_df = (df[hue]
            .groupby(df[x])
            .value_counts(normalize=True)
            .rename(y)
            .reset_index())
df0 = prop_df[prop_df.is_open==0]
print(f'bike parking - % of closed shops')
for parking,closed in zip(df0.BikeParking, round(df0.proportion*100,0)):
    print(f'{parking:^12} - {closed:^15}')

#================================================================
separator()
print('Extract information about Credit Card acceptance by the shop:\n')
attribute = 'BusinessAcceptsCreditCards'
l_CreditCards = []
for dict_att in df_fashion.attributes:
    try:
        l_CreditCards.append(int(ast.literal_eval(ast.literal_eval(dict_att)[attribute])))
    except:
        l_CreditCards.append(np.nan)
# add new column to the datafarme
df_fashion['CreditCards'] = l_CreditCards
attribute = 'CreditCards'

df = df_fashion
x, y, hue = attribute, 'proportion', 'is_open'
hue_order = [1, 0]
prop_df = (df[hue]
            .groupby(df[x])
            .value_counts(normalize=True)
            .rename(y)
            .reset_index())
df0 = prop_df[prop_df.is_open==0]
print(f'Credit Cards accepted - % of closed shops')
for parking,closed in zip(df0.CreditCards, round(df0.proportion*100,0)):
    print(f'{parking:^20} - {closed:^15}')


#================================================================
#                  Feature engineering
#================================================================
separator()
t = pd.to_timedelta((time.time_ns() - start_time)*10**(-9), unit='s')
print(f" Running time is:  {t}")
separator()
#================================================================
separator()
print('Percentage of closed shop dependent on the group of review count:\n')
df_copy = df_fashion.copy()
df_copy["review_count_group"] = pd.qcut(df_copy['review_count'], 3, labels=False) #[3:5], (5:11], (11:2216]))
df_fashion = df_copy.copy()

attribute = 'review_count_group'

# plot data: total ammount and proportions of shops closed and open grouped by garage
df = df_fashion
x, y, hue = attribute, 'proportion', 'is_open'
hue_order = [1, 0]

# second subplot
prop_df = (df[hue]
            .groupby(df[x])
            .value_counts(normalize=True)
            .rename(y)
            .reset_index())
df0 = prop_df[prop_df.is_open==0]
print(f'Review counts group - % of closed shops')
for parking,closed in zip(df0.review_count_group, round(df0.proportion*100,0)):
    print(f'{parking:^19} - {closed:^15}')


#================================================================
l_h_tuesday = []
for item in df_fashion.hours:
    try:
        l_h_tuesday.append(abs(pd.to_datetime(ast.literal_eval(item)['Tuesday'].split('-')[1],format='%H:%M') - 
        pd.to_datetime(ast.literal_eval(item)['Tuesday'].split('-')[0],format='%H:%M')).total_seconds()/3600)
    except:
        l_h_tuesday.append(np.nan)

l_h_sunday = []
for item in df_fashion.hours:
    try:
        l_h_sunday.append(abs(pd.to_datetime(ast.literal_eval(item)['Sunday'].split('-')[1],format='%H:%M') - 
        pd.to_datetime(ast.literal_eval(item)['Sunday'].split('-')[0],format='%H:%M')).total_seconds()/3600)
    except:
        l_h_sunday.append(np.nan)

df_copy = df_fashion.copy()

df_copy['h_tuesday'] = l_h_tuesday
df_copy['h_sunday'] = l_h_sunday
df_copy["h_tuesday_group"] = pd.qcut(df_copy['h_tuesday'], 3, labels=False) # [(-0.001, 8.0] < (8.0, 11.0] < (11.0, 23.75]]
df_copy["h_sunday_group"] = pd.qcut(df_copy['h_sunday'], 3, labels=False) # [(-0.001, 6.0] < (6.0, 8.0] < (8.0, 23.75]]
df_fashion = df_copy.copy()

#================================================================
separator()
print('Percentage of closed shop dependent on the Tuesday working hours duration:\n')
attribute = 'h_tuesday_group'

# plot data: total ammount and proportions of shops closed and open grouped by garage
df = df_fashion
x, y, hue = attribute, 'proportion', 'is_open'
hue_order = [1, 0]

# second subplot
prop_df = (df[hue]
            .groupby(df[x])
            .value_counts(normalize=True)
            .rename(y)
            .reset_index())
df0 = prop_df[prop_df.is_open==0]
print(f'Tuesday working hours duration - % of closed shops')
for tuesday,closed in zip(df0.h_tuesday_group, round(df0.proportion*100,0)):
    print(f'{tuesday:^30} - {closed:^15}')


#================================================================
separator()
print('Percentage of closed shop dependent on the Sunday working hours duration:\n')
attribute = 'h_sunday_group'

# plot data: total ammount and proportions of shops closed and open grouped by garage
df = df_fashion
x, y, hue = attribute, 'proportion', 'is_open'
hue_order = [1, 0]

# second subplot
prop_df = (df[hue]
            .groupby(df[x])
            .value_counts(normalize=True)
            .rename(y)
            .reset_index())
df0 = prop_df[prop_df.is_open==0]
print(f'Sunday working hours duration - % of closed shops')
for sunday,closed in zip(df0.h_sunday_group, round(df0.proportion*100,0)):
    print(f'{sunday:^28} - {closed:^15}')


separator()
print('Save data as "fashion.csv" :\n')
df_fashion.to_csv('data_clean/fashion.csv',index=False)

#================================================================
#                    Cleaning data
#================================================================
df_fashion=pd.read_csv("data_clean/fashion.csv")
try:
    df_fashion= df_fashion[
        ['attributes', 'business_id', 'categories', 'city', 'hours',
        'name', 'review_count', 'stars', 'price', 'garage', 'street', 'lot',
        'BikeParking', 'CreditCards', 'start_year', 'latest_year',
        'review_count_group', 'h_tuesday', 'h_sunday', 'h_tuesday_group',
        'h_sunday_group', 'is_open']
                    ]
except:
    df_fashion= df_fashion[
        ['attributes', 'business_id', 'categories', 'city', 'hours',
        'name', 'review_count', 'stars', 'price', 'garage', 'street', 'lot',
        'BikeParking', 'CreditCards',
        'review_count_group', 'h_tuesday', 'h_sunday', 'h_tuesday_group',
        'h_sunday_group', 'is_open']
                    ]
df_fashion.to_csv('data_clean/fashion.csv',index=False)

separator()
print('Missing values fo data:')
print(df_fashion.isna().sum())

df=pd.read_csv("data_clean/fashion.csv")
try:
    df.start_year = df.start_year.fillna(df.start_year.mean())
    df.latest_year  = df.latest_year.fillna(df.latest_year.mean())
except:
    pass
df.price = df.price.fillna(df.price.mean())
df.lot = df.lot.fillna(df.price.mean())
df.garage = df.garage.fillna(df.garage.mean())
df.street = df.street.fillna(df.street.mean())
df.BikeParking  = df.BikeParking.fillna(df.BikeParking.mean())
df.CreditCards = df.CreditCards.fillna(df.CreditCards.mean())
df.h_tuesday_group = df.h_tuesday_group.fillna(df.h_tuesday_group.mean())
df.h_sunday_group = df.h_sunday_group.fillna(df.h_sunday_group.mean())
df.h_tuesday = df.h_tuesday.fillna(df.h_tuesday.mean())
df.h_sunday = df.h_sunday.fillna(df.h_sunday.mean())

separator()
print('Missing values after cleaning:')
print(df.isna().sum())

df.to_csv('data_clean/fashion_for_DecisionTree.csv',index=False)

#================================================================
#                 Decision tree
#================================================================
df = pd.read_csv('data_clean/fashion_for_DecisionTree.csv')
df_copy = df.copy()
target = df['is_open']
df = df.drop(['attributes','review_count', 'business_id', 'categories', 'city', 'hours',
                'name', 'h_tuesday', 'h_sunday'], axis = 1)
features = df.drop(['is_open'], axis = 1)

#================================================================
#                       All Features
#================================================================
separator()
print(f'Start Decision tree for all data (N={len(features.columns)}):')

print('Our taget is variable: is_open')
print('Correlation coefficients our feature variables with our target:\n')
print(df.corr().round(2).iloc[0:(len(df.columns)-1),-1].abs().sort_values(ascending=False))

separator()
X = features.copy()
y = target.copy()
i_split = 0.30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=i_split, random_state=None)
# define the model
# Limit max depth
model = RandomForestClassifier(n_estimators=1000, criterion='entropy')
# model  = tree.DecisionTreeClassifier(random_state=None, criterion='entropy', splitter='best')
#model = DecisionTreeRegressor(criterion='friedman_mse', max_depth=5)
# fit the model
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average='weighted')
recall = recall_score(y_test, predictions, average='weighted')
accuracy_train = accuracy_score(y_train, model.predict(X_train))
precision_train = precision_score(y_train, model.predict(X_train), average='weighted')
recall_train = recall_score(y_train, model.predict(X_train), average='weighted')
print(f'Accuracy(test):{accuracy:.2f}')
print(f'Precision(test):{precision:.2f}')
print(f'Recall(test):{recall:.2f}\n')
print(f'Accuracy(train):{accuracy_train:.2f}')
print(f'Precision(train):{precision_train:.2f}')
print(f'Recall(train):{recall_train:.2f}\n')

print('Confusion matrix (test):')
df_conf_matrix = confusion_matrix(y_test,predictions)
print(df_conf_matrix)
print('')

# Get numerical feature importances
importances = model.feature_importances_
labels = df.columns[:df.shape[1]-1].values.tolist()
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(labels, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
labels =  [x for l in feature_importances for x in l]
labels = labels[0:len(labels):2]
#================================================================
#                   Cross-validation
#================================================================
print('')
t = pd.to_timedelta((time.time_ns() - start_time)*10**(-9), unit='s')
print(f" Running time is:  {t}\n")

cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=None)
cross_val = cross_val_score(model, X, y, scoring='precision', cv=cv)
print(f'Cross-validation: {cross_val}')

#================================================================
#                   Features without Yelp data
#================================================================
df = pd.read_csv('data_clean/fashion_for_DecisionTree.csv')
df_copy = df.copy()
target = df['is_open']
df = df.drop(['attributes','review_count', 'business_id', 'categories', 'city', 'hours',
                'name', 'h_tuesday', 'h_sunday', 'stars', 'review_count_group'], axis = 1)
features = df.drop(['is_open'], axis = 1)

separator()
print(f'Start Decision tree for data without Yelp reviews (N={len(features.columns)}):')

print('Our taget is variable: is_open')
print('Correlation coefficients our feature variables with our target:\n')
print(df.corr().round(2).iloc[0:(len(df.columns)-1),-1].abs().sort_values(ascending=False))

separator()
X = features.copy()
y = target.copy()
i_split = 0.30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=i_split, random_state=None)
# define the model
# Limit max depth
model = RandomForestClassifier(n_estimators=1000, criterion='entropy')
# model  = tree.DecisionTreeClassifier(random_state=None, criterion='entropy', splitter='best')
#model = DecisionTreeRegressor(criterion='friedman_mse', max_depth=5)
# fit the model
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average='weighted')
recall = recall_score(y_test, predictions, average='weighted')
accuracy_train = accuracy_score(y_train, model.predict(X_train))
precision_train = precision_score(y_train, model.predict(X_train), average='weighted')
recall_train = recall_score(y_train, model.predict(X_train), average='weighted')
print(f'Accuracy(test):{accuracy:.2f}')
print(f'Precision(test):{precision:.2f}')
print(f'Recall(test):{recall:.2f}\n')
print(f'Accuracy(train):{accuracy_train:.2f}')
print(f'Precision(train):{precision_train:.2f}')
print(f'Recall(train):{recall_train:.2f}\n')

print('Confusion matrix (test):')
df_conf_matrix = confusion_matrix(y_test,predictions)
print(df_conf_matrix)
print('')

# Get numerical feature importances
importances = model.feature_importances_
labels = df.columns[:df.shape[1]-1].values.tolist()
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(labels, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
labels =  [x for l in feature_importances for x in l]
labels = labels[0:len(labels):2]


#================================================================
#                   Cross-validation
#================================================================
print('')
t = pd.to_timedelta((time.time_ns() - start_time)*10**(-9), unit='s')
print(f" Running time is:  {t}\n")

cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=None)
cross_val = (cross_val_score(model, X, y, scoring='precision', cv=cv))
print(f'Cross-validation: {cross_val}')

separator()
t = pd.to_timedelta((time.time_ns() - start_time)*10**(-9), unit='s')
print(f" Running time is:  {t}")
separator()


print('\n')
print('-'*70)
print('='*70)
print('End:')
print('\n')