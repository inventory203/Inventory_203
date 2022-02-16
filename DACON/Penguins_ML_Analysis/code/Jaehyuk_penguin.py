import inline as inline
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings(action='ignore')

import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

data = pd.read_excel('/Users/choejaehyeog/Downloads/penguins_lter.xlsx', engine='openpyxl')
# print(data.head())_

data = data.drop('Sample Number', axis = 1)

# function that returns the col that has NA value
def check_missing_col(dataframe):
    missing_col = []
    countNA = 0
    for i, col in enumerate(dataframe.columns):
        missing_values = sum(dataframe[col].isna())
        is_missing = True if missing_values >= 1 else False
        if is_missing:
            countNA += 1
            missing_col.append([col, dataframe[col].dtype])
            # print(f'The column that has NA value is {col} and has {missing_values} NA values')
    if countNA == 0:
        print('No NA values')
    return missing_col

missing_col = check_missing_col(data)
# function that handles the NA value
def manage_missing_value(dataframe, missing_col):
    temp = dataframe.copy()
    for col,dtype in missing_col:
        # if it's categorical data, remove that row
        if dtype == 'O':
            temp = temp.dropna(subset=[col])
        # if it's numerical data, fill that row with 0
        elif dtype == float:
            temp.loc[:, col] = temp[col].fillna(temp[col].mean())
    return temp

data = manage_missing_value(data, missing_col)
check_missing_col(data)
print("=======================Finished data pre-processing=======================")
# divide the data into test dataset and training dataset
train, test = train_test_split(data)

# This time need to handle categorical data to numerical data: By incoding
# Using the label-incoding, covert categorical data to 0, 1, 2
def make_label_map(dataframe):
    label_maps = {}
    for col in dataframe.columns:
        if dataframe[col].dtype == 'object':
            label_map = {'unknown':0}
            for i, key in enumerate(train[col].unique()):
                label_map[key] = i+1
            label_maps[col] = label_map
    return label_maps

# print(make_label_map(train))

# assign incoded value to each categorical variance
def label_encoder(dataframe, label_map):
    for col in dataframe.columns:
        if dataframe[col].dtype == 'object':
            dataframe[col] = dataframe[col].map(label_map[col])
            dataframe[col] = dataframe[col].fillna(label_map[col]['unknown']) #if theres any NA, fill it with unknown
    return dataframe

label_map = make_label_map(train) #generate label map
labeled_train = label_encoder(train,label_map)

# evaluate your model
def RMSE(true, pred):
    score = np.sqrt(np.mean(np.square(true-pred)))
    return score

# show the pair plotted graph
train_copy = train.copy()
num_feature = []
cat_feature = []
for col in train_copy.columns:
    if train_copy[col].dtype==float:
        num_feature.append(col)
    else:
        cat_feature.append(col)

num_df = train_copy[num_feature]
cat_df = train_copy[cat_feature]

sns.pairplot(train_copy[num_feature], corner = True)
plt.show()

# show the correlation between various variance
train_copy = train_copy.corr() #shows the correlation

fig, ax = plt.subplots(figsize=(12,12)) #select the size of the pic
# make the mask. Role of mask is to hide the cell if the value is True
mask = np.zeros_like(train_copy, dtype = np.bool) #fill the array with 0 that has same size with its parameter
mask[np.triu_indices_from(mask)] = True #triu = triangle upper. This generate a mask for the upper triangle of a given correlation matrix

sns.heatmap(train_copy,
            cmap='RdYlBu_r',
            annot=True,  # annotate the value
            mask=mask,  # select the part of mask if it will not be shown
            linewidths=.5,
            vmin=-1, vmax=1
            )
plt.show()

print("=======================Finished EDA=======================")

from sklearn.linear_model import LinearRegression
train_valid_copy = train.copy()
X = np.asarray(train_valid_copy.drop('Body Mass (g)', axis = 1))
# X = np.asarray(train['Flipper Length (mm)']).reshape(250,1)
Y = np.asarray(train['Body Mass (g)'])

X_train, X_test, y_train, y_test = train_test_split(X, Y, shuffle= True)
linReg = LinearRegression()
linReg.fit(X_train, y_train)

def RMSE(true, pred):
    score = np.sqrt(np.mean(np.square(true-pred)))
    return score

real_answer = Y.copy()
error = RMSE(y_test, linReg.predict(X_test))
print("For linear regression, error that can be found is: ",error)


from sklearn import linear_model
# To avoid overfitting, lets use regularizations: Ridge, Lasso
# Regularization helps the model to progress its training by declining the coefficient
# when alpha increases, coef decreases.
alphas = [0.05, 0.5, 5, 10]

# Ridge regression
ridge_error_list = []
for i in alphas:
    reg = linear_model.Ridge(alpha=i)
    reg.fit(X_train, y_train)
    error = RMSE(y_test, reg.predict(X_test))
    ridge_error_list.append(error)

ridge_best_error = min(ridge_error_list)
best_ridge_alpha = alphas[ridge_error_list.index(ridge_best_error)]
print("For Ridge regression, best error is:", ridge_best_error," when alpha is ",best_ridge_alpha)

# Lasso regression
lasso_error_list = []

for i in alphas:
    lasso = linear_model.Lasso (alpha = i)
    lasso.fit(X_train, y_train)
    error = RMSE(y_test, lasso.predict(X_test))
    lasso_error_list.append(error)

lasso_best_error = min(lasso_error_list)
best_lasso_alpha = alphas[lasso_error_list.index(lasso_best_error)]
print("For Lasso regression, best error is:",lasso_best_error," when alpha is ",best_lasso_alpha)

print("=======================Finised training=======================")

# kfold validation
from sklearn.model_selection import KFold
target = labeled_train["Body Mass (g)"]
feature = labeled_train.drop(['Body Mass (g)'], axis=1)

lr = LinearRegression()
lr_rid = linear_model.Ridge(alpha = best_ridge_alpha)
lr_las = linear_model.Lasso(alpha = best_lasso_alpha)

kfold = KFold(n_splits=5)

cv_rmse = []
cv_rmse_rid = []
cv_rmse_las = []
n_iter = 0

for train_index, test_index in kfold.split(feature):

    x_train, x_test = feature.iloc[train_index], feature.iloc[test_index]
    y_train, y_test = target.iloc[train_index], target.iloc[test_index]

    lr = lr.fit(x_train, y_train)  # train the model
    lr_rid = lr_rid.fit(x_train, y_train)
    lr_las = lr_las.fit(x_train, y_train)

    pred = lr.predict(x_test)  # predict the dataset
    pred_rid = lr_rid.predict(x_test)
    pred_las = lr_las.predict(x_test)
    n_iter += 1  # increase the iteration

    error = RMSE(y_test, pred)  # get the score of RMSE
    error_rid = RMSE(y_test, pred_rid)
    error_las = RMSE(y_test, pred_las)

    train_size = x_train.shape[0]  # size of train set
    test_size = x_test.shape[0]  # size of validation set
    cv_rmse.append(error)
    cv_rmse_rid.append(error_rid)
    cv_rmse_las.append(error_las)

print('\n==> Avg mean of this equation(RMSE) by simple linear regression is {}.'.format(np.mean(cv_rmse)))  # check the model's avg error
print('\n==> Avg mean of this equation(RMSE) by ridge regression is {}.'.format(np.mean(cv_rmse_rid)))
print('\n==> Avg mean of this equation(RMSE) by lasso regression is {}.'.format(np.mean(cv_rmse_las)))
print("=======================Finished Validation=======================")
