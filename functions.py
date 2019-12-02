import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from functools import partial
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.ensemble import RandomForestClassifier

def removeNonAlphanumeric(df) :
    for c in df.columns :
        if df[c].dtype == "O" :
            df[c] = df[c].str.replace('\t', '')
            df[c] = df[c].str.replace('[^a-zA-Z0-9]', '')
    df = df.replace('',np.nan)
    return df

def toNumeric(df):
    for c in df.columns :
        if df[c].dtype == "O" and all(df[c].str.isnumeric()):
            df[c] = pd.to_numeric(df[c])
    return df
            
    
def handleMissing(df, method = "drop",train = True, imputerDict = {}):
    if method == "drop" :
        df = df.dropna(inplace= True)
    elif type(method) == int :
        df.fillna(method, inplace= True)
    else :
        if method != "most_frequent":
            print("For non numeric columns, most frequent strategy is used")
        for c in df.columns :
            if train :
                imp = SimpleImputer(missing_values=np.nan, strategy=method if df[c].dtype!="O" else "most_frequent")
                imp = imp.fit(df[[c]])
                imputerDict[c] = imp 
            df[c] = imputerDict[c].transform(df[[c]])
            
    return df,imputerDict


class HandleMissingTransformer(BaseEstimator, TransformerMixin):
    """Custom scaling transformer"""
    def __init__(self, method):
        self.method = method
        self.imputerDict = {}
        

    def fit(self, df ):
        if self.method != "drop" and type(self.method) != int :
            if self.method != "most_frequent":
                print("For non numeric columns, most frequent strategy is used")
            for c in df.columns :
                imp = SimpleImputer(missing_values=np.nan, strategy=self.method if df[c].dtype!="O" else "most_frequent")
                imp = imp.fit(df[[c]])
                self.imputerDict[c] = imp 
        return self
            
                
        
    def transform(self, df):
        if self.method == "drop" :
            df = df.dropna(inplace= True)
        elif type(self.method) == int :
            df.fillna(self.method, inplace= True)
        else :
            for c in df.columns : 
                df[c] = self.imputerDict[c].transform(df[[c]])
        return df  
    
def target_variable_exploration(df, target, xlabel, ylabel, title) :
    positive_class = df[target].value_counts()[1]
    negative_class = df[target].value_counts()[0]
    positive_per = positive_class / df.shape[0] * 100
    negative_per = negative_class / df.shape[0] * 100

    plt.figure(figsize=(8, 8))

    sns.countplot(df[target]);
    plt.xlabel(xlabel, size=15, labelpad=15)
    plt.ylabel(ylabel, size=15, labelpad=15)
    plt.xticks((0, 1), ['Negative class ({0:.2f}%)'.format(negative_per), 'Positive class ({0:.2f}%)'.format(positive_per)])
    plt.tick_params(axis='x', labelsize=13)
    plt.tick_params(axis='y', labelsize=13)

    plt.title(title, size=15, y=1.05)

    plt.show()
    

def plot_numeric(train_data, numeric_features, target) :
    # Looping through and Plotting Numeric features
    for column in numeric_features:    
        # Figure initiation
        fig = plt.figure(figsize=(18,12))

        ### Distribution plot
        sns.distplot(train_data[column], ax=plt.subplot(221));
        # X-axis Label
        plt.xlabel(column, fontsize=14);
        # Y-axis Label
        plt.ylabel('Density', fontsize=14);
        # Adding Super Title (One for a whole figure)
        plt.suptitle('Plots for '+column, fontsize=18);

        ### Distribution per Positive / Negative class Value
        # Not Survived hist
        classes = train_data[target].unique()
        sns.distplot(train_data.loc[train_data[target]==classes[0], column].dropna(),
                     color='red', label=str(classes[0]), ax=plt.subplot(222));
        # Survived hist
        sns.distplot(train_data.loc[train_data[target]==classes[1], column].dropna(),
                     color='blue', label=str(classes[1]), ax=plt.subplot(222));
        # Adding Legend
        plt.legend(loc='best')
        # X-axis Label
        plt.xlabel(column, fontsize=14);
        # Y-axis Label
        plt.ylabel('Density per '+ str(classes[0])+' / '+str(classes[1]), fontsize=14);

        ### Average Column value per positive / Negative Value
        sns.barplot(x=target, y=column, data=train_data, ax=plt.subplot(223));
        # X-axis Label
        plt.xlabel('Positive or Negative?', fontsize=14);
        # Y-axis Label
        plt.ylabel('Average ' + column, fontsize=14);

        ### Boxplot of Column per Positive / Negative class Value
        sns.boxplot(x=target, y=column, data=train_data, ax=plt.subplot(224));
        # X-axis Label
        plt.xlabel(str(classes[0])+'  or '+ str(classes[0])+' ?', fontsize=14);
        # Y-axis Label
        plt.ylabel(column, fontsize=14);
        # Printing Chart
        plt.show()
        
def plot_categ(train_data, target, nominal_features) :
    # Looping through and Plotting Categorical features
    for column in nominal_features:
    # Figure initiation
        fig = plt.figure(figsize=(18,12))
        
        ### Number of occurrences per categoty - target pair
        ax = sns.countplot(x=column, hue=target, data=train_data, ax = plt.subplot(211));
        # X-axis Label
        plt.xlabel(column, fontsize=14);
        # Y-axis Label
        plt.ylabel('Number of occurrences', fontsize=14);
        # Adding Super Title (One for a whole figure)
        plt.suptitle('Plots for '+column, fontsize=18);
        # Setting Legend location 
        plt.legend(loc=1);

        ### Adding percents over bars
        # Getting heights of our bars
        height = [p.get_height() if np.isnan(p.get_height()) == 0 else 0 for p in ax.patches] #  get nan if
        # Counting number of bar groups 
        ncol = int(len(height)/2)
        # Counting total height of groups
        total = [height[i] + height[i + ncol] for i in range(ncol)] * 2
        # Looping through bars
        for i, p in enumerate(ax.patches):    
            # Adding percentages
            ax.text(p.get_x()+p.get_width()/2, height[i]*1.01 + 10,
                    '{:1.0%}'.format(height[i]/total[i]), ha="center", size=14) 

        classes = train_data[target].unique()
        ### Positive class percentage for every value of feature
        if train_data[target].dtype == 'O':
            ### Adding new column with numerical target names
            train_data[target+str('_id')] = train_data[target].map({classes[0]:0 , classes[1]: 1})
            sns.pointplot(x=column, y=target+str('_id'), data=train_data, ax = plt.subplot(212));
        # X-axis Label
        plt.xlabel(column, fontsize=14);
        # Y-axis Label
        plt.ylabel(' Positive class percentage', fontsize=14);
        # Printing Chart
        plt.show()
        
        
def featureEng(numerical_features, categorical_features):
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    t =  ColumnTransformer([('Scaler', numeric_transformer, numerical_features),('OneHotEncod', categorical_transformer, categorical_features)])
    preproc = Pipeline(steps=[('preprocessor', t)])
    #print(t.get_feature_names())
    return t

def getCategFeat(df, n, target):
    """
    -- n : min modalities for numerical feat
    """
    return [c for c in df.columns if (df[c].dtype == 'O' or df[c].nunique()<n) and c!=target]

class selectFeaturesTransformer(BaseEstimator, TransformerMixin):
    """Custom scaling transformer"""
    def __init__(self, k=10,method='RF',discreteCol=[]):
        self.k = k
        self.method = method
        self.order = []
        self.discreteCol = discreteCol
        
        
        

    def fit(self, X_train,y_train):
        if self.method == "Mutual Information" :
            discrete_mutual_info_classif = partial(mutual_info_classif, 
                                                   discrete_features=self.discreteCol)
            featS = SelectKBest(k=self.k, score_func=discrete_mutual_info_classif).fit(X_train,y_train )
            self.order = np.flip(featS.scores_.argsort())
            #self.selectedColumns = [columns_eng[i]  for i in self.order[:self.k]]
            #return X_train[:,order_mi[:self.k]]
        
        else :
            rfModel = RandomForestClassifier(random_state =0).fit(X_train, y_train)
            order = np.flip(rfModel.feature_importances_.argsort())
            self.order = np.flip(rfModel.feature_importances_.argsort())
            #self.selectedColumns = [columns_eng[i]  for i in order_rf[:self.k]]
            #return X_train[:,order_[:self.k]]
        return self
            
                
        
    def transform(self, X_train):
            return X_train[:,self.order[:self.k]]
        