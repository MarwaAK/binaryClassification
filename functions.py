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
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from IPython.display import display
import time

def removeNonAlphanumeric(df) :
    """ 
    Remove non-alphanumeric characters from data values
    Input :
        df -- dataframe 
    Output :
        df -- cleaned dataframe
    """
    for c in df.columns :
        if df[c].dtype == "O" :
            df[c] = df[c].str.replace('\t', '')
            df[c] = df[c].str.replace('[^a-zA-Z0-9]', '')
    df = df.replace('',np.nan)
    return df

def toNumeric(df):
    """" 
    Convert string column corresponding to numerical values to numerical columns
    Input : 
        df -- dataframe 
    Output :
        df -- dataframe with converted columns
    """
    for c in df.columns :
        if df[c].dtype == "O" and all(df[c].str.isnumeric()):
            df[c] = pd.to_numeric(df[c])
    return df
            


class HandleMissingTransformer(BaseEstimator, TransformerMixin):
    """Customized transformer to handles missing data"""
    
    def __init__(self, method,constant = ''):
        '''' 
        Initialise The transformer
        Inputs :
            method -- method used to replace or impute missing data (drop/constant/most_frequent/median/mean)
            constant -- if constant method is selected, the value of the constant must be specified
        '''
        self.method = method
        self.constant = constant
        self.imputerDict = {}
        

    def fit(self, df ):
        '''
        If impute method is selected i.e self.method not in ['drop', 'constant'], we must fit an imputer
        Input : 
            df -- data with missing values
        '''
        if self.method not in ['drop', 'constant'] :
            if self.method != "most_frequent":
                print("For non numerical columns, most frequent strategy is used")
            for c in df.columns :
                imp = SimpleImputer(missing_values=np.nan, strategy=self.method if df[c].dtype!="O" else "most_frequent")
                imp = imp.fit(df[[c]])
                self.imputerDict[c] = imp 
        return self
            
                
        
    def transform(self, df):
        """
        If impute method is selected, impute missing values using imput_dict created in fit function
        Input : 
            df -- data with missing values
        """
        if self.method == "drop" :
            df = df.dropna(inplace= True)
        elif self.method == 'constant' :
            df.fillna(self.constant, inplace= True)
        else :
            for c in df.columns : 
                df[c] = self.imputerDict[c].transform(df[[c]])
        return df  
    
def target_variable_exploration(df, target, xlabel, ylabel, title, positive=1) :
    """ 
    plots the distribution of the classes
    Input :
        df -- dataframe containing classes
        target -- class column
        xlabel
        ylabel 
        title
        positive -- modality corresponding to positive class
    """
    negative =  [c for c in df[target].unique() if c !=positive][0]
    positive_class = df[target].value_counts()[positive]
    negative_class = df[target].shape[0] - positive_class
    positive_per = positive_class / df.shape[0] * 100
    negative_per = negative_class / df.shape[0] * 100
    plt.figure(figsize=(8, 8))
    sns.countplot(df[target], order=[positive, negative]);
    plt.xlabel(xlabel, size=15, labelpad=15)
    plt.ylabel(ylabel, size=15, labelpad=15)
    plt.xticks((0, 1), [ 'Positive class ({0:.2f}%)'.format(positive_per), 'Negative class ({0:.2f}%)'.format(negative_per)])
    plt.tick_params(axis='x', labelsize=13)
    plt.tick_params(axis='y', labelsize=13)
    plt.title(title, size=15, y=1.05)
    plt.show()
    

def plot_numeric(data, numeric_features, target) :
    """ 
    plots analysing numerical features
    Inputs : 
        data -- dataframe containing features to plot
        numeric_features -- list of numerical features
        target -- target column name
     """
    # Looping through and Plotting Numeric features
    for column in numeric_features:    
        # Figure initiation
        fig = plt.figure(figsize=(18,12))

        ### Distribution plot
        sns.distplot(data[column], ax=plt.subplot(221));
        # X-axis Label
        plt.xlabel(column, fontsize=14);
        # Y-axis Label
        plt.ylabel('Density', fontsize=14);
        # Adding Super Title (One for a whole figure)
        plt.suptitle('Plots for '+column, fontsize=18);

        ### Distribution per Positive / Negative class Value
        # Not Survived hist
        classes = data[target].unique()
        sns.distplot(data.loc[data[target]==classes[0], column].dropna(),
                     color='red', label=str(classes[0]), ax=plt.subplot(222));
        # Survived hist
        sns.distplot(data.loc[data[target]==classes[1], column].dropna(),
                     color='blue', label=str(classes[1]), ax=plt.subplot(222));
        # Adding Legend
        plt.legend(loc='best')
        # X-axis Label
        plt.xlabel(column, fontsize=14);
        # Y-axis Label
        plt.ylabel('Density per '+ str(classes[0])+' / '+str(classes[1]), fontsize=14);

        ### Average Column value per positive / Negative Value
        sns.barplot(x=target, y=column, data=data, ax=plt.subplot(223));
        # X-axis Label
        plt.xlabel('Positive or Negative?', fontsize=14);
        # Y-axis Label
        plt.ylabel('Average ' + column, fontsize=14);

        ### Boxplot of Column per Positive / Negative class Value
        sns.boxplot(x=target, y=column, data=data, ax=plt.subplot(224));
        # X-axis Label
        plt.xlabel('Positive or Negative ?', fontsize=14);
        # Y-axis Label
        plt.ylabel(column, fontsize=14);
        # Printing Chart
        plt.show()
        
def plot_categ(train_data, target, nominal_features,positive =1) :
    """ 
    plots analysing nominal categorical features
    Inputs : 
        data -- dataframe containing features to plot
        nominal_features -- list of nominal features
        target -- target column name
     """
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

        negative = train_data[target].unique()[0] if train_data[target].unique()[0] != positive else train_data[target].unique()[1]
        ### Positive class percentage for every value of feature
        
        sns.pointplot(x=train_data[column], y=train_data[target].map({negative:0 , positive: 1}), ax = plt.subplot(212));
        # X-axis Label
        plt.xlabel(column, fontsize=14);
        # Y-axis Label
        plt.ylabel(' Positive class percentage', fontsize=14);
        # Printing Chart
        plt.show()

def correlationMap(df, target) :
    """ 
    Correlation Heatmap
    Inputs : 
        df -- dataframe containing features to plot
        target -- target column name
     """
    classes = df[target].unique()
    if data[target].dtype == 'O' :
        df[target+'_id'] = (df[target]== classes[0]).astype(int) #encode string target 
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corr, vmax=.8,annot=True, square=True)
    if data[target].dtype == 'O' :
        df.drop([target+'_id'], axis=1, inplace=True)
    # fix for matplotlib bug that cuts off top/bottom of seaborn viz
    b, t = plt.ylim() # Gets the values for bottom and top
    b += 0.5 # Add 0.5 to the bottom
    t -= 0.5 # Subtract 0.5 from the top
    plt.ylim(b, t) # update the ylim(bottom, top) values
    plt.show()
    
        
def featureEng(numerical_features, categorical_features):
    """ 
    create pipeline for feature preprocessing 
    Inputs : 
        numerical_features -- list of numerical features
        categorical_features -- list of categorical features
    Outputs :
        preproc -- pipeline with feature preprocessing steps
     """
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    t =  ColumnTransformer([('Scaler', numeric_transformer, numerical_features),('OneHotEncod', categorical_transformer, categorical_features)])
    preproc = Pipeline(steps=[('preprocessor', t)])
    return preproc



def getCategFeat(df, n, target):
    """
    get dataframe's categorical features 
    Inputs :
        df     -- dataframe  
        n      -- min modalities for numerical features
        target -- target column name
    """
    return [c for c in df.columns if (df[c].dtype == 'O' or df[c].nunique()<n) and c!=target]

class selectFeaturesTransformer(BaseEstimator, TransformerMixin):
    """Custom scaling transformer"""
    def __init__(self, k=10,method='RF',discreteCol=[]):
        """ 
        initialize transformer
        Inputs : 
            k -- number of features to keep
            method -- method to use, either 'Mutual Information or RF
            discreteCol -- if Mutual Information is used, specify indexes of discrete columns
        """
        self.k = k
        self.method = method
        self.order = []
        self.discreteCol = discreteCol
        
        
        

    def fit(self, X_train,y_train):
        """
        Fit the transformer on data
        Input :
            X_train -- features array
            Y_train -- labels array
        Output :
            fitted transformer
        """
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
        """
        apply fitted transformer to select features
        Input :
            X_train -- features array
        Output :
            array containing only selected features
        """
            return X_train[:,self.order[:self.k]]
        
def train(X_train, y_train, classifiers, names,parameters, parameters_featuresSelection, crossVal = True):
    """ 
    training process
    Inputs : 
        X_train -- features array
        Y_train -- labels array
        classifiers -- list of classifiers to test
        names -- list of classifiers names
        parameters -- tuning parameters corresponding for classifiers
        parameters_featuresSelection -- parameters for fearures selection
        crossVal -- whether to use cross validation or not
     """
    results = pd.DataFrame()
    for name, clf in zip(names, classifiers):
        print('############# ', name, ' #############')
        start = time.time()
        #print(params[name])
        FSelector = selectFeaturesTransformer()
        pipeline = Pipeline([('FeatureSelection',FSelector),('Classifier',clf)])
        parameters[name]['FeatureSelection__method'] = parameters_featuresSelection['FeatureSelection__method']
        parameters[name]['FeatureSelection__k']= parameters_featuresSelection['FeatureSelection__k']
        if crossVal:
            classifier = GridSearchCV(pipeline, parameters[name], cv=3)
        else:
            classifier = pipeline
        #print(classifier)
        classifier.fit(X_train, y_train)
        # All results
        means = classifier.cv_results_['mean_test_score']
        stds = classifier.cv_results_['std_test_score']
        r = pd.DataFrame(means,columns = ['mean_test_score'])
        r['std_test_score'] = stds
        r['params'] = classifier.cv_results_['params']
        r['classifier'] = name
        
        print('Training time (Cross Validation = ',crossVal,') :',(time.time()-start)/len(means))
        display(r.sort_values(by=['mean_test_score','std_test_score'],ascending =False))
        results = pd.concat([results, r], ignore_index=True)
        #for mean, std, params in zip(means, stds, classifier.cv_results_['params']):
        #print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    results_sorted = results.sort_values(by=['mean_test_score','std_test_score'],ascending =False)
    return results_sorted