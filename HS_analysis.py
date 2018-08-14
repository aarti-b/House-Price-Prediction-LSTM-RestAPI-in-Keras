# data analysis and pre processing
from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import warnings
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, 
	help="path to csv data file")
ap.add_argument("-a", "--analysis", type = bool, default = False,
	help="It will run analysis function")
ap.add_argument("-f", "--final_data", type=bool, default = False, help="it will run final_data function")
args = vars(ap.parse_args())

path = args["path"] 
def load_data(path):
    
    data = pd.read_csv(path)
    return data

# data analysis
def analysis(path):
    
    data = load_data(path)
    print("The Shape of data is :", data.shape)
    print("\n Name of Columns in data :", data.columns)
    #print("\n Data type of each Column :", data.dtype())
    print("\n Summary of data :", data.describe())
    num_cols = data._get_numeric_data().columns
    print("\n Columns having Numerical data :", (num_cols, len(num_cols)))
    # get categorical columns
    categorical_columns = list(set(data.columns) - set(num_cols))
    print("Columns having Categorical data:", (categorical_columns, len(categorical_columns)))
    
    print("Some Statistics of the Housing Price:\n")
    print(data['SalePrice'].describe())
    print("\nThe median of the Housing Price is: ", data['SalePrice'].median(axis = 0))

    print("Skewness: %f" % data['SalePrice'].skew())
    print("Kurtosis: %f" % data['SalePrice'].kurt())
    
    #Autocorrelation of SalePrice Series
    acorr = data['SalePrice'].autocorr(lag=2)   # at lag1 it is -0.0014076
    print('Auto Correlation of Sale price at lag 2 is :', acorr)
    
    # Group by Neighborhood Category for average SP in particular neighborhood
    nbr_groups = data.groupby("Neighborhood")['SalePrice'].mean()
    print(nbr_groups)
    plt.figure(figsize=(18,6));plt.title('Neighborhood vs mean value of Sale Price in Specific Neighborhood')
    plt.plot(nbr_groups);plt.ylabel('Mean Value of Sale Price');plt.show()
    
    ## Group by Neighborhood Category for average OverallQuall in particular neighborhood
    nbr_qual_groups = data.groupby("Neighborhood")['OverallQual'].mean()
    print(nbr_qual_groups)
    plt.figure(figsize=(18,6));plt.title('Neighborhood vs mean value of Quality in Specific Neighborhood')
    plt.plot(nbr_qual_groups);plt.ylabel('Mean Value of Quality');plt.show()
    
    # normalize salePrice
    trainx = data['SalePrice'].values.reshape(-1,1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_SP = scaler.fit_transform(trainx)
    
    # plot of normalized and non normalized saleprice
    plt.figure(figsize=(8,6))
    
    plt.subplot(2,2,1)
    sns.distplot(normalized_SP, kde = False, color = 'b', hist_kws={'alpha': 0.9})
    plt.title('Normalized SalePrice')
    
    plt.subplot(2,2,2)
    sns.distplot(data['SalePrice'], kde = False, color = 'b', hist_kws={'alpha': 0.9})
    plt.title('UnNormalized SalePrice')
    plt.show()

    
    #correlation plot

    corr = data.select_dtypes(include = ['float64', 'int64']).iloc[:, 1:].corr()
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, vmax=1, square=True)

    # correlation of features with saleprice
    
    cor_dict = corr['SalePrice'].to_dict()
    del cor_dict['SalePrice']
    print("List the numerical features decendingly by their correlation with Sale Price:\n")
    for ele in sorted(cor_dict.items(), key = lambda x: -abs(x[1])):
        print("{0}: \t{1}".format(*ele))
        
    #plot correlation curve
    plt.figure(figsize=(30,6))
    lists = sorted(cor_dict.items()) # sorted by key, return a list of tuples
    x, y = zip(*lists) # unpack a list of pairs into two tuples
    plt.plot(x, y);plt.title('Correlation of Different features w.r.t SalePrice')
    plt.show()
    
    #plot of over all quality wrt SalePrice 
    plt.figure(figsize=(8,6))
    sns.regplot(x = 'OverallQual', y = 'SalePrice', data = data, color = 'Orange');plt.show()
    
    #make plot with different features wrt sale price and also plot their individual distribution
    plt.figure(figsize=(8,6))
    sns.distplot(data['OverallQual'], kde = False, color = 'b', hist_kws={'alpha': 0.9})

    #box plot YrSold/saleprice

    var = 'YrSold'   
    
    data_tmp = pd.concat([data['SalePrice'], data[var]], axis=1)
    f, ax = plt.subplots(figsize=(8, 6))
    fig = sns.boxplot(x=var, y="SalePrice", data=data_tmp)
    fig.axis(ymin=0, ymax=800000)
    plt.show()
    
    #scatterplot
    #sns.set()
    #cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
    #sns.pairplot(data[cols], size = 2.5)
    #plt.show();
    
#outlier removal function
def get_median_filtered(signal, threshold=3):
    signal = signal.copy()
    difference = np.abs(signal - np.median(signal))
    median_difference = np.median(difference)
    if median_difference == 0:
        s = 0
    else:
        s = difference / float(median_difference)
    mask = s > threshold
    signal[mask] = np.median(signal)
    return signal

def final_data(path):
    
    data = load_data(path)
    corr = data.select_dtypes(include = ['float64', 'int64']).iloc[:, 1:].corr()
    cor_dict = corr['SalePrice'].to_dict()
    #Missing data and drop columns 
    
    total = data.isnull().sum().sort_values(ascending=False)
    percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    missing_data.head(20)
    
    data_new = data.drop((missing_data[missing_data['Total'] > 1]).index,1)
    #data_new = data.drop(data.loc[data['Electrical'].isnull()].index)
    data_new.isnull().sum().max()
    # remove columns based on correlation
    
    num__col = data_new._get_numeric_data().columns
    
    categorical_col = list(set(data_new.columns) - set(num__col))
    #print("Length of Categorical Columns : ", len(categorical_col)) 
    
    #filter out columns whose correlation with saleprice is less than 0.4
    columns = [i for i in cor_dict if cor_dict[i] >= 0.4] 

    data_filter = data_new[columns]
    data_cat = pd.get_dummies(data_new[categorical_col])
    final_data =  pd.concat([data_filter, data_cat], axis = 1)
    print("The shape of data is : ", final_data.shape)
        
    return final_data
        
    
if __name__=='__main__':
    
    if args["analysis"] == True:
        analysis(path)
    elif args["final_data"] == True:
        final_data(path)
    else:
        print("No function executed")
    
    #filtered_data = get_median_filtered(data['SalePrice'])
    #print(filtered_data.shape) ##(no column is removed)
   