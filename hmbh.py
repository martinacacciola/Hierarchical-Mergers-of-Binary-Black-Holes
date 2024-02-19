import numpy as np
import pandas as pd
import polars as pl 

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import hvplot
import seaborn as sns

import numpy.typing as npt
from typing import Optional, Tuple, Callable, Union, List

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import MinMaxScaler, label_binarize

from xgboost.sklearn import XGBClassifier

import shap

""" This Python module, hmbh.py, provides a set of tools for data processing and machine learning tasks. 
It includes the use of libraries such as sklearn for machine learning algorithms and model evaluation, xgboost for gradient boosting framework, 
and shap for model interpretation.
"""


def create_dataset(path: str, folders: List[str], mets: List[str], cols: List[str]) -> pl.DataFrame:

    """
    Function that takes in input a list of folders and a list of metallicities.
    It returns a polars lazy DataFrame with all the data from the folders and metallicities.

    Parameters
    --------
    path : str
        Path to the data parent folder (remember to add the final '/')
    folders : List[str]

        List of selected folders
    mets : List[str]

        List of selected metallicities
    cols : List[str]

        List of selected columns original names

    Return
    --------
    pl.Dataframe 
        `polars` DataFrame with all the data from the folders and metallicities
    """

    name = 'nth_generation.txt'

    col_name = ['c' + str(i) for i in range(28)]

    for i in range(len(folders)):
        for j in range(len(mets)):

            
            file_q = pl.scan_csv(path + folders[i] + '/Dyn/' + 
                                mets[j] + '/' + name, 
                                separator = ' ', has_header=True,
                                new_columns = col_name).select(cols) # select bold columns -> most important for this analysis
            
            
            met = pl.Series([float(mets[j]) for k in range(file_q.collect().height)]) # add metallicity column

            # univocal ID given by the original ID . metallicity folder . hosting obj folder
            new_id = pl.Series(file_q.select(pl.col('c0')).cast(pl.String).collect() + '.' + str(j) + '.' + str(i))#.cast(pl.Float64)

            # hosting object label set: 0 -> GC, 1 -> nSC, 2 -> ySC
            label = pl.Series([i for _ in range(file_q.collect().height)])

            file_q = file_q.with_columns(label.alias('label'), met.alias('met'), new_id.alias('c0')) 

            if i == 0 and j==0:
                df = file_q.collect()
                
            else:
                
                df = df.vstack(file_q.collect())

    df = df.filter(pl.col('c13') <= 13.6*1_000) # remove nth merges that take longer than Hubble time ~ 13.6 Gyr

    return df


def rename_columns(df: pl.DataFrame, new_cols: List[str]) -> pl.DataFrame:

    """
    Function that takes in input a DataFrame and a list of new column names (apart from the last two columns).
    It returns a DataFrame with renamed columns.

    Parameters
    --------
    df : DataFrame 
        `polars` or `pandas` DataFrame

    new_cols : List[str] 
        List of new column names
    
    Return
    --------
    pl.Dataframe or pd.DataFrame
        `polars` or `pandas` DataFrame with renamed columns
    """

    old_cols = df.columns

    rename_dict = {old_cols[i]: new_cols[i] for i in range(len(old_cols)-2)}
    df = df.rename(rename_dict)

    return df


def get_label_ngen(df: pl.DataFrame) -> pl.DataFrame:

    """
    Function that takes in input a DataFrame and returns a new column with the label for the nth generation.
     - 0 if the ID is unique, meaning that the binary systems did not evolve further (from the 2nd generation); 
     - 1 if the ID is not unique, meaning that the binary systems evolved further (from the 2nd generation).

    Parameters
    ---------
    df : pl.Dataframe
        input DataFrame
    
    Return
    --------
    pl.DataFrame
        DataFrame with new column 'label_ngen'
    """

    id_counts = df.group_by('ID').agg(pl.count('ID').alias('count'))

    # Join the count information back to the original DataFrame
    df = df.join(id_counts, on='ID', how='left')

    # Add the new column based on your condition
    df = df.with_columns(
        pl.when(df['count'] > 1)
        .then(1)
        .otherwise(0)
        .alias('label_ngen')
    )

    # Drop the temporary count column if needed
    df = df.drop('count')

    return df


def data_preprocessing(df: pl.DataFrame, n_sample: Union[int, bool], label: str, test_size: float, balanced_label: bool=True, random_state: int=42) -> Tuple[pd.DataFrame, pd.DataFrame, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    
    """
    Function that takes in input a `polars` DataFrame and returns the preprocessed data for the ML model.

    Parameters
    ----------
    df : pl.DataFrame
        Input `polars` DataFrame, that will be converted into a `pandas` DataFrame
    
    n_samples : int or None
        Number of samples to select. If None all the dataset is used
        
    label : str
        Column name of the label to use
        
    test_size : float
        Test size for the train/test split
        
    balanced_label : if True, the label will be balanced. Default is True
    
    random_state : int
        Random state for reproducibility
    
    Return
    -------
    Returns a tuple with the preprocessed data: X, y, X_train, y_train, X_test, y_test
    
    """

    df =  df.to_pandas()

    if balanced_label == True:

        if len(df[label].unique()) == 2:

            label_counts = df[label].value_counts()
        
            # Separate majority and minority classes
            df_majority = df[df[label]==label_counts.idxmax()]
            df_minority = df[df[label]==label_counts.idxmin()]

            # Downsample majority class
            df_majority_downsampled = resample(df_majority, 
                                            replace=False,    # sample without replacement
                                            n_samples=df_minority.shape[0],     # to match minority class
                                            random_state=random_state) # reproducible results
            
            # Combine minority class with downsampled majority class
            df_balanced = pd.concat([df_majority_downsampled, df_minority])

        else: 

            label_counts = df[label].value_counts()

            df_majority = df[df[label]==label_counts.idxmax()]
            df_minority = df[df[label]==label_counts.idxmin()]
            df_middle = df[(df[label] != label_counts.idxmax()) & (df[label] != label_counts.idxmin())]

            # Downsample majority class
            df_majority_downsampled = resample(df_majority, 
                                            replace=False,    # sample without replacement
                                            n_samples=df_minority.shape[0],     # to match minority class
                                            random_state=random_state) # reproducible results
            
            df_middle_downsampled = resample(df_middle,
                                            replace=False,
                                            n_samples=df_minority.shape[0],
                                            random_state=random_state)

            # Combine minority class with downsampled majority class
            df_balanced = pd.concat([df_majority_downsampled, df_middle_downsampled, df_minority])

        # Display new class counts
        print('Label count before balancing:\n', df[label].value_counts(), '\n')
        print('Label count after balancing:\n', df_balanced[label].value_counts())
        
        if n_sample == None:
            
            df_sample = df_balanced
         
        else:

            # Now let's select n shuffled samples
            df_sample = df_balanced.sample(n=n_sample, random_state=random_state)

    else:
        
        if n_sample == None:
            
            df_sample = df
            
        else:
            
            df_sample = df.sample(n=n_sample, random_state=random_state)

    # Define features and target
    X = df_sample.drop(['ID', 'label', 'n_gen', 'label_ngen'], axis=1)
    y = df_sample[label]

    print()
    print('Label count after sampling:\n', y.value_counts())

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    print()
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)
    print('y_train shape:', y_train.shape)
    print('y_test shape:', y_test.shape)

    # Initialize the scaler
    scaler = MinMaxScaler()

    # Fit the scaler to the training data and transform
    X_train = scaler.fit_transform(X_train)

    # Transform the test data
    X_test = scaler.transform(X_test)

    return X, y, X_train, y_train, X_test, y_test


def simple_RF(X_train: npt.NDArray, y_train: npt.NDArray, X_test: npt.NDArray, random_state: int=42) -> Tuple[RandomForestClassifier, npt.NDArray]:
    
    """
    Function that takes in input the preprocessed data and returns a simple Random Forest model.

    Parameters
    ---------
    X_train : npt.NDArray
        Training features

    y_train : npt.NDArray
        Training target

    X_test : npt.NDArray
        Testing features

    random_state : int 
        Random state for reproducibility

    Return
    --------
    Returns a tuple with the model, the predictions target: model, y_pred
    
    """

    # Initialize the Random Forest Classifier
    RF = RandomForestClassifier(n_estimators=10, class_weight='balanced_subsample', criterion='entropy', random_state=random_state)

    # Fit the model to the training data
    RF.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = RF.predict(X_test)

    return RF, y_pred


def model_evaluation(model: Union[RandomForestClassifier, XGBClassifier], X: pd.DataFrame, y: pd.DataFrame, X_train: npt.NDArray, y_train: npt.NDArray, X_test: npt.NDArray, y_test: npt.NDArray, y_pred: npt.NDArray, bar_plot: bool=False) -> None:
    
    """
    Function that takes in input the model and the preprocessed data and returns the evaluation of the model: Training Score, Test Score, Confusion Matrix and Feature Importances.

    Parameters
    ---------
    model : Union[RandomForestClassifier, XGBClassifier] 
        Random Forest or XGBoost classifier model

    X : pd.Dataframe
        Features dataframe

    y : pd.Dataframe
        Target dataframe

    X_train : npt.NDArray
        Training features

    y_train : npt.NDArray 
        Training target

    X_test : npt.NDArray
        Testing features

    y_test : npt.NDArray
        Testing target

    y_pred : npt.NDArray
        Model predictions

    bar_plot : bool. Default is False
        If True, the function will return the bar plot of the feature importances
    
    Return
    -------
    Returns the evaluation of the model: Training Score, Test Score, Confusion Matrix and Feature Importances. 
    """

    #std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)

    # Get feature importances
    importances = model.feature_importances_

    # Convert the importances into one-dimensional 1darray with corresponding df column names as axis labels
    f_importances = pd.Series(importances, X.columns)

    # Sort the array in descending order of the importances
    f_importances.sort_values(ascending=False, inplace=True)

    print()
    print("Training Score:       ", model.score(X_train, y_train))
    print("Test score (Accuracy):", model.score(X_test, y_test))
    print()

    # Plot the confusion matrix
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap=plt.cm.Greys, normalize='true', text_kw={'fontsize': 11}, ax=ax)
    plt.title('Confusion Matrix')
    plt.show()

    if bar_plot == True:

        # Make the bar plot from f_importances
        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(111) 
        f_importances.plot(x='Features', y='Importance', kind='bar', rot=45, fontsize=10, ax=ax, color='tab:blue', edgecolor='black', alpha=0.8)#, yerr=std)
        plt.title('Built-in Feature Importances')
        plt.show()


def gridsearch_RF(param_grid: dict, cv: int, X_train: npt.NDArray, y_train: npt.NDArray, X_test: npt.NDArray, n_jobs: int, random_state: int=42, verbose: int=1) -> Tuple[RandomForestClassifier, npt.NDArray, dict]:

    """
    Function that takes in input the parameter grid and the preprocessed data and returns the best model and the predictions.

    Parameters
    ----------
    param_grid : dict
        Parameter grid for the Grid Search

    cv : int 
        Cross-Validation folds

    X_train : npt.NDArray
        Training features

    y_train : npt.NDArray
        Training target

    X_test : npt.NDArray 
        Testing features

    random_state : int
        Random state for reproducibility

    verbose : int 
        Verbosity level

    n_jobs : int
        Number of jobs to run in parallel

    Return
    ----------
    Return a tuple with the best model and the predictions: model, y_pred

    """

    RF = RandomForestClassifier(criterion='entropy', class_weight='balanced_subsample', random_state=random_state)

    # Initialize the grid search model
    grid_search = GridSearchCV(estimator=RF, param_grid=param_grid, cv=cv, verbose=verbose, n_jobs=n_jobs, return_train_score=True)

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)

    # Get the best parameters
    best_params = grid_search.best_params_
    best_index = grid_search.best_index_

    print()
    print('Best parameters: \n', best_params, '\n')
    print('ID of the best combination: \n', best_index)

    # Fit the model with the best parameters
    RF_best = RandomForestClassifier(**best_params)
    RF_best.fit(X_train, y_train)

    # Predict on the test set
    y_pred = RF_best.predict(X_test)

    return RF_best, y_pred, grid_search


def gridsearch_scores(grid_search: dict) -> None:

    """
    Function that takes in input the grid search results and returns the mean test and train scores plot.

    Parameters
    -----------
    grid_search : dict 
        Grid search results
    
    Return
    ---------
    Returns a plot with the mean test and train scores.
    """
    # Get mean test scores
    mean_test_scores = grid_search.cv_results_['mean_test_score']
    mean_train_scores = grid_search.cv_results_['mean_train_score']

    best_params = grid_search.best_params_
    title = ', '.join([f"{key}={value}" for key, value in best_params.items()])

    best_index = int(grid_search.best_index_)

    # Plot mean test scores
    plt.figure(figsize=(10, 6))

    plt.plot(range(1, len(mean_test_scores)+1), mean_test_scores, label='Mean Test Score', color='tab:blue', linewidth=2)
    plt.plot(range(1, len(mean_train_scores)+1), mean_train_scores, label='Mean Train Score', color='tab:orange', linewidth=2)
    plt.xlabel('Index of hyperparameter combination')
    plt.vlines(x=best_index, ymin=0, ymax=1.2, color='tab:green', label='Best combination', linestyles='--', linewidth=1.3)
    plt.ylabel('Mean Score')
    plt.legend(loc='best')
    plt.ylim(0.5, 1.05)
    plt.xlim(-1, len(mean_test_scores)+1)
    plt.yticks(np.arange(0.5, 1.05, 0.1))
    plt.title(title)
    plt.show()
    

def simple_XGB(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, n_jobs: int, random_state: int = 42, params: dict = None) -> Tuple[XGBClassifier, np.ndarray]:
    
    """
    Function that takes in input the preprocessed data and returns a simple XGBoost model.

    Parameters
    ---------

    X_train : npt.NDArray
        Training features

    y_train : npt.NDarray 
        Training target

    X_test : npt.NDArray
        Testing features

    n_jobs: int
        Number of jobs to run in parallel

    random_state : int 
        Random state for reproducibility

    params : dict 
        Parameters for XGBoost
    
    Return
    ---------
    Returns a tuple with the model, the predictions target: model, y_pred
    """

    # Default parameters if not provided
    if params is None:
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss'
        }

    # Initialize XGBClassifier
    XGB = XGBClassifier(random_state=random_state, tree_method='approx', n_jobs=n_jobs, **params,)

    # Fit the model to the training data
    XGB.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = XGB.predict(X_test)

    return XGB, y_pred


def gridsearch_XGB(param_grid: dict, cv: int, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, n_jobs: int, random_state: int=42, verbose: int=1) -> Tuple[XGBClassifier, np.ndarray, dict]:
    
    """
    Function that takes in input the parameter grid and the preprocessed data and returns the best model and the predictions.

    Parameters
    ----------

    param_grid : dict 
        Parameter grid for the Grid Search
    
    cv : int
        Cross-Validation folds

    X_train : npt.NDArray
        Training features
    
    y_train : npt.NDArray
        Training target
    
    X_test : npt.NDArray
        Testing features
    
    n_jobs: int
        Number of jobs to run in parallel

    random_state : int 
        Random state for reproducibility

    verbose : int 
        Verbosity level
    
    Return
    ---------
    Returns a tuple with the best model and the predictions: model, y_pred
    """

    # Initialize the XGBoost classifier
    XGB = XGBClassifier(objective='binary:logistic', random_state=random_state)

    # Initialize the grid search model
    grid_search = GridSearchCV(estimator=XGB, param_grid=param_grid, cv=cv, verbose=verbose, n_jobs=n_jobs, return_train_score=True)

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)

    # Get the best parameters
    best_params = grid_search.best_params_
    best_index = grid_search.best_index_

    print()
    print('Best parameters: \n', best_params, '\n')
    print('ID of the best combination: \n', best_index)

    # Fit the model with the best parameters
    XGB_best = XGBClassifier(objective='binary:logistic', tree_method='approx', **best_params)
    XGB_best.fit(X_train, y_train)

    # Predict on the test set
    y_pred = XGB_best.predict(X_test)

    return XGB_best, y_pred, grid_search


def plot_learning_curve(model: Union[RandomForestClassifier, XGBClassifier], X_train: npt.NDArray, y_train: npt.NDArray, cv: Union[int, bool], title: str, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    model : Union[RandomForestClassifier, XGBClassifier]
        Classifier model to use for the learning curve. Must implement the `fit` and `predict` methods.

    X_train : npt.NDArray 
        Training vector

    y : npt.NDArray
        Target relative to X_train for classification or regression

    cv : int
        Cross-validation folds

    title : str
        Title for the plot

    n_jobs : int
        Number of jobs to run in parallel

    train_sizes : array-like, shape (n_ticks,), dtype float or int. Default `np.linspace(.1, 1.0, 5)`
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise, it is interpreted as absolute sizes of the training sets.
    """

    plt.figure()
    
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, shuffle=True)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, 
                     alpha=0.1, color="tab:red")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, 
                     alpha=0.1, color="tab:green")
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="CV (Test) score")

    plt.title(title)
    plt.xlabel("Training Examples")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.grid()

    plt.show()


def plot_ROC(model: Union[RandomForestClassifier, XGBClassifier], X_test: npt.NDArray, y_test: npt.NDArray, title: str):

    n_classes = len(y_test.unique())

    if n_classes == 2:
    
        # Predict probabilities of the positive class
        y_score = model.predict_proba(X_test)[:, 1]

        # Compute ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='tab:orange', lw=2, label='ROC curve positive class (AUC = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Binary-class (Test Set)\n' + title)
        plt.legend(loc="lower right")
        plt.show()
    

    else:
        # Predict probabilities of each class for the testing set
        y_score = model.predict_proba(X_test)
        
        # Binarize the labels
        y_bin = label_binarize(y_test, classes=np.unique(y_test))

        # Compute ROC curve and AUC for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot ROC curve for each class
        plt.figure()
        colors = ['tab:blue', 'tab:green', 'tab:orange']
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2, label='ROC curve (class {0}) (AUC = {1:0.2f})'.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Multi-class (Test Set)\n' + title)
        plt.legend(loc='lower right')
        plt.show()


def shap_explainer(model: Union[RandomForestClassifier, XGBClassifier], data: Union[pd.DataFrame, npt.NDArray], n_sample: Union[int, bool]) -> List:

    """
    Function that takes in input the model and the data and returns the shap values.

    Parameters
    ----------

    model: Union[RandomForestClassifier, XGBClassifier] 
        Random Forest or XGBoost Classifier model

    data: Union[pd.DataFrame, npt.NDArray]
        The background dataset to use for integrating out features.

    n_sample: int or None
        Number of samples to select. Note that the “interventional” 
        option requires a background dataset and its runtime scales 
        linearly with the size of the background dataset you use.  
        Anywhere from 100 to 1000 random background samples are good sizes to use.
    
    Return
    -------
    List
        Return a list of arrays containing the shap values
    """
    
    expl = shap.TreeExplainer(model)

    if n_sample == None:
        shap_values = expl.shap_values(data)
    else:
        shap_values = expl.shap_values(data[:n_sample])

    return shap_values


def plot_shap_violin(shap_values: List, data: Union[pd.DataFrame, npt.NDArray], X: pd.DataFrame, n_sample: int) -> None:

    """
    Function that takes in input the shap values and the data, used to train the Explainer, and returns the violin plot of the shap values,
    for a local inspection.

    Parameters
    ----------
    shap_values: List
        List of arrays containing the shap values

    data: Union[pd.DataFrame, npt.NDArray]
        Matrix of feature values. Data must be the one used to train the Explainer. 
        The data will be used in the violin plot.

    X: pd.DataFrame
        Features dataframe. Used to retrieve the features names. Must be provided to label the y-axis of the violin plot.

    n_sample: int
        Number of samples to select. Must be the same number used to train the Explainer.

    Return
    ---------
    Returns a plot with the violin plot of the shap values.
    """
    
    a = len(shap_values)

    if n_sample == None:

        if a == 2:

            fig = plt.figure()

            ax0 = fig.add_subplot(211)
            ax0.set_title('Class 0 - No Evol')
            shap.summary_plot(shap_values[0], data, plot_type='violin', feature_names=X.columns, show=False)
            ax0.set_xlabel('SHAP values (impact on output model)', fontsize=11)

            ax1 = fig.add_subplot(212)
            ax1.set_title('Class 1 - Evol')
            shap.summary_plot(shap_values[1], data, plot_type='violin', feature_names=X.columns, show=False)
            ax1.set_xlabel('SHAP values (impact on output model)', fontsize=11)

            plt.subplots_adjust(hspace = 50)
            plt.gcf().set_size_inches(8,12)
            plt.tight_layout()
            plt.show()

        else: 
            fig = plt.figure()

            ax0 = fig.add_subplot(311)
            ax0.set_title('Class 0 - GC')
            shap.summary_plot(shap_values[0], data, plot_type='violin', feature_names=X.columns, show=False)
            ax0.set_xlabel('SHAP values (impact on output model)', fontsize=11)

            ax1 = fig.add_subplot(312)
            ax1.set_title('Class 1 - YSC')
            shap.summary_plot(shap_values[1], data, plot_type='violin', feature_names=X.columns, show=False)
            ax1.set_xlabel('SHAP values (impact on output model)', fontsize=11)

            ax2 = fig.add_subplot(313)
            plt.title('Class 2 - NSC')
            shap.summary_plot(shap_values[2], data, plot_type='violin', feature_names=X.columns, show=False)
            plt.xlabel('SHAP values (impact on output model)', fontsize=11)

            plt.subplots_adjust(hspace = 50)
            plt.gcf().set_size_inches(8,12)
            plt.tight_layout()
            plt.show()    
    else:
     
        if a == 2:

            fig = plt.figure()

            ax0 = fig.add_subplot(211)
            ax0.set_title('Class 0 - No Evol')
            shap.summary_plot(shap_values[0], data[:n_sample], plot_type='violin', feature_names=X.columns, show=False)
            ax0.set_xlabel('SHAP values (impact on output model)', fontsize=11)

            ax1 = fig.add_subplot(212)
            ax1.set_title('Class 1 - Evol')
            shap.summary_plot(shap_values[1], data[:n_sample], plot_type='violin', feature_names=X.columns, show=False)
            ax1.set_xlabel('SHAP values (impact on output model)', fontsize=11)

            plt.subplots_adjust(hspace = 50)
            plt.gcf().set_size_inches(8,10)
            plt.tight_layout()
            plt.show()

        else: 
            fig = plt.figure()

            ax0 = fig.add_subplot(311)
            ax0.set_title('Class 0 - GC')
            shap.summary_plot(shap_values[0], data[:n_sample], plot_type='violin', feature_names=X.columns, show=False)
            ax0.set_xlabel('SHAP values (impact on output model)', fontsize=11)

            ax1 = fig.add_subplot(312)
            ax1.set_title('Class 1 - YSC')
            shap.summary_plot(shap_values[1], data[:n_sample], plot_type='violin', feature_names=X.columns, show=False)
            ax1.set_xlabel('SHAP values (impact on output model)', fontsize=11)

            ax2 = fig.add_subplot(313)
            plt.title('Class 2 - NSC')
            shap.summary_plot(shap_values[2], data[:n_sample], plot_type='violin', feature_names=X.columns, show=False)
            plt.xlabel('SHAP values (impact on output model)', fontsize=11)

            plt.subplots_adjust(hspace = 50)
            plt.gcf().set_size_inches(8,12)
            plt.tight_layout()
            plt.show()


def plot_shap_bar(shap_values: List, X: pd.DataFrame) -> None:

    """
    Function that takes in input the shap values, used to train the Explainer, and returns the bar plot of the shap values,
    either for a global or local inspection.

    Parameters
    ---------
    shap_values: List
        List of arrays containing the shap values

    X: pd.DataFrame
        Features dataframe. Used to retrieve the features names. Must be provided to label the y-axis of the violin plot.

    Return
    ---------
    Returns a plot with the bar plot of the shap values.
    """

    a = len(shap_values)

    shap_mean = np.mean(np.abs(shap_values), axis=1)
    col = X.columns
    shap_df = pd.DataFrame(shap_mean, columns=col)
    
    if a == 2:
        class_names = ['No Evol - 0', 'Evol - 1']

        # Global bar plot
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(1,1,1)
        ax.set_title('Global Insight', fontsize=14)
        shap.summary_plot(shap_values, plot_type='bar', feature_names=X.columns, plot_size=(8,5), show=False, class_names=class_names)

        # Single class bar plot
        fig, ax = plt.subplots(1, 2, figsize=(26,10))
        for i in range(a):
            shap_df1 = shap_df.iloc[i,:].sort_values()
            ax[i].barh(shap_df1.index, shap_df1, color='tab:blue')
            ax[i].set_title(f'{class_names[i]}', fontsize=15)
            ax[i].set_xlabel('mean(|SHAP value|)', fontsize=13)
            plt.suptitle('Pin point insight', fontsize=25)


    else: 
        class_names = ['GC - 0', 'YSC - 1', 'NSC - 2']
    
        # Global bar plot
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(1,1,1)
        ax.set_title('Global Insight', fontsize=14)
        shap.summary_plot(shap_values, plot_type='bar', feature_names=X.columns, plot_size=(8,5), show=False, class_names=class_names)
        
        # Single class bar plot
        fig, ax = plt.subplots(1, 3, figsize=(26,10))
        for i in range(a):
            shap_df1 = shap_df.iloc[i,:].sort_values()
            ax[i].barh(shap_df1.index, shap_df1, color='tab:blue', edgecolor='black', alpha=0.9)
            ax[i].set_title(f'{class_names[i]}', fontsize=15)
            ax[i].set_xlabel('mean(|SHAP value|)', fontsize=13)
            plt.suptitle('Local Insight', fontsize=25)