import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns


# Increase matplotlib font
plt.rcParams.update({'font.size': 16})

__all__ = ["plot_variable_over_time", "print_distribution_two_variables", "boxplot_three_variables", "plot_confusion_matrix"]

def plot_variable_over_time(data_before, data_after, variable_name, user_id="AS14.01"):
    """
    Plot a variable over time before and after cleaning.

    Parameters:
    - data_before: Original dataframe before cleaning (with 'time', 'variable', 'value')
    - data_after: Cleaned dataframe (same format)
    - variable_name: Name of the variable to plot
    - user_id: ID of the user to plot
    """
    # Filter data
    before = data_before[(data_before['variable'] == variable_name) & (data_before['id'] == user_id)].copy()
    after = data_after[(data_after['id'] == user_id)].copy()

    # Select the specific column from data_after
    after_value = after[variable_name].copy()

    # Convert time if not already datetime
    if not pd.api.types.is_datetime64_any_dtype(before['time']):
        before['time'] = pd.to_datetime(before['time'])
    if not pd.api.types.is_datetime64_any_dtype(after['time']):
        after['time'] = pd.to_datetime(after['time'])

    # Sort by time 
    before = before.sort_values('time')
    after = after.sort_values('time')

    # Plot
    plt.figure(figsize=(14, 6))
    plt.scatter(before['time'], before['value'], label='Before Cleaning', alpha=0.5, s=15)
    plt.scatter(after['time'], after_value, label='After Cleaning', alpha=0.75, s=10)
    plt.title(f'Temporal Plot of "{variable_name}" for User {user_id} Before and After Cleaning')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plt.show()

def print_distribution_two_variables(data, variables = ['mood', 'activity']):
    
    # Distribution of mood values
    plt.figure(figsize=(14, 9))
    for index, variable in enumerate(variables):
        var_data = data[data['variable'] == variable]
        plt.subplot(2, 1, index + 1)
        
        # Create histogram
        plt.hist(var_data['value'], bins=min(int(var_data['value'].unique().size), 100), 
                                alpha=0.5, label=variable, density=True)
        
        # Add mean and median lines
        plt.axvline(var_data['value'].mean(), color='k', linestyle='dashed', linewidth=1, label='Mean')
        plt.axvline(var_data['value'].median(), color='r', linestyle='dashed', linewidth=1, label='Median')
        
        # if variable == 'activity':
            # Add proper density estimation using KDE
        density = gaussian_kde(var_data['value'])
        x_vals = np.linspace(var_data['value'].min(), var_data['value'].max(), min(int(var_data['value'].unique().size), 100))
        plt.plot(x_vals, density(x_vals), color='g', label='Distribution')


                
        plt.legend()
        plt.title(f'Distribution of {variable} values')
        plt.xlabel(f'{variable} ({var_data["value"].min():.2f} - {var_data["value"].max():.2f})')
        plt.ylabel('Density')
    plt.tight_layout()
    plt.show()

def boxplot_three_variables(data, variables=['mood', 'activity', 'screen']):
   # Boxplot of different variables
    plt.figure(figsize=(12, 9))
    for index, var in enumerate(variables):
        var_data = data[data['variable'] == var]['value']
        if not var_data.empty:
            plt.subplot(1, 3, index + 1)
            plt.boxplot(var_data.astype('float'), patch_artist=True)
            plt.title(f'{var}')
            plt.ylabel('Value')
    plt.tight_layout()
    plt.show()   

def plot_confusion_matrix(y_test, y_pred_rf, y_pred_rnn, label_encoder):
    # Confusion matrices
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title('Random Forest')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    
    plt.subplot(1, 2, 2)
    cm_rnn = confusion_matrix(y_test, y_pred_rnn)
    sns.heatmap(cm_rnn, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title('RNN')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    
    plt.tight_layout()
    plt.show()
    