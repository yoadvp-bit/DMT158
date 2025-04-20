import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from multiprocessing import Pool, cpu_count
import warnings
import tqdm
from functools import partial
import traceback

from rnn_classifier import RNNClassifier
from annoyimputer import AnnoyKNNImputer
from plots import *

warnings.filterwarnings('ignore')

# Task 1A: Exploratory Data Analysis
def exploratory_data_analysis(data):
    print("\n=== Task 1A: Exploratory Data Analysis ===")
    
    # Basic information
    print(f"Dataset shape: {data.shape}")
    print("\nData types:")
    print(data.dtypes)
    
    # Convert time to datetime
    data['time'] = pd.to_datetime(data['time'])
    
    # Check for missing values
    print("\nMissing values per column:")
    print(data.isnull().sum())
    
    # Basic statistics
    print("\nBasic statistics for numerical values:")
    print(data['value'].describe())
    
    # Count unique values for categorical variables
    print("\nUnique IDs:", data['id'].nunique())
    print("Unique variables:", data['variable'].nunique())
    print("Variables in dataset:", data['variable'].unique())
    
    #Find variable with maximum value
    max_value = data['value'].max()
    max_variable = data[data['value'] == max_value]['variable'].values[0]
    print(f"Maximum value in dataset: {max_value} for variable: {max_variable}")

    print_distribution_two_variables(data, ['mood', 'activity'])
    print_distribution_two_variables(data, ['appCat.communication', 'appCat.social'])
    
    boxplot_three_variables(data, ['mood', 'activity', 'screen'])
    boxplot_three_variables(data, ['appCat.entertainment', 'appCat.communication', 'appCat.social'])
    
    return data

# Used in Task 1B to process each user singularly when performing imputation.
def process_user(user, df):
        user_mask = df['id'] == user
        user_data = df.loc[user_mask].copy()
        
        # Sort user data by time
        user_data = user_data.sort_values('time')
        
        # Handle binary variables (call, sms)
        binary_vars = ['call', 'sms']
        for var in binary_vars:
            if var in user_data.columns:
                # Most missing values for binary vars are 0 (no calls/sms)
                user_data[var] = user_data[var].fillna(0)
        
        # Handle psychological measures with interpolation
        psych_vars = ['circumplex.arousal', 'circumplex.valence', 'activity']
        for var in psych_vars:
            if var in user_data.columns:
                try:
                    # First set the time as index for time-weighted interpolation
                    user_data_indexed = user_data.set_index('time')
                    
                    # Now we can use time interpolation
                    user_data_indexed[var] = user_data_indexed[var].interpolate(
                        method='time', limit=20, limit_direction='both'
                    )
                    
                    # Reset index to get back our time column
                    user_data = user_data_indexed.reset_index()
                    
                    # Fill any remaining with median
                    if user_data[var].isnull().any():
                        user_data[var] = user_data[var].fillna(user_data[var].median())

                except Exception as e:
                    # Fallback to linear interpolation if time interpolation fails
                    user_data[var] = user_data[var].interpolate(method='linear', limit=6)
                    # Fill any remaining with median
                    if user_data[var].isnull().any():
                        user_data[var] = user_data[var].fillna(user_data[var].median())
        
        # Handle app usage and screen time with linear interpolation
        app_vars = [col for col in user_data.columns if col.startswith('appCat') or col == 'screen']
        for var in app_vars:
            if var in user_data.columns:
                # Use linear interpolation for app data
                user_data[var] = user_data[var].interpolate(method='linear', limit=8)
                
                # For remaining gaps, use time-of-day patterns (efficient implementation)
                if user_data[var].isnull().any():
                    # Group by hour of day to find patterns
                    user_data['hour'] = user_data['time'].dt.hour
                    hourly_medians = user_data.groupby('hour')[var].median()
                    
                    # Apply hour-based medians to missing values
                    for idx in user_data[user_data[var].isnull()].index:
                        hour = user_data.loc[idx, 'hour']
                        if hour in hourly_medians.index and not pd.isna(hourly_medians[hour]):
                            user_data.loc[idx, var] = hourly_medians[hour]
                    
                    # Clean up
                    user_data = user_data.drop('hour', axis=1)
        
        # Use KNN for any remaining missing values
        numeric_cols = user_data.select_dtypes(include=[np.number]).columns.tolist()
        
        if user_data[numeric_cols].isnull().sum().sum() > 0:
            try:
                user_data_numeric = user_data[numeric_cols]
                imputer = AnnoyKNNImputer(n_neighbors=min(50, len(user_data_numeric)-1))
                
                # If enough data, impute
                if len(user_data_numeric) >= 3:
                    imputer.fit(user_data_numeric.values)
                    imputed_data = imputer.transform(user_data_numeric.values)
                    user_data[numeric_cols] = imputed_data
            except Exception as e:
                print("Failed KNN")
                print(traceback.format_exc())
                pass
        
        # Final pass for any remaining values
        for col in numeric_cols:
            if user_data[col].isnull().any():
                # Forward fill
                user_data[col] = user_data[col].fillna(method='ffill')
                # Backward fill
                user_data[col] = user_data[col].fillna(method='bfill')
                # Finally, use column median if still missing
                if user_data[col].isnull().any():
                    col_median = user_data[col].median()
                    user_data[col] = user_data[col].fillna(col_median)
        
        return user_data
    
# Task 1B: Data Cleaning
def data_cleaning(data):
    print("\n=== Task 1B: Data Cleaning ===")
    print("Identifying and removing outliers...")

    # Remove outliers based on domain knowledge
    cleaned_data = data.copy()
    
    # Group by variable type to handle different variables appropriately
    for variable in cleaned_data['variable'].unique():
        var_data = cleaned_data[cleaned_data['variable'] == variable]
        
        # Different approaches for different variable types
        if variable == 'mood':
            # Mood should be between 1-10
            invalid_indices = var_data[~var_data['value'].between(1, 10)].index
            cleaned_data.drop(invalid_indices, inplace=True)
        elif variable in ['circumplex.arousal', 'circumplex.valence']:
            # Circumplex values should be between -2 and 2
            invalid_indices = var_data[~var_data['value'].between(-2, 2)].index
            cleaned_data.drop(invalid_indices, inplace=True)
        elif variable == 'activity':
            # Activity between 0 and 1
            invalid_indices = var_data[~var_data['value'].between(0, 1)].index
            cleaned_data.drop(invalid_indices, inplace=True)
        elif variable in ['call', 'sms']:
            # Should be 0 or 1
            invalid_indices = var_data[~var_data['value'].isin([0, 1])].index
            cleaned_data.drop(invalid_indices, inplace=True)
        else:
            # For other variables (app usage, screen time), remove extreme outliers using IQR method
            Q1 = var_data['value'].quantile(0.10)
            Q3 = var_data['value'].quantile(0.90)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_indices = var_data[(var_data['value'] < lower_bound) | 
                                      (var_data['value'] > upper_bound)].index
            cleaned_data.drop(outlier_indices, inplace=True)
    
    print(f"Removed {len(data) - len(cleaned_data)} outliers")
    
    # Create pivot table for imputation
    print("\nImputing missing values...")
    pivot_data = cleaned_data.pivot_table(
        index=['id', 'time'],
        columns='variable',
        values='value'
    ).reset_index()

    print("Null values before imputation:")
    print(pivot_data.isnull().sum())

    # Prepare data for parallel processing
    imputed_data = pivot_data.copy()
    
    # Convert time to datetime once for all operations
    imputed_data['time'] = pd.to_datetime(imputed_data['time'])
    users = list(imputed_data['id'].unique())
    
    # Determine number of processes to use
    n_processes = max(1, min(cpu_count() - 1, len(users)))
    print(f"Using {n_processes} processes for parallel imputation")
    
    # Create a partial function with the dataframe
    process_user_partial = partial(process_user, df=imputed_data)
    
    # Process users in parallel
    user_data_list = []
    with Pool(processes=n_processes) as pool:
        for result in tqdm.tqdm(pool.imap(process_user_partial, users), total=len(users)):
            user_data_list.append(result)
    
    # Combine results into a single dataframe
    final_data = pd.concat(user_data_list, ignore_index=True)
    
    # Convert time back to string format if original was string
    if isinstance(data['time'].iloc[0], str):
        final_data['time'] = final_data['time'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')
    
    missing_before = pivot_data.isnull().sum().sum()
    missing_after = final_data.isnull().sum().sum()
    print(f"\nMissing values before imputation: {missing_before}")
    print(f"Missing values after imputation: {missing_after}")
    if missing_before > 0:
        print(f"Imputed {missing_before - missing_after} values ({(missing_before - missing_after) / missing_before * 100:.2f}%)")

    return final_data

def feature_engineering(cleaned_data):
    print("\n=== Task 1C: Feature Engineering ===")

    daily_pivot = cleaned_data.sort_values(['id', 'time'])
    
    #Convert time to date for daily aggregation if needed
    if 'date' not in daily_pivot.columns:
        daily_pivot['time'] = pd.to_datetime(daily_pivot['time'], format='%Y-%m-%d %H:%M:%S.%f')
        daily_pivot['date'] = daily_pivot['time'].dt.date
    
    # If we still need to aggregate by day (in case there are multiple entries per day)
    daily_agg = daily_pivot.groupby(['id', 'date']).mean().reset_index()
    
    # Sort by user and date
    daily_agg = daily_agg.sort_values(['id', 'date'])
    
    # Create instances with 5-day history
    instances = []
    for user in daily_agg['id'].unique():
        user_data = daily_agg[daily_agg['id'] == user].reset_index(drop=True)
        
        if len(user_data) < 6:
            continue
        
        for i in range(5, len(user_data)):
            target_day = user_data.iloc[i]
            
            if 'mood' not in target_day or pd.isna(target_day['mood']):
                continue
            
            # Create features from 5-day history
            instance = {
                'id': user,
                'target_date': target_day['date'],
                'target_mood': target_day['mood']
            }
            
            # Add features from previous 5 days
            for j in range(1, 6):
                prev_day = user_data.iloc[i-j]
                day_num = 6-j
                
                for col in prev_day.index:
                    if col in ['id', 'time', 'date', 'index']:
                        continue
                    
                    if not pd.isna(prev_day[col]):
                        instance[f'{col}_day{day_num}'] = prev_day[col]
                    else:
                        instance[f'{col}_day{day_num}'] = np.nan
            
            # Add more advanced features
            # Mood trend (linear regression slope over 5 days)
            mood_days = [instance.get(f'mood_day{j}', np.nan) for j in range(1, 6)]
            if not any(pd.isna(mood) for mood in mood_days):
                instance['mood_trend'] = np.polyfit(range(5), mood_days, 1)[0]
            
            # Mood volatility (standard deviation over 5 days)
            if not any(pd.isna(mood) for mood in mood_days):
                instance['mood_volatility'] = np.std(mood_days)
            
            # Moving averages for mood
            instance['mood_ma3'] = np.nanmean([instance.get(f'mood_day{j}', np.nan) for j in range(3, 6)])
            instance['mood_ma5'] = np.nanmean([instance.get(f'mood_day{j}', np.nan) for j in range(1, 6)])
            
            # Activity features (if available)
            if any(f'activity_day{j}' in instance for j in range(1, 6)):
                activity_days = [instance.get(f'activity_day{j}', np.nan) for j in range(1, 6)]
                instance['activity_ma3'] = np.nanmean([instance.get(f'activity_day{j}', np.nan) for j in range(3, 6)])
                
                # Activity trend
                valid_activities = [act for act in activity_days if not pd.isna(act)]
                if len(valid_activities) >= 3:
                    valid_indices = [i for i, act in enumerate(activity_days) if not pd.isna(act)]
                    instance['activity_trend'] = np.polyfit(valid_indices, valid_activities, 1)[0]
            
            # Screen time features (if available)
            if any(f'screen_day{j}' in instance for j in range(1, 6)):
                instance['screen_ma3'] = np.nanmean([instance.get(f'screen_day{j}', np.nan) for j in range(3, 6)])
            
            # App usage patterns (for app categories if available)
            app_columns = [col for col in prev_day.index if 'appCat' in str(col)]
            if app_columns:
                # Total app usage over 5 days
                for app_col in app_columns:
                    app_values = [instance.get(f'{app_col}_day{j}', 0) for j in range(1, 6)]
                    app_values = [0 if pd.isna(val) else val for val in app_values]
                    instance[f'{app_col}_total'] = sum(app_values)
                
                # Calculate app diversity (number of different app categories used)
                app_diversity = 0
                for j in range(1, 6):
                    day_app_usage = {col: instance.get(f'{col}_day{j}', 0) for col in app_columns}
                    day_app_usage = {k: v for k, v in day_app_usage.items() if not pd.isna(v) and v > 0}
                    app_diversity += len(day_app_usage)
                instance['app_diversity'] = app_diversity
            
            instances.append(instance)
    
    # Create dataframe from instances
    features_df = pd.DataFrame(instances)
    
    # Handle any remaining missing values
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    features_df[numeric_cols] = features_df[numeric_cols].fillna(features_df[numeric_cols].mean())
    
    print(f"Created {len(features_df)} instances with temporal features")
    print(f"Feature categories: {len(features_df.columns) - 3} features created") 
    
    return features_df

def apply_classification(features_df):
    print("\n=== Task 2A: Classification ===")
    
    # Prepare the dataset for classification
    # Convert mood values to discrete classes
    features_df['mood_class'] = pd.cut(
        features_df['target_mood'],
        # NOTE: SINCE THE NUMBER ARE INTEGERS, ONE CLASS WILL HAVE MORE VALUES ASSIGNED. 
        # DISCUSS THIS IN REPORT. WE CHOSE LOW CAUSE IT WAS THE LEAST POPULATED.
        bins=[0, 4, 7, 10], 
        labels=['low', 'medium', 'high']
    )
    
    # Check class distribution
    class_counts = features_df['mood_class'].value_counts()
    print("\nClass distribution:")
    print(class_counts)
    
    # Identify classes with too few samples
    problematic_classes = class_counts[class_counts < 2].index.tolist()
    if problematic_classes:
        print(f"\nWarning: Classes with fewer than 2 samples: {problematic_classes}")
        print("These will be removed to allow stratification")
        
        # Filter out instances with problematic classes
        features_df = features_df[~features_df['mood_class'].isin(problematic_classes)]
        print(f"Remaining instances: {len(features_df)}")
        
        # If after removing problematic classes we have too few instances, disable stratification
        if len(features_df) < 10 or any(count < 2 for count in features_df['mood_class'].value_counts()):
            print("Too few instances remain. Disabling stratification.")
            stratify = None
        else:
            stratify = features_df['mood_class']
    else:
        stratify = features_df['mood_class']
    
    # Select features and target
    features = features_df.drop(['id', 'target_date', 'target_mood', 'mood_class'], axis=1)
    target = features_df['mood_class']
    
    # Handle any remaining missing values
    features = features.fillna(features.mean())
    
    # Standardize the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Split the data into training and testing sets
    if len(features_df) <= 1:
        print("ERROR: Not enough data for train/test split. Please review data preparation steps.")
        return None, None, None, None
    
    # Use stratify only if we have enough data in each class
    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, target, test_size=0.3, random_state=42, 
        stratify=stratify if isinstance(stratify, pd.Series) and len(stratify) > 1 else None
    )
    
    print(f"Training set: {X_train.shape[0]} instances")
    print(f"Testing set: {X_test.shape[0]} instances")
    
    # Apply classification algorithm 1: Random Forest
    print("\nTraining Random Forest classifier...")
    
    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    
    # Use GridSearchCV for hyperparameter optimization
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=min(5, np.min(np.bincount(pd.factorize(y_train)[0]))),  # Ensure CV doesn't exceed class counts
        scoring='accuracy'
    )
    
    rf_grid.fit(X_train, y_train)
    
    # Get best model and make predictions
    rf_best = rf_grid.best_estimator_
    y_pred_rf = rf_best.predict(X_test)
    
    print(f"Best Random Forest parameters: {rf_grid.best_params_}")
    print("\nRandom Forest Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_rf))
    
    # Feature importance
    feature_names = features.columns
    importances = rf_best.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\nTop 10 important features:")
    for i in range(min(10, len(feature_names))):
        print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
    
    # Classification algorithm 2: RNN with PyTorch
    print("\n=== Training PyTorch RNN classifier ===")
    
    # Prepare data for PyTorch
    # Encode targets
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    # Convert to PyTorch tensors
    # Reshape input to [batch_size, seq_len, input_size]
    X_train_tensor = torch.FloatTensor(X_train.reshape(X_train.shape[0], 1, X_train.shape[1]))
    y_train_tensor = torch.LongTensor(y_train_encoded)
    X_test_tensor = torch.FloatTensor(X_test.reshape(X_test.shape[0], 1, X_test.shape[1]))
    y_test_tensor = torch.LongTensor(y_test_encoded)
    
    # Create datasets and data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    batch_size = min(32, len(X_train) // 2)  # Adjust batch size based on data size
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize the model
    input_size = X_train.shape[1]
    hidden_size = 64
    num_layers = 2
    num_classes = len(np.unique(y_train_encoded))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    rnn_model = RNNClassifier(input_size, hidden_size, num_layers, num_classes).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(rnn_model.parameters(), lr=0.001)
    
    # Train the model
    num_epochs = 50
    print_every = max(1, num_epochs // 5)  # Print progress roughly 5 times
    
    # For early stopping
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    best_model_state = None
    
    print(f"\nTraining RNN model for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        rnn_model.train()
        total_loss = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Forward pass
            outputs = rnn_model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        rnn_model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                outputs = rnn_model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        val_accuracy = correct / total
        avg_val_loss = val_loss / len(test_loader)
        
        # Print progress
        if (epoch + 1) % print_every == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {total_loss/len(train_loader):.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}, '
                  f'Val Accuracy: {val_accuracy:.4f}')
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = rnn_model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load the best model
    if best_model_state:
        rnn_model.load_state_dict(best_model_state)
    
    # Evaluate the model
    rnn_model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        all_preds = []
        
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            outputs = rnn_model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
        
        test_acc = correct / total
        print(f'\nRNN Test Accuracy: {test_acc:.4f}')
    
    # Convert predictions back to original classes
    y_pred_rnn = label_encoder.inverse_transform(all_preds)
    
    # Print classification report
    print("\nRNN Classification Report:")
    print(classification_report(y_test, y_pred_rnn))
    
    # Compare models
    print("\n=== Model Comparison ===")
    print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
    print(f"RNN Accuracy: {accuracy_score(y_test, y_pred_rnn):.4f}")

    plot_confusion_matrix(y_test, y_pred_rf, y_pred_rnn, label_encoder)
        
    return rf_best, rnn_model, X_test, y_test, y_pred_rf, y_pred_rnn

def convert_dataframe_format(original_df):
    """
    Convert the original feature-engineered dataframe into a simplified format with consolidated columns.
    
    Args:
        original_df (pd.DataFrame): The original dataframe with detailed features
        
    Returns:
        pd.DataFrame: A simplified dataframe with consolidated features
    """
    # Create a copy to avoid modifying the original
    simplified_df = original_df[['id', 'target_date', 'target_mood']].copy()
    
    # Copy mood values for each day
    for day in range(1, 6):
        if f'mood_day{day}' in original_df.columns:
            simplified_df[f'mood_day{day}'] = original_df[f'mood_day{day}']
    
    # Copy screen time values for each day
    for day in range(1, 6):
        if f'screen_day{day}' in original_df.columns:
            simplified_df[f'screen_day{day}'] = original_df[f'screen_day{day}']
    
    # Calculate consolidated app usage for each day
    for day in range(1, 6):
        app_columns = [col for col in original_df.columns if f'appCat.' in col and f'_day{day}' in col]
        if app_columns:
            simplified_df[f'app_usage_day{day}'] = original_df[app_columns].sum(axis=1)
    
    # Copy mood analytics
    analytics_cols = ['mood_ma3', 'mood_ma5', 'mood_trend', 'mood_volatility']
    for col in analytics_cols:
        if col in original_df.columns:
            simplified_df[col] = original_df[col]
    
    # Reorder columns
    ordered_columns = [
        'id', 'target_date', 'target_mood',
        'mood_day5', 'app_usage_day5',
        'mood_day4', 'app_usage_day4',
        'mood_day3', 'app_usage_day3',
        'mood_day2', 'app_usage_day2',
        'mood_day1', 'app_usage_day1',
        'mood_ma3', 'mood_ma5', 'mood_trend', 'mood_volatility',
        'screen_day5', 'screen_day4', 'screen_day3', 'screen_day2', 'screen_day1'
    ]
    
    # Filter to only include columns that exist in the dataframe
    final_columns = [col for col in ordered_columns if col in simplified_df.columns]
    
    return simplified_df[final_columns]


# Main execution
if __name__ == "__main__":
    df = pd.read_csv('dataset_mood_smartphone.csv')
    print("Data loaded successfully!")

    # Task 1A: Exploratory Data Analysis
    data = exploratory_data_analysis(df)
    
    # Task 1B: Data Cleaning
    cleaned_data = data_cleaning(data)
    cleaned_data = pd.read_csv('cleaned_data.csv')

    plot_variable_over_time(data, cleaned_data, "appCat.social", "AS14.01")

    # save to csv for further analysis
    cleaned_data.to_csv('cleaned_data.csv', index=False)
    
    # Task 1C: Feature Engineering
    features_df = feature_engineering(cleaned_data)
    
    # Task 2A: Classification
    rf_best, rnn_model, X_test, y_test, y_pred_rf, y_pred_rnn = apply_classification(features_df)

    features_df = convert_dataframe_format(features_df)
    features_df.to_csv('data/features.csv', index=False)