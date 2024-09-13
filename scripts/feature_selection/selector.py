from dython.nominal import associations
from sklearn.feature_selection import mutual_info_classif, chi2, SelectKBest
import pandas as pd
import numpy as np

def select_vars(corr_df):

    melted_df = corr_df.reset_index().melt(id_vars='index', var_name='variable', value_name='corr_value')
    filtered_df = melted_df[(melted_df['index'] != melted_df['variable']) & (melted_df['corr_value'].abs() > 0.85)].copy()
    filtered_df[['var1', 'var2']] = filtered_df.apply(lambda row: sorted([row['index'], row['variable']]), axis=1, result_type='expand')
    filtered_df['pair'] = filtered_df['var1'] + ' vs ' + filtered_df['var2']
    result_df = filtered_df.drop_duplicates(subset='pair')[['pair', 'corr_value']].reset_index(drop=True)

    return result_df

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■


def choose_variable_to_drop(pair_df):

    var_count = {}

    # Count appearances of each variable
    for pair in pair_df['pair']:
        var1, var2 = pair.split(' vs ')
        var_count[var1] = var_count.get(var1, 0) + 1
        var_count[var2] = var_count.get(var2, 0) + 1

    # Determine the variable to drop based on the highest count
    variable_to_drop = max(var_count, key=var_count.get)

    return variable_to_drop

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■

    
def corr_comparison(result_df_1, result_df_2):
    # Extract pairs
    pairs_1 = set(result_df_1['pair'])
    pairs_2 = set(result_df_2['pair'])
    
    # Find common pairs
    common_pairs = pairs_1.intersection(pairs_2)
    
    # Find pairs unique to each DataFrame
    unique_to_df_1 = pairs_1.difference(pairs_2)
    unique_to_df_2 = pairs_2.difference(pairs_1)

    # Filter the DataFrames to get rows corresponding to common pairs and unique pairs
    common_pairs_df_1 = result_df_1[result_df_1['pair'].isin(common_pairs)]
    common_pairs_df_2 = result_df_2[result_df_2['pair'].isin(common_pairs)]
    unique_to_df_1 = result_df_1[result_df_1['pair'].isin(unique_to_df_1)]
    unique_to_df_2 = result_df_2[result_df_2['pair'].isin(unique_to_df_2)]
    
    # Merge the common pairs DataFrames to keep correlation values from both DataFrames
    common_pairs_df = pd.merge(common_pairs_df_1, common_pairs_df_2, on='pair', suffixes=('_df1', '_df2'))
    # Calculate the absolute difference and filter based on the threshold
    common_pairs_df['abs_diff'] = (common_pairs_df['corr_value_df1'] - common_pairs_df['corr_value_df2']).abs()
    filtered_common_pairs_df = common_pairs_df[common_pairs_df['abs_diff'] > 0.01]
    
    # Drop the 'abs_diff' column if no longer needed
    filtered_common_pairs_df = filtered_common_pairs_df.drop(columns=['abs_diff'], axis=1)
    
    return filtered_common_pairs_df, unique_to_df_1, unique_to_df_2

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■


def mutual_information(df):
    
    # Identify categorical columns
    categorical_cols = df.columns.tolist()
    
    # Convert categorical columns to category dtype if not already
    for col in df.columns:
        df[col] = df[col].astype('category')
    
    # Create a matrix to store mutual information
    mi_matrix = pd.DataFrame(index=categorical_cols, columns=categorical_cols)
    
    # Calculate mutual information for each pair of categorical columns
    for col1 in categorical_cols:
        for col2 in categorical_cols:
            if col1 == col2:
                mi_matrix.at[col1, col2] = 0.0
            else:
                # Use a temporary DataFrame to avoid modification of the original df
                temp_df = df[[col1, col2]].dropna()
                mi = mutual_info_classif(
                    temp_df[[col1]].apply(lambda x: x.cat.codes), 
                    temp_df[col2].cat.codes, 
                    discrete_features=True
                )
                mi_matrix.at[col1, col2] = mi[0]
    
    # Display the mutual information matrix
    return mi_matrix