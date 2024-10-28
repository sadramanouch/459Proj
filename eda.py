import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def performEDA(X, y):
    # Combine features and target into one DataFrame
    data = X.copy()
    data['quality'] = y.reset_index(drop=True)
    
    # Basic Information
    print("\nBasic Information:")
    print(data.info())
    
    # Statistical Summary
    print("\nStatistical Summary:")
    print(data.describe())
    
    # Check for Missing Values
    print("\nMissing Values:")
    print(data.isnull().sum())
    
    # Create a directory to save plots
    import os
    plots_dir = 'eda_plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    # Plot Histograms of Key Features
    data.hist(bins=15, figsize=(15, 10))
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'histograms.png'))
    plt.close()
    
    # Box Plots to Detect Outliers
    plt.figure(figsize=(15, 10))
    sns.boxplot(data=data.drop('quality', axis=1))
    plt.title('Box Plot of Features')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'boxplots.png'))
    plt.close()
    
    # Correlation Heatmap
    plt.figure(figsize=(12, 10))
    corr_matrix = data.corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'correlation_heatmap.png'))
    plt.close()
    
    # Distribution of Target Variable
    plt.figure(figsize=(8, 6))
    sns.countplot(x='quality', data=data)
    plt.title('Distribution of Wine Quality')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'quality_distribution.png'))
    plt.close()
    
    # Pairplot for Relationships between Features
    sns.pairplot(data, hue='quality', vars=data.columns[:-1], corner=True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'pairplot.png'))
    plt.close()
    
    # Print Correlation with Target Variable
    print("\nCorrelation with Target Variable:")
    print(corr_matrix['quality'].sort_values(ascending=False))
