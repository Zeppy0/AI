import pandas as pd
from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt

# Load the datasets
file_path_1 = "FAOSTAT_data_en_11-1-2024.csv"
file_path_2 = "FAOSTAT_data_1-10-2022.csv"

data1 = pd.read_csv(file_path_1)
data2 = pd.read_csv(file_path_2)

# Enhanced Rule-Based Anomaly Detection
def enhanced_anomaly_detection(df, window=5, deviation_factor=1.5):
    # Calculate rolling mean and standard deviation
    df['Rolling_Mean'] = df['Value'].rolling(window=window, min_periods=1).mean()
    df['Rolling_Std'] = df['Value'].rolling(window=window, min_periods=1).std()

    # Define anomaly thresholds
    df['Anomaly'] = df.apply(lambda row: (
        'High Positive Anomaly' if row['Value'] > row['Rolling_Mean'] + (deviation_factor * row['Rolling_Std'])
        else 'High Negative Anomaly' if row['Value'] < row['Rolling_Mean'] - (deviation_factor * row['Rolling_Std'])
        else 'Normal'
    ), axis=1)
    df['Anomaly_Binary'] = df['Anomaly'].apply(lambda x: 1 if x != 'Normal' else 0)  # Binary label for ROC

    return df[['Area', 'Year', 'Months', 'Value', 'Anomaly', 'Anomaly_Binary']]

# Apply anomaly detection
data1_anomalies = enhanced_anomaly_detection(data1)
data2_anomalies = enhanced_anomaly_detection(data2)

# Enhanced Rule-Based Trend Categorization
def enhanced_trend_categorization(df):
    trend_results = []
    for area, group in df.groupby('Area'):
        group = group.sort_values(by='Year')
        start_value = group['Value'].iloc[0]
        end_value = group['Value'].iloc[-1]
        years = group['Year'].iloc[-1] - group['Year'].iloc[0]
        
        # Calculate percentage change and average annual change
        pct_change = (end_value - start_value) / abs(start_value) * 100 if start_value != 0 else 0
        avg_annual_change = (end_value - start_value) / years if years != 0 else 0

        # Categorize trends
        if pct_change > 20 and avg_annual_change > 0.5:
            trend = 'Strong Increasing Trend'
        elif pct_change > 5 and avg_annual_change > 0.1:
            trend = 'Moderate Increasing Trend'
        elif pct_change < -20 and avg_annual_change < -0.5:
            trend = 'Strong Decreasing Trend'
        elif pct_change < -5 and avg_annual_change < -0.1:
            trend = 'Moderate Decreasing Trend'
        else:
            trend = 'Stable Trend'

        trend_results.append({'Area': area, 'Trend': trend})

    return pd.DataFrame(trend_results)

# Apply trend categorization
data1_trends = enhanced_trend_categorization(data1)
data2_trends = enhanced_trend_categorization(data2)

# Function to calculate accuracy
def calculate_accuracy(true_values, predicted_values):
    return accuracy_score(true_values, predicted_values)

# ROC curve generation and display
def generate_roc_curve(true_values, predicted_probs, title):
    fpr, tpr, _ = roc_curve(true_values, predicted_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

# Generate ROC curves and calculate AUC
if 'True_Anomaly' in data1.columns:
    true_anomalies_data1 = data1['True_Anomaly'].apply(lambda x: 1 if x != 'Normal' else 0)
    predicted_probs_data1 = data1_anomalies['Anomaly_Binary']
    generate_roc_curve(true_anomalies_data1, predicted_probs_data1, "Data1 Anomaly Detection ROC Curve")

if 'True_Anomaly' in data2.columns:
    true_anomalies_data2 = data2['True_Anomaly'].apply(lambda x: 1 if x != 'Normal' else 0)
    predicted_probs_data2 = data2_anomalies['Anomaly_Binary']
    generate_roc_curve(true_anomalies_data2, predicted_probs_data2, "Data2 Anomaly Detection ROC Curve")

# Calculate anomaly detection accuracy
accuracy_data1_anomaly = None
accuracy_data2_anomaly = None
if 'True_Anomaly' in data1.columns:
    accuracy_data1_anomaly = calculate_accuracy(true_anomalies_data1, data1_anomalies['Anomaly_Binary'])
    print("Data1 Anomaly Detection Accuracy:", accuracy_data1_anomaly)

if 'True_Anomaly' in data2.columns:
    accuracy_data2_anomaly = calculate_accuracy(true_anomalies_data2, data2_anomalies['Anomaly_Binary'])
    print("Data2 Anomaly Detection Accuracy:", accuracy_data2_anomaly)

# Calculate trend categorization accuracy
accuracy_data1_trend = None
accuracy_data2_trend = None
if 'True_Trend' in data1.columns:
    data1_with_trend = pd.merge(data1, data1_trends, on='Area')
    accuracy_data1_trend = calculate_accuracy(data1_with_trend['True_Trend'], data1_with_trend['Trend'])
    print("Data1 Trend Categorization Accuracy:", accuracy_data1_trend)

if 'True_Trend' in data2.columns:
    data2_with_trend = pd.merge(data2, data2_trends, on='Area')
    accuracy_data2_trend = calculate_accuracy(data2_with_trend['True_Trend'], data2_with_trend['Trend'])
    print("Data2 Trend Categorization Accuracy:", accuracy_data2_trend)

# Display the enhanced anomalies and trends
print("\nData1 Enhanced Anomalies:\n", data1_anomalies.head())
print("\nData1 Enhanced Trends:\n", data1_trends.head())
print("\nData2 Enhanced Anomalies:\n", data2_anomalies.head())
print("\nData2 Enhanced Trends:\n", data2_trends.head())

# Calculate and display overall accuracy for both datasets
def calculate_overall_accuracy(anomaly_accuracy, trend_accuracy):
    accuracies = []
    if anomaly_accuracy is not None:
        accuracies.append(anomaly_accuracy)
    if trend_accuracy is not None:
        accuracies.append(trend_accuracy)
    
    if accuracies:
        return sum(accuracies) / len(accuracies)
    else:
        return 0  # No valid accuracy data

# Calculate overall accuracy for data1 and data2
overall_accuracy_data1 = calculate_overall_accuracy(accuracy_data1_anomaly, accuracy_data1_trend)
overall_accuracy_data2 = calculate_overall_accuracy(accuracy_data2_anomaly, accuracy_data2_trend)

# Display overall accuracy
print("\nData1 Overall Accuracy Rate: {:.2f}%".format(overall_accuracy_data1 * 100))
print("Data2 Overall Accuracy Rate: {:.2f}%".format(overall_accuracy_data2 * 100))
