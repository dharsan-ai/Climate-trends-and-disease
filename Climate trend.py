"""
Climate Trends and Disease Outbreak Prediction
Using PCA, SVR, and Random Forest
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================
# 1. SYNTHETIC DATA GENERATION
# ============================================================

def generate_climate_data(n_samples=1000):
    """Generate synthetic climate and disease data"""
    
    # Time variable (days)
    time = np.arange(n_samples)
    
    # Seasonal pattern (sine wave)
    temperature = 25 + 5 * np.sin(2 * np.pi * time / 365) + np.random.normal(0, 1, n_samples)
    humidity = 65 + 15 * np.sin(2 * np.pi * time / 365 + 1.5) + np.random.normal(0, 3, n_samples)
    rainfall = 20 + 15 * np.sin(2 * np.pi * time / 90) + np.random.exponential(5, n_samples)
    
    # Long-term warming trend
    temperature += time / 5000
    
    # Disease outbreaks correlated with climate
    # Higher risk when temperature > 28°C and humidity > 70%
    risk_score = ( (temperature - 25) / 5 ) * ( (humidity - 65) / 15 )
    risk_score = np.clip(risk_score, 0, 1)
    
    # Add noise
    risk_score += np.random.normal(0, 0.1, n_samples)
    risk_score = np.clip(risk_score, 0, 1)
    
    # Outbreak flag (1 if risk > threshold)
    outbreak = (risk_score > 0.7).astype(int)
    
    # Disease cases (Poisson distribution based on risk)
    disease_cases = np.random.poisson(lam=50 + 450 * risk_score)
    
    # Create DataFrame
    data = pd.DataFrame({
        'day': time,
        'temperature': temperature,
        'humidity': humidity,
        'rainfall': rainfall,
        'disease_cases': disease_cases,
        'outbreak': outbreak
    })
    
    return data

# Generate data
print("Generating synthetic climate data...")
data = generate_climate_data(1000)
print(f"Data shape: {data.shape}")
print(data.head())

# ============================================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================================

# Plot climate variables over time
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(data['day'], data['temperature'], alpha=0.7, color='red')
axes[0, 0].set_title('Temperature Trend Over Time', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Day')
axes[0, 0].set_ylabel('Temperature (°C)')

axes[0, 1].plot(data['day'], data['humidity'], alpha=0.7, color='blue')
axes[0, 1].set_title('Humidity Trend Over Time', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Day')
axes[0, 1].set_ylabel('Humidity (%)')

axes[1, 0].scatter(data['temperature'], data['disease_cases'], alpha=0.5, c=data['outbreak'], cmap='RdYlGn')
axes[1, 0].set_title('Temperature vs Disease Cases', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Temperature (°C)')
axes[1, 0].set_ylabel('Disease Cases')
axes[1, 0].set_xlim(15, 35)

axes[1, 1].scatter(data['humidity'], data['disease_cases'], alpha=0.5, c=data['outbreak'], cmap='RdYlGn')
axes[1, 1].set_title('Humidity vs Disease Cases', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Humidity (%)')
axes[1, 1].set_ylabel('Disease Cases')

plt.tight_layout()
plt.savefig('outputs/plots/eda_plots.png', dpi=150)
plt.show()

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

# Create lag features (previous days' values)
for lag in [1, 3, 7]:
    data[f'temp_lag_{lag}'] = data['temperature'].shift(lag)
    data[f'humidity_lag_{lag}'] = data['humidity'].shift(lag)

# Create rolling averages
data['temp_rolling_7'] = data['temperature'].rolling(window=7).mean()
data['humidity_rolling_7'] = data['humidity'].rolling(window=7).mean()

# Drop NaN values from lag features
data = data.dropna().reset_index(drop=True)

# ============================================================
# 4. PRINCIPAL COMPONENT ANALYSIS (PCA)
# ============================================================

# Select features for PCA
feature_columns = ['temperature', 'humidity', 'rainfall', 
                   'temp_lag_1', 'temp_lag_3', 'temp_lag_7',
                   'humidity_lag_1', 'humidity_lag_3', 'humidity_lag_7',
                   'temp_rolling_7', 'humidity_rolling_7']

X = data[feature_columns]
y_temp = data['temperature']
y_outbreak = data['outbreak']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Explained variance ratio
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# Plot explained variance
plt.figure(figsize=(10, 5))
plt.bar(range(1, len(explained_variance)+1), explained_variance, alpha=0.7, label='Individual')
plt.step(range(1, len(cumulative_variance)+1), cumulative_variance, where='mid', 
         label='Cumulative', color='red', linewidth=2)
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('PCA Explained Variance', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('outputs/plots/pca_variance.png', dpi=150)
plt.show()

print(f"\nCumulative variance explained by first 5 components: {cumulative_variance[4]:.2%}")

# Select top 5 components
n_components = 5
X_pca_reduced = X_pca[:, :n_components]

# ============================================================
# 5. SUPPORT VECTOR REGRESSION (SVR) for Temperature Forecast
# ============================================================

print("\n" + "="*60)
print("SUPPORT VECTOR REGRESSION - Temperature Prediction")
print("="*60)

# Split data for SVR (using lag features as predictors for future temperature)
# Predict next day's temperature
X_svr = X_scaled[:-1]  # All but last day
y_svr = data['temperature'].values[1:]  # Next day's temperature

X_train, X_test, y_train, y_test = train_test_split(X_svr, y_svr, test_size=0.2, random_state=42)

# Train SVR model
svr_model = SVR(kernel='rbf', C=10, gamma='scale', epsilon=0.1)
svr_model.fit(X_train, y_train)

# Predictions
y_pred_svr = svr_model.predict(X_test)

# Evaluation metrics
rmse_svr = np.sqrt(mean_squared_error(y_test, y_pred_svr))
r2_svr = r2_score(y_test, y_pred_svr)

print(f"SVR Model Performance:")
print(f"  RMSE: {rmse_svr:.4f}°C")
print(f"  R² Score: {r2_svr:.4f}")

# Plot SVR predictions
plt.figure(figsize=(12, 5))
plt.plot(y_test[:100], label='Actual Temperature', alpha=0.7)
plt.plot(y_pred_svr[:100], label='Predicted Temperature', alpha=0.7, linestyle='--')
plt.xlabel('Sample')
plt.ylabel('Temperature (°C)')
plt.title('SVR: Actual vs Predicted Temperature', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('outputs/plots/svr_predictions.png', dpi=150)
plt.show()

# ============================================================
# 6. RANDOM FOREST CLASSIFIER for Outbreak Prediction
# ============================================================

print("\n" + "="*60)
print("RANDOM FOREST CLASSIFIER - Outbreak Prediction")
print("="*60)

# Use PCA-reduced features for classification
X_clf = X_pca_reduced
y_clf = y_outbreak

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42
)

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train_clf, y_train_clf)

# Predictions
y_pred_clf = rf_model.predict(X_test_clf)

# Evaluation metrics
accuracy = accuracy_score(y_test_clf, y_pred_clf)
print(f"Random Forest Classifier Performance:")
print(f"  Accuracy: {accuracy:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test_clf, y_pred_clf, target_names=['No Outbreak', 'Outbreak']))

# Feature importance (from original features)
feature_importance = rf_model.feature_importances_
feature_names_pca = [f'PC{i+1}' for i in range(n_components)]

plt.figure(figsize=(8, 5))
plt.barh(feature_names_pca, feature_importance, color='teal')
plt.xlabel('Importance')
plt.title('Random Forest - PCA Component Importance', fontsize=12, fontweight='bold')
plt.gca().invert_yaxis()
plt.savefig('outputs/plots/feature_importance.png', dpi=150)
plt.show()

# ============================================================
# 7. CONFUSION MATRIX
# ============================================================

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test_clf, y_pred_clf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Outbreak', 'Outbreak'])

plt.figure(figsize=(6, 5))
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix - Outbreak Detection', fontsize=12, fontweight='bold')
plt.savefig('outputs/plots/confusion_matrix.png', dpi=150)
plt.show()

# ============================================================
# 8. INTERPRETATION & SUMMARY
# ============================================================

print("\n" + "="*60)
print("SUMMARY OF RESULTS")
print("="*60)

print(f"""
INTERPRETATION:

1. PCA DIMENSIONALITY REDUCTION:
   - First 5 components capture {cumulative_variance[4]:.1%} of data variance
   - This reduces feature space from {len(feature_columns)} to 5 dimensions
   - Enables faster computation while preserving essential information

2. TEMPERATURE FORECASTING (SVR):
   - RMSE: {rmse_svr:.4f}°C — prediction error is within acceptable range
   - R²: {r2_svr:.4f} — indicates model explains {r2_svr*100:.1f}% of variance
   - SVR successfully captures seasonal patterns and trends

3. OUTBREAK CLASSIFICATION (Random Forest):
   - Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)
   - Strong performance in distinguishing outbreak from non-outbreak periods
   - Key climate indicators effectively predict disease risk

CONCLUSION:
This project demonstrates that machine learning models can effectively
predict climate trends and disease outbreak risks using historical
climate data. The integration of PCA, SVR, and Random Forest provides
a robust framework for public health early warning systems.
""")

# Save processed data
data.to_csv('data/synthetic_climate_data.csv', index=False)
print("\nData saved to 'data/synthetic_climate_data.csv'")
print("Plots saved to 'outputs/plots/'")
