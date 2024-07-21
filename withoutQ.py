import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, OneHotEncoder, LabelEncoder, RobustScaler, PowerTransformer
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.cluster import KMeans
import time

df = pd.read_csv("C:/Users/Saniya Walia/PycharmProjects/MLiot83-pandas/.venv/Scripts/model_training_dataset_deisel.csv")

numerical_features = df.select_dtypes(include=['number']).columns.tolist()
categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
datetime_columns = df.select_dtypes(include=['datetime64']).columns.tolist()

new_df = copy.deepcopy(df)

new_df[numerical_features] = new_df[numerical_features].fillna(new_df[numerical_features].median())
new_df_1 = new_df[numerical_features].dropna(axis=1, how='all')

Q1 = new_df[numerical_features].quantile(0.25)
Q3 = new_df[numerical_features].quantile(0.75)
IQR = Q3 - Q1
new_df[numerical_features] = new_df[numerical_features][~((new_df[numerical_features] < (Q1 - 1.5 * IQR)) | (new_df[numerical_features] > (Q3 + 1.5 * IQR))).any(axis=1)]

imputer = KNNImputer(n_neighbors=5)
imputed_numerical = imputer.fit_transform(df[numerical_features])
imputed_numerical_df = pd.DataFrame(imputed_numerical, columns=new_df_1.columns)
new_df_1 = imputed_numerical_df

new_df[categorical_features] = df[categorical_features].apply(lambda col: col.fillna(col.mode()[0]))

for col in datetime_columns:
    new_df[col] = pd.to_numeric(pd.to_datetime(df[col]))

high_cardinality_threshold = 10
encoded_dfs = []
for col in categorical_features:
    if new_df[col].nunique() >= high_cardinality_threshold:
        le = LabelEncoder()
        new_df[col] = le.fit_transform(new_df[col])
    else:
        encoder = OneHotEncoder(drop='first')
        encoded_categorical = encoder.fit_transform(new_df[[col]])
        encoded_categorical_df = pd.DataFrame(encoded_categorical.toarray(),
                                              columns=encoder.get_feature_names_out([col]))
        encoded_dfs.append(encoded_categorical_df)
        new_df.drop(columns=[col], inplace=True)

if encoded_dfs:
    encoded_categorical_df = pd.concat(encoded_dfs, axis=1)
    new_df = pd.concat([new_df, encoded_categorical_df], axis=1)

pt = PowerTransformer()
new_df_1 = pd.DataFrame(pt.fit_transform(new_df_1), columns=new_df_1.columns)

poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(new_df_1)
poly_feature_names = poly.get_feature_names_out(new_df_1.columns)
df_poly = pd.DataFrame(poly_features, columns=poly_feature_names)
new_df_1 = pd.concat([new_df_1, df_poly], axis=1)

scaler = RobustScaler()
minmax_scaler = MinMaxScaler()
scaled_data = minmax_scaler.fit_transform(scaler.fit_transform(new_df_1))

new_df_1 = pd.DataFrame(scaled_data, columns=new_df_1.columns)  # Convert scaled_data back to a DataFrame

new_df_1.rename(columns={'FuelEfficiency': 'target'}, inplace=True)

if new_df_1.isnull().sum().sum() > 0:
    raise ValueError("Data still contains NaN values after processing.")

y = new_df_1['target'].copy()

unique_values = len(y.nunique())
total_values = len(y)
unique_percentage = (unique_values / total_values) * 100

kmeans = KMeans(n_clusters=2, random_state=0).fit(y.values.reshape(-1, 1))
cluster_centers = kmeans.cluster_centers_
threshold = np.mean(cluster_centers)

if unique_percentage < threshold:
    task_type = 'classification'
else:
    task_type = 'regression'

X = new_df_1.drop(columns=['target'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classification_models = {
    'KNeighborsClassifier': KNeighborsClassifier(),
    'RandomForestClassifier': RandomForestClassifier(random_state=42)
}

regression_models = {
    'KNeighborsRegressor': KNeighborsRegressor(),
    'RandomForestRegressor': RandomForestRegressor(random_state=42)
}

evaluation_metric = input("Enter evaluation metric ('accuracy' or 'runtime'): ").strip().lower()


def evaluate_models(models, X_train, X_test, y_train, y_test, metric, task_type):
    results = {}
    for name, model in models.items():
        start_time = time.time()
        pipeline = Pipeline([
            ('model', model)
        ])
        pipeline.fit(X_train, y_train)
        runtime = time.time() - start_time

        if task_type == 'classification':
            y_pred = pipeline.predict(X_test)
            score = accuracy_score(y_test, y_pred)
        else:
            y_pred = pipeline.predict(X_test)
            score = r2_score(y_test, y_pred)

        results[name] = {'score': score, 'runtime': runtime, 'model': pipeline}

    return results


if task_type == 'classification':
    results = evaluate_models(classification_models, X_train, X_test, y_train, y_test, evaluation_metric, task_type)
else:
    results = evaluate_models(regression_models, X_train, X_test, y_train, y_test, evaluation_metric, task_type)

if evaluation_metric == 'accuracy':
    best_model_name = max(results, key=lambda x: results[x]['score'])
else:
    best_model_name = min(results, key=lambda x: results[x]['runtime'])

best_model = results[best_model_name]['model']
y_pred = best_model.predict(X_test)

print(f"Best Model: {best_model_name}")
print(f"Score: {results[best_model_name]['score']:.2f}")
print(f"Runtime: {results[best_model_name]['runtime']:.4f} seconds")

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title(f'Predicted vs Actual for {best_model_name}')
plt.show()
