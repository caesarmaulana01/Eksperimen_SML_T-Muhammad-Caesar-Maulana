from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from joblib import dump
import pandas as pd
import numpy as np

# Custom transformer untuk Label Encoding pada fitur dengan <= 2 nilai unik
class LabelEncoderTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.encoders = {}
        for col in X.columns:
            if X[col].nunique() <= 2 and X[col].dtype == 'object':
                le = LabelEncoder()
                le.fit(X[col])
                self.encoders[col] = le
        return self

    def transform(self, X):
        X_ = X.copy()
        for col, le in self.encoders.items():
            X_[col] = le.transform(X_[col])
        return X_

# Fungsi utama pipeline preprocessing
def preprocess_pipeline(data: pd.DataFrame, target_column: str, save_path: str):
    df = data.copy()

    # Pisahkan target
    y = df[target_column]
    X = df.drop(columns=[target_column])

    # Kolom numerik & kategorikal (sebelum encoding)
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    # Definisikan pipeline preprocessing
    preprocessor = ColumnTransformer(transformers=[
        # Label Encode kolom kategorikal dengan <= 2 nilai unik
        ('label_enc', Pipeline(steps=[
            ('label_encoder', LabelEncoderTransformer())
        ]), categorical_features),

        # Imputasi dan scaling kolom numerik
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', MinMaxScaler(feature_range=(0, 5)))
        ]), numeric_features),

        # OneHot Encoding untuk sisanya
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
        ]), categorical_features),
    ])

    # Bagi data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Fitting & transformasi
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Simpan pipeline
    dump(preprocessor, save_path)
    print(f"Pipeline preprocessing berhasil disimpan ke {save_path}")

    return X_train_transformed, X_test_transformed, y_train, y_test

# Panggil pipeline dan simpan hasilnya
if __name__ == '__main__':
    df = pd.read_csv('employee_data.csv')  # File input
    target_column = 'Attrition'
    save_pipeline_path = 'preprocessing/preprocessing_pipeline.joblib'

    X_train, X_test, y_train, y_test = preprocess_pipeline(
        df, target_column=target_column, save_path=save_pipeline_path
    )

    # Gabungkan hasil preprocessing (fitur + target)
    combined_X = np.vstack((X_train, X_test))
    combined_y = pd.concat([y_train, y_test], axis=0).values.reshape(-1, 1)
    final_data = np.hstack((combined_X, combined_y))

    # Simpan ke CSV
    pd.DataFrame(final_data).to_csv(
        'preprocessing/employee_data_preprocessed.csv', index=False, header=False
    )
    print("Preprocessed data berhasil disimpan ke preprocessing/employee_data_preprocessed.csv")
