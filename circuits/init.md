I need you to write a Python function that creates engineered features based on the macro conditions we discussed. 

Requirements:

1. Write a single function with this exact signature:
   def engineer_features(df: pd.DataFrame) -> pd.DataFrame:

2. For EACH of the 24 conditions, create a boolean column named `cond_01`, `cond_02`, ... `cond_24` that is True when that condition is met.

3. Use pandas operations only. Handle missing/empty values with .fillna() or .isna() as appropriate.

4. For string comparisons, use case-insensitive matching: df['col'].str.lower().str.contains('value')

5. For date comparisons, assume date columns are already datetime type. If comparing to today, use pd.Timestamp.now().

6. For numeric comparisons involving empty strings, convert first: pd.to_numeric(df['col'], errors='coerce')

7. Return the dataframe with ALL original columns PLUS the new cond_XX boolean columns.

8. Add a comment above each condition with the plain-English description from the macro.

Example structure:
```python
import pandas as pd

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Condition 1: Invoice amount is greater than zero AND status is active
    df['cond_01'] = (pd.to_numeric(df['invoice'], errors='coerce') > 0) & (df['status'].str.lower() == 'active')
    
    # Condition 2: ...
    df['cond_02'] = ...
    
    return df
```

Write the complete function for all 24 conditions based on the macro logic I provided earlier. Do not summarize or skip any conditions.

train.py
```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
import joblib
import json

# You'll paste the generated function here
from feature_engineering import engineer_features

def load_and_combine_reports(file_paths: list[str]) -> pd.DataFrame:
    """Load multiple Excel reports and combine them."""
    dfs = []
    for i, path in enumerate(file_paths):
        df = pd.read_excel(path)
        df['report_number'] = i + 1  # Track which report each row came from
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def prepare_features(df: pd.DataFrame, categorical_cols: list[str], label_col: str):
    """Encode categorical features and separate X/y."""
    df = engineer_features(df)
    
    # Identify boolean condition columns
    cond_cols = [c for c in df.columns if c.startswith('cond_')]
    
    # Encode remaining categorical columns (not one-hot, just label encode for trees)
    encoders = {}
    for col in categorical_cols:
        if col in df.columns and col != label_col:
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col].astype(str).fillna('MISSING'))
            encoders[col] = le
    
    encoded_cols = [c + '_encoded' for c in categorical_cols if c in df.columns and c != label_col]
    
    # Numeric columns (exclude IDs, labels, non-features)
    exclude = {label_col, 'circuit_id', 'report_number'} | set(categorical_cols)
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
    
    feature_cols = cond_cols + encoded_cols + numeric_cols
    
    X = df[feature_cols].fillna(0)
    y = df[label_col]
    
    return X, y, feature_cols, encoders

def train_model(X, y, feature_cols):
    """Train Random Forest with class weights to handle imbalance."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluation
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    
    print(f"Train accuracy: {train_acc:.3f}")
    print(f"Test accuracy:  {test_acc:.3f}")
    print(f"CV accuracy:    {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nTop 10 features:")
    print(importance.head(10).to_string(index=False))
    
    return model

def main():
    # === CONFIGURE THESE ===
    file_paths = ['report_01.xlsx', 'report_02.xlsx']  # Add all report paths
    label_col = 'label'  # Your label column name
    categorical_cols = ['col1', 'col2']  # List your categorical columns
    circuit_id_col = 'circuit_id'  # Column identifying unique circuits
    # ========================
    
    print("Loading data...")
    df = load_and_combine_reports(file_paths)
    print(f"Total rows: {len(df)}")
    print(f"Unique circuits: {df[circuit_id_col].nunique()}")
    print(f"Label distribution:\n{df[label_col].value_counts(normalize=True)}")
    
    print("\nPreparing features...")
    X, y, feature_cols, encoders = prepare_features(df, categorical_cols, label_col)
    print(f"Feature count: {len(feature_cols)}")
    
    print("\nTraining model...")
    model = train_model(X, y, feature_cols)
    
    # Save artifacts
    joblib.dump(model, 'model.joblib')
    joblib.dump(encoders, 'encoders.joblib')
    with open('feature_cols.json', 'w') as f:
        json.dump(feature_cols, f)
    
    print("\nSaved: model.joblib, encoders.joblib, feature_cols.json")

if __name__ == '__main__':
    main()
```

evaluate.py
```python
import pandas as pd
import numpy as np
import joblib
import json

from feature_engineering import engineer_features

def load_artifacts():
    model = joblib.load('model.joblib')
    encoders = joblib.load('encoders.joblib')
    with open('feature_cols.json') as f:
        feature_cols = json.load(f)
    return model, encoders, feature_cols

def prepare_features_for_eval(df: pd.DataFrame, encoders: dict, feature_cols: list):
    """Apply same transformations as training."""
    df = engineer_features(df)
    
    for col, le in encoders.items():
        if col in df.columns:
            # Handle unseen categories
            df[col + '_encoded'] = df[col].astype(str).fillna('MISSING').apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
    
    X = df[feature_cols].fillna(0)
    return X

def flag_anomalies(df: pd.DataFrame, model, X, label_col: str, circuit_id_col: str):
    """Flag circuits based on model disagreement and uncertainty."""
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    df = df.copy()
    df['ml_prediction'] = predictions
    df['ml_confidence'] = probabilities.max(axis=1)
    df['ml_entropy'] = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)
    
    # Flag 1: Model disagrees with macro label
    df['flag_disagreement'] = df[label_col] != df['ml_prediction']
    
    # Flag 2: Low confidence (bottom 10%)
    confidence_threshold = df['ml_confidence'].quantile(0.10)
    df['flag_low_confidence'] = df['ml_confidence'] < confidence_threshold
    
    # Flag 3: High entropy (top 10% - prediction spread across classes)
    entropy_threshold = df['ml_entropy'].quantile(0.90)
    df['flag_high_entropy'] = df['ml_entropy'] > entropy_threshold
    
    # Flag 4: Temporal anomaly - label changed between reports for same circuit
    df = df.sort_values([circuit_id_col, 'report_number'])
    df['prev_label'] = df.groupby(circuit_id_col)[label_col].shift(1)
    df['flag_label_changed'] = (df['prev_label'].notna()) & (df[label_col] != df['prev_label'])
    
    # Combined flag
    df['flag_any'] = (
        df['flag_disagreement'] | 
        df['flag_low_confidence'] | 
        df['flag_high_entropy'] |
        df['flag_label_changed']
    )
    
    return df

def main():
    # === CONFIGURE THESE ===
    input_file = 'report_to_evaluate.xlsx'  # Or combined reports
    label_col = 'label'
    circuit_id_col = 'circuit_id'
    output_file = 'flagged_circuits.csv'
    # ========================
    
    print("Loading model artifacts...")
    model, encoders, feature_cols = load_artifacts()
    
    print("Loading data...")
    df = pd.read_excel(input_file)
    
    # If evaluating multiple reports, use load_and_combine_reports from train.py
    if 'report_number' not in df.columns:
        df['report_number'] = 1
    
    print("Preparing features...")
    X = prepare_features_for_eval(df, encoders, feature_cols)
    
    print("Flagging anomalies...")
    results = flag_anomalies(df, model, X, label_col, circuit_id_col)
    
    # Summary
    print(f"\nTotal circuits: {len(results)}")
    print(f"Flagged (any):        {results['flag_any'].sum()} ({results['flag_any'].mean():.1%})")
    print(f"  - Disagreement:     {results['flag_disagreement'].sum()}")
    print(f"  - Low confidence:   {results['flag_low_confidence'].sum()}")
    print(f"  - High entropy:     {results['flag_high_entropy'].sum()}")
    print(f"  - Label changed:    {results['flag_label_changed'].sum()}")
    
    # Export flagged only
    flagged = results[results['flag_any']].sort_values('ml_confidence')
    flagged.to_csv(output_file, index=False)
    print(f"\nExported {len(flagged)} flagged circuits to {output_file}")

if __name__ == '__main__':
    main()
```

feature_engineering.py stub
```python
import pandas as pd

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Paste the function generated by your internal model here.
    """
    df = df.copy()
    
    # Condition 1: [description]
    # df['cond_01'] = ...
    
    # ... conditions 2-24 ...
    
    return df
```
