import json

with open('pred_m.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # Update Feature Engineering cell
        if "cols_to_drop = ['UDI', 'Product ID', 'Type', 'Target', 'Failure Type']" in source:
            cell['source'] = [
                "cols_to_drop = ['UDI', 'Product ID', 'Target', 'Failure Type']\n",
                "X = df.drop(columns=cols_to_drop, errors='ignore')\n",
                "y = df['Failure Type']\n",
                "\n",
                "# Feature Engineering\n",
                "X = pd.get_dummies(X, columns=['Type'])\n",
                "X['Temp_Diff'] = X['Process temperature [K]'] - X['Air temperature [K]']\n",
                "X['Power'] = X['Rotational speed [rpm]'] * X['Torque [Nm]']\n",
                "X['Strain'] = X['Tool wear [min]'] * X['Torque [Nm]']\n"
            ]
            
        # Update SMOTE cell to not use SMOTE
        elif "# Handle Class Imbalance using SMOTE" in source:
            cell['source'] = [
                "# We will use Random Forest's class_weight='balanced_subsample' instead of SMOTE\n",
                "# This is often better for extreme tabular imbalances\n",
                "X_train_resampled = X_train_scaled\n",
                "y_train_resampled = y_train_encoded"
            ]
            
        # Update Model Training cell
        elif "# Train Random Forest Classifier with class weights" in source:
            cell['source'] = [
                "# Train an optimized Random Forest Classifier\n",
                "from sklearn.ensemble import RandomForestClassifier\n",
                "from sklearn.metrics import classification_report, confusion_matrix, accuracy_score\n",
                "\n",
                "rf_model = RandomForestClassifier(\n",
                "    n_estimators=200,\n",
                "    max_depth=15,\n",
                "    min_samples_split=5,\n",
                "    min_samples_leaf=2,\n",
                "    class_weight='balanced_subsample',\n",
                "    random_state=42,\n",
                "    n_jobs=-1\n",
                ")\n",
                "\n",
                "rf_model.fit(X_train_resampled, y_train_resampled)\n",
                "print(\"Optimized Random Forest model trained successfully!\")"
            ]

with open('pred_m.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
