"""
Flexible CSV handler for various formats
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import re

class CSVMapper:
    """Maps various CSV formats to standard schema"""
    
    # Common field mappings
    FIELD_MAPPINGS = {
        'age': ['age', 'patient_age', 'pt_age', 'edad', 'age_years'],
        'gender': ['gender', 'sex', 'patient_gender', 'pt_gender', 'sexo', 'm_f', 'male_female'],
        'appointment_date': ['appointment_date', 'appt_date', 'visit_date', 'fecha_cita', 'scheduled_date'],
        'no_show': ['no_show', 'noshow', 'missed', 'absence', 'did_not_attend', 'dna'],
        'readmitted': ['readmitted', 'readmission', 'readmit', 'readmitted_30days', 'readmission_30'],
        'length_of_stay': ['length_of_stay', 'los', 'days_in_hospital', 'stay_duration'],
        'diagnosis': ['diagnosis', 'diagnoses', 'dx', 'diag', 'diagnostic'],
        'medications': ['medications', 'meds', 'drugs', 'prescriptions', 'rx'],
    }
    
    @staticmethod
    def detect_delimiter(file_content: str) -> str:
        """Detect CSV delimiter"""
        delimiters = [',', ';', '\t', '|']
        delimiter_counts = {}
        
        first_line = file_content.split('\n')[0]
        for delimiter in delimiters:
            delimiter_counts[delimiter] = first_line.count(delimiter)
        
        return max(delimiter_counts, key=delimiter_counts.get)
    
    @staticmethod
    def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names"""
        df = df.copy()
        
        # Clean column names
        df.columns = [col.lower().strip().replace(' ', '_') for col in df.columns]
        df.columns = [re.sub(r'[^\w\s]', '', col) for col in df.columns]
        
        # Map to standard names
        for standard, variations in CSVMapper.FIELD_MAPPINGS.items():
            for col in df.columns:
                if col in variations:
                    df.rename(columns={col: standard}, inplace=True)
                    break
        
        return df
    
    @staticmethod
    def detect_target_column(df: pd.DataFrame) -> Optional[str]:
        """Auto-detect likely target column"""
        potential_targets = []
        
        for col in df.columns:
            # Binary columns are likely targets
            if df[col].nunique() == 2:
                potential_targets.append(col)
            
            # Common target patterns
            if any(pattern in col.lower() for pattern in ['target', 'label', 'outcome', 'result', 'class']):
                return col
            
            # Healthcare specific targets
            if any(pattern in col.lower() for pattern in ['no_show', 'readmit', 'mortality', 'diagnosis']):
                return col
        
        # Return binary column if only one exists
        if len(potential_targets) == 1:
            return potential_targets[0]
        
        return None
    
    @staticmethod
    def infer_data_types(df: pd.DataFrame) -> pd.DataFrame:
        """Infer and convert data types"""
        df = df.copy()
        
        for col in df.columns:
            # Try to convert to numeric
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                # Try to convert to datetime
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    # Keep as string
                    pass
        
        return df
    
    @staticmethod
    def process_csv(file_content: str, target_hint: Optional[str] = None) -> Dict[str, Any]:
        """Process CSV with automatic field detection"""
        import io
        
        # Detect delimiter
        delimiter = CSVMapper.detect_delimiter(file_content)
        
        # Read CSV
        df = pd.read_csv(io.StringIO(file_content), delimiter=delimiter)
        
        # Standardize columns
        df = CSVMapper.standardize_column_names(df)
        
        # Infer data types
        df = CSVMapper.infer_data_types(df)
        
        # Detect target
        if target_hint:
            target = target_hint
        else:
            target = CSVMapper.detect_target_column(df)
        
        # Generate report
        report = {
            'original_shape': df.shape,
            'delimiter_detected': delimiter,
            'columns_mapped': list(df.columns),
            'target_detected': target,
            'data_types': df.dtypes.astype(str).to_dict(),
            'sample_data': df.head(3).to_dict(),
            'processing_suggestions': []
        }
        
        # Add suggestions
        if df.isnull().sum().sum() > 0:
            report['processing_suggestions'].append('Data contains missing values - will be imputed')
        
        if any(df.select_dtypes(include=['object']).columns):
            report['processing_suggestions'].append('Categorical columns detected - will be encoded')
        
        return report, df

# Update API endpoint
@app.post("/data/analyze")
async def analyze_data_format(file: UploadFile = File(...)):
    """Analyze and map CSV format automatically"""
    try:
        content = await file.read()
        content_str = content.decode('utf-8')
        
        report, processed_df = CSVMapper.process_csv(content_str)
        
        return {
            "status": "success",
            "analysis": report,
            "ready_for_training": True if report['target_detected'] else False,
            "message": "Data analyzed successfully. " + 
                      (f"Target column '{report['target_detected']}' detected." 
                       if report['target_detected'] 
                       else "Please specify target column for training.")
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "suggestion": "Please ensure file is valid CSV/TSV format"
        }

@app.post("/train/auto")
async def train_auto_detect(
    file: UploadFile = File(...),
    model_id: str = "auto_model",
    target_hint: Optional[str] = None,
    model_type: str = "random_forest"
):
    """Train with automatic field detection"""
    try:
        content = await file.read()
        content_str = content.decode('utf-8')
        
        # Process CSV
        report, df = CSVMapper.process_csv(content_str, target_hint)
        
        if not report['target_detected']:
            raise HTTPException(
                status_code=400, 
                detail="Could not detect target column. Please specify with target_hint parameter."
            )
        
        target = report['target_detected']
        
        # Clean data
        cleaner = DataCleaner()
        df = cleaner.clean_dataframe(df, target_col=target)
        
        # Prepare features
        X = df.drop(columns=[target])
        y = df[target]
        
        # Encode categoricals
        X = cleaner.encode_categoricals(X)
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if model_type == "random_forest":
            model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        elif model_type == "gradient_boosting":
            model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
        else:
            model = LogisticRegression(max_iter=1000, random_state=42)
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Save model with mapping info
        model_data = {
            'model': model,
            'features': list(X.columns),
            'target': target,
            'model_type': model_type,
            'accuracy': float(accuracy),
            'field_mappings': report['columns_mapped'],
            'trained_at': datetime.now().isoformat()
        }
        
        MODELS[model_id] = model_data
        
        if model_storage:
            model_storage.save_model(model_id, model_data)
        
        return {
            "status": "success",
            "model_id": model_id,
            "accuracy": float(accuracy),
            "target_detected": target,
            "columns_mapped": report['columns_mapped'],
            "original_columns": list(pd.read_csv(io.StringIO(content_str), nrows=0).columns),
            "message": f"Model trained successfully with auto-detected target '{target}'"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
