
# CSV Mapper for flexible format handling
import re

class CSVMapper:
    """Maps various CSV formats to standard schema"""
    
    FIELD_MAPPINGS = {
        'age': ['age', 'patient_age', 'pt_age', 'edad', 'age_years'],
        'gender': ['gender', 'sex', 'patient_gender', 'pt_gender', 'sexo', 'm_f', 'male_female'],
        'appointment_date': ['appointment_date', 'appt_date', 'visit_date', 'fecha_cita', 'scheduled_date'],
        'no_show': ['no_show', 'noshow', 'missed', 'absence', 'did_not_attend', 'dna'],
        'readmitted': ['readmitted', 'readmission', 'readmit', 'readmitted_30days', 'readmission_30'],
        'length_of_stay': ['length_of_stay', 'los', 'days_in_hospital', 'stay_duration'],
    }
    
    @staticmethod
    def detect_delimiter(file_content: str) -> str:
        delimiters = [',', ';', '\t', '|']
        delimiter_counts = {}
        first_line = file_content.split('\n')[0]
        for delimiter in delimiters:
            delimiter_counts[delimiter] = first_line.count(delimiter)
        return max(delimiter_counts, key=delimiter_counts.get)
    
    @staticmethod
    def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [col.lower().strip().replace(' ', '_') for col in df.columns]
        df.columns = [re.sub(r'[^\w\s]', '', col) for col in df.columns]
        
        for standard, variations in CSVMapper.FIELD_MAPPINGS.items():
            for col in df.columns:
                if col in variations:
                    df.rename(columns={col: standard}, inplace=True)
                    break
        return df
    
    @staticmethod
    def detect_target_column(df: pd.DataFrame) -> Optional[str]:
        potential_targets = []
        for col in df.columns:
            if df[col].nunique() == 2:
                potential_targets.append(col)
            if any(pattern in col.lower() for pattern in ['target', 'label', 'outcome', 'result', 'class', 'no_show', 'readmit', 'mortality']):
                return col
        if len(potential_targets) == 1:
            return potential_targets[0]
        return None
    
    @staticmethod
    def infer_data_types(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    pass
        return df
    
    @staticmethod
    def process_csv(file_content: str, target_hint: Optional[str] = None) -> tuple:
        delimiter = CSVMapper.detect_delimiter(file_content)
        df = pd.read_csv(io.StringIO(file_content), delimiter=delimiter)
        df = CSVMapper.standardize_column_names(df)
        df = CSVMapper.infer_data_types(df)
        
        if target_hint:
            target = target_hint
        else:
            target = CSVMapper.detect_target_column(df)
        
        report = {
            'original_shape': df.shape,
            'delimiter_detected': delimiter,
            'columns_mapped': list(df.columns),
            'target_detected': target,
            'data_types': df.dtypes.astype(str).to_dict(),
            'sample_data': df.head(3).to_dict(),
            'processing_suggestions': []
        }
        
        if df.isnull().sum().sum() > 0:
            report['processing_suggestions'].append('Data contains missing values - will be imputed')
        if any(df.select_dtypes(include=['object']).columns):
            report['processing_suggestions'].append('Categorical columns detected - will be encoded')
        
        return report, df
# CSV handler included below

# Add these endpoints to your API

@app.post("/data/analyze")
async def analyze_data_format(file: UploadFile = File(...)):
    """Analyze CSV format and suggest mappings"""
    try:
        content = await file.read()
        content_str = content.decode('utf-8')
        
        report, processed_df = CSVMapper.process_csv(content_str)
        
        return {
            "status": "success",
            "analysis": report,
            "ready_for_training": True if report['target_detected'] else False
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/train/auto", dependencies=[Depends(verify_api_key)])
async def train_auto_format(
    file: UploadFile = File(...),
    model_id: str = "auto_model",
    target_hint: Optional[str] = None
):
    """Train with automatic format detection"""
    try:
        content = await file.read()
        mapper = CSVMapper()
        
        # Auto-process the CSV
        report, df = mapper.process_csv(content.decode('utf-8'), target_hint)
        
        if not report['target_detected']:
            return {"error": "Cannot detect target. Please specify target_hint parameter"}
        
        # Continue with standard training using detected target
        # ... rest of training code
        
        return {
            "status": "success",
            "model_id": model_id,
            "detected_format": report
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
