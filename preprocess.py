from preprocess import BikePreprocessor, extract_brand, extract_cc, parse_mileage, parse_power_bhp, parse_kms, normalize_owner
# rebuild full_pipe using the imported BikePreprocessor, fit, then:
from joblib import dump
dump(full_pipe, "artifacts/full_pipeline.joblib")
