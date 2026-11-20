"""Online shoppers session-level Revenue classification; primary metric log loss.

Session aggregates, including PageValues, are retrospective predictors. No
claim of pre-session purchase forecasting or verified visitor independence.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

NUMERIC = ('Administrative','Administrative_Duration','Informational','Informational_Duration',
           'ProductRelated','ProductRelated_Duration','BounceRates','ExitRates','PageValues','SpecialDay')
CATEGORICAL = ('Month','OperatingSystems','Browser','Region','TrafficType','VisitorType','Weekend')
FEATURES = (*NUMERIC,*CATEGORICAL)
MONTHS = {'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'June':6,'Jun':6,'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}


def binary(series, name, missing=False):
    values = series.astype('string').str.strip().str.lower().map({'true':1,'false':0,'1':1,'0':0})
    if (values.isna() & series.notna()).any() or (not missing and values.isna().any()):
        raise ValueError(f'{name} must contain Boolean or 0/1 values')
    return values


def prepare_data(raw):
    required = {'Revenue',*FEATURES}
    if not required.issubset(raw) or not raw.index.is_unique:
        raise ValueError(f'Require unique source indices and shoppers columns; missing {sorted(required-set(raw))}')
    if raw.Month.isna().any() or not raw.Month.isin(MONTHS).all():
        raise ValueError('Month must contain recognized, nonmissing month names; no chronology can be inferred from row order')
    profile = dict(source_rows=len(raw),raw_exact_duplicates_removed=int(raw.duplicated().sum()),
        dataset='Online shoppers purchasing intention',target='Revenue',primary_metric='log_loss',
        date_resolution='Month only; no recorded full date/year',source_missing_values=raw.isna().sum().to_dict())
    frame = raw.drop_duplicates().copy()
    frame['Revenue'] = binary(frame.Revenue,'Revenue').astype(int)
    if set(frame.Revenue.unique())!={0,1}:raise ValueError('Both Revenue classes required')
    frame['month_order'] = frame.Month.map(MONTHS).map(lambda x:f'{x:02d}')
    for column in NUMERIC:
        frame[column] = pd.to_numeric(frame[column],errors='raise').astype(float)
        if np.isinf(frame[column]).any():raise ValueError(f'Infinite {column}')
    frame['Weekend'] = binary(frame.Weekend,'Weekend',missing=True).map({0:'False',1:'True'})
    for column in CATEGORICAL:
        frame[column] = frame[column].astype('string').astype(object).where(frame[column].notna(),np.nan)
    profile.update(eligible_rows=len(frame),positives=int(frame.Revenue.sum()),positive_rate=float(frame.Revenue.mean()))
    return frame[['Revenue','month_order',*FEATURES]],profile


def build_models():
    # Reuse the frozen seven-model configuration, with explicit categorical IDs.
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from dataset_models import build_models as build
    template = pd.DataFrame({c:pd.Series(dtype=float if c in NUMERIC else object) for c in FEATURES})
    return build(template)


if __name__=='__main__':
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from benchmark_a import main
    main('shoppers')
