
from src.preprocess import build_preprocess_pipeline, NUM_FEATURES, CAT_FEATURES

def test_pipeline_columns():
    preprocess = build_preprocess_pipeline()
    # check that transformers are configured with expected columns
    cols = []
    for name, transformer, column_list in preprocess.transformers:
        cols.extend(column_list)
    assert set(cols) == set(NUM_FEATURES + CAT_FEATURES)
