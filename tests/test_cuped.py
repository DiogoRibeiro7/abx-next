import pandas as pd
from abx_next import ABFrame
from abx_next.analysis.cuped import cuped_adjust

def test_cuped_adds_column_and_theta_float():
    df = pd.DataFrame({
        "user_id": [1,2,3,4],
        "group": ["control","control","treatment","treatment"],
        "metric": [1.0, 1.2, 1.1, 1.3],
        "exposed": [True, True, True, True],
        "baseline": [0.9, 1.0, 1.0, 1.1],
    })
    ab = ABFrame(df); ab.validate()
    out, theta = cuped_adjust(ab, covariate=df["baseline"])
    assert "metric_cuped" in out.columns
    assert isinstance(theta, float)
