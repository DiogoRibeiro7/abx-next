import pandas as pd

from abx_next import ABFrame
from abx_next.analysis.triggered import diff_in_means, filter_exposed


def test_filter_exposed_and_diff():
    df = pd.DataFrame({
        "user_id": [1,2,3,4],
        "group": ["control","control","treatment","treatment"],
        "metric": [1.0, 1.2, 1.1, 1.5],
        "exposed": [True, False, True, True],
    })
    ab = ABFrame(df)
    ab.validate()
    fx = filter_exposed(ab)
    assert fx.shape[0] == 3
    stats = diff_in_means(fx)
    assert set(stats) == {"n_c","n_t","mean_c","mean_t","diff","se","z"}
