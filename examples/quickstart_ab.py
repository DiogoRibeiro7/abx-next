import pandas as pd

from abx_next import ABFrame, cuped_adjust, diff_in_means, filter_exposed, srm_from_frame

# Fake data
df = pd.DataFrame({
    "user_id": range(1000),
    "group": ["control"] * 500 + ["treatment"] * 500,
    "metric": pd.Series(range(1000)).astype(float) * 0.01 + 0.5,
    "exposed": [True] * 1000,
    "baseline": pd.Series(range(1000)).astype(float) * 0.005 + 0.2,
})

ab = ABFrame(df)
ab.validate()

# Guardrail: SRM
print(srm_from_frame(df))

# CUPED (using baseline as covariate)
df_adj, theta = cuped_adjust(ab, covariate=df["baseline"])
print("theta =", theta)

# Triggered + diff-in-means on adjusted metric
df_exp = filter_exposed(ABFrame(df_adj))
print(diff_in_means(df_exp, use_column="metric_cuped"))
