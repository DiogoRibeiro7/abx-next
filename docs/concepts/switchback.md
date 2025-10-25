# Switchback designs

Switchbacks randomize by time blocks instead of by user to mitigate network
effects. The helpers in `abx_next.design.switchback` validate periods and label
event logs so downstream queries stay tidy.

```python
import pandas as pd

from abx_next import assign_switchback
from abx_next.design.switchback import label_events_by_period, validate_period

validate_period("1h")

timestamps = pd.date_range("2024-05-01", periods=6, freq="30min")
assignments = assign_switchback(timestamps, period="1h", seed=7)
print(assignments)

events = pd.DataFrame(
    {
        "event_id": range(1, 7),
        "ts": timestamps,
        "metric": [1.0, 0.8, 1.2, 1.1, 0.9, 1.3],
    }
)

labeled = label_events_by_period(events, ts_col="ts", period_assign=assignments)
print(labeled)
```

Use the labeled output to compute per-period aggregates or to join back into
warehouse tables. The utilities guarantee timezone alignment and will raise
explicit errors when logs drift from the intended cadence.
