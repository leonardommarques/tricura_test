from pathlib import Path

import pandas as pd
import pandas as pd

# Show all rows and columns
pd.set_option('display.max_rows', None)     # None shows all rows
pd.set_option('display.max_columns', None)  # None shows all columns

# Prevent column truncation and ensure they stay on one line
pd.set_option('display.width', 1000)        # Increase total display width
pd.set_option('display.max_colwidth', None) # Show full content of each cell

# Stop pandas from wrapping columns to a new line
pd.set_option('display.expand_frame_repr', False)

data_dir = Path("data")
parquet_files = sorted(data_dir.glob("*.parquet"))

#############################################################################3

aux_file = 'vitals.parquet'
df = pd.read_parquet('data/' + aux_file)

df.head(20)

df['vital_type'].unique()


df[df['vital_type']=='3fe20e80-80b2-50ea-8d18-966b6c016b05'].head(10)


'''
change how vitals are used. the unique values of column vital_type are ['Pain Level', 'Weight', 'BP - Systolic', 'Blood Sugar', 'Pulse', 'Respiration', 'O2 sats', 'Temperature']
and the values of these vitals are in column value


remove adl_change_sum_30d, adl_change_sum_90d from model and from features notebook



remove gg_response_code_mean_ from features and model


remove gg_response_code_mean_, adl_change_sum_30d, adl_change_sum_90d form the model, from features notebook, and features code


'''



############################################################################################################33
from pathlib import Path

import pandas as pd

aux_file = 'incidents.parquet'
df = pd.read_parquet('data/' + aux_file)

# Optional: drop voided rows if strikeout means inactive
if "strikeout" in df.columns:
    df = df.loc[~df["strikeout"].fillna(False)]

counts = df["incident_type"].value_counts(dropna=False)
pct = (counts / counts.sum() * 100).round(2)

out = pd.DataFrame({"incident_type": counts.index, "count": counts.values, "pct_of_incidents": pct.values})

out['pct_Acc'] = out['pct_of_incidents'].cumsum()

print(out.to_string(index=False))



# get the counts and pecent grouping by incident_type and incident_location
# Count by incident_type + incident_location
out = (
    df
    .groupby(["incident_type", "incident_location"], dropna=False)
    .size()
    .reset_index(name="count")
)

# Percentage over total incidents
out["pct_of_incidents"] = (
    out["count"] / out["count"].sum() * 100
).round(2)

# Sort descending
out = out.sort_values("count", ascending=False)

out['pct_Acc'] = out['pct_of_incidents'].cumsum()

print(out.head(10).to_string(index=False))

