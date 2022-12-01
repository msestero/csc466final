# unzips all .bz2 files
!bzip2 -d *.bz2

import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split

# main getter function to read in the data from the files
# You can read in multiple files, but for our purposes we
# used one file since the file sizes were so large

def get_df():
  dfs = []
  for year in range(2007, 2008):
    with open(f"{year}.csv", "r") as csvfile:
        dfs.append(pd.read_csv(csvfile))
  data = pd.concat(dfs)
  
  # Converts all Delay times into a boolean if there was a delay.
  data["WeatherDelay"] = pd.Series(map(lambda x: int(x > 0), data["WeatherDelay"]))
  data["CarrierDelay"] = pd.Series(map(lambda x: int(x > 0), data["CarrierDelay"]))
  data["NASDelay"] = pd.Series(map(lambda x: int(x > 0), data["NASDelay"]))
  data["SecurityDelay"] = pd.Series(map(lambda x: int(x > 0), data["SecurityDelay"]))
  data["LateAircraftDelay"] = pd.Series(map(lambda x: int(x > 0) , data["LateAircraftDelay"]))
  return data

# get
main_df = get_df()

# display
main_df.head()

# for reference while coding
main_df.columns

# Our initial data is too large, we take a random 50_000
df = main_df.sample(50000)
df.reset_index(inplace=True)

# Useful Sklearn tool to transform all of the columns in a df in one call.
ct = make_column_transformer(
    (StandardScaler(), ["WeatherDelay", "CarrierDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay"]),
    (OneHotEncoder(), ['Month', 'Origin', 'Dest', 'UniqueCarrier']),
    remainder="drop"
)

# Transform the data to be used
# Re-attach the data to be predicted (DepDelay)
# Drop any rows with invalid data or NaN 
encoded_df = pd.DataFrame(ct.fit_transform(df).toarray())
encoded_df["DepDelay"] = df["DepDelay"]
encoded_df.dropna(inplace=True)
encoded_df.head()

# split up the data into train and test sets
# Use a test size of 30%
all_cols_exp_delay = encoded_df.loc[:, encoded_df.columns != 'DepDelay']
X_train, X_test, y_train, y_test = train_test_split(all_cols_exp_delay, encoded_df["DepDelay"], test_size=0.30)

# Creating the model and predictions
neigh = KNeighborsRegressor(n_neighbors=100)
neigh.fit(X_train, y_train)
pred = neigh.predict(X_test)

# Score
# (1 - U/V)
# U = ((y_true - y_pred) ** 2).sum()
# V = ((y_true - y_true.mean()) ** 2).sum()
neigh.score(X_test, y_test)

pred = pd.Series(pred)
y_test = y_test.reset_index(drop=True)
pred_act_df = pd.concat([pred, y_test], axis=1)
pred_act_df = pred_act_df.rename(columns={0: "Predicted", "DepDelay": "Actual"})

#chart for first 50
num_bars = 50
disp_df = pred_act_df.sample(n=num_bars)
disp_df = disp_df.reset_index(drop=True)
disp_df.plot.bar(ylabel="Minutes Delayed", xlabel="Random Flight", figsize=(14, 7), width=0.95, title="Actual vs. Predicted Delay Time")

# Only look at delays
onlyDelays = pred_act_df[pred_act_df["Actual"] > 0]
onlyDelays = onlyDelays.sample(n=num_bars)
onlyDelays = onlyDelays.reset_index(drop=True)
onlyDelays.plot.bar(ylabel="Minutes Delayed", 
                    xlabel="Random Flight", 
                    figsize=(14, 7), 
                    width=0.95, 
                    title="Actual vs. Predicted Delay Time (Delay)")

# Only look at non delays
nonDelays = pred_act_df[pred_act_df["Actual"] <= 0]
nonDelays = nonDelays.sample(n=num_bars)
nonDelays = nonDelays.reset_index(drop=True)
nonDelays.plot.bar(ylabel="Minutes Delayed", 
                    xlabel="Random Flight", 
                    figsize=(14, 7), 
                    width=0.95, 
                    title="Actual vs. Predicted Delay Time (No Delay)")
