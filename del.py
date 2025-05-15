# %%


import pandas as pd
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

columns = [
    "intercolumnar_distance",
    "upper_margin",
    "lower_argin",
    "exploitation",
    "row_number",
    "modular_ratio",
    "interlinear_spacing",
    "weight",
    "peak_number",
    "modular_ratio/interlinear_spacing",
    "class",
]
df = pd.read_csv(
    r"C:\Users\saman\Downloads\bank-full.csv",
    delimiter=";",
)


unique_values = sorted(df["job"].unique())
value_map = {value: index for index, value in enumerate(unique_values)}
df["Task"] = df["job"].map(value_map)
df.drop(columns=["job"], inplace=True)


num_cols = [
    "age",
    "balance",
    "day",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "Task",
]
df[num_cols] = df[num_cols].astype(int)

cat_cols = [col for col in df.columns if df[col].dtype == "object" and col != "y"]
ohe = OneHotEncoder(drop="first", sparse_output=False)
transformer = ColumnTransformer(
    transformers=[("cat", ohe, cat_cols)], remainder="passthrough"
)

X = transformer.fit_transform(df.drop("y", axis=1))
feature_names = transformer.get_feature_names_out()

X_df = pd.DataFrame(X, columns=feature_names)
y = df["y"].map({"yes": 1, "no": 0})
X_df.to_csv("bank_data.csv")
y.to_csv("bank_target.csv")


