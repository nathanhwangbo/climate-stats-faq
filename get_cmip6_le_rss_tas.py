import intake
import xarray as xr
from tqdm import tqdm
import pyarrow as pa
import polars as pl
import numpy as np
import os.path


var_to_use = ["rsus", "rsds", "tas"]
exps = ["historical"]
savedir = "D:data\\CMIP6\\processed_data\\"

# Open catalog
col_url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
col = intake.open_esm_datastore(col_url)

# Search for the variable and experiments
cat = col.search(variable_id=var_to_use, table_id="Amon", experiment_id=exps)

# Work with a local copy of the catalog dataframe
df = cat.df.dropna(subset=["zstore"]).copy()

# Create a combined ID of model and member
df["source_member"] = df["source_id"].astype(str) + "." + df["member_id"].astype(str)

# Keep only model/member pairs that have all variables in both experiments ----------
counts = df.groupby(["source_member", "variable_id"]).size().unstack()
valid_sources = counts.dropna().index.tolist()
df = df[df["source_member"].isin(valid_sources)]


# Keep only members that have the "standard forcings" (f1), and the first set of physics (p1) ----
is_p1f1 = df["member_id"].str.endswith("p1f1")
df = df[is_p1f1]

# remove duplicates
df = df.drop_duplicates(
    subset=["source_id", "member_id", "experiment_id", "variable_id"]
)

# Keep only source_ids that have at least 20 ensemble members
ens_member_counts = (
    df.groupby(["source_id", "variable_id"]).size().reset_index(name="count")
)
min_n_members = ens_member_counts.groupby("source_id")["count"].min()
sources_to_keep = min_n_members[min_n_members >= 20].index.tolist()
df = df[df["source_id"].isin(sources_to_keep)]

##### temp: just get CESM2 members
df = df[df["source_id"] == "CESM2"]

# View example keys (you can check one with df.iloc[0])
print(df)


# Group by model/member
grouped = df.groupby(["source_id", "member_id"])

# next(iter(grouped)) # look at the first group
for (model, member), group_df in tqdm(grouped, desc="Processing models"):
    if os.path.exists(f"{savedir}monthly_rss_tas_{member}_{model}.nc"):
        # skip if file already exists
        continue
    try:
        # get paths for rus and rds, and tas
        zstores = dict(zip(group_df["variable_id"], group_df["zstore"]))
        if not set(var_to_use).issubset(zstores):
            continue  # skip incomplete

        # load rsds, rsus, and tas
        ds = xr.open_mfdataset(
            list(zstores.values()), engine="zarr", combine="by_coords", join="override"
        )
        # subtract to get rss.
        ds["rss"] = ds["rsds"] - ds["rsus"]

        # save to file
        ds[["rss", "tas"]].to_netcdf(f"{savedir}monthly_rss_tas_{member}_{model}.nc")

    except Exception as e:
        print(f"Skipped {model}_{member}: {e}")


error_messages = """Skipped NorCPM1_r11i1p1f1: cannot align objects with join='override' with matching indexes along dimension 'time' that don't have the same size
Processing models:  83%|████████▎ | 130/157 [55:29<07:23, 16.42s/it]
Skipped NorCPM1_r12i1p1f1: cannot align objects with join='override' with matching indexes along dimension 'time' that don't have the same size
Processing models:  83%|████████▎ | 131/157 [55:32<05:21, 12.37s/it]
Skipped NorCPM1_r13i1p1f1: cannot align objects with join='override' with matching indexes along dimension 'time' that don't have the same size
Processing models:  85%|████████▍ | 133/157 [55:53<04:14, 10.60s/it]
Skipped NorCPM1_r15i1p1f1: cannot align objects with join='override' with matching indexes along dimension 'time' that don't have the same size
Processing models:  85%|████████▌ | 134/157 [55:56<03:11,  8.32s/it]
Skipped NorCPM1_r16i1p1f1: cannot align objects with join='override' with matching indexes along dimension 'time' that don't have the same size
Processing models:  86%|████████▌ | 135/157 [55:59<02:28,  6.75s/it]
Skipped NorCPM1_r17i1p1f1: cannot align objects with join='override' with matching indexes along dimension 'time' that don't have the same size
Processing models:  87%|████████▋ | 136/157 [56:02<01:58,  5.64s/it]
Skipped NorCPM1_r18i1p1f1: cannot align objects with join='override' with matching indexes along dimension 'time' that don't have the same size
Processing models:  87%|████████▋ | 137/157 [56:05<01:36,  4.83s/it]
Skipped NorCPM1_r19i1p1f1: cannot align objects with join='override' with matching indexes along dimension 'time' that don't have the same size
Processing models:  88%|████████▊ | 138/157 [56:08<01:21,  4.29s/it]
Skipped NorCPM1_r1i1p1f1: cannot align objects with join='override' with matching indexes along dimension 'time' that don't have the same size
Processing models:  89%|████████▊ | 139/157 [56:11<01:11,  3.95s/it]
Skipped NorCPM1_r20i1p1f1: cannot align objects with join='override' with matching indexes along dimension 'time' that don't have the same size
Processing models:  90%|█████████ | 142/157 [56:52<02:13,  8.90s/it]
Skipped NorCPM1_r23i1p1f1: cannot align objects with join='override' with matching indexes along dimension 'time' that don't have the same size
Processing models:  91%|█████████ | 143/157 [56:55<01:40,  7.16s/it]
Skipped NorCPM1_r24i1p1f1: cannot align objects with join='override' with matching indexes along dimension 'time' that don't have the same size
Processing models:  94%|█████████▎| 147/157 [57:51<01:48, 10.81s/it]
Skipped NorCPM1_r28i1p1f1: cannot align objects with join='override' with matching indexes along dimension 'time' that don't have the same size
Processing models:  94%|█████████▍| 148/157 [57:54<01:16,  8.47s/it]
Skipped NorCPM1_r29i1p1f1: cannot align objects with join='override' with matching indexes along dimension 'time' that don't have the same size
Processing models:  96%|█████████▌| 150/157 [58:15<01:02,  8.96s/it]
Skipped NorCPM1_r30i1p1f1: cannot align objects with join='override' with matching indexes along dimension 'time' that don't have the same size
Processing models:  97%|█████████▋| 153/157 [58:54<00:41, 10.28s/it]
Skipped NorCPM1_r5i1p1f1: cannot align objects with join='override' with matching indexes along dimension 'time' that don't have the same size
Processing models:  99%|█████████▉| 156/157 [59:31<00:10, 10.58s/it]
Skipped NorCPM1_r8i1p1f1: cannot align objects with join='override' with matching indexes along dimension 'time' that don't have the same size
Processing models: 100%|██████████| 157/157 [59:34<00:00, 22.76s/it]
Skipped NorCPM1_r9i1p1f1: cannot align objects with join='override' with matching indexes along dimension 'time' that don't have the same size"""

import re

errored_norcpm_members = re.findall(r"r\d.*(?=:)", error_messages)
errored_norcpm_members
