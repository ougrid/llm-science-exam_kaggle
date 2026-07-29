import glob
import os

import pandas as pd

print("contents of /kaggle/input:", os.listdir("/kaggle/input"))
candidates = glob.glob("/kaggle/input/**/test.csv", recursive=True)
print("test.csv candidates found:", candidates)
assert candidates, "test.csv not found anywhere under /kaggle/input — competition data did not mount"

test = pd.read_csv(candidates[0])
submission = pd.DataFrame({"id": test["id"], "prediction": "A B C"})
submission.to_csv("submission.csv", index=False)
print(submission.head())
print(f"wrote {len(submission)} rows")
