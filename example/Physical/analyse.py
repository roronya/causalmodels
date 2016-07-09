import causalmodels as cm
import pandas as pd

data = pd.read_table("res.txt", header=None, delim_whitespace=True)
data.columns = ["theta1", "theta2", "omega1", "omega2"]
model = cm.DirectLiNGAM()
result = model.fit(data.values, data.columns)
print(result.matrix)
result.draw()
