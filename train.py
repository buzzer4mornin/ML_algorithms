import pandas as pd
import numpy as np

shape = (50, 4460)

data = np.random.normal(size=shape)

data[:, 1000] += data[:, 2000]

df = pd.DataFrame(data)

c = df.corr().abs()

s = c.unstack()
so = s.sort_values(ascending=False)

#print(so[-2047761:-2047741])
print(so[1000:1010])