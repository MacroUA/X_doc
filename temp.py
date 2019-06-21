import pandas as pd

a = {
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [7, 8, 9]
    }

df = pd.DataFrame(a)
df['A'][0]=23
print(df)