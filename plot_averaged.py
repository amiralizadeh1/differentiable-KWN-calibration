import pandas as pd
import matplotlib.pyplot as plt

# File names
files = ["TND_point_top_bottom_averaged", "MPR_point_top_bottom_averaged", "Hardness_point_top_bottom_averaged"]

for file in files:
    df = pd.read_excel(f"{file}.xlsx")
    
    plt.figure(figsize=(10, 6))
    for i in range(len(df)):
        plt.plot(df.iloc[i, 0], df.iloc[i, 1], marker='o', label=f'Row {i+1}')
    
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(file)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{file}_plot.png")
    plt.close()
    print(f"Created {file}_plot.png")
