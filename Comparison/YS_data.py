import numpy as np
import matplotlib.pyplot as plt

x1_1 = [0.01, 0.5, 1.00, 4.00, 8.00, 24.0]
y1_1 = [105.53746620455607, 156.35180385548145, 202.2801482183788, 260.42345716276094, 268.7296494801256, 259.4462454581457]
legend1_1 = '180°C, Al–0.4Mg–1.32Si, gupta01fig5'



x2_1 = [0.01, 0.5, 1.00, 4.00, 8.00,  24.0]
y2_1 = [68.40388739380685, 87.45930594972064, 130.45604315665005, 233.0619022052382, 238.4364873666349, 229.15310198324022]
legend2_1 = '180°C, Al–0.4Mg–0.72Si, gupta01fig5'



x3_1 = [0.01, 0.5, 1.00, 4.00, 8.00,  24.0]
y3_1 = [28.338775981430604, 29.31595972816799, 29.31595972816799, 90.87948634047204, 128.01304651263604, 148.0456115381168]
lenged3_1 = '180°C, Al–0.4Mg–0.28Si, gupta01fig5'


x1_2 = np.array([1, 2, 3, 8, 24, 48, 72, 120, 168])
y1_2 = np.array([74, 89, 127, 187, 219, 230, 252, 280, 255])
legend1_2 = "150°C, Al-0.5Mg-0.43Si, Sekhar18"



x2_2 = np.array([1, 3, 5, 7, 9])
y2_2 = np.array([250.3, 294.4, 331.9, 336.8, 329.1])
legend2_2 = "175°C,  (6082) Al−1.01Mg−0.89Si−0.14Cr−0.13Fe, Liu23"



x3_2 = np.array([0.5, 1, 1.5, 2, 2.5, 3, 4, 6, 8, 12, 24, 48, 72, 96, 120, 168, 336, 504])
y3_2 = np.array([94.1, 155.9, 151.5, 172.2, 181.3, 176.3, 227.3, 246.5, 264.3, 243.6, 216.7, 227.3, 216.5, 207.5, 192.3, 174.4, 137.6, 103.3])
e3_2 = np.array([2.5, 3.8, 2.8, 4.5, 4.8, 5.4, 5.3, 6.8, 7.4, 5.1, 6.7, 6.0, 6.1, 6.8, 6.2, 5.2, 4.1, 2.8])
legend3_2 = '175°C, (6063) Al-0.5Mg-0.435Si-0.178Fe-0.001Cu-0.082 Mn-0.006Cr-0.067Zn-0.023Ti, sekhar19'


x1_3 = [0.01, 0.5, 1.00, 4.00, 8.00,  24.0]
y1_3 = [120.95238694978822, 168.09524850419976, 212.85715782182737, 265.238107031074, 271.9047623949346, 264.2857225033156]

legend1_3 = "180°C, Al0.4Mg1.32Si, gupta01fig10"



x2_3 = [0.01, 0.5, 1.00, 4.00, 8.00,  24.0]
y2_3 = [149.04763060992826, 194.76192445531427, 258.57143350198936, 315.23810400353665, 320.00000847710453, 313.8095272118991]
legend2_3 = "180°C, Al0.6Mg1.32Si, gupta01fig10"



x3_3 = [0.01, 0.5, 1.00, 4.00, 8.00,  24.0]
y3_3 = [155.71430413901294, 177.61905745133552, 225.23810218701425, 329.5238174242403, 346.666666262995, 329.04760699513713]
legend3_3 = ", 180°C, Al0.8Mg1.32Si gupta01fig10"



# Create the figure and axis objects
fig, ax = plt.subplots(figsize=(12, 8))

# Plot for set 1
ax.plot(x1_1, y1_1, label=legend1_1, marker='o', color='blue')
ax.plot(x2_1, y2_1, label=legend2_1, marker='s', color='red')
ax.plot(x3_1, y3_1, label=lenged3_1, marker='^', color='green')

# Plot for set 2
ax.plot(x1_2, y1_2, 'o-', label=legend1_2, color='orange')
ax.plot(x2_2, y2_2, 's--', label=legend2_2, color='purple')
ax.errorbar(x3_2, y3_2, yerr=e3_2, fmt='^-', color='brown', ecolor='lightgray', elinewidth=2, capsize=4, label=legend3_2)

# Plot for set 3
ax.plot(x1_3, y1_3, 'o-', label=legend1_3, color='cyan')
ax.plot(x2_3, y2_3, 's--', label=legend2_3, color='magenta')
ax.plot(x3_3, y3_3, '^-', label=legend3_3, color='yellow')

# Set the labels and title
ax.set_xlabel('Time (hours)', fontsize=12)
ax.set_ylabel('Yield Strength (MPa)', fontsize=12)
ax.set_title('Comparison of Yield Strength vs Time for Different Aluminium Alloys', fontsize=14)

# Add grid and legend
ax.grid(True)
ax.legend()

# Show the plot
plt.show()