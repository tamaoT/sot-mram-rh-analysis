#%%
#one loop
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read data
df = np.loadtxt(fname=r"d:\spintec\BTMA60B_3_0.40Hz_0.03V_1004mT_300.0K.dat",
                dtype="float",
                delimiter="\t",
                skiprows=1)

# delate first row
df = df[1:]

# convert into DataFrame and name
df_named = pd.DataFrame(df, columns=["T", "R"])

# T:-0.2〜0.2(to make it easy)
df_filtered = df_named[(df_named['T'] > -0.2) & (df_named['T'] < 0.2)]

#each loop has 4000 data
df_cut = df_filtered.iloc[:4000]

fig, ax = plt.subplots()

# Plot the resistance vs field
ax.plot(df_cut["T"], df_cut["R"], label='R vs T')

# Add a vertical line at T = 0
ax.axvline(x=0, color='red', linestyle='--', label='Field = 0(T)')
ax.axvline(x=-0.0124, color='blue', linestyle='--', label='Boffset')
mid_x = (0 + -0.0124) / 2
top_y = df_cut["R"].max() 
ax.annotate('', xy=(0, top_y), xytext=(-0.0124, top_y),
            arrowprops=dict(arrowstyle='<->', color='green'))
#ax.text(mid_x, top_y + 0.02, f'Bc offset = 0.0124 T',  # 0.02 is arange
        #ha='center', va='bottom', color='green')

# Add a marker or line at T = 0.1536 under the x-axis
# ax.annotate('Center T = 0.1536',
#             xy=(0.1536, ax.get_ylim()[0]),  # bottom of the y-axis
#             xytext=(0.1536, ax.get_ylim()[0] - 0.02),  # offset below
#             ha='center',
#             arrowprops=dict(arrowstyle='-|>', color='blue'),
#             fontsize=9, color='blue')

# Labeling
ax.set_xlabel('Field (T)')
ax.set_ylabel('Resistance ($\Omega$)')
ax.legend()

plt.tight_layout()
plt.show()


#%%
#all loop

# --- Split into loops ---
# Example: assume 400 data points per loop
points_per_loop = 400
num_loops = len(df_filtered) // points_per_loop

# --- Plot all loops ---
fig, ax = plt.subplots()

for i in range(num_loops):
    start = i * points_per_loop
    end = start + points_per_loop
    df_loop = df_filtered.iloc[start:end]
    ax.plot(df_loop["T"], df_loop["R"], alpha=0.5, label=f"Loop {i+1}")

# --- Optional: vertical line and annotation ---
#ax.axvline(x=0, color='red', linestyle='--', label='Field = 0(T)')

ax.set_xlabel('Field (T)')
ax.set_ylabel('Resistance ($\Omega$)')
ax.set_title("all R vs T Loops at 300K")
#ax.legend(loc='best', fontsize='small', ncol=2)
plt.tight_layout()
plt.show()
# %%
#extract magnetic properties
R_AP = df_cut["R"].max()
R_P = df_cut["R"].min()
TMR = (R_AP - R_P) / R_P
df_cut_clean = df_cut.drop_duplicates(subset="T")
dR_dT = np.gradient(df_cut_clean["R"], df_cut_clean["T"])#R, T列をそれぞれ微分
T_vals = df_cut_clean["T"]
idx_min = np.argmin(dR_dT)
idx_max = np.argmax(dR_dT)

sorted_indices = np.argsort(dR_dT)[::-1] 
selected_Ts = []
selected_indices = []

for idx in sorted_indices:
    T_candidate = T_vals.iloc[idx]
    # candidate should be far away more than 0.05
    if all(abs(T_candidate - t) >= 0.05 for t in selected_Ts):
        selected_Ts.append(T_candidate)
        selected_indices.append(idx)
    if len(selected_Ts) == 2:
        break
    
T1, T2 = selected_Ts
Bc_minus = min(T1, T2)
Bc_plus = max(T1, T2)

Bc = abs(Bc_plus - Bc_minus) / 2
offset = (Bc_plus + Bc_minus) / 2
print(f"R_AP(max R):{R_AP}")
print(f"R_P (min R):{R_P}")
print(f"TMR:{TMR*100}%")
print(f"B_Cplus:{Bc_plus}, B_Cminus:{Bc_minus}")
print(f"Bc:{Bc}")
print(f"Offset:{offset}")
# %%
#you can plot how it looks when it is differenciated
plt.plot(T_vals, dR_dT)
plt.title("dR/dT vs T")
plt.xlabel("T")
plt.ylabel("dR/dT")
plt.grid()
plt.show()
# %%
