import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

cov_std = np.load("results/cov_fn_std.npy")
cov_mpc = np.load("results/cov_fn_mpc.npy")
min_index = np.argmin(cov_mpc[:, -1])
cov_mpc = np.delete(cov_mpc, min_index, axis=0)

NUM_EPISODES = cov_std.shape[0]
NUM_STEPS = cov_std.shape[1]

mean_std = np.mean(cov_std, axis=0)
mean_mpc = np.mean(cov_mpc, axis=0)
std_std = np.std(cov_std, axis=0)
std_mpc = np.std(cov_mpc, axis=0)

eff_mpc = np.load("results/effect_mpc.npy")
eff_std = np.load("results/effect_std.npy")
min_index = np.argmin(eff_mpc[:, -1])
eff_mpc = np.delete(eff_mpc, min_index, axis=0)

eff_mean_mpc = np.mean(eff_mpc, axis=0)
eff_mean_std = np.mean(eff_std, axis=0)
eff_std_mpc = np.std(eff_mpc, axis=0)
eff_std_std = np.std(eff_std, axis=0)

t = np.arange(NUM_STEPS)

lim_range = 21
apf = 14

fig, ax = plt.subplots(figsize=(8,10))
lw = 5
ax.plot(t, mean_std, color="tab:blue", linewidth=lw, label=f"[{lim_range}]+[{apf}]")
ax.plot(t, mean_mpc, color="tab:orange", linewidth=lw, label="HMPCC")
ax.fill_between(t, mean_std - std_std, mean_std + std_std,  color='tab:blue', alpha=0.25)
ax.fill_between(t, mean_mpc - std_mpc, mean_mpc + std_mpc, color='tab:orange', alpha=0.25)
# ax.errorbar(t, mean_mpc, yerr=std_mpc)
s_labels = 32
s_legend = 28
s_ticks = 28
ax.set_xlabel("Time [s]", fontsize=s_labels)
ax.set_ylabel(r"$\mathcal{H}$", fontsize=s_labels)
plt.grid()
plt.xticks(
    ticks=[0, 20, 40, 60, 80, 100], 
    labels=["0", "2", "4", "6", "8", "10"], 
    fontsize=s_ticks
)
plt.yticks(fontsize=s_ticks)
plt.legend(fontsize=s_legend)
# plt.tight_layout()
plt.savefig("pics/cov_rate_vert.pdf", bbox_inches='tight')
# plt.show()

fig, ax = plt.subplots(figsize=(8,10))
lw = 5
ax.plot(t, eff_mean_std, color="tab:blue", linewidth=lw, label=f"[{lim_range}]+[{apf}]")
ax.plot(t, eff_mean_mpc, color="tab:orange", linewidth=lw, label="HMPCC")
ax.fill_between(t, eff_mean_std - eff_std_std, eff_mean_std + eff_std_std,  color='tab:blue', alpha=0.25)
ax.fill_between(t, eff_mean_mpc - eff_std_mpc, eff_mean_mpc + eff_std_mpc, color='tab:orange', alpha=0.25)
# ax.errorbar(t, mean_mpc, yerr=std_mpc)
s_labels = 32
s_legend = 28
s_ticks = 28
ax.set_xlabel("Time [s]", fontsize=s_labels)
ax.set_ylabel(r"$\mathcal{E}$", fontsize=s_labels)
plt.grid()
plt.xticks(
    ticks=[0, 20, 40, 60, 80, 100], 
    labels=["0", "2", "4", "6", "8", "10"], 
    fontsize=s_ticks
)
plt.yticks(fontsize=s_ticks)
plt.legend(fontsize=s_legend)
# plt.tight_layout()
plt.savefig("pics/effect_vert.pdf", bbox_inches='tight')
# plt.show()

r_nums = [4, 6, 8, 10]
eff_std = np.zeros((10, len(r_nums)))
h_std = np.zeros((10, len(r_nums)))
eff_mpc = np.zeros_like(eff_std)
h_mpc = np.zeros_like(h_std)

for i, r in enumerate(r_nums):
    data_std = np.loadtxt(f'results/std_maze_eval_{r}r.txt', skiprows=1, usecols=(3, 4))
    eff_std[:, i] = data_std[:, 0]
    h_std[:, i] = data_std[:, 1]
    data_mpc = np.loadtxt(f"results/maze_eval_{r}r.txt", skiprows=1, usecols=(3, 4))
    eff_mpc[:, i] = data_mpc[:, 0]
    h_mpc[:, i] = data_mpc[:, 1]


avg_eff_std = np.mean(eff_std, axis=0)
std_eff_std = np.std(eff_std, axis=0) / np.sqrt(len(r_nums))
std_eff_std = np.clip(std_eff_std, None, 1-avg_eff_std)
avg_h_std = np.mean(h_std, axis=0)
std_h_std = np.std(h_std, axis=0) / np.sqrt(len(r_nums))
std_h_std = np.clip(std_h_std, None, 1-avg_h_std)
avg_eff_mpc = np.mean(eff_mpc, axis=0)
std_eff_mpc = np.std(eff_mpc, axis=0) / np.sqrt(len(r_nums))
std_eff_mpc = np.clip(std_eff_mpc, None, 1-avg_eff_mpc)
avg_h_mpc = np.mean(h_mpc, axis=0)
std_h_mpc = np.std(h_mpc, axis=0) / np.sqrt(len(r_nums))
std_h_mpc = np.clip(std_h_mpc, None, 1-avg_h_mpc)


bw = 0.8
fig, ax = plt.subplots(figsize=(8,10))
ax.bar(x=np.array(r_nums)-bw/2, 
       height=avg_eff_std,
       yerr=std_eff_std,
       width=bw,
       color="tab:blue",
       label=f"[{lim_range}]+[{apf}]",
       capsize=4)
ax.bar(x=np.array(r_nums)+bw/2, 
       height=avg_eff_mpc,
       yerr=std_eff_mpc,
       width=bw,
       color="tab:orange",
       label="HMPCC",
       capsize=4)
plt.xticks(ticks=r_nums, fontsize=s_ticks)
plt.yticks(fontsize=s_ticks)
ax.set_xlabel("N", fontsize=s_labels)
ax.set_ylabel(r"$\mathcal{E}$", fontsize=s_labels)
ax.grid(True)
ax.legend(fontsize=s_legend)
ax.set_axisbelow(True)
plt.savefig("pics/eff_bar.pdf", bbox_inches='tight')

fig, ax = plt.subplots(figsize=(8,10))
ax.bar(x=np.array(r_nums)-bw/2, 
       height=avg_h_std,
       yerr=std_h_std,
       width=bw,
       color="tab:blue",
       label=f"[{lim_range}]+[{apf}]",
       capsize=4)
ax.bar(x=np.array(r_nums)+bw/2, 
       height=avg_h_mpc,
       yerr=std_h_mpc,
       width=bw,
       color="tab:orange",
       label="HMPCC",
       capsize=4)
plt.xticks(ticks=r_nums, fontsize=s_ticks)
plt.yticks(fontsize=s_ticks)
ax.set_xlabel("N", fontsize=s_labels)
ax.set_ylabel(r"$\mathcal{H}$", fontsize=s_labels)
ax.grid(True)
ax.legend(fontsize=s_legend)
ax.set_axisbelow(True)
plt.savefig("pics/h_bar.pdf", bbox_inches='tight')
# plt.show()


# WITH HUMANS
h_nums = [3, 6, 9]
eff_std = np.zeros((10, len(h_nums)))
h_std = np.zeros((10, len(h_nums)))
coll_std = np.zeros((10, len(h_nums)))
eff_mpc = np.zeros_like(eff_std)
h_mpc = np.zeros_like(h_std)
coll_mpc = np.zeros_like(coll_std)
for i, h in enumerate(h_nums):
    data_std = np.loadtxt(f'results/std_eval_00{h}_uni.txt', skiprows=1, usecols=(2, 3, 4))
    coll_std[:, i] = data_std[:, 0]
    eff_std[:, i] = data_std[:, 1]
    h_std[:, i] = data_std[:, 2]
    data_mpc = np.loadtxt(f"results/eval_00{h}.txt", skiprows=1, usecols=(2, 3, 4))
    coll_mpc[:, i] = data_mpc[:, 0]
    eff_mpc[:, i] = data_mpc[:, 1]
    h_mpc[:, i] = data_mpc[:, 2]

avg_eff_std = np.mean(eff_std, axis=0)
std_eff_std = np.std(eff_std, axis=0) / np.sqrt(len(r_nums))
std_eff_std = np.clip(std_eff_std, None, 1-avg_eff_std)
avg_h_std = np.mean(h_std, axis=0)
std_h_std = np.std(h_std, axis=0) / np.sqrt(len(r_nums))
std_h_std = np.clip(std_h_std, None, 1-avg_h_std)
avg_eff_mpc = np.mean(eff_mpc, axis=0)
std_eff_mpc = np.std(eff_mpc, axis=0) / np.sqrt(len(r_nums))
std_eff_mpc = np.clip(std_eff_mpc, None, 1-avg_eff_mpc)
avg_h_mpc = np.mean(h_mpc, axis=0)
std_h_mpc = np.std(h_mpc, axis=0) / np.sqrt(len(r_nums))
std_h_mpc = np.clip(std_h_mpc, None, 1-avg_h_mpc)

coll_tot_std = np.sum(coll_std, axis=0)
coll_tot_mpc = np.sum(coll_mpc, axis=0)

print("=================")
print("==== HMPCC ======")
print("Collisions: ", coll_tot_mpc)
print(f"Eff: {avg_eff_mpc}" )
print("Eff std: ", std_eff_mpc)
print("H : ", avg_h_mpc)
print("std: ", std_h_mpc)


print("=================")
print("==== STD ======")
print("Collisions: ", coll_tot_std)
print(f"Eff: {avg_eff_std}" )
print("Eff std: ", std_eff_std)
print("H: ", avg_h_std)