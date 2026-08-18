import numpy as np
import matplotlib.pyplot as plt

# ================= 数据（都来自 repo.md） =================

# 位宽扫描：量化误差（%）
x = np.array([8, 7, 6, 5, 4, 3, 2])
abs_y     = np.array([-0.0504, 0.0542, -0.0528, 0.5330, 3.0015, 23.7184, 910.2188])
abs_ch0_y = np.array([0.0110, 0.0124, 0.0397, 0.2821, 1.0269, 6.5312, 233.6379])
abs_ch1_y = np.array([0.0305, -0.0033, 0.0921, 0.2644, 0.9050, 11.0916, 430.7149])
zp_y      = np.array([0.0231, 0.0281, 0.1470, 0.4214, 2.0489, 10.9011, 239.2861])
zp_ch0_y  = np.array([-0.0202, 0.0353, -0.0331, 0.0435, 0.8670, 4.8702, 51.5138])
zp_ch1_y  = np.array([-0.0231, -0.0216, 0.0388, 0.3227, 0.8114, 5.0887, 88.0759])


methods    = ["baseline", "absmax", "abs ch0", "abs ch1", "zp", "zp ch0", "zp ch1"]
ppl_lmhead = [57.6772, 59.2826, 58.0181, 60.4896, 59.0323, 57.7052, 60.6983]



fig1, ax1 = plt.subplots(figsize=(10, 6))

# 取绝对值：让"误差"始终非负，对数坐标才能画全
ax1.plot(x, np.abs(abs_y),     marker="o", linestyle="-",  label="absmax")
ax1.plot(x, np.abs(abs_ch0_y), marker="s", linestyle="-",  label="absmax ch0")
ax1.plot(x, np.abs(abs_ch1_y), marker="^", linestyle="-",  label="absmax ch1")
ax1.plot(x, np.abs(zp_y),      marker="v", linestyle="--", label="zero-point")
ax1.plot(x, np.abs(zp_ch0_y),  marker="D", linestyle="--", label="zp ch0")
ax1.plot(x, np.abs(zp_ch1_y),  marker="x", linestyle="--", label="zp ch1")

ax1.set_yscale("log")
ax1.grid(True, which="both", linestyle="--", alpha=0.4)
ax1.set_title("Quantization Error vs Bit Width", fontsize=14)
ax1.set_xlabel("Bit Width", fontsize=12)
ax1.set_ylabel("Absolute Loss (%)", fontsize=12)
ax1.legend(loc="upper left", fontsize=10)

plt.tight_layout()
plt.savefig("BabyGPT-Quantization/error_vs_bitwidth.png", dpi=200, bbox_inches="tight")
plt.show()



fig2, ax2 = plt.subplots(figsize=(10, 6))


colors = ["gray"] + ["steelblue"] * 3 + ["indianred"] * 3
bars = ax2.bar(methods, ppl_lmhead, color=colors)


ax2.axhline(y=57.6772, color="gray", linestyle="--", linewidth=1.2, label="FP32 baseline")

ax2.set_ylim(57, 61)
ax2.set_title("lm_head Quantization Sensitivity", fontsize=14)
ax2.set_xlabel("Method", fontsize=12)
ax2.set_ylabel("Perplexity", fontsize=12)


for bar in bars:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2, h,
             f"{h:.2f}", ha="center", va="bottom", fontsize=9)

ax2.legend(loc="upper left", fontsize=10)
plt.tight_layout()
plt.savefig("BabyGPT-Quantization/lmhead_sensitivity.png", dpi=200, bbox_inches="tight")
plt.show()
