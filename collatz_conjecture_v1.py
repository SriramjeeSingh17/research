import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.mixture import GaussianMixture


# 1) collatz conjecture series for any number n until it reaches 1
def collatz_conjecture(x):
    series = [x]
    while x != 1:
        if x % 2 == 0:
            x = x // 2
        else:
            x = 3*x + 1
        series.append(x)
    return series

number_list = [27, 941, 8842, 52315, 234712, 6543127, 82134563, 234576829, 1232675423]

for i in number_list:
    li = len(collatz_conjecture(i))
    print(f"number: {i}, collatz_conjecture_length: {li}")


# 2) finding length of collatz conjecture series for a range of numbers (until n) and frequency of the lengths
def len_cc_series(n):
    length_cc = []
    number = []
    for i in range(1, n+1):
        l = len(collatz_conjecture(i))
        number.append(i)
        length_cc.append(l)
    length_cc_counts = Counter(length_cc)
    length_cc_counts_top = length_cc_counts.most_common(20)
    for rank, (val, count) in enumerate(length_cc_counts_top, 1):
        print(f"{rank}. length_cc: {val}, length_cc_counts: {count}")
    return number, length_cc, length_cc_counts

numbers, lengths, lengths_counts = len_cc_series(1000)

plt.figure(figsize=(12, 6))
plt.scatter(numbers, lengths, s=10, alpha=0.6, label="Collatz Length")
coeffs = np.polyfit(numbers, lengths, deg=1)
poly_eq = np.poly1d(coeffs)
trendline = poly_eq(numbers)
plt.plot(numbers, trendline, color="red", linewidth=2, label="Trend Line")
plt.title("Collatz Conjecture: Numbers vs Sequence Length")
plt.xlabel("Number")
plt.ylabel("Collatz Sequence Length")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("collatz_trend.png", dpi=300)
plt.show()


# 3) plotting lengths and its frequency for numbers (until n) and normal curve fitting (2 normal curves (a Gaussian mixture) captures the shape better)

def get_lengths(n):
    lengths = [len(collatz_conjecture(i)) for i in range(1, n+1)]
    return lengths

def plot_length_distribution_mixture(n):
    lengths = get_lengths(n)
    length_counts = Counter(lengths)
    x = np.array(list(length_counts.keys()))
    y = np.array(list(length_counts.values()))
    data = np.repeat(x, y).reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(data)
    x_smooth = np.linspace(min(x), max(x), 400).reshape(-1, 1)
    logprob = gmm.score_samples(x_smooth)
    responsibilities = gmm.predict_proba(x_smooth)
    pdf_individual = responsibilities * np.exp(logprob)[:, np.newaxis]
    pdf_total = np.exp(logprob)
    scale = sum(y) * (x_smooth[1] - x_smooth[0])
    pdf_total *= scale
    pdf_individual *= scale

    plt.figure(figsize=(10, 5))
    plt.bar(x, y, width=1.0, color='lightblue', edgecolor='black', alpha=0.7, label="Counts")
    plt.plot(x_smooth, pdf_total, 'r-', linewidth=2, label="Mixture Fit (2 Gaussians)")
    plt.plot(x_smooth, pdf_individual[:,0], 'g--', linewidth=2, label="Gaussian 1")
    plt.plot(x_smooth, pdf_individual[:,1], 'm--', linewidth=2, label="Gaussian 2")

    plt.xlabel("Collatz Sequence Length")
    plt.ylabel("Frequency (Count)")
    plt.title(f"Collatz Sequence Length Distribution (1 to {n})")
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_length_distribution_mixture(1000)

