import matplotlib.pyplot as plt


def plot_surprisals(surprisals: list, ymax=None, bar_col=None, title=None):
    tokens, surprisal_vals = zip(*surprisals[0])

    if ymax is None:
        ymax = max(surprisal_vals) + 2
        
    if bar_col is None:
        bar_col = "#1f77b4"

    if title is None:
        title = "Surprisal for each input"

    plt.figure(figsize=(8, 4))
    plt.bar(tokens, surprisal_vals, color=bar_col, width=0.9)
    plt.xlabel('')
    plt.ylabel('')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.ylim(0, ymax)
    plt.tight_layout()
    plt.show()
