import pandas as pd
import os
import matplotlib.pyplot as plt


def save_plot(df , plot_name , plot_dir):
    df.plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)

    unique_filename = plot_name

    os.makedirs(plot_dir, exist_ok=True)
    plotpath = os.path.join(plot_dir)
    plt.savefig(plotpath)
    plt.show()