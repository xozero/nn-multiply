import pickle

import numpy as np
import scipy.stats as st
import matplotlib as mpl
from matplotlib import pyplot as plt


mpl.font_manager._rebuild()
plt.rc('font', family='Raleway')
# n = 6
# color = plt.cm.Greens(np.linspace(.3, 1, n))[::-1]
# mpl.rcParams['axes.prop_cycle'] = plt.cycler('color', color)
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.Set2.colors)


def plot(kv):
    tags = {
        '7_high2_50': '50',
        '7_high2_200': '200',
        '7_high2_500': '500',
        '7_high2_2000': '2000'
    }

    fig, axs = plt.subplots(1, 1, figsize=(5, 3.5))

    def calc_ci(ax, key, arr):
        # arr = arr[~np.isnan(arr)]
        # arr = arr[arr != 0.]
        mean = np.mean(arr, axis=0)
        # ci = st.t.interval(
        #     0.95,
        #     len(arr) - 1,
        #     loc=np.mean(arr, axis=0),
        #     scale=st.sem(arr, axis=0)
        # )

        x = np.arange(len(mean))

        ax.plot(x, mean, label=tags[key])
        # ax.fill_between(x, ci[0], ci[1], alpha=.2)

    for k, v in kv.items():
        if k not in tags:
            continue
        v = np.array(v)
        calc_ci(axs, k, v[0][:, :1000])  # r2

        axs.legend()
        axs.set_ylabel(r'$R^2$')
        axs.set_xlabel("Epoch")

    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    plt.legend(frameon=False)
    plt.ylim((0.93, 1.002))
    plt.tight_layout()
    plt.show()
    fig.savefig('comp.png')
    fig.savefig('comp.pdf')


def plot2(kv):
    tags = {
        '7_high2_2000': '2000 (20 nodes)',
        '7_high2_2000_fc100': '2000 (100 nodes)'
    }

    fig, axs = plt.subplots(1, 1, figsize=(5, 3.5))

    def calc_ci(ax, key, arr):
        # arr = arr[~np.isnan(arr)]
        # arr = arr[arr != 0.]
        mean = np.mean(arr, axis=0)
        # ci = st.t.interval(
        #     0.95,
        #     len(arr) - 1,
        #     loc=np.mean(arr, axis=0),
        #     scale=st.sem(arr, axis=0)
        # )

        x = np.arange(len(mean))

        ax.plot(x, mean, label=tags[key])
        # ax.fill_between(x, ci[0], ci[1], alpha=.2)

    for k, v in kv.items():
        if k not in tags:
            continue
        v = np.array(v)
        calc_ci(axs, k, v[0][:, :1000])  # r2

        axs.legend()
        axs.set_ylabel(r'$R^2$')
        axs.set_xlabel("Epoch")

    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    plt.legend(frameon=False)
    plt.ylim((0.93, 1.002))
    plt.tight_layout()
    plt.show()
    fig.savefig('comp2.png')
    fig.savefig('comp2.pdf')


def run():
    with open('comp.pickle', 'rb') as f:
        kv = pickle.load(f)
    plot(kv)
    plot2(kv)


if __name__ == '__main__':
    run()
