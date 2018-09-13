# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 15:25:32 2018

@author: William Lord
"""

import scipy.stats as st


def plot_utility_bars():
    """
    Plot bar chart for utility data

    Keyword arguments:
    - data1: array containing first data set
    - data2: array containing second data set

    The function plots the mean accumulated utility for different number of
    allowed action attempts, for the two models.
    """
#    data1_means = (5, 15, 30)  # , 40)
#    data2_means = (6, 13, 32)  # , 38)
#    data1_std = (2, 3, 4)  # , 5)
#    data2_std = (2, 3, 3)  # , 6)
    data1_means = [4.2189999]
    data2_means = [3.73699999]
    data3_means = [4.098]
    data1_error = (0.06915851358)
    data2_error = (0.1259607082)
    data3_error = (0.07616823485)
    n = len(data1_means)
    ind = np.arange(n)    # the x locations for the groups
    w = 0.2
    labels = [str(i + 1) for i in range(n)]
    labels = [2]

    # Pull the formatting out here
    bar_kwargs = {'width': w, 'color': 'black', 'yerr': data1_error,
                  'ecolor': 'grey', 'capsize': 5,  # 'tick_label': labels
                  }

    fig, ax = plt.subplots()
    ax.bar(ind, data1_means, **bar_kwargs)

    bar_kwargs = {'width': w, 'color': 'gray', 'yerr': data2_error,
                  'ecolor': 'black', 'capsize': 5, 'tick_label': labels
                  }
    ax.bar(ind + w, data2_means, **bar_kwargs)
    
    bar_kwargs = {'width': w, 'color': 'black', 'yerr': data3_error,
                  'ecolor': 'black', 'capsize': 5,  # 'tick_label': labels
                  }
    ax.bar(ind + 2*w, data3_means, **bar_kwargs)

    def plot_significance(i, text, data1_means, data3_means):
        # TEST OF P-VALUE ARROWS
        (y_bottom, y_top) = ax.get_ylim()
        y_height = y_top - y_bottom
        x = ind[i] + w
        y = max(data1_means[i], data3_means[i]) + (y_height * 0.03)
        props = {'connectionstyle': 'bar', 'arrowstyle': '-', 'shrinkA': 6,
                 'shrinkB': 0
                 }

        ax.text(x + w, y_height + 0.02 * y_height, text, ha='center',
                va='bottom', size='small'
                )
        ax.annotate('', xy=(x, y), xytext=(x + w, y), arrowprops=props)

    for i in range(n):
        (t, p) = st.ttest_ind(data1_means[i], data2_means[i], equal_var=False)
        if p <= 0.001:
            plot_significance(i, '***', data1_means, data2_means)
        elif p <= 0.01:
            plot_significance(i, '**', data1_means, data2_means)
        elif p <= 0.05:
            plot_significance(i, '*', data1_means, data2_means)
        elif p <= 0.1:
            plot_significance(i, '****', data1_means, data2_means)

#    (y_bottom, y_top) = ax.get_ylim()
#    y_height = y_top - y_bottom
#    ax.text(0.2, y_height - 0.1*y_height,
#            '*) p < 0.05, **) p < 0.01, ***) p < 0.001', ha='left',
#            va='bottom', size='x-small'
#            )

    plt.ylabel('Utility')
    plt.xlabel('Action attempts')
    plt.ylim(3, 4.3)
    
plot_utility_bars()