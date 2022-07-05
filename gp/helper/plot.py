import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as mticker

def plot_1D_regression(context_x, context_y, target_x, target_y, \
    idx = 0, *args, **kargs):
    '''
        Plots the predictive distribution (mean, var) and context points

        Args:
            All arguments are "NP arrays"
            target_x : [batch_size, the number of total_points, x_size(dimension)] 
            target_y : [batch_size, the number of total_points, y_size(dimension)] 
            context_x : [batch_size, the number of context_points, x_size(dimension)] 
            context_y : [batch_size, the number of context_points, y_size(dimension)] 
            pred_y : [batch_size, the number of total_point, y_size(dimension)] same as target_y
            var  : [batch_size, the number of total_point, y_size(dimension)] same as var
    '''
    # Path
    result_path = kargs.get("result_path", "./")
    file_name = kargs.get("file_name", "regression_result.png")
    FILE_NAME = os.path.join(result_path, file_name)

    # Pred
    pred = kargs.get("pred_mu", None)
    std = kargs.get("pred_sigma", None)

    
    if target_x.shape[0] == 1:
        context_x = np.squeeze(context_x[idx], axis=-1)
        context_y = np.squeeze(context_y[idx], axis=-1)
        target_x = np.squeeze(target_x[idx], axis=-1)
        target_y = np.squeeze(target_y[idx], axis=-1)
        pred = np.squeeze(pred[idx], axis=-1)
        std = np.squeeze(std[idx], axis=-1)
    else:
        context_x = np.squeeze(context_x[idx])
        context_y = np.squeeze(context_y[idx])
        target_x = np.squeeze(target_x[idx])
        target_y = np.squeeze(target_y[idx])
        pred = np.squeeze(pred[idx])
        std = np.squeeze(std[idx])
        
    # scatter plot
    plt.plot(context_x, context_y, 'ko', markersize=5)
    
    
    # Line plot
    plt.plot(target_x, target_y, 'k:', linewidth=1)
    if pred is not None and std is not None:
        # line plot
        plt.plot(target_x, pred, 'b', linewidth=2)

        # var
        plt.fill_between(target_x, \
            pred - std,\
            pred + std,
            alpha = 0.2,
            facecolor='#65c9f7',
            interpolate=True
        )
        
    # Make a plot pretty
    TITLE = kargs.get("title", "1D regression plot")
    ITERATION = kargs.get("iteration", None)

    # Title
    plt.title(TITLE, fontweight='bold', loc='center')
    
    # Label
    plt.xlabel("{} iterations".format(ITERATION)) if not ITERATION is None else None
    
    # Axis
    plt.yticks([-2, 0, 2], fontsize=16)
    plt.xticks([-4, 0, 4], fontsize=16)
    #plt.ylim([-1, 1])
    plt.grid(False)
    ax = plt.gca()
    ax.set_facecolor('white')

    # Save
    plt.savefig(FILE_NAME, dpi=300)

    # Close plt
    plt.clf()


def plot_attention_weights_heat_map(context_x, target_x, attention_weights, \
    idx=0, *args, **kargs):
    '''
        Plot heatap for attention weights

        Args:
            All arguments are "NP arrays"
            context_x : [batch_size, the number of context_points, x_size(dimension)] 
            target_x : [batch_size, the number of context_points, x_size(dimension)] 
            attention_weights_dict : [batch_size, the num of target set, the num of context set]
    '''
    # Formatting
    f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
    g = lambda x,pos : "${}$".format(f._formatSciNotation('%1.10e' % x))
    
    
    # Path
    result_path = kargs.get("result_path", "./")
    file_name = kargs.get("file_name", "heatmap_result.png")
    FILE_NAME = os.path.join(result_path, file_name)

    # Make a plot pretty
    TITLE = kargs.get("title", "1D regression plot")
    ITERATION = kargs.get("iteration", None)
    
    # data pre-processing
    if target_x.shape[0] == 1:
        context_x = np.squeeze(context_x[idx], axis=-1)
        target_x = np.squeeze(target_x[idx], axis=-1)
        attention_weights = attention_weights[idx] if attention_weights is not None else None
    else:
        context_x = np.squeeze(context_x[idx])
        target_x = np.squeeze(target_x[idx])
        attention_weights = np.squeeze(attention_weights[idx]) if attention_weights is not None else None

    # Sorted 
    context_x_idx = sorted(range(context_x.shape[0]), key=lambda k : context_x[k])
    target_x_idx = sorted(range(target_x.shape[0]), key=lambda k : target_x[k])

    # rearrange the attention_weight
    context_x = context_x[context_x_idx]
    attention_weights = attention_weights[:, context_x_idx]

    # Make a figure
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(attention_weights, cmap=plt.cm.Blues)

    # color-bar
    cbar = plt.colorbar(heatmap)

    # put the major ticks at the middle of each cell
    ax.set_xticks(np.arange(attention_weights.shape[1]) + 0.5, minor=False) # target
    ax.set_yticks(np.arange(attention_weights.shape[0]) + 0.5, minor=False) # input

    # extra columns row
    ax.set_xlim(0, int(attention_weights.shape[1])) # context
    ax.set_ylim(0, int(attention_weights.shape[0])) # target

    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    # source words -> column labels
    ax.set_xticklabels(context_x, minor=False)
    ax.set_yticklabels(target_x, minor=False)
    
    # set ticks
    plt.xticks(rotation=45, fontsize=8)
    plt.yticks(rotation=25, fontsize=8)

    # formatting
    # ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    # ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

    # Title
    plt.title(TITLE, fontweight='bold', loc='center')

    # tight layout
    plt.tight_layout()
    
    # Save
    plt.savefig(FILE_NAME, dpi=300)

    # Close plt
    plt.clf()

if __name__ == "__main__":

    # Test
    a = [2,3,1,0,4]

    # Sort 함수
    idx = sorted(range(len(a)), key=lambda k : a[k])
    s = sorted(a)

    print(idx)
    print(s)

    for i in idx:
        print(a[i])
