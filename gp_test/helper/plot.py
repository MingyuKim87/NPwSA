import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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
    # plt.title(TITLE, fontweight='bold', loc='center')
    
    # Label
    plt.xlabel("{} iterations".format(ITERATION)) if not ITERATION is None else None
    
    # Axis
    plt.yticks([-2, 0, 2], fontsize=16)
    plt.xticks([-4, 0, 4], fontsize=16)
    #plt.ylim([-1, 1])
    plt.grid(False)
    ax = plt.gca()
    ax.set_facecolor('white')

    # make pretty
    plt.tight_layout()

    # Save
    plt.savefig(FILE_NAME, dpi=300)

    # Close plt
    plt.close()


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
    if np.isclose(np.std(attention_weights), 0):
        heatmap.set_clim(0.0, 0.5)

    # put the major ticks at the middle of each cell
    ax.set_xticks(np.arange(attention_weights.shape[1]) + 0.5, minor=False) # context
    ax.set_yticks(np.arange(attention_weights.shape[0], step=10) + 0.5, minor=False) # target

    # extra columns row
    ax.set_xlim(0, int(attention_weights.shape[1])) # context
    ax.set_ylim(0, int(attention_weights.shape[0])) # target

    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    # source words -> column labels
    ax.set_xticklabels(np.round_(context_x, 2), minor=False)
    ax.set_yticklabels(np.round_(target_x[np.arange(attention_weights.shape[0], step=10)],2), minor=False)
    
    # set ticks
    plt.xticks(rotation=45, fontsize=8)
    plt.yticks(rotation=25, fontsize=8)

    # formatting
    # ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    # ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

    # Title
    # plt.title(TITLE, fontweight='bold', loc='center')

    # tight layout
    plt.tight_layout()
    
    # Save
    plt.savefig(FILE_NAME, dpi=300)

    # Close plt
    plt.close()

def plot_training_curve(train_file_path, val_file_path=None, k=20, **kargs):
    '''
        Args:
            train_file_path : training result path
            val_file_path : val result path    
            k : paraemter for exponential moving average
    '''

    # Path
    result_path = kargs.get("result_path", "./")
    file_name = kargs.get("file_name", "training_curve.png")
    FILE_NAME = os.path.join(result_path, file_name)
    
    # Make a title
    TITLE = kargs.get("title", "training curve plot")

    # train
    train_temp = pd.read_csv(train_file_path, header=None)
    train_temp.columns = ["epochs", "loss", "negative_likelihood", "kld", "kld_in_attention"]
    train_temp['type'] = 'train'

    # add the EMA loss
    train_temp['EMA_loss'] = train_temp["loss"].ewm(k).mean()
    
    # val
    if val_file_path is not None:
        val_temp = pd.read_csv(val_file_path, header=None)
        val_temp.columns = ["epochs", "loss"]
        val_temp['type'] = 'val'

        # add the EMA loss
        val_temp['EMA_loss'] = val_temp["loss"].ewm(k).mean()
        
    # Make a plot pretty
    sns.set_style("darkgrid")
    
    # integrated dataframe
    integrated_df = pd.concat([train_temp, val_temp]) if val_file_path is not None else train_temp

    # plot a line 
    ax = sns.lineplot(data=integrated_df, x='epochs', y='loss', hue='type', alpha=0.3, legend=False)
    ax = sns.lineplot(data=integrated_df, x='epochs', y='EMA_loss', hue='type')

    # make pretty
        # Title
    # plt.title(TITLE, fontweight='bold', loc='center')
    plt.tight_layout()
    
    # save fig
    plt.savefig(FILE_NAME, dpi=300)

    # Close plt
    plt.close()

if __name__ == "__main__":
    train_file_path = "./save_models/two_gp/fixed_steps_200000_fixed_params_iwae_20210324/ANP/test/ANP_result_during_training.txt"
    val_file_path = "./save_models/two_gp/fixed_steps_200000_fixed_params_iwae_20210324/ANP/test/ANP_val_result_during_training.txt"
    plot_training_curve(train_file_path, val_file_path)

