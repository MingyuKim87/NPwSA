import os
import numpy as np
import matplotlib.pyplot as plt

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
        context_x = np.squeeze(context_x[idx], axis=0)
        context_y = np.squeeze(context_y[idx], axis=0)
        target_x = np.squeeze(target_x[idx], axis=0)
        target_y = np.squeeze(target_y[idx], axis=0)
        pred = np.squeeze(pred[idx], axis=0)
        std = np.squeeze(std[idx], axis=0)
    else:
        context_x = np.squeeze(context_x[idx])
        context_y = np.squeeze(context_y[idx])
        target_x = np.squeeze(target_x[idx])
        target_y = np.squeeze(target_y[idx])
        pred = np.squeeze(pred[idx])
        std = np.squeeze(std[idx])

    if context_y.shape[-1] != 1 or target_y.shape[-1] != 1:
        for i in range(context_y.shape[-1]):
            # temporarily assign containers
            temp_context_y = context_y[:, i]
            temp_target_y = target_y[:, i]
            temp_pred = pred[:, i]
            temp_std = std[:, i]

            # file name
            FILE_NAME_ = FILE_NAME + "_" + str(i)
            
            # scatter plot
            plt.plot(target_x, temp_target_y, 'r*', markersize=3)

            # scatter plot
            plt.plot(context_x, temp_context_y, 'ko', markersize=3)
            
            if pred is not None and std is not None:
                # line plot
                plt.plot(target_x, temp_pred, 'b', linewidth=1)

                # var
                plt.fill_between(target_x, \
                    temp_pred - temp_std,\
                    temp_pred + temp_std,
                    alpha = 0.2,
                    facecolor='#65c9f7',
                    interpolate=True
                )
            
            # Make a plot pretty
            TITLE = kargs.get("title", "1D regression plot")
            ITERATION = kargs.get("iteration", None)

            # Title
            plt.title(TITLE + "_" + str(i), fontweight='bold', loc='center')
        
            # Label
            plt.xlabel("{} iterations".format(ITERATION)) if not ITERATION is None else None
        
            '''
            # Axis
            plt.yticks([-2, 0, 2], fontsize=16)
            plt.xticks([-4, 0, 4], fontsize=16)
            #plt.ylim([-1, 1])
            '''
            plt.grid(False)
            
            ax = plt.gca()
            ax.set_facecolor('white')

            # Save
            plt.savefig(FILE_NAME_, dpi=300)

            # Close plt
            plt.clf()

    else:
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
    


    
