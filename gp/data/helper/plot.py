import numpy as np
import matplotlib.pyplot as plt

def plot_test(context_x, context_y, target_x, target_y):
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
    result_path = "."
    file_name = '/data_test.png'
    
    RESULT_PATH = result_path
    FILE_NAME = RESULT_PATH + file_name

    plt.plot(target_x, target_y, 'k:', linewidth=1)
    plt.plot(context_x, context_y, 'ko', markersize=10)

    # Make a plot pretty
    plt.title("data_test", fontweight='bold', loc='center')
    #plt.xlabel("{} iterations".format(iteration))
    plt.yticks([-2, 0, 2], fontsize=16)
    plt.xticks([-4, 0, 4], fontsize=16)
    #plt.ylim([-1, 1])
    plt.grid(False)
    ax = plt.gca()
    ax.set_facecolor('white')

    plt.savefig(FILE_NAME, dpi=300)

    plt.clf()
