from decision_tree_classifier import DecisionTree_Classifier
import numpy as np
from matplotlib.patches import BoxStyle
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors


# Need to find the split_value 

def visualise_decision_tree(node, tree, ax, x, y, grid = [], grid_x = [], grid_y = [], max_x=5, max_y=5, depth=0, max_depth=3):

    kwargs = dict(
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='b'),
                ha="center",
                va="center",
                arrowprops=dict(arrowstyle="<-",edgecolor=plt.rcParams["text.color"]),
                fontsize=5
                )
    segments = []

    dim_x = grid_x[y,x]
    dim_y = grid_y[y,x]
    if depth < max_depth:
            if node.parent is None:
                # root
                text = f'Feature: {node.attribute} \n Split at: {node.value}'
                ax.annotate(text, xy=(dim_x,dim_y), **kwargs)

            if node.left != None:         
                if node.left.leaf:

                    xl = x - np.power(2,(max_depth-depth-1))
                    yl = y - 1
                    dim_xl = grid_x[yl,xl]
                    dim_yl = grid_y[yl,xl]

                    segments.append([[dim_x, dim_y], [dim_xl, dim_yl]]) 
                    text = f'Leaf: {node.left.label}'
                    kwargs['bbox']['facecolor']='green'
                    ax.annotate(text, xy=(dim_xl,dim_yl), **kwargs)
                    line_segments = LineCollection(segments, linewidths=1, linestyle='solid')
                    ax.add_collection(line_segments)
                    segments.remove([[dim_x, dim_y], [dim_xl, dim_yl]])
                    
                    
                else:
                    xl = x - np.power(2,(max_depth-depth-1))
                    yl = y - 1   
                    dim_xl = grid_x[yl,xl]
                    dim_yl = grid_y[yl,xl]

                    segments.append([[dim_x, dim_y], [dim_xl, dim_yl]]) 
                    text = f'Feature: {node.left.attribute}\n Split at: {node.left.value}'
                    ax.annotate(text, xy=(dim_xl,dim_yl), **kwargs)
                    kwargs['bbox']['facecolor']='white'
                    line_segments = LineCollection(segments, linewidths=1, linestyle='solid')
                    ax.add_collection(line_segments)
                    visualise_decision_tree(node.left, tree, ax, xl, yl, grid, grid_x, grid_y,max_x, max_y, depth=depth + 1, max_depth=max_depth)
                   
            if node.right != None:
                if node.right.leaf:
                    xr = x + np.power(2,(max_depth-depth-1))
                    yr = y - 1

                    dim_xr = grid_x[yr,xr]
                    dim_yr = grid_y[yr,xr]

                    segments.append([[dim_x, dim_y], [dim_xr, dim_yr]])
                    text = f'Leaf: {node.right.label}'
                    kwargs['bbox']['facecolor']='green'
                    ax.annotate(text, xy=(dim_xr,dim_yr), **kwargs)
                    line_segments = LineCollection(segments, linewidths=1, linestyle='solid')
                    segments.remove([[dim_x, dim_y], [dim_xr, dim_yr]])
                    ax.add_collection(line_segments)
                    
                else:
                    xr = x + np.power(2,(max_depth-depth-1))
                    yr = y - 1

                    dim_xr = grid_x[yr,xr]
                    dim_yr = grid_y[yr,xr]

                    segments.append([[dim_x, dim_y], [dim_xr, dim_yr]])
                    text = f'Feature: {node.right.attribute} \n Split at:{node.right.value}'
                    kwargs['bbox']['facecolor']='white'
                    ax.annotate(text, xy=(dim_xr,dim_yr), **kwargs)
                    line_segments = LineCollection(segments, linewidths=1, linestyle='solid')
                    ax.add_collection(line_segments)
                    visualise_decision_tree(node.right, tree, ax, xr, yr, grid, grid_x, grid_y,max_x, max_y, depth=depth + 1,max_depth=max_depth)

    # print('Left : ',left_pos,' Right : ',right_pos)
                    

def test_visualise():
    dataset = np.loadtxt("wifi_db/clean_dataset.txt", dtype=float)
    # from sklearn.tree import DecisionTreeClassifier
    # from sklearn import tree
    # clf = DecisionTreeClassifier(criterion = "entropy")
    # clf.fit(dataset[:,:-1], dataset[:,-1])
    # tree.plot_tree(clf)
    max_depth = 10
    grid = np.zeros((max_depth+1,np.power(2,max_depth+1)))
    grid_x = np.zeros((max_depth+1,np.power(2,max_depth+1)))
    grid_y = np.zeros((max_depth+1,np.power(2,max_depth+1)))
    scale = 5
    step = scale/max_depth
    for i in range(max_depth+1):
        grid_y[i,:] = 0.1*i

        for j in range(len(grid_x[i])):
            grid_x[i,j] = (scale)*(j - int(np.power(2,max_depth+1)/2)) + 2
    dtree = DecisionTree_Classifier()
    dtree.fit(dataset)
    
    
    fig, ax = plt.subplots(figsize=(10,10))
    visualise_decision_tree(node=dtree.dtree, tree=dtree, grid = grid, grid_x = grid_x, grid_y = grid_y,x=np.power(2,max_depth), y=max_depth, ax=ax, max_depth=max_depth, max_x=6, max_y=5)
    ax.margins(0.2, 0.2)  
    ax.axis('off')
   
    plt.ylim(0.45, 1)
    plt.show()
     
if __name__ == "__main__":
    test_visualise()
    
