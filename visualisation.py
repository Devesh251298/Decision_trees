from decision_tree_classifier import DecisionTree_Classifier
import numpy as np
from matplotlib.patches import BoxStyle
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors


# Need to find the split_value 

def visualise_decision_tree(node, tree, ax, x, y, max_x=5, max_y=5, depth=0, max_depth=8):

    kwargs = dict(
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='b'),
                ha="center",
                va="center",
                arrowprops=dict(arrowstyle="<-",edgecolor=plt.rcParams["text.color"]),
                fontsize=8
                )
    segments = []
    
    if depth <= max_depth:
            if node.parent is None:
                # root
                text = f'Feature: {node.attribute} \n Split at: {node.value}'
                ax.annotate(text, xy=(x,y), **kwargs)


            if node.left != None:
                
                if node.left.leaf:
                    xl = x - 0.1
                    yl = y - 0.1   
                    segments.append([[x, y], [xl, yl]]) 
                    text = f'Leaf: {node.left.label}'
                    kwargs['bbox']['facecolor']='green'
                    ax.annotate(text, xy=(xl,yl), **kwargs)
                    line_segments = LineCollection(segments, linewidths=1, linestyle='solid')
                    ax.add_collection(line_segments)
                    segments.remove([[x, y], [xl, yl]])
                    
                    
                else:
                    xl = x - 0.5
                    yl = y - 0.1    
                    segments.append([[x, y], [xl, yl]]) 
                    text = f'Feature: {node.left.attribute}\n Split at: {node.left.value}'
                    ax.annotate(text, xy=(xl,yl), **kwargs)
                    kwargs['bbox']['facecolor']='white'
                    line_segments = LineCollection(segments, linewidths=1, linestyle='solid')
                    ax.add_collection(line_segments)
                    visualise_decision_tree(node.left, tree, ax, xl, yl, max_x, max_y, depth=depth + 1)
                   
            if node.right != None:
                
                if node.right.leaf:
                    xr = x + 0.1
                    yr = y - 0.1 
                    segments.append([[x, y], [xr, yr]])
                    text = f'Leaf: {node.right.label}'
                    kwargs['bbox']['facecolor']='green'
                    ax.annotate(text, xy=(xr,yr), **kwargs)
                    line_segments = LineCollection(segments, linewidths=1, linestyle='solid')
                    segments.remove([[x, y], [xr, yr]])
                    ax.add_collection(line_segments)
                    
                else:
                    xr = x + 0.5
                    yr = y - 0.1
                    segments.append([[x, y], [xr, yr]])
                    text = f'Feature: {node.right.attribute} \n Split at:{node.right.value}'
                    kwargs['bbox']['facecolor']='white'
                    ax.annotate(text, xy=(xr,yr), **kwargs)
                    line_segments = LineCollection(segments, linewidths=1, linestyle='solid')
                    ax.add_collection(line_segments)
                    visualise_decision_tree(node.right, tree, ax, xr, yr, max_x, max_y, depth=depth + 1)
                    

def test_visualise():
    dataset = np.loadtxt("wifi_db/clean_dataset.txt", dtype=float)
    # from sklearn.tree import DecisionTreeClassifier
    # from sklearn import tree
    # clf = DecisionTreeClassifier(criterion = "entropy")
    # clf.fit(dataset[:,:-1], dataset[:,-1])
    # tree.plot_tree(clf)

    dtree = DecisionTree_Classifier()
    dtree.fit(dataset)
    
    
    fig, ax = plt.subplots(figsize=(10,10))
    visualise_decision_tree(node=dtree.dtree, tree=dtree, x=0, y=1, ax=ax, max_depth=8, max_x=6, max_y=5)
    ax.set_ylim(0.25,1.1)
    ax.set_xlim(-2,2)
    ax.axis('off')
   
  
    plt.show()
     
if __name__ == "__main__":
    test_visualise()
    