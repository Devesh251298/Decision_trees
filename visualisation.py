from decision_tree_classifier import DecisionTree_Classifier
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def visualise_decision_tree(node, tree, ax, x, y, grid = [], grid_x = [], grid_y = [], max_x=5, max_y=5, depth=0, max_depth=3):

    # Parameters for text boxes in tree
    kwargs = dict(
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='b'),
                ha="center",
                va="center",
                arrowprops=dict(arrowstyle="<-",edgecolor=plt.rcParams["text.color"]),
                fontsize=5
                )
    # Empty list for points defining the lines that connect the nodes
    segments = [] 

    # Grid to scale dimensions at each level of depth - for both x and y coordinates
    dim_x = grid_x[y,x]
    dim_y = grid_y[y,x]

    # Loop until max_depth has been reached
    if depth < max_depth:

        # Check if node is node root 
        if node.parent is None:
            # Root
            text = f'Feature: {node.attribute} \n Split at: {node.value}' 
            # Add a text box at the position of the current node with the attribute and value of the node
            ax.annotate(text, xy=(dim_x,dim_y), **kwargs)

        # if node to the left of current node is not empty
        if node.left != None:    
            #  If the node to the left of the current node is a leaf
            if node.left.leaf:
                # Update position of the node and scale in the coordinate system 
                xl = x - np.power(2,(max_depth-depth-1))
                yl = y - 1
                dim_xl = grid_x[yl,xl]
                dim_yl = grid_y[yl,xl]

                # Append points defining the line that connects parent-child node
                segments.append([[dim_x, dim_y], [dim_xl, dim_yl]]) 

                # Add a text box at the position of the leaf node 
                text = f'Leaf: {node.left.label}'
                kwargs['bbox']['facecolor']='green'
                ax.annotate(text, xy=(dim_xl,dim_yl), **kwargs)

                # Turn the points in segments list into lines
                line_segments = LineCollection(segments, linewidths=1, linestyle='solid')
                ax.add_collection(line_segments)
                segments.remove([[dim_x, dim_y], [dim_xl, dim_yl]])
                
            # If node is not a leaf
            else: 
                # Update position of the node and scale in the coordinate system 
                xl = x - np.power(2,(max_depth-depth-1))
                yl = y - 1   
                dim_xl = grid_x[yl,xl]
                dim_yl = grid_y[yl,xl]

                # Append points defining the line that connects parent-child node
                segments.append([[dim_x, dim_y], [dim_xl, dim_yl]]) 

                # Add a text box at the position of the node displaying the attribute and the splitting value
                text = f'Feature: {node.left.attribute}\n Split at: {node.left.value}'
                ax.annotate(text, xy=(dim_xl,dim_yl), **kwargs)
                kwargs['bbox']['facecolor']='white' # Change default colour of the text box to white for the non-terminal node

                # Turn the points in segments list into lines
                line_segments = LineCollection(segments, linewidths=1, linestyle='solid')
                ax.add_collection(line_segments)

                # Run function recursively for the node to the left, adding a level of depth and with the updated x,y parameters
                visualise_decision_tree(node.left, tree, ax, xl, yl, grid, grid_x, grid_y,max_x, max_y, depth=depth + 1, max_depth=max_depth)
                
        #  If the node to the right of the current node is a leaf      
        if node.right != None:
            #  If the node to the right of the current node is a leaf
            if node.right.leaf:
                # Update position of the node and scale in the coordinate system 
                xr = x + np.power(2,(max_depth-depth-1))
                yr = y - 1
                dim_xr = grid_x[yr,xr]
                dim_yr = grid_y[yr,xr]

                # Append points defining the line that connects parent-child node
                segments.append([[dim_x, dim_y], [dim_xr, dim_yr]])

                # Add a text box at the position of the leaf node 
                text = f'Leaf: {node.right.label}'
                kwargs['bbox']['facecolor']='green' # green box for a leaf node
                ax.annotate(text, xy=(dim_xr,dim_yr), **kwargs)

                # Turn the points in segments list into lines
                line_segments = LineCollection(segments, linewidths=1, linestyle='solid')
                segments.remove([[dim_x, dim_y], [dim_xr, dim_yr]])
                ax.add_collection(line_segments)

             # If node is not a leaf
            else:
                # Update position of the node and scale in the coordinate system 
                xr = x + np.power(2,(max_depth-depth-1))
                yr = y - 1
                dim_xr = grid_x[yr,xr]
                dim_yr = grid_y[yr,xr]

                # Append points defining the line that connects parent-child node
                segments.append([[dim_x, dim_y], [dim_xr, dim_yr]])
                
                 # Add a text box at the position of the node displaying the attribute and the splitting value
                text = f'Feature: {node.right.attribute} \n Split at:{node.right.value}'
                kwargs['bbox']['facecolor']='white'                
                ax.annotate(text, xy=(dim_xr,dim_yr), **kwargs)

                 # Turn the points in segments list into lines
                line_segments = LineCollection(segments, linewidths=1, linestyle='solid')
                ax.add_collection(line_segments)

                # Run function recursively for the node to the right, adding a level of depth and with the updated x,y parameters
                visualise_decision_tree(node.right, tree, ax, xr, yr, grid, grid_x, grid_y,max_x, max_y, depth=depth + 1,max_depth=max_depth)


def test_visualise():

    # Load dataset and fit it with the decision tree classifier
    dataset = np.loadtxt("wifi_db/clean_dataset.txt", dtype=float)
    dtree = DecisionTree_Classifier()
    dtree.fit(dataset)

    max_depth = dtree.compute_depth(dtree.dtree, 0)

    # Initialise grid for scaling of decision tree visualisation
    grid = np.zeros((max_depth+1,np.power(2,max_depth+1)))
    grid_x = np.zeros((max_depth+1,np.power(2,max_depth+1)))
    grid_y = np.zeros((max_depth+1,np.power(2,max_depth+1)))

    # Scaling factor
    scale = 5
    for i in range(max_depth+1):
        # Compute the different scales of the grid for x and y according to the depth of the graph
        grid_y[i,:] = 0.1*i

        for j in range(len(grid_x[i])):
            grid_x[i,j] = (scale)*(j - int(np.power(2,max_depth+1)/2)) + 2
    
    # Initialise figure
    fig, ax = plt.subplots(figsize=(10,10))
    visualise_decision_tree(node=dtree.dtree, tree=dtree, grid = grid, grid_x = grid_x, grid_y = grid_y,x=np.power(2,max_depth), y=max_depth, ax=ax, max_depth=max_depth, max_x=6, max_y=5)
    ax.margins(0.2, 0.2)  
    ax.axis('off') # Remove axis
   
    plt.ylim(0.1*(max_depth-5.5), 0.1*max_depth)
    plt.show()
     
if __name__ == "__main__":
    test_visualise()
    
