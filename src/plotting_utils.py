import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors as clrs

  
   

    
def plot_qs_and_actions(qvalues,num_actions,targets,range_x,range_y):
    

    X, Y = np.meshgrid(range(1,range_x+1),range(1,range_y+1))
    
    cmap=plt.cm.jet
    norm = clrs.BoundaryNorm(np.arange(-.5,num_actions+.5), cmap.N)
    
    for qvalue,target in zip(qvalues,targets):

        max_q = np.max(qvalue[1:,1:],axis=2)
        
        fig = plt.figure(figsize=(10,14))
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312,projection="3d")
        ax3 = fig.add_subplot(313)

        
        

        im1 = ax1.imshow(np.argmax(qvalue[1:,1:],axis=2),norm=norm,cmap=cmap,origin="lower")
             
        #TODO instead of scatter use vlines and hlines to mark target area.
        ax1.scatter(target["vaxis"],target["haxis"],\
                    color="red",label="target area",marker="s",s=1000,facecolors='none', edgecolors='r',lw=2)
        ax3.scatter(target["vaxis"],target["haxis"],\
                    color="red",label="target area",marker="s",s=1000,facecolors='none', edgecolors='r',lw=2)
                    
        surf = ax2.plot_surface(X, Y, max_q, rstride=1, cstride=1, cmap=cm.coolwarm,
                linewidth=0, antialiased=False)
    
        im3 = ax3.imshow(max_q,origin="lower",cmap=cm.coolwarm)


        cbar = fig.colorbar(im1,ticks=np.arange(num_actions),spacing="uniform",ax=ax1)   
        fig.colorbar(surf, shrink=0.5, aspect=5, ax = ax2)
        fig.colorbar(im3,ax=ax3)
        ax1.legend(bbox_to_anchor=[1.1,1.15])

        ax1.set_title("Action per state")
        ax1.set_xlabel("Width")
        ax1.set_ylabel("Height")
        ax2.set_ylabel("Height")
        ax2.set_xlabel("Width")
        ax2.set_title("Max Q value per state")
        ax3.set_ylabel("Height")
        ax3.set_xlabel("Width")

        cbar.ax.set_yticklabels(['Increase width D=1', 'decrease width D=1',\
                                 'increase height D=1','decrease height D=1',\
                                'Increase width D=2', 'decrease width D=2',\
                                 'increase height D=2','decrease height D=2']);

        ax1.set_xlim(1,range_x-1)
        ax1.set_ylim(1,range_y-1)

        plt.show()
        
        

def plot_variances(variances_dict,figrows=8,figcols=16,log=True):
    
    title_tmplt = "Variance per position for each node in "
    cmap=plt.cm.Reds
  
    identity = lambda x: x
    funk = np.log if log else identity  
    shift = 1 if log else 0
    label = "$\log$(variance+1) per position" if log else "variance per position"
    

    all_log_variances = [funk(var+shift) for _,var in variances_dict.items()]
    _min,_max = (np.min(all_log_variances), np.max(all_log_variances))
    
    
    
    norm = clrs.Normalize(_min,_max)
    
    for name,var in variances_dict.items():
        fig, axes = plt.subplots(figrows,figcols,figsize=(10,5))
        axes = axes.flatten()
        fig.suptitle(title_tmplt+name,fontsize=18)
        for idx,plot in enumerate(range(var.shape[-1])):
            im = axes[idx].imshow(funk(var[:,:,idx]+shift),cmap=cmap,norm=norm)
            axes[idx].set_xticks([])
            axes[idx].set_yticks([])
        plt.subplots_adjust(bottom=0.1, right=0.9, top=0.9)
        cax = plt.axes([1.05, 0.25, 0.025, 0.45])
        cbar = plt.colorbar(im,cax=cax)

        cbar.set_label(label)
        
        plt.show()
