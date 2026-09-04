def draw_tilt_and_corr_density_shade(T, Corr, uniname, saveFigures, n_bins = 100, title = None):
    """ 
    Generate the Glazer plot. 

    Args:
        T (numpy.ndarray): Tilt data.
        Corr (numpy.ndarray): Correlation data.
        uniname (str): User-defined name for printing and figure saving.
        saveFigures (bool): Whether to save the figure.
        n_bins (int): Number of bins for the histogram.
        title (str): Title of the plot.

    Returns:
        list of floats: tilt correlation polarity (TCP) values of each axis.
    """
    
    fig_name=f"traj_tilt_corr_density_{uniname}.png"
    
    corr_power = 2.5
    fill_alpha = 0.5
    
    T_a = T[:,:,0].reshape((T.shape[0]*T.shape[1]))
    T_b = T[:,:,1].reshape((T.shape[0]*T.shape[1]))
    T_c = T[:,:,2].reshape((T.shape[0]*T.shape[1]))
    tup_T = (T_a,T_b,T_c)
    assert len(tup_T) == 3
    assert Corr.shape[2] == 3 or Corr.shape[2] == 6

    if Corr.shape[2] == 3:
        C = (Corr[:,:,0].reshape((Corr.shape[0]*Corr.shape[1])),
             Corr[:,:,1].reshape((Corr.shape[0]*Corr.shape[1])),
             Corr[:,:,2].reshape((Corr.shape[0]*Corr.shape[1])))
    elif Corr.shape[2] == 6:
        C = (np.concatenate((Corr[:,:,0],Corr[:,:,1]),axis=0).reshape((Corr.shape[0]*2*Corr.shape[1])),
             np.concatenate((Corr[:,:,2],Corr[:,:,3]),axis=0).reshape((Corr.shape[0]*2*Corr.shape[1])),
             np.concatenate((Corr[:,:,4],Corr[:,:,5]),axis=0).reshape((Corr.shape[0]*2*Corr.shape[1])))

    figs, axs = plt.subplots(3, 1)
    labels = [r'$\mathit{a}$',r'$\mathit{b}$',r'$\mathit{c}$']
    colors = ["C0", "C1", "C2"]
    #rgbcode = np.array([[0,1,0,fill_alpha],[0,0,1,fill_alpha],[1,0,0,fill_alpha]])
    por = [0,0,0]
    for i in range(3):
        
        y,binEdges=np.histogram(tup_T[i],bins=n_bins,range=[-45,45])
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        yt=y/max(y)
        axs[i].plot(bincenters,yt,label = labels[i], color = colors[i],linewidth = 2.4)
        #axs[i].text(0.03, 0.82, labels[i], horizontalalignment='center', fontsize=14, verticalalignment='center', transform=axs[i].transAxes)

        y,binEdges=np.histogram(C[i],bins=n_bins,range=[-45,45]) 
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        yc=y/max(y)
        yy=yt*yc

        axs[i].fill_between(bincenters, yy, 0, facecolor = colors[i], alpha=fill_alpha, interpolate=True)
        axs[i].text(0.03, 0.82, labels[i], horizontalalignment='center', fontsize=16, verticalalignment='center', transform=axs[i].transAxes, style='italic')
        
        axs[i].set_ylim(bottom=0)
        
        parneg = np.sum(np.power(yc,corr_power)[bincenters<0])
        parpos = np.sum(np.power(yc,corr_power)[bincenters>0])
        por[i] = (-parneg+parpos)/(parneg+parpos)
        
    for ax in axs.flat:
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_ylabel('Counts (a.u.)', fontsize = 15) # Y label
        ax.set_xlabel(r'Tilt Angle ($\degree$)', fontsize = 15) # X label
        ax.set_xlim([-45,45])
        ax.set_xticks([-45,-30,-15,0,15,30,45])
        ax.set_yticks([])
        
    axs[0].xaxis.set_ticklabels([])
    axs[1].xaxis.set_ticklabels([])
    axs[0].yaxis.set_ticklabels([])
    axs[1].yaxis.set_ticklabels([])
    axs[2].yaxis.set_ticklabels([])
    axs[0].set_xlabel("")
    axs[0].set_ylabel("")
    axs[1].set_xlabel("")
    axs[2].set_ylabel("")

    if not title is None:
        axs[0].set_title(title,fontsize=17)
        
    if saveFigures:
        plt.savefig(fig_name, dpi=350,bbox_inches='tight')
        
    plt.show()
    
    div = 0.35
    scal = 4/3
    for ci, cval in enumerate(por):
        if abs(cval) > div:
            cval = np.sign(cval)*(np.power(((np.abs(cval)-div)/(1-div)),1/scal)*(1-div)+div)
            por[ci] = np.round(cval,4)
        else:
            cval = np.sign(cval)*(np.power(np.abs(cval)/div,scal)*div)
            por[ci] = np.round(cval,4)
    
    return por
