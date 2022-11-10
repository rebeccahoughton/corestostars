import numpy as np
import matplotlib.pyplot as plt
import imf_master.imf.imf as imf

#Open multiplicities file
fractions = np.loadtxt("multiplicities",usecols=(0,1,2,3,4,5,6,7),skiprows=2)

#Formatting
title_size="14"
tick_size="10"
label_size="12"

# The Maschberger core mass function
def maschberger(alpha3,beta,mu,mlo,mup,n):
    '''
    Generates the Maschberger IMF. 
    '''
    oma3 = 1 - alpha3
    omb  = 1 - beta
    Gmlo = (1+(mlo/mu)**oma3)**omb
    Gmup = (1+(mup/mu)**oma3)**omb
    Gm = np.random.uniform(0,1,size=n)*(Gmup-Gmlo) + Gmlo
    Mc = mu*(Gm**(1/omb)-1)**(1/oma3)
    return(Mc)

# Plotting multiplicity fractions
def MF_CSF(ndim,fractions,horizontal=True):
    '''
    Plot the multiplicity, triple higher order fractions, and 
    the companion star fractions from Table 1 of Offner et al. (2022).
    '''

    # Unpacking data and getting rid of NaNs
    MF_obs = fractions[:,2][~np.isnan(fractions[:,2])]/100
    MF_yerr = fractions[:,3][~np.isnan(fractions[:,3])]/100
    THF_obs = fractions[:,4][~np.isnan(fractions[:,4])]/100
    THF_yerr = fractions[:,5][~np.isnan(fractions[:,5])]/100
    CSF_obs = fractions[:,6][~np.isnan(fractions[:,6])]
    CSF_yerr = fractions[:,7][~np.isnan(fractions[:,7])]
    x_err = (fractions[:,1][~np.isnan(fractions[:,1])]-fractions[:,0][~np.isnan(fractions[:,0])])/2

    # Getting the midpoints of mass ranges for x array
    x = fractions[:,0][~np.isnan(fractions[:,0])]+x_err

    # Label setup for either horizontal (2x columns) or vertical (2x rows) plots
    if horizontal==False:
        fig,ax = plt.subplots(2,1,figsize=[5,7],sharex=True,gridspec_kw={'hspace':0.0,'wspace':0.0})
        ax[0].set_ylabel("Multiplicity / \n Triple High-order Fractions",fontsize=label_size)
        ax[1].set_ylabel("Companion Star Fraction",fontsize=label_size)
        ax[1].set_xlabel(r"Primary Mass ($M_\odot$)",fontsize=label_size)
        ax[0].set_ylim(-0.05,1.1)
        ax[1].set_ylim(-0.15,3.2)
    else:
        fig,ax = plt.subplots(ndim,2,figsize=[10,4*ndim],sharex=True,gridspec_kw={'hspace':0,'wspace':0.2})
    fig.get_constrained_layout()

    # Reshape axes for plotting in a loop
    ax = np.reshape(ax, (ndim*2))

    # Offner et al. (2022) data
    for j in range(ndim):
        ax[j*2].errorbar(  x, MF_obs,xerr=x_err, yerr=MF_yerr,color="k",marker="s",ms=5,markerfacecolor="b",label="MF",ls=' ',capsize=1.2,lw=1.0)
        ax[j*2+1].errorbar(x,CSF_obs,xerr=x_err,yerr=CSF_yerr,color="k",marker="s",ms=5,markerfacecolor="limegreen",label="CSF",ls=' ',capsize=1.2,lw=1.0)#,lw=2.0)
        ax[j*2].errorbar(  x,THF_obs,xerr=x_err,yerr=THF_yerr,color="k",marker="o",ms=5,markerfacecolor="r",label="THF",ls=' ',capsize=1.2,lw=1.0)
    
    # Legend on top plot only
    ax[0].legend(loc="upper left",fontsize=label_size)
    ax[1].legend(loc="upper left",fontsize=label_size)

    # Formatting
    for i in range(2*ndim):
        ax[i].set_xscale("log")
        ax[i].tick_params(axis="both",which="both",direction="in",top=True,right=True,labelsize=12)

    # Extra formatting for horizontal plots
    if horizontal==True:
        ax[-1].set_xlabel(r"Primary Mass ($M_\odot$)")
        ax[-2].set_xlabel(r"Primary Mass ($M_\odot$)")

        # Gets the labels right when adding extra rows of subplots
        fig.add_subplot(111, frameon=False)
        # Hide tick and tick label of the big axis
        plt.tick_params(labelcolor='none',which='both',top=False,bottom=False,left=False,right=False)
        plt.ylabel(r"$\bf{Multiplicity}$ / Triple High-order Fractions",labelpad=-0.5)
        axa = fig.add_subplot(122, frameon=False)
        # Hide tick and tick label of the big axis
        plt.tick_params(labelcolor='none',which='both',top=False,bottom=False,left=False,right=False)
        axa.set_ylabel("Companion Star Fraction")#,labelpad=-0.5)
    
    return(fig,ax)


def IMF_plot(Masch=True,Kroupa=True,Salpeter=True,Chabrier05=True,horizontal=True):
    '''
    Plot any of the Maschberger, Kroupa, Salpeter, or Chabrier forms
    of the initial mass function. 
    '''
    # Mass array for the x axis
    mass = np.logspace(np.log10(0.01),np.log10(100),25)

    font = {'fontname':'DejaVu Sans'}

    # Defining the different IMF functions
    masch_single = maschberger(2.3,1.4,0.2,0.01,150,int(5e6))
    masch_system = maschberger(2.3,2.0,0.2,0.01,150,int(5e6))
    kroupa = imf.Kroupa(mmin=0.01)
    salpeter = imf.Salpeter(mmin=0.3)
    chabrier_ln = imf.ChabrierLogNormal()
    chabrier_pl = imf.ChabrierPowerLaw()
    chabrier05_system = imf.ChabrierPowerLaw(lognormal_width=0.55*np.log(10),
                                             lognormal_center=0.2, alpha=2.35)
    chabrier05_single = imf.ChabrierPowerLaw(lognormal_width=0.69*np.log(10),
                                             lognormal_center=0.08, alpha=2.35)

    # Setting limits for the Kroupa and Salpeter IMFs
    ind_s=np.where(mass>0.3)[0][0]
    ind_k=np.where(mass>0.01)[0][0]

    # Colours (from David Nichols colourblind friendly IBM)
    cols = ["#FE6100","#648FFF","#DC267F"] # Could swap chabrier to #785EF0, yellow: #FFB000
    #cols = ["#f5591b","#009E73","#E69F00"] # Plots before changing to system and ss IMFs
    #      Salpeter    Chabrier    Masch

    if horizontal==True:
        fig,ax = plt.subplots(1,2,figsize=[10,4],sharey=True,gridspec_kw={'hspace':0,'wspace':0.1})
        ax[0].set_ylabel(r"$dN/dlog M$",fontsize=label_size,**font)
        ax[0].set_xlabel(r"Stellar Mass ($M_\odot$)",fontsize=label_size,**font)
        ax[1].set_xlabel(r"Stellar Mass ($M_\odot$)",fontsize=label_size,**font)
    else:
        fig,ax = plt.subplots(2,1,figsize=[6,8],sharex=True,gridspec_kw={'hspace':0.0,'wspace':0.0})
        ax[0].set_ylabel("dN/dlog M",fontsize=label_size,**font)
        ax[1].set_ylabel("dN/dlog M",fontsize=label_size,**font)
        ax[1].set_xlabel(r"Stellar Mass ($\rm{M}_\odot$)",fontsize=label_size,**font)
    fig.get_constrained_layout()

    # Logging axes and setting limits 
    for i in range(2):
        #Setup
        ax[i].tick_params(axis="both",which="both",direction="in",top=True,right=True,labelsize=12,pad=4.2)
        ax[i].set_xscale("log")
        ax[i].set_yscale('log')
        ax[i].set_xlim(0.01,100)
        ax[i].set_ylim(0.0002,1.4)
        ax[i].axvspan(0,0.08,color="k",alpha=0.1)

    # Parameters
    line_width = 4.0
    tp = 0.52
    
    # Note: The Salpeter and Maschberger functions here are scaled by a factor of 1/np.log(10),
    # which I think comes from something related to equation 2 of Chabrier (2003). Everything 
    # is now scaled to match the scaling in Maschberger (2013). 

    # Salpeter
    for i in range(2):
        ax[i].plot(mass[ind_s:],(salpeter(mass,integral_form=False)[ind_s:])*mass[ind_s:]/np.log(10),
                   label="Salpeter55",lw=line_width,alpha=tp,color=cols[0]) if Salpeter==True else None

    # Kroupa
    ax[1].plot(mass[ind_k:],(kroupa(mass,integral_form=False)[ind_k:])*mass[ind_k:]/np.log(10),
               label="Kroupa01",lw=line_width,alpha=tp,color=cols[1]) if Kroupa==True else None

    # Chabrier 2005 
    ax[0].plot(mass,(chabrier05_system(mass,integral_form=False))*mass,label="Chabrier05",
              lw=line_width,alpha=tp,color=cols[1]) if Chabrier05==True else None
    ax[1].plot(mass,(chabrier05_single(mass,integral_form=False))*mass,label="Chabrier05",
              lw=line_width,alpha=tp,color=cols[1]) if Chabrier05==True else None

    # Calculating the histogram of the Maschberger function data
    y_masch, x_masch =np.histogram(np.log10(masch_system),25,density=True)
    ax[0].plot(10**x_masch[:-1],y_masch/np.log(10),label="Masch13",alpha=tp,lw=line_width,c=cols[2]) if Masch==True else None
    #Single
    y_masch, x_masch =np.histogram(np.log10(masch_single),25,density=True)
    ax[1].plot(10**x_masch[:-1],y_masch/np.log(10),label="Masch13",alpha=tp,lw=line_width,c=cols[2]) if Masch==True else None
    
    # Adding text to the top corner of the plots 
    ax[0].text(13.1,0.64,"System IMF",fontsize=11)
    ax[1].text(7.8,0.64,"Single star IMF",fontsize=11)

    return(fig,ax[0],ax[1])