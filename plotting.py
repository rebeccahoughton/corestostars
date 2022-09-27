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
    Plot the multiplicity and triple higher order fractions on the
    left plot and the companion star fractions on the right plot.
    '''
    c_size = len(fractions[:,0])
    #colors = plt.cm.RdYlBu(np.linspace(0,1,c_size))
    colors1 = plt.cm.autumn(np.linspace(0,1,int(c_size/2)+5))
    colors2 = plt.cm.Blues(np.linspace(0,1,int(c_size/2)+2))
    colors = np.vstack((colors1[:-3,:],colors2[3:,:]))
    #colors = plt.cm.jet(np.linspace(0,1,len(fractions[:,0])))

    if horizontal==False:
        fig,ax = plt.subplots(2,1,figsize=[5,8],sharex=True,gridspec_kw={'hspace':0.07,'wspace':0.0})
        ax[0].set_ylabel(r"$\bf{Multiplicity}$ / Triple High-order Fractions",labelpad=-0.5)
        ax[1].set_ylabel("Companion Star Fraction",labelpad=-0.5)
        ax[1].set_xlabel(r"Primary Mass ($M_\odot$)",labelpad=-0.5)
    else:
        fig,ax = plt.subplots(ndim,2,figsize=[10,4*ndim],sharex=True,gridspec_kw={'hspace':0,'wspace':0.2})
    fig.get_constrained_layout()

    #Getting data
    MF_obs = fractions[:,2][~np.isnan(fractions[:,2])]/100
    MF_yerr = fractions[:,3][~np.isnan(fractions[:,3])]/100
    THF_obs = fractions[:,4][~np.isnan(fractions[:,4])]/100
    THF_yerr = fractions[:,5][~np.isnan(fractions[:,5])]/100
    CSF_obs = fractions[:,6][~np.isnan(fractions[:,6])]
    CSF_yerr = fractions[:,7][~np.isnan(fractions[:,7])]
    x_err = (fractions[:,1][~np.isnan(fractions[:,1])]-fractions[:,0][~np.isnan(fractions[:,0])])/2

    x = fractions[:,0][~np.isnan(fractions[:,0])]+x_err

    ax = np.reshape(ax, (ndim*2))

    # Offner et al. (2022) data
    for j in range(ndim):
        #Plotting
        # for i in range(len(MF_obs)):
        #     ax[j*2].errorbar(x[i],MF_obs[i]/100,xerr=x_err[i],yerr=MF_yerr[i]/100,color=colors1[i],marker="s",markersize=4,label="MF")#,lw=4.0)
        #     ax[j*2+1].errorbar(x[i],CSF_obs[i],xerr=x_err[i],yerr=CSF_yerr[i],color=colors2[i],marker="^",markersize=4,label="THF")#,lw=2.0)
        #     ax[j*2].errorbar(x[i],THF_obs[i]/100,xerr=x_err[i],yerr=THF_yerr[i]/100,color=colors3[i],marker="o",markersize=4,label="CSF")

        ax[j*2].errorbar(  x, MF_obs,xerr=x_err, yerr=MF_yerr,color="k",marker="s",ms=5,markerfacecolor="b",label="MF",ls=' ',capsize=1.2,lw=1.0)
        ax[j*2+1].errorbar(x,CSF_obs,xerr=x_err,yerr=CSF_yerr,color="k",marker="s",ms=5,markerfacecolor="limegreen",label="CSF",ls=' ',capsize=1.2,lw=1.0)#,lw=2.0)
        ax[j*2].errorbar(  x,THF_obs,xerr=x_err,yerr=THF_yerr,color="k",marker="o",ms=5,markerfacecolor="r",label="THF",ls=' ',capsize=1.2,lw=1.0)
        ax[j*2].legend(loc="upper left")
        ax[j*2+1].legend(loc="upper left")

    # Formatting
    for i in range(2*ndim):
        ax[i].set_xscale("log")
        ax[i].tick_params(axis="both",which="both",direction="in",top=True,right=True)

    if horizontal==True:
        ax[-1].set_xlabel(r"Primary Mass ($M_\odot$)",labelpad=-0.5)
        ax[-2].set_xlabel(r"Primary Mass ($M_\odot$)",labelpad=-0.5)

        # Gets the labels right when adding extra rows of subplots
        fig.add_subplot(111, frameon=False)
        # Hide tick and tick label of the big axis
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.ylabel(r"$\bf{Multiplicity}$ / Triple High-order Fractions",labelpad=-1)
        axa = fig.add_subplot(122, frameon=False)
        # Hide tick and tick label of the big axis
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        axa.set_ylabel("Companion Star Fraction",labelpad=-1)
    
    return(ax)


def IMF_plot(Masch=True,Kroupa=True,Salpeter=True,Chabrier05=True,horizontal=True):
    mass = np.logspace(np.log10(0.01),np.log10(100),100)
    font = {'fontname':'DejaVu Sans'}

    masch = maschberger(2.3,2.0,0.2,0.01,150,int(5e6))
    kroupa = imf.Kroupa(mmin=0.01)
    salpeter = imf.Salpeter(mmin=0.3)
    chabrier_ln = imf.ChabrierLogNormal()
    chabrier_pl = imf.ChabrierPowerLaw()
    chabrier2005 = imf.ChabrierPowerLaw(lognormal_width=0.55*np.log(10),
                                        lognormal_center=0.2, alpha=2.35)

    # Setting limits for the Kroupa and Salpeter IMFs
    ind_s=np.where(mass>0.3)[0][0]
    ind_k=np.where(mass>0.01)[0][0]

    cols = ["orange","lawngreen","mediumseagreen"]

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
    
    # Actually doing the plotting
    for i in range(2):
        ax[i].plot(mass[ind_s:],(salpeter(mass,integral_form=False)[ind_s:])*mass[ind_s:]/np.log(10),
                   label="Salpeter55",lw=4.5,alpha=0.6,color=cols[0]) if Salpeter==True else None
        ax[i].plot(mass[ind_k:],(kroupa(mass,integral_form=False)[ind_k:])*mass[ind_k:]/np.log(10),
                   label="Kroupa01",lw=4.5,alpha=0.6,color=cols[1]) if Kroupa==True else None
        #ax.plot(mass,(chabrier2005(mass,integral_form=False))*mass/np.log(10),label="Chabrier05",
        #          lw=4.5,alpha=0.4,color=cols[2]) if Chabrier05==True else None
        
        #Slightly more involved for the Maschberger IMF
        y_masch, x_masch =np.histogram(np.log10(masch),25,density=True)
        ax[i].plot(10**x_masch[:-1],y_masch*0.2,label="Masch13",alpha=0.6,lw=4.5,c=cols[2]) if Masch==True else None
        
        #Setup
        ax[i].tick_params(axis="both",which="both",direction="in",top=True,right=True,labelsize=12,pad=4.2)
    
        ax[i].set_xscale("log")
        ax[i].set_yscale('log')
        ax[i].set_xlim(0.01,100)
        ax[i].set_ylim(0.0001,0.5)
        ax[i].axvspan(0,0.08,color="k",alpha=0.1)

    #ax[0].set_title("IMF (System)",fontsize=title_size,**font)
    #ax[1].set_title("IMF (All stars)",fontsize=title_size,**font)
    c = 3 if horizontal==True else 0
    ax[0].text(16-c,0.25,"System IMF")
    ax[1].text(6-c,0.25,"Individual star IMF")

    ax[0].set_ylabel("dN/dlog M",fontsize=label_size,**font)

    return(ax[0],ax[1])