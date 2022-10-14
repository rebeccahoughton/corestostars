from ast import Continue
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import imf_master.imf.imf as imf
from sys import exit
from tqdm import tqdm
# My modules
from decay import *
from plotting import *

start_time = time.time()

#Formatting
title_size="14"
tick_size="10"
label_size="12"

#For plotting multiplicities 
stypes = [ "Y",  "T",  "L", "M","K","G", "G","F","A","B","B","O"]
mtypes = [0.02,0.055,0.075,0.15,0.3,0.6,1.00,1.5,2.4,  5,  8, 17,50]
#mtypes = [0.02,0.055,0.08,0.1,0.15,0.3,0.65,1.25,1.6,2.5,5.0,8.0,17,50]

#Open multiplicities file
fractions = np.loadtxt("multiplicities",usecols=(0,1,2,3,4,5,6,7),skiprows=2)

#-----------------------------------------------------------
#---------------------Functions-----------------------------
#-----------------------------------------------------------

# The Maschberger core mass function
def maschberger(alpha3,beta,mu,mlo,mup,n):
    oma3 = 1 - alpha3
    omb  = 1 - beta
    Gmlo = (1+(mlo/mu)**oma3)**omb
    Gmup = (1+(mup/mu)**oma3)**omb
    Gm = np.random.uniform(0,1,size=n)*(Gmup-Gmlo) + Gmlo
    Mc = mu*(Gm**(1/omb)-1)**(1/oma3)
    return(Mc)
   
# Number of stars per core
def N_stars(Nmin,Nmax,n,M_be,random=True,mdep=False,mbe=True):
    """
    Generating the number of stars for each core. Either generate
    randomly between Nmin and Nmax (random=True), with more stars
    procuced as core mass increases (mdep=True), or as a function 
    of Bonnor-Ebert Mass (mbe=True).
    """

    Ns = np.zeros((n),dtype=int)

    # Random fragmentation
    if random==True:
        Ns = np.random.randint(Nmin,Nmax,n)

    # Slight mass dependence 
    elif mdep==True:
        for i in range(n):
            if Mc[i]<=1.0:
                Ns[i]=np.random.randint(1,4,size=1)
            elif 1.0<Mc[i]<5.0:
                Ns[i]=np.random.randint(2,5,size=1)
            elif 5.0<=Mc[i]<20:# 5.<=Mc[i]<10:
                Ns[i]=np.random.randint(3,6,size=1)
            else:
                Ns[i]=np.random.randint(3,7,size=1) if Mc[i]<50 else np.random.randint(4,8,size=1)

    # elif mdep==True:
    #     for i in range(n):
    #         if Mc[i]<=1.0:
    #             Ns[i]=np.random.randint(2,4,size=1)
    #         elif 1.0<Mc[i]<8:
    #             Ns[i]=np.random.randint(2,5,size=1)
    #         elif 8<=Mc[i]<17:# 5.<=Mc[i]<10:
    #             Ns[i]=np.random.randint(3,6,size=1)
    #         else:
    #             Ns[i]=np.random.randint(4,6,size=1)# if Mc[i]<50 else np.random.randint(4,8,size=1)
            

    # Calculate the number of stars based on M_be
    elif mbe==True:
        for i in range(n):
            Ns[i] = 0 if Mc[i]<M_be[i] else (Mc[i]/M_be[i]).astype(int)
            Ns[i]=Nmax if Ns[i]>Nmax else Ns[i]
                
    return(Ns)

# Number of stars ejected
def N_ejected(n,Ns,random=True,rule=False):
    '''
    Generate the number of stars ejected from each core either
    randomly (random=True) or with a specific set of rules based
    on the number of stars in the system (rule=True)
    '''

    Ns_ej = np.ones((n),dtype=int)

    # Number of ejected stars randomly generated
    if random==True:
        if min(Ns)==0:
            for i in range(n):
                Ns_ej[i]=0 if Ns[i]==0 else np.random.randint(0,Ns[i],1)
        else:
            for i in range(n):
                Ns_ej[i] = np.random.randint(0,Ns[i],1)# if Ns[i]>2 else 0

    elif rule==True:
        # Manually defining the number of ejected stars
        # based on the initial multiplicity

        for i in range(n):
            ran = np.random.uniform(size=1)
            if Ns[i] >= 6:
                Ns_ej[i]=Ns[i]-4
            elif Ns[i] == 5:
                Ns_ej[i] = Ns[i]-3# if ran<0.9 else Ns[i]-2
            elif Ns[i] == 4:
                Ns_ej[i] = 2 if ran>0.5 else 1
            elif 2 <= Ns[i] <= 3:
                ran = np.random.uniform(size=1)
                Ns_ej[i] = 1# if ran>0.1 else 0
            # elif 4>Ns[i]>=2 and ran>0.3:
            #     Ns_ej[i] = 1
            else:
                Ns_ej[i] = 0
    
    return(Ns_ej)

# Split up the core mass into star masses
def get_masses(Mc,Ns,eta,m_stars_all,m_sys,SFE):
    '''
    Assigns masses to each of the stars from a flat mass
    ratio distribution. 
    '''

    # Set up zeroing arrays to start with
    m_temp = np.zeros((Ns))
    m_stars = np.zeros((Ns))

    # For Bonnor-Ebert mass model when there can be 0 stars
    if Ns==0:
        m_stars=np.zeros((1))
        return(m_stars)

    # For a single star that gets all of the mass
    if Ns==1:
        m_temp = np.random.uniform(low=0,high=1.0,size=Ns)

    # For multiple systems
    if Ns>1:
        q = np.random.uniform(low=0.2,high=1.0,size=Ns-1)
        m_temp[0]=Mc
        for i in range(Ns-1):
            m_temp[i+1]=q[i]*m_temp[i]
    
    # Multiply the mass fraction by the mass of the core, then times by an SFE factor 
    if SFE=="fixed":
        m_stars = -np.sort(-m_temp*Mc*eta/sum(m_temp))
    elif SFE=="vary":
        eta_new = eta + np.random.uniform(low=0,high=(1.0-eta),size=1)
        #eta_new = eta + np.random.uniform(low=max(0,eta-0.2*eta),high=min(1,eta+0.4*eta),size=1)
        m_stars = -np.sort(-m_temp*Mc*eta_new/sum(m_temp))
    elif SFE=="random":
        eta_new = np.random.uniform(low=0,high=1.0,size=1)
        m_stars = -np.sort(-m_temp*Mc*eta_new/sum(m_temp)) 

    # Add to a list of all stars
    [m_stars_all.append(m_stars[x]) for x in range(Ns)]

    # Add system masses to m_sys array
    # m_sys.append(sum(m_stars[:-Ns_ej[i]]))
    # print("Ns: {} and Ns_ej: {}".format(Ns[i],Ns_ej[i]))
    # [m_sys.append(m_stars[-(i+1)]) for i in range(Ns_ej[i])]

    # # If a system splits into several bound systems
    # if Nsys[i]>1:
    #     start=0
    #     for i in range(Nsys):
    #         m_sys.append(sum(m_stars[start:start+split[i]]))
    #         start+=split[i]
    #     # Counting the ejected stars as well
    #     [m_sys.append(m_stars[-(i+1)]) for i in range(Ns_ej)]

    return(m_stars)

# Counting the stars for when high order systems decay into two multiple systems
def count_stars(Ns,Ns_ej,m_stars,M_p,m_sys,Nfin,Nsys,split):

    '''
    Getting the primary masses of each system, and the final number of stars
    once some companions have been ejected. We use mmin to define the 
    minimum mass of stars we want included in the IMF and mmin_comp to
    define the minimum mass of the companions we want included in the 
    multiplicity statistics. 
    '''

    mmin = 0.001
    mmin_comp = 0.001#0.2*m_stars[0]
    ind = len(np.where(m_stars<mmin_comp)[0])
    ind = ind if ind<Ns else Ns-1

    # Add masses of all primary/single stars to an array
    if Nsys==1:
        if m_stars[0]>mmin:
            M_p.append(m_stars[0])
            Nfin.append(Ns-max(ind,Ns_ej)) if m_stars[0]>mtypes[3] else Nfin.append(Ns-Ns_ej)
            
    elif Nsys>1:
        # For Nsys bound systems
        split_ind=0
        for x in range(Nsys):
            if m_stars[split[0]]>mmin:
                M_p.append(m_stars[split_ind])
                Nfin.append(split[x])
            split_ind=split[x]

    # Sort out the ejected stars as well
    for j in range(Ns_ej):
        if m_stars[-(j+1)]>mmin:
            M_p.append(m_stars[-(j+1)])
            Nfin.append(1)

    # Getting the system masses
    # If a system splits into several bound systems
    if Nsys>1:
        start=0
        for i in range(Nsys):
            m_sys.append(sum(m_stars[start:start+split[i]]))
            start+=split[i]
    # For a single bound system and ejected companions
    else:
        m_sys.append(sum(m_stars[:-Ns_ej]))
        
    # Counting the ejected stars as well
    [m_sys.append(m_stars[-(i+1)]) for i in range(Ns_ej)]

    return(M_p,Nfin)


# Get the mass ratio distribution for M, G, and A stars
def mass_ratio(m1,m2,q_m,q_g,q_a):
    # Append to list representing the correct mass range
    q_m.append(m2/m1) if 0.10<m1 and m1<=0.51 else None
    q_g.append(m2/m1) if 0.79<m1 and m1<=1.08 else None
    q_a.append(m2/m1) if 1.73<m1 and m1<=3.70 else None
    #return(q_m,q_g,q_a)

# Main loop for assigning star masses and calculating multiplicity
def main(Mc,Ns,Ns_ej,eta,SFE):
    M_p = []
    Nfin = []
    m_stars_all = []
    m_sys = []
    # Mass-ratio distribution lists
    q_m = []
    q_g = []
    q_a = []

    # Multiplicity counter array
    mult = np.zeros((Nmax,len(mtypes)-1))

    count=0
    #Start loop here rather than inside the split_mass function
    for i in range(n): # Was previously n-1 for some reason
        # if Ns[i]-Ns_ej[i]==4:
        #     count+=1
        m_stars = get_masses(Mc[i],Ns[i],eta,m_stars_all,m_sys,SFE)
        M_p,Nfin = count_stars(Ns[i],Ns_ej[i],m_stars,M_p,m_sys,Nfin,Nsys[i],split[i])
    
        if len(m_stars)>1:
            mass_ratio(m_stars[0],m_stars[1],q_m,q_g,q_a) 

    # print("Quadruples:",count*100/n,"%")
    # exit()

    return(M_p,Nfin,m_stars_all,m_sys,mult,q_m,q_g,q_a)

# Calculating multiplicity fractions
def multiplicity(Nmin,Nmax,mult):
    '''
    Calculate MF and CSF for all spectral types
    '''
    Nmin = min(Nfin)
    Nmax = max(Nfin)

    MF = np.zeros(len(mtypes)-1)
    CSF = np.zeros(len(mtypes)-1)
    THF = np.zeros(len(mtypes)-1)

    for i in range(len(mtypes)-1):
        MF[i]  = (sum(mult[:,i])-mult[0,i])/sum(mult[:,i])
        top=0
        for j in range(Nmax-Nmin):
            top+=(j+1)*mult[j+1,i]
        CSF[i] = top/sum(mult[:,i])
        THF[i] = (sum(mult[:,i])-mult[0,i]-mult[1,i])/sum(mult[:,i])

    mult_df = pd.DataFrame(mult,#index=["N=1","N=2","N=3","N=4","N=5","N=6","N=7"],
              columns=stypes)

    Gs = mult_df["G"].sum(axis=1)
    Gs = 100*Gs/sum(Gs)
    As = mult_df["A"]#.sum(axis=0)
    As = 100*As/sum(As)
    #print("The ratio of G type stars is: {} : {} : {} : {}".format(round(Gs[0],1),round(Gs[1],1),round(Gs[2],1),round(Gs[3],1)))
    #print("The ratio of A type stars is: {} : {} : {} : {}".format(round(As[0],1),round(As[1],1),round(As[2],1),round(As[3],1)))

    return(MF,CSF,THF)#,ratios)

    

#------------------------------------------------------------------
#---------------------START OF PROGRAM-----------------------------
#------------------------------------------------------------------
np.random.seed(625)
n = int(1e5)  #Number of cores

#Star formation efficiency

#Bonnor-Ebert Mass
#M_be = np.full((n),0.6)
#M_be = np.random.normal(loc=1.0,scale=0.2,size=n)
M_be = np.random.uniform(0.5,2.5,n)

# Generate core masses from the Maschberger IMF
# For the L3 form of the IMF, parameters are 2.3, 1.4, 0.2, 0.01, 150
# For the B4 form of the IMF, parameters are 2.3, -0.15, 0.15, 0.01, 150 
Mc = maschberger(2.3,1.9,1.0,0.05,300,n)

# Defining variables
Nmin = 2
Nmax = 6

# Number of stars
Ns = N_stars(Nmin,Nmax,n,M_be,random=False,mdep=True,mbe=False)

# Number of ejected stars
Ns_ej = N_ejected(n,Ns,random=True,rule=False)
Nsys = np.ones((n))
split = []
[split.append([1]) for i in range(n)]

# Ejected stars using the Sterzik and Durisen decay probabilities
Ns_ej,Nsys,split = decay(Ns,n)

# arr = np.array([Ns,Ns_ej])
# arr = np.swapaxes(arr,0,1)
# inds = np.where(arr[:,0]==4)[0]
# # print("Array of Ns_ej values when Ns=2:",arr[inds,1])
# print("mean ejected stars when Ns=4: {}".format(np.mean(arr[inds,1])))
# exit()

# Multiplicity plots
ndim = 1
ndim2= 4

x = mtypes[:-1] + np.diff(mtypes)/2

etas = [0.3,0.5,0.9,r"$U\rm{[0,1]}$"]
#etas = [0.6]
lss = [(0,(5,1)),"dotted","dashed","dashdot"]

lss = [(0,(5,1)),(0,(3,1.5,1,1.5,1,1.5)),"dashed","dashdot"]
SFES = ["fixed","fixed","fixed","random"]

#Colours for IMFs with different eta values
cols2 = ["blue","purple","#c904c9","red"]
cols = ["midnightblue","mediumblue","dodgerblue","cyan","darkred","red","orangered","orange",
        "darkgreen","green","forestgreen","limegreen"]

# Generating multiplicity plot
figMF,ax = MF_CSF(ndim,fractions,horizontal=False)

# Plotting the IMFs   
figIMF,ax1,ax2 = IMF_plot(Masch=True,Kroupa=True,Salpeter=True,Chabrier05=True,horizontal=False)

#CMF
y_CMF, x_CMF =np.histogram(np.log10( Mc),25,density=True)
ax1.plot(10**x_CMF[:-1],y_CMF*0.2,label="CMF",c="k",lw=1.5,alpha=0.8)
ax2.plot(10**x_CMF[:-1],y_CMF*0.2,label="CMF",c="k",lw=1.5,alpha=0.8)

# Nmin_arr = [1,1,2,2,2,3,3]
# Nmax_arr = [6,7,6,7,8,7,8]

for i in range(ndim2):
    # for j in range(ndim):
    #     Nmin = Nmin_arr[j]#1+j
    #     Nmax = Nmax_arr[j]#5+j

    #     Ns = N_stars(Nmin,Nmax,n,M_be,random=True,mdep=False,mbe=False)

    #     # Number of ejected stars
    #     Ns_ej = N_ejected(n,Ns,random=True,rule=False)
    #     Nsys = np.ones((n))
    #     split = []
    #     [split.append([1]) for i in range(n)]

    # SFE = "fixed", "vary", or "random"
    M_p,Nfin,m_stars_all,m_sys,mult,q_m,q_g,q_a = main(Mc,Ns,Ns_ej,etas[i],SFES[i])

    # Multiplicity array and boundary edges
    mult, lim1, lim2 = np.histogram2d(Nfin,M_p,bins=[np.arange(0,max(Nfin)+1)+0.5,mtypes])

    #Multiplicities
    MF,CSF,THF=multiplicity(Nmin,Nmax,mult)

        # ax[j*2].plot(x,MF,c="b",label="MF",lw=1.5,alpha=0.8,ls=lss[i])
        # ax[j*2].plot(x,THF,c="r",label="THF",lw=1.5,alpha=0.8,ls=lss[i])
        # ax[j*2+1].plot(x,CSF,c="g",label="CSF",lw=1.5,alpha=0.8,ls=lss[i])

    ax[0].plot(x,MF,c=cols[i],label="MF",ls=lss[i],lw=1.5)
    ax[0].plot(x,THF,c=cols[i+4],label="THF",ls=lss[i],lw=1.5)
    ax[1].plot(x,CSF,c=cols[i+8],label="CSF",ls=lss[i],lw=1.5)

    #Primary star IMF
    y_IMF, x_IMF =np.histogram(np.log10(M_p),25,density=True)
    ax1.plot(10**x_IMF[:-1],y_IMF*0.2,label=r"IMF: $\eta$={}".format(etas[i]),ls=lss[i],color=cols2[i])
    # Single star IMF
    y_IMF, x_IMF =np.histogram(np.log10(m_stars_all),25,density=True)
    ax2.plot(10**x_IMF[:-1],y_IMF*0.2,label=r"IMF, $\eta$={}".format(etas[i]),ls=lss[i],color=cols2[i])
    ax2.legend(loc="lower left",ncol=2,columnspacing=0.8,labelspacing=0.4,fontsize=9.5)#,bbox_to_anchor=[0.4,0.0])

# figMF.savefig("mult_sd.pdf",bbox_inches='tight')
# figIMF.savefig("IMF_sd.pdf",bbox_inches='tight')

# Mass-ratio distributions
# fig,ax = plt.subplots()
# ax.hist(q_m,10,density=True,histtype='step',color='g',label='M',linewidth=1.5)
# ax.hist(q_g,10,density=True,histtype='step',color='c',label='G',linewidth=1.5)
# ax.hist(q_a,10,density=True,histtype='step',color='r',label='A',linewidth=1.5)
# ax.set_xlim(0.21,0.99)
# #ax.set_ylim(0.6,1.8)
# ax.set_xlabel(r"$q = m_1/m_2$")
# ax.set_ylabel("Scaled histogram")
# ax.legend(loc="lower right")

plt.show()