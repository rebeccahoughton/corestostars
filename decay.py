import numpy as np

# Decay based on the conditions in Sterzik & Durisen 1998
def decay(Ns,n):

    Ns_ej = np.ones((n),dtype=int)
    Nsys  = np.ones((n),dtype=int)
    split = [[]]

    for i in range(n):
        Nsys[i]=1
        ran = np.random.uniform(0,1,1)

        if Ns[i] == 2:
            Ns_ej[i] = 0# if ran>0.7 else 0

        elif Ns[i] == 3:
            Ns_ej[i] = 1 if ran<0.874 else 0
            
        elif Ns[i] == 4:
            if ran<=0.751:
                # Binary and 2 ejected stars
                Ns_ej[i] = 2
            elif 0.751<ran<=0.932:
                # Triple and 1 ejected star
                Ns_ej[i] = 1
            elif 0.932<ran<=0.969:
                # Quadruple and no ejected stars
                Ns_ej[i] = 0
            elif 0.969<ran<=0.981:
                # 2 binaries
                Nsys[i]=2
                Nsplit=[2,2]
                Ns_ej[i] = 0
            else:
                Ns_ej[i] = 0 #but forms 2 binaries

        elif Ns[i] == 5:
            if ran<=0.532:
                # Forms 1 binary and 3 singles
                Ns_ej[i] = Ns[i] - 2
            elif 0.532<ran<=0.872:
                # Forms 1 triple and 2 singles
                Ns_ej[i] = Ns[i] - 3
            elif 0.872<ran<=0.934:
                # forms 1 quadruple and 1 single
                Ns_ej[i] = Ns[i] - 4 
            elif 0.934<ran<=0.975:
                #TWO BINARIES AND A SINGLE
                Nsys[i]=2
                Nsplit=[2,2,1]
                Ns_ej[i] = 0   
            elif 0.975<ran<=0.983:
                #ONE BINARY AND ONE TRIPLE
                Nsys[i]=2
                Nsplit=[3,2]
                Ns_ej[i] = 0
            else:
                Ns_ej[i] = 0

        elif Ns[i] == 6:
            if ran<=0.313:
                # Forms 1 binary and 4 singles
                Ns_ej[i] = Ns[i] - 2
            elif 0.313<ran<=0.812:
                # Forms 1 triple and 3 singles
                Ns_ej[i] = Ns[i] - 3
            elif 0.812<ran<=0.899:
                # forms 1 quadruple and 2 singles
                Ns_ej[i] = Ns[i] - 4
            elif 0.899<ran<=0.969:
                #TWO BINARIES AND 2 SINGLES
                Nsys[i]=2
                Nsplit=[2,2,1,1]
                Ns_ej[i] = Ns[i] - 2   
            elif 0.969<ran<=0.995:
                #TRIPLE PLUS BINARY PLUS SINGLE
                Nsys[i]=2
                Nsplit=[3,2,1]
                Ns_ej[i] = Ns[i] - 5
            else:
                Ns_ej[i] = 0

        elif Ns[i]>=7:
            if ran<=0.094:
                # Forms 1 binary and 5 singles
                Ns_ej[i] = Ns[i] - 2
            elif 0.094<ran<=0.752:
                # Forms 1 triple and 4 singles
                Ns_ej[i] = Ns[i] - 3
            elif 0.752<ran<=0.864:
                # forms 1 quadruple and 3 singles
                Ns_ej[i] = Ns[i] - 4
            elif 0.864<ran<=0.963:
                #TWO BINARIES AND 3 SINGLES
                Nsys[i]=2
                Nsplit=[2,2,1,1,1] if Ns[i]==7 else [2,2,1,1,1,1]
                Ns_ej[i] = Ns[i] - 4   
            elif 0.963<ran<=0.995:
                #TRIPLE PLUS BINARY PLUS SINGLES
                Nsys[i]=2
                Nsplit=[3,2,1,1] if Ns[i]==7 else [3,2,1,1,1]
                Ns_ej[i] = Ns[i] - 5
            else:
                Ns_ej[i] = 0

        else:
            Ns_ej[i]=0

        if Nsys[i]==1:
            split.append([1])
        else:
            split.append(Nsplit)
    
    del(split[0])
    split = np.array(split,dtype=object)
    
    return(Ns_ej,Nsys,split)


# # From original testing - not properly extrapolated values
# elif Ns[i] == 6 or Ns[i]==7:
#     if ran<=0.25:
#         # Forms 1 binary and 4/5 singles
#         Ns_ej[i] = Ns[i] - 2
#     elif 0.25<ran<=0.5:
#         # Forms 1 triple and 3/4 singles
#         Ns_ej[i] = Ns[i] - 3
#     elif 0.5<ran<=0.8:
#         # forms 1 quadruple and 2/3 singles
#         Ns_ej[i] = Ns[i] - 4
#     elif 0.8<ran<=0.89:
#         #TWO BINARIES AND 2/3 SINGLES
#         Nsys[i]=2
#         Nsplit=[2,2,1,1] if Ns[i]==6 else [2,2,1,1,1]
#         Ns_ej[i] = Ns[i] - 2   
#     elif 0.89<ran<=0.993:
#         #TRIPLE PLUS BINARY PLUS SINGLE
#         Nsys[i]=2
#         Nsplit=[3,2,1,1] if Ns[i]==6 else [3,2,1,1,1]
#         Ns_ej[i] = Ns[i] - 3
#     else:
#         Ns_ej[i] = 0