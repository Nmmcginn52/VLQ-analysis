#!/usr/bin/python



import os

exe = os.system

from math import *

import random

import numpy

from numpy import ndarray

from scipy.interpolate import InterpolatedUnivariateSpline


MW = 80.385
MZ = 91.1876
g = 0.652954
thetaw = acos(MW/MZ)
mmu = 0.1056584    #physics muon mass
mt = 173.21       #top quark pole mass central value (uncertantiy \pm 0.874 GeV)
GF = (sqrt(2)/8.)*pow(g/MW,2)
v = (1/sqrt(sqrt(2.0)*GF))/sqrt(2.0)             #174 GeV
#gelsm = (g/cos(thetaw))*(pow(sin(thetaw),2) - 1/2.)   # g_L^Z SM : for mu or e or tau
#gersm = (g/cos(thetaw))*pow(sin(thetaw),2)            # g_R^Z SM : for mu or e or tau
mbpole = 4.18



#listfrac = ['SM','doublet','singlet']


def thetap(y1, y2):
    if y1 == y2 or y1 == 0 or y2 == 0:
        return 0
    else:
        return y1 + y2 - (2*y1*y2/(y1-y2))*log(y1/y2)

def thetam(y1, y2):
    if y1 == y2 or y1 == 0 or y2 == 0:
        return 0
    else:
        return 2*sqrt(y1*y2)*(((y1+y2)/(y1-y2))*log(y1/y2) - 2)

def Delta(y1, y2):
    return -1 - pow(y1,2) - pow(y2,2) + 2*y1 + 2*y2 + 2*y1*y2

def ff(y1, y2):
    if Delta(y1, y2) > 0:
        return -2*sqrt(Delta(y1,y2))*(atan((y1-y2+1)/sqrt(Delta(y1,y2))) - atan((y1-y2-1)/sqrt(Delta(y1,y2))))
    elif Delta(y1, y2) == 0:
        return 0
    elif Delta(y1, y2) < 0:
        return sqrt(-Delta(y1, y2))*log((y1+y2-1+sqrt(-Delta(y1,y2)))/(y1+y2-1-sqrt(-Delta(y1,y2))))

def chip(y1, y2):
    if y1 == y2 or y1 == 0 or y2 == 0:
        return 0
    else:
        return (y1+y2)/2. - pow(y1-y2,2)/3. +(pow(y1-y2,3)/6. - (1/2.)*((pow(y1,2)+pow(y2,2))/(y1-y2)))*log(y1/y2) + ((y1-1)/6.)*ff(y1,y1) + ((y2-1)/6.)*ff(y2,y2) + (1/3. - (y1+y2)/6. - pow(y1-y2,2)/6.)*ff(y1,y2)

def chim(y1,y2):
    if y1 == y2 or y1 == 0 or y2 == 0:
        return 0
    else:
        return -sqrt(y1*y2)*(2 + (y1 - y2 - (y1+y2)/(y1-y2))*log(y1/y2) + (ff(y1,y1)+ff(y2,y2))/2. - ff(y1,y2)  )


def psip(y1,y2):
    if y1 == 0 or y2 == 0:
        return 0
    else:
        return (22*y1 + 14*y2)/9. - (1/9.)*log(y1/y2) + ((11*y1 + 1)/18.)*ff(y1,y1) + ((7*y2 - 1)/18.)*ff(y2,y2)

def psim(y1,y2):
    if y1 == 0 or y2 == 0:
        return 0
    else:
        return -sqrt(y1*y2)*( 4 + (ff(y1,y1) + ff(y2,y2))/2.  )
 

def unitstep(x):
    if x > 0:
        return 1
    else:
        return 0

def phase(x,y,z):
    return (x-y-z)**2 - 4*y*z
    

exe("rm t1ewptallhelp-s6.dat")
#exe("rm t1ewptallhelp-0.5Yukawa.dat")

rng = random.SystemRandom()


for i in xrange(0,2000000):#1000000):#1500000):
    MTT = rng.uniform(500.,4000.)#random.uniform(100.,500.)
    MBB = rng.uniform(500.,4000.)#1000
    MQQ = rng.uniform(500.,4000.)
    kaT = rng.uniform(-1.0,1.0)
    kaQ = rng.uniform(-1.0,1.0)
    ka = rng.uniform(-1.0,1.0)
    kab = rng.uniform(-1.0,1.0)
    beta = rng.uniform(atan(0.3),atan(50.0))#random.uniform(atan(0.3),atan(20.))
    lamQ = rng.uniform(-1.0,1.0)
    lamB = rng.uniform(-1.0,1.0)
    lam = rng.uniform(-1.0,1.0)#rng.choice([-rng.uniform(0.5,1.0),rng.uniform(0.5,1.0)])
    lamba = rng.uniform(-1.0,1.0)#rng.choice([-rng.uniform(0.5,1.0),rng.uniform(0.5,1.0)])
    yt0 = mt/(v*sin(beta))#random.uniform(-0.001,0.001)
    yb0 = mbpole/(v*cos(beta))

    # Mass of the up type quarks

    # Needed for Newton's method of iteration : for up type quark sector mixing to have the top quark pole mass
    def myfunction(xx):
        class myarray(ndarray):    
            @property
            def H(self):
                return self.conj().T
        mups1 = numpy.array([[xx*v*sin(beta),0,kaT*v*sin(beta)],[kaQ*v*sin(beta),MQQ,ka*v*sin(beta)],[0,kab*v*sin(beta),MTT]]).view(myarray)
        mupmupda1 = numpy.dot(mups1, mups1.H)
        mupdamup1 = numpy.dot(mups1.H, mups1)
        cevaluesl1, UL1 = numpy.linalg.eig(mupmupda1)
        idxul1 = cevaluesl1.argsort()
        cevaluesl1 = cevaluesl1[idxul1]
        UL1 = UL1[:,idxul1]
        cevaluesr1, UR1 = numpy.linalg.eig(mupdamup1)
        idxur1 = cevaluesr1.argsort()
        cevaluesr1 = cevaluesr1[idxur1]
        UR1 = UR1[:,idxur1]
        if abs(cevaluesl1[0] - cevaluesr1[0]) > 0.00000001 or abs(cevaluesl1[1] - cevaluesr1[1]) > 0.00000001 or abs(cevaluesl1[2] - cevaluesr1[2]) > 0.00000001:
            return -20000
        elif cevaluesl1[0] < 0.:
            return -20000
        else:
            return sqrt(cevaluesl1[0]) - mt

    #Well define myfunction(ymu0) change (-sign)
    while True:
        yt1 = yt0 + 0.001
        if myfunction(yt0) > -10000:
            break
        yt0 = yt1
        
    #Now begin Newton's method
    def derivative(f):
        def compute(x, dx):
            return (f(x+dx) - f(x))/dx
        return compute
    def newtons_method(f, x, dx=0.0001, tolerance=0.005):
        df = derivative(f)
        while True:
            x1 = x - f(x)/df(x, dx)
            t = abs(x1 - x)
            if t < tolerance:
                break
            x = x1
        return x

    try:
        yt = newtons_method(myfunction, yt0)
    except ZeroDivisionError:
        continue


    # Mass of the up type quarks
    class myarray(ndarray):    
        @property
        def H(self):
            return self.conj().T    
    mups = numpy.array([[yt*v*sin(beta),0,kaT*v*sin(beta)],[kaQ*v*sin(beta),MQQ,ka*v*sin(beta)],[0,kab*v*sin(beta),MTT]]).view(myarray)
    mupmupda = numpy.dot(mups, mups.H)
    mupdamup = numpy.dot(mups.H, mups)
    cevaluesl, UL = numpy.linalg.eig(mupmupda)
    idxul = cevaluesl.argsort()
    cevaluesl = cevaluesl[idxul]
    UL = UL[:,idxul]
    cevaluesr, UR = numpy.linalg.eig(mupdamup)
    idxur = cevaluesr.argsort()
    cevaluesr = cevaluesr[idxur]
    UR = UR[:,idxur]
    ULda = UL.H
    URda = UR.H
    # Remove the wrong sign of UL or UR: make ULda.mups.UR[1][1] = me4 and ULda.mups.UR[2][2] = me5 positive
    helpcheckch = numpy.dot(ULda, numpy.dot(mups, UR))
    helpdiagch = numpy.array([[1*numpy.sign(helpcheckch[0][0]),0,0],[0,1*numpy.sign(helpcheckch[1][1]),0],[0,0,1*numpy.sign(helpcheckch[2][2])]])
    UL = numpy.dot(UL, helpdiagch)
    ULda = numpy.dot(helpdiagch, ULda)
    if abs(cevaluesl[0] - cevaluesr[0]) > 0.00000001 or abs(cevaluesl[1] - cevaluesr[1]) > 0.00000001 or abs(cevaluesl[1] - cevaluesr[1]) > 0.00000001:
        #print 'Error in the charged lepton eigenvalues'
        continue
    if cevaluesl[0] < 0.:
        continue
    onetest = numpy.dot(numpy.dot(ULda, mupmupda), UL)
    twotest = numpy.dot(numpy.dot(URda, mupdamup), UR)
    restart1 = False
    restart2 = False
    for ii in range(0,3):
        for jj in range(0,3):
            if abs(onetest[ii][jj] - twotest[ii][jj]) > 0.00000001:
                restart1 = True
                restart2 = True
                break
        if restart1:
            break
    if restart2:
        continue

    # Mass of the down type quarks
    # Needed for Newton's method of iteration : for up type quark sector mixing to have the top quark pole mass
    def myfunction2(xx):
        class myarray(ndarray):    
            @property
            def H(self):
                return self.conj().T
        mdowns1 = numpy.array([[xx*v*cos(beta),0,lamB*v*cos(beta)],[lamQ*v*cos(beta),MQQ,lam*v*cos(beta)],[0,lamba*v*cos(beta),MBB]]).view(myarray)
        mdownmdownda1 = numpy.dot(mdowns1, mdowns1.H)
        mdowndamdown1 = numpy.dot(mdowns1.H, mdowns1)
        evaluesl1, VL1 = numpy.linalg.eig(mdownmdownda1)
        idxdl1 = evaluesl1.argsort()
        evaluesl1 = evaluesl1[idxdl1]
        VL1 = VL1[:,idxdl1]
        evaluesr1, VR1 = numpy.linalg.eig(mdowndamdown1)
        idxdr1 = evaluesr1.argsort()
        evaluesr1 = evaluesr1[idxdr1]
        VR1 = VR1[:,idxdr1]
        if abs(evaluesl1[0] - evaluesr1[0]) > 0.00000001 or abs(evaluesl1[1] - evaluesr1[1]) > 0.00000001 or abs(evaluesl1[2] - evaluesr1[2]) > 0.00000001:
            return -20000
        elif evaluesl1[0] < 0.:
            return -20000
        else:
            return sqrt(evaluesl1[0]) - mbpole

    #Well define myfunction2(ymu0) change (-sign)
    while True:
        yb1 = yb0 + 0.001
        if myfunction2(yb0) > -10000:
            break
        yb0 = yb1
        
    #Now begin Newton's method
    def derivative(f):
        def compute(x, dx):
            return (f(x+dx) - f(x))/dx
        return compute
    def newtons_method(f, x, dx=0.0001, tolerance=0.005):
        df = derivative(f)
        while True:
            x1 = x - f(x)/df(x, dx)
            t = abs(x1 - x)
            if t < tolerance:
                break
            x = x1
        return x

    try:
        yb = newtons_method(myfunction2, yb0)
    except ZeroDivisionError:
        continue



    # Mass of the down type quarks
    class myarray(ndarray):    
        @property
        def H(self):
            return self.conj().T
    mdowns = numpy.array([[yb*v*cos(beta),0,lamB*v*cos(beta)],[lamQ*v*cos(beta),MQQ,lam*v*cos(beta)],[0,lamba*v*cos(beta),MBB]]).view(myarray)
    mmda = numpy.dot(mdowns, mdowns.H)
    mdam = numpy.dot(mdowns.H, mdowns)
    # After obtaining the eigenvalues and eigenvectors we sort "both of them" simultaneously.
    evaluesl, VL = numpy.linalg.eig(mmda)
    #fractionEV = evaluesl
    #fractionVL = VL
    idxvl = evaluesl.argsort()
    evaluesl = evaluesl[idxvl]
    VL = VL[:,idxvl]
    evaluesr, VR = numpy.linalg.eig(mdam)
    idxvr = evaluesr.argsort()
    evaluesr = evaluesr[idxvr]
    VR = VR[:,idxvr]
    VLda = VL.H
    VRda = VR.H
    # Remove the sign ambiguity of VL or VR : make VLda.mdowns.VR[1][1] = mb4 and VLda.mdowns.VR[2][2] = mb5 positive
    helpcheck = numpy.dot(VLda, numpy.dot(mdowns, VR))
    helpdiag = numpy.array([[1,0,0],[0,1*numpy.sign(helpcheck[1][1]),0],[0,0,1*numpy.sign(helpcheck[2][2])]])
    VL = numpy.dot(VL, helpdiag)
    VLda = numpy.dot(helpdiag, VLda)
    #fractionVLda = fractionVL.H
    if abs(evaluesl[0] - evaluesr[0]) > 0.00000001 or abs(evaluesl[1] - evaluesr[1]) > 0.00000001 or abs(evaluesl[2] - evaluesr[2])  > 0.00000001:
        continue
    if evaluesl[0] < 0. or evaluesl[1] < 0. or evaluesl[2] < 0.:
        continue
    onetestd = numpy.dot(numpy.dot(VLda, mmda), VL)
    twotestd = numpy.dot(numpy.dot(VRda, mdam), VR)
    restartd1 = False
    restartd2 = False
    for ii in range(0,3):
        for jj in range(0,3):
            if abs(onetestd[ii][jj] - twotestd[ii][jj]) > 0.00000001:
                restartd1 = True
                restartd2 = True
                break
        if restartd1:
            break
    if restartd2:
        continue

    
    
    
    # Physical masses
    mt4 = sqrt(cevaluesl[1])
    mt5 = sqrt(cevaluesl[2])
    mb4 = sqrt(evaluesl[1])
    mb5 = sqrt(evaluesl[2])
    mHH = random.uniform(500,4000)
    # Do NOT impose the condition mb4 > mHH because this decouples the doublet-like t4!!!
    # But impose mb4 > mHH/2 in order to exclude the possibility of H -> b4 b4
    # For VLL, do NOT impose the condition me4 > mHH (for nu4 analysis) or mnu4 > mHH (for e4 analysis)!
    if mt4 < mHH + mt or mb4 < mHH + mt:#if mt4 + mt > mHH:# or mt4 < mHH/2. or mt5 < mHH or mb4 < mHH/2.:# or mnv < mHH/2.  or abs((mnv - 130.000000)/130.000000) > 0.02:
        # or me4 < mHH/2. #or me4 < mHH:
        continue
    
    # New couplings
    lamHt34 = ( (MQQ/v)*(ULda[0][1])*UR[1][1] + (MTT/v)*(ULda[0][2])*UR[2][1])/tan(beta)
    lamHrt34 = ( (MQQ/v)*(URda[0][1])*UL[1][1] + (MTT/v)*(URda[0][2])*UL[2][1])/tan(beta)    
    lamHt35 = ( (MQQ/v)*(ULda[0][1])*UR[1][2] + (MTT/v)*(ULda[0][2])*UR[2][2])/tan(beta)
    lamHrt35 = ( (MQQ/v)*(URda[0][1])*UL[1][2] + (MTT/v)*(URda[0][2])*UL[2][2])/tan(beta)
    lamHt45 = ( (MQQ/v)*(ULda[1][1])*UR[1][2] + (MTT/v)*(ULda[1][2])*UR[2][2])/tan(beta)
    lamHrt45 = ( (MQQ/v)*(URda[1][1])*UL[1][2] + (MTT/v)*(URda[1][2])*UL[2][2])/tan(beta)        
    lamHt44 = (-mt4/v + (MQQ/v)*(ULda[1][1])*UR[1][1] + (MTT/v)*(ULda[1][2])*UR[2][1])/tan(beta)    
    lamHrt44 = (-mt4/v + (MQQ/v)*(URda[1][1])*UL[1][1] + (MTT/v)*(URda[1][2])*UL[2][1])/tan(beta)
    lamHt55 = (-mt5/v + (MQQ/v)*(ULda[2][1])*UR[1][2] + (MTT/v)*(ULda[2][2])*UR[2][2])/tan(beta)    
    lamHrt55 = (-mt5/v + (MQQ/v)*(URda[2][1])*UL[1][2] + (MTT/v)*(URda[2][2])*UL[2][2])/tan(beta)

    lamHtest = UR[2][0]
    lamHtest1 = UL[2][0]
    lamHtest2 = UR[1][0]
    lamHtest3 = ULda[1][1]
    lamHtest4 = ULda[1][2]
    
    #Charged Higgs coupling
    #lamHpm43 = -(sin(beta))*( yb*(ULda[1][0])*VR[0][0] + lamB*(ULda[1][0])*VR[2][0] + lamQ*(ULda[1][1])*VR[0][0] + lam*(ULda[1][1])*VR[2][0] ) + ( yt*(ULda[1][0])*VR[0][0] + kab*(ULda[1][2])*VR[1][0] )*cos(beta)
    lamHpm43 = -(sin(beta))*( yb*(ULda[1][0])*VR[0][0] + lamB*(ULda[1][0])*VR[2][0] + lamQ*(ULda[1][1])*VR[0][0] + lam*(ULda[1][1])*VR[2][0] ) + ( kab*(ULda[1][2])*VR[1][0])*cos(beta)
    
    #lamHpmr43 = -(sin(beta))*lamba*(VLda[0][2])*UR[1][1] + (cos(beta))*( kaT*(VLda[0][0])*UR[2][1] + kaQ*(VLda[0][1])*UR[0][1] + ka*(VLda[0][1])*UR[2][1] )
    lamHpmr43 = -(sin(beta))*lamba*(VLda[0][2])*UR[1][1] + (cos(beta))*( yt*(VLda[0][0])*UR[0][1] + kaT*(VLda[0][0])*UR[2][1] + kaQ*(VLda[0][1])*UR[0][1] + ka*(VLda[0][1])*UR[2][1] )
    
    #lamHpmb43 = -(sin(beta))*lamba*(VLda[1][2])*UR[1][0] + (cos(beta))*( kaT*(VLda[1][0])*UR[2][0] + kaQ*(VLda[1][1])*UR[0][0] + ka*(VLda[1][1])*UR[2][0] )
    lamHpmb43 = -(sin(beta))*lamba*(VLda[1][2])*UR[1][0] + (cos(beta))*( yt*(VLda[1][0])*UR[0][0] + kaT*(VLda[1][0])*UR[2][0] + kaQ*(VLda[1][1])*UR[0][0] + ka*(VLda[1][1])*UR[2][0] )
    
    #lamHpmbr43 = -(sin(beta))*( yb*(ULda[0][0])*VR[0][1] + lamB*(ULda[0][0])*VR[2][1] + lamQ*(ULda[0][1])*VR[0][1] + lam*(ULda[0][1])*VR[2][1] ) + (yt*(ULda[0][0])*VR[0][1] + kab*(ULda[0][2])*VR[1][1] )*cos(beta)
    lamHpmbr43 = -(sin(beta))*( yb*(ULda[0][0])*VR[0][1] + lamB*(ULda[0][0])*VR[2][1] + lamQ*(ULda[0][1])*VR[0][1] + lam*(ULda[0][1])*VR[2][1] ) +( kab*(ULda[0][2])*VR[1][1] )*cos(beta)


    
    lamHb34 = -( (MQQ/v)*(VLda[0][1])*VR[1][1] + (MBB/v)*(VLda[0][2])*VR[2][1])*tan(beta)
    lamHrb34 = -( (MQQ/v)*(VRda[0][1])*VL[1][1] + (MBB/v)*(VRda[0][2])*VL[2][1])*tan(beta)    
    lamHb35 = -( (MQQ/v)*(VLda[0][1])*VR[1][2] + (MBB/v)*(VLda[0][2])*VR[2][2])*tan(beta)
    lamHrb35 = -( (MQQ/v)*(VRda[0][1])*VL[1][2] + (MBB/v)*(VRda[0][2])*VL[2][2])*tan(beta)    
    lamHb45 = -( (MQQ/v)*(VLda[1][1])*VR[1][2] + (MBB/v)*(VLda[1][2])*VR[2][2])*tan(beta)
    lamHrb45 = -( (MQQ/v)*(VRda[1][1])*VL[1][2] + (MBB/v)*(VRda[1][2])*VL[2][2])*tan(beta)    
    lamHb44 = (mb4/v - (MQQ/v)*(VLda[1][1])*VR[1][1] - (MBB/v)*(VLda[1][2])*VR[2][1])*tan(beta)    
    lamHrb44 = (mb4/v - (MQQ/v)*(VRda[1][1])*VL[1][1] - (MBB/v)*(VRda[1][2])*VL[2][1])*tan(beta)
    lamHb55 = (mb5/v - (MQQ/v)*(VLda[2][1])*VR[1][2] - (MBB/v)*(VLda[2][2])*VR[2][2])*tan(beta)    
    lamHrb55 = (mb5/v - (MQQ/v)*(VRda[2][1])*VL[1][2] - (MBB/v)*(VRda[2][2])*VL[2][2])*tan(beta)
    
    gLW44t = (g/sqrt(2))*(ULda[1][0]*VL[0][1]*0.99915 + ULda[1][1]*VL[1][1])
    gLW43t = (g/sqrt(2))*(ULda[1][0]*VL[0][0]*0.99915 + ULda[1][1]*VL[1][0])
    gLW45t = (g/sqrt(2))*(ULda[1][0]*VL[0][2]*0.99915 + ULda[1][1]*VL[1][2])
    gLW33t = (g/sqrt(2))*(ULda[0][0]*VL[0][0]*0.99915 + ULda[0][1]*VL[1][0])
    gLW34t = (g/sqrt(2))*(ULda[0][0]*VL[0][1]*0.99915 + ULda[0][1]*VL[1][1])
    gLW35t = (g/sqrt(2))*(ULda[0][0]*VL[0][2]*0.99915 + ULda[0][1]*VL[1][2])
    gLW55t = (g/sqrt(2))*(ULda[2][0]*VL[0][2]*0.99915 + ULda[2][1]*VL[1][2])
    gLW53t = (g/sqrt(2))*(ULda[2][0]*VL[0][0]*0.99915 + ULda[2][1]*VL[1][0])
    gLW54t = (g/sqrt(2))*(ULda[2][0]*VL[0][1]*0.99915 + ULda[2][1]*VL[1][1])
    gRW44t = (g/sqrt(2))*(URda[1][1]*VR[1][1])
    gRW43t = (g/sqrt(2))*(URda[1][1]*VR[1][0])
    gRW45t = (g/sqrt(2))*(URda[1][1]*VR[1][2])
    gRW33t = (g/sqrt(2))*(URda[0][1]*VR[1][0])
    gRW34t = (g/sqrt(2))*(URda[0][1]*VR[1][1])
    gRW35t = (g/sqrt(2))*(URda[0][1]*VR[1][2])
    gRW55t = (g/sqrt(2))*(URda[2][1]*VR[1][2])
    gRW53t = (g/sqrt(2))*(URda[2][1]*VR[1][0])
    gRW54t = (g/sqrt(2))*(URda[2][1]*VR[1][1])
        
    gLZ44t = (g/cos(thetaw))*((1./2. - (2./3.)*(sin(thetaw))**2) - (1./2.)*ULda[1][2]*UL[2][1])
    gLZ43t = (g/cos(thetaw))*( - (1./2.)*ULda[1][2]*UL[2][0])
    gLZ45t = (g/cos(thetaw))*( - (1./2.)*ULda[1][2]*UL[2][2])
    gLZ33t = (g/cos(thetaw))*((1./2. - (2./3.)*(sin(thetaw))**2) - (1./2.)*ULda[0][2]*UL[2][0])
    gLZ34t = (g/cos(thetaw))*( - (1./2.)*ULda[0][2]*UL[2][1])
    gLZ35t = (g/cos(thetaw))*( - (1./2.)*ULda[0][2]*UL[2][2])
    gLZ55t = (g/cos(thetaw))*((1./2. - (2./3.)*(sin(thetaw))**2) - (1./2.)*ULda[2][2]*UL[2][2])
    gLZ53t = (g/cos(thetaw))*( - (1./2.)*ULda[2][2]*UL[2][0])
    gLZ54t = (g/cos(thetaw))*( - (1./2.)*ULda[2][2]*UL[2][1])
    gRZ44t = (g/cos(thetaw))*(-(2./3.)*(sin(thetaw))**2 + (1./2.)*URda[1][1]*UR[1][1])
    gRZ43t = (g/cos(thetaw))*(1./2.)*URda[1][1]*UR[1][0]
    gRZ45t = (g/cos(thetaw))*(1./2.)*URda[1][1]*UR[1][2]
    gRZ33t = (g/cos(thetaw))*(-(2./3.)*(sin(thetaw))**2 + (1./2.)*URda[0][1]*UR[1][0])
    gRZ34t = (g/cos(thetaw))*(1./2.)*URda[0][1]*UR[1][1]
    gRZ35t = (g/cos(thetaw))*(1./2.)*URda[0][1]*UR[1][2]
    gRZ55t = (g/cos(thetaw))*(-(2./3.)*(sin(thetaw))**2 + (1./2.)*URda[2][1]*UR[1][2])
    gRZ53t = (g/cos(thetaw))*(1./2.)*URda[2][1]*UR[1][0]
    gRZ54t = (g/cos(thetaw))*(1./2.)*URda[2][1]*UR[1][1]

    gLZ33b = (g/cos(thetaw))*((-1./2. + (1./3.)*(sin(thetaw))**2) + (1./2.)*VLda[0][2]*VL[2][0])
    gLZ44b = (g/cos(thetaw))*((-1./2. + (1./3.)*(sin(thetaw))**2) + (1./2.)*VLda[1][2]*VL[2][1])
    gLZ55b = (g/cos(thetaw))*((-1./2. + (1./3.)*(sin(thetaw))**2) + (1./2.)*VLda[2][2]*VL[2][2])
    gLZ34b = (g/cos(thetaw))*(1./2.)*VLda[0][2]*VL[2][1]
    gLZ43b = (g/cos(thetaw))*(1./2.)*VLda[1][2]*VL[2][0]
    gLZ35b = (g/cos(thetaw))*(1./2.)*VLda[0][2]*VL[2][2]
    gLZ53b = (g/cos(thetaw))*(1./2.)*VLda[2][2]*VL[2][0]
    gLZ45b = (g/cos(thetaw))*(1./2.)*VLda[1][2]*VL[2][2]
    gLZ54b = (g/cos(thetaw))*(1./2.)*VLda[2][2]*VL[2][1]
    gRZ33b = (g/cos(thetaw))*((1./3.)*(sin(thetaw))**2 - (1./2.)*VRda[0][1]*VR[1][0])
    gRZ44b = (g/cos(thetaw))*((1./3.)*(sin(thetaw))**2 - (1./2.)*VRda[1][1]*VR[1][1])
    gRZ55b = (g/cos(thetaw))*((1./3.)*(sin(thetaw))**2 - (1./2.)*VRda[2][1]*VR[1][2])
    gRZ34b = (g/cos(thetaw))*(- (1./2.)*VRda[0][1]*VR[1][1])
    gRZ43b = (g/cos(thetaw))*(- (1./2.)*VRda[1][1]*VR[1][0])
    gRZ35b = (g/cos(thetaw))*(- (1./2.)*VRda[0][1]*VR[1][2])
    gRZ53b = (g/cos(thetaw))*(- (1./2.)*VRda[2][1]*VR[1][0])
    gRZ45b = (g/cos(thetaw))*(- (1./2.)*VRda[1][1]*VR[1][2])
    gRZ54b = (g/cos(thetaw))*(- (1./2.)*VRda[2][1]*VR[1][1])

    
    #mytest = (g/cos(thetaw))*( (-1/2.0 + pow(sin(thetaw),2))*(URda[0][1])*UR[1][0] + (pow(sin(thetaw),2))*( (URda[0][0])*UR[0][0] + (URda[0][2])*UR[2][0] ) )    
    #csfwd = (pow(gelsm*gLZmu,2)*7 + pow(gelsm*gRZmu,2) + pow(gersm*gLZmu,2) + pow(gersm*gRZmu,2)*7)/2.
    #csbwd = (pow(gelsm*gLZmu,2) + pow(gelsm*gRZmu,2)*7 + pow(gersm*gLZmu,2)*7 + pow(gersm*gRZmu,2))/2.
    #Afbmuz = (csfwd - csbwd)/(csfwd + csbwd)
    #cvb = (2/3.)*pow(sin(thetaw),2) - 1/2.
    #vab = -1/2.
    #cau = 1/2.
    #cvu = 1/2. - (4/3.)*pow(sin(thetaw),2)
    #Nc = 3
    #gbL = (cvb + cab)/2.
    #gbr = (cvb - cab)/2.
    #guL = (cvu + cau)/2.
    #guR = (cvu - cau)/2.
    # I guess this is a global fit from the Z-pole observables?
    # Z-pole observables
    # Gamma(W -> \mu \nu_\mu) ratio
    # Muon lifetime
    # Neutral current variables : neutrino scattering
    # I guess this is a global fit from the Z-pole observables?
    # effective angle sin^2 \theta : effective coupling at the resonance
    # DY pair production e4e4, 
        
    
    # Oblique corrections


    yu3 = cevaluesl[0]/MZ**2
    yu4 = cevaluesl[1]/MZ**2
    yu5 = cevaluesl[2]/MZ**2
    yd3 = evaluesl[0]/MZ**2
    yd4 = evaluesl[1]/MZ**2
    yd5 = evaluesl[2]/MZ**2
    ydSM3 = (mbpole**2)/MZ**2
    yuSM3 = (mt**2)/MZ**2

    
    #gW3L44t = (g/2.)*((1./2. - (2./3.)*(sin(thetaw))**2) - (1./2.)*ULda[1][2]*UL[2][1])
    #gW3L33t = (g/2.)*((1./2. - (2./3.)*(sin(thetaw))**2) - (1./2.)*ULda[0][2]*UL[2][0])
    #gW3L55t = (g/2.)*((1./2. - (2./3.)*(sin(thetaw))**2) - (1./2.)*ULda[2][2]*UL[2][2])
    #gW3L43t = (g/2.)*(- (1./2.)*ULda[1][2]*UL[2][0])
    #gW3L53t = (g/2.)*(- (1./2.)*ULda[2][2]*UL[2][0])
    #gW3L54t = (g/2.)*(- (1./2.)*ULda[2][2]*UL[2][1])
    #gW3R44t = (g/2.)*(-(2./3.)*(sin(thetaw))**2 + (1./2.)*URda[1][1]*UR[1][1])    
    #gW3R33t = (g/2.)*(-(2./3.)*(sin(thetaw))**2 + (1./2.)*URda[0][1]*UR[1][0])
    #gW3R55t = (g/2.)*(-(2./3.)*(sin(thetaw))**2 + (1./2.)*URda[2][1]*UR[1][2])
    #gW3R43t = (g/2.)*(1./2.)*URda[1][1]*UR[1][0]
    #gW3R53t = (g/2.)*(1./2.)*URda[2][1]*UR[1][0]
    #gW3R54t = (g/2.)*(1./2.)*URda[2][1]*UR[1][1]
    
    #gW3L44b = (g/2.)*((-1./2. + (1./3.)*(sin(thetaw))**2) + (1./2.)*VLda[1][2]*UL[2][1])
    #gW3L33b = (g/2.)*((-1./2. + (1./3.)*(sin(thetaw))**2) + (1./2.)*VLda[0][2]*UL[2][0])
    #gW3L55b = (g/2.)*((-1./2. + (1./3.)*(sin(thetaw))**2) + (1./2.)*VLda[2][2]*UL[2][2])
    #gW3L43b = (g/2.)*((1./2.)*VLda[1][2]*VL[2][0])
    #gW3L53b = (g/2.)*((1./2.)*VLda[2][2]*VL[2][0])
    #gW3L54b = (g/2.)*((1./2.)*VLda[2][2]*VL[2][1])
    #gW3R44b = (g/2.)*((1./3.)*(sin(thetaw))**2 - (1./2.)*VRda[1][1]*VR[1][1])    
    #gW3R33b = (g/2.)*((1./3.)*(sin(thetaw))**2 - (1./2.)*VRda[0][1]*VR[1][0])
    #gW3R55b = (g/2.)*((1./3.)*(sin(thetaw))**2 - (1./2.)*VRda[2][1]*VR[1][2])
    #gW3R43b = (g/2.)*(-1./2.)*VRda[1][1]*VR[1][0]
    #gW3R53b = (g/2.)*(-1./2.)*VRda[2][1]*VR[1][0]
    #gW3R54b = (g/2.)*(-1./2.)*VRda[2][1]*VR[1][1]
    gW3L44t = (g)*((1./2. - (2./3.)*(sin(thetaw))**2) - (1./2.)*ULda[1][2]*UL[2][1])
    gW3L33t = (g)*((1./2. - (2./3.)*(sin(thetaw))**2) - (1./2.)*ULda[0][2]*UL[2][0])
    gW3L55t = (g)*((1./2. - (2./3.)*(sin(thetaw))**2) - (1./2.)*ULda[2][2]*UL[2][2])
    gW3L43t = (g)*(- (1./2.)*ULda[1][2]*UL[2][0])
    gW3L53t = (g)*(- (1./2.)*ULda[2][2]*UL[2][0])
    gW3L54t = (g)*(- (1./2.)*ULda[2][2]*UL[2][1])
    gW3R44t = (g)*(-(2./3.)*(sin(thetaw))**2 + (1./2.)*URda[1][1]*UR[1][1])    
    gW3R33t = (g)*(-(2./3.)*(sin(thetaw))**2 + (1./2.)*URda[0][1]*UR[1][0])
    gW3R55t = (g)*(-(2./3.)*(sin(thetaw))**2 + (1./2.)*URda[2][1]*UR[1][2])
    gW3R43t = (g)*(1./2.)*URda[1][1]*UR[1][0]
    gW3R53t = (g)*(1./2.)*URda[2][1]*UR[1][0]
    gW3R54t = (g)*(1./2.)*URda[2][1]*UR[1][1]
    
    gW3L44b = (g)*((-1./2. + (1./3.)*(sin(thetaw))**2) + (1./2.)*VLda[1][2]*UL[2][1])
    gW3L33b = (g)*((-1./2. + (1./3.)*(sin(thetaw))**2) + (1./2.)*VLda[0][2]*UL[2][0])
    gW3L55b = (g)*((-1./2. + (1./3.)*(sin(thetaw))**2) + (1./2.)*VLda[2][2]*UL[2][2])
    gW3L43b = (g)*((1./2.)*VLda[1][2]*VL[2][0])
    gW3L53b = (g)*((1./2.)*VLda[2][2]*VL[2][0])
    gW3L54b = (g)*((1./2.)*VLda[2][2]*VL[2][1])
    gW3R44b = (g)*((1./3.)*(sin(thetaw))**2 - (1./2.)*VRda[1][1]*VR[1][1])    
    gW3R33b = (g)*((1./3.)*(sin(thetaw))**2 - (1./2.)*VRda[0][1]*VR[1][0])
    gW3R55b = (g)*((1./3.)*(sin(thetaw))**2 - (1./2.)*VRda[2][1]*VR[1][2])
    gW3R43b = (g)*(-1./2.)*VRda[1][1]*VR[1][0]
    gW3R53b = (g)*(-1./2.)*VRda[2][1]*VR[1][0]
    gW3R54b = (g)*(-1./2.)*VRda[2][1]*VR[1][1]

    
    

    mspdg = 0.096
    mcpole = 1.27
    yd2 = (mspdg**2)/MZ**2
    yu2 = (mcpole**2)/MZ**2
    #gLW42 = (g/sqrt(2))*(ULda[1][0])*(0.22492 + 0.97351 + 0.0411)
    gLW42t = (g/sqrt(2))*(ULda[1][0])*0.0403
    gLW52t = (g/sqrt(2))*(ULda[2][0])*0.0403
    gLW24b = (g/sqrt(2))*(VL[0][1])*0.0411
    gLW25b = (g/sqrt(2))*(VL[0][2])*0.0411

    mdpdg = 0.0047
    mupdg = 0.0022
    yd1 = (mdpdg**2)/MZ**2
    yu1 = (mupdg**2)/MZ**2
    #gLW41 = (g/sqrt(2))*(ULda[1][0])*(0.97434 + 0.22506 + 0.00357)
    gLW41t = (g/sqrt(2))*(ULda[1][0])*0.00875
    gLW51t = (g/sqrt(2))*(ULda[2][0])*0.00875
    gLW14b = (g/sqrt(2))*(VL[0][1])*0.00357
    gLW15b = (g/sqrt(2))*(VL[0][2])*0.00357



    
    # substract the original SM 33 contribution
    smgLW33 = (g/sqrt(2))*(0.00875 + 0.0403 + 0.99915)


    
    # neglect the contributions by the massless particles
    # \Delta T
    #delT = (1./(8.*pi*pow(sin(thetaw)*cos(thetaw),2)))*(1/pow(g,2))*( pow(gLW43,2)*thetap(yu4,yd3) + pow(gLW33,2)*thetap(yu3,yd3) + pow(gLW41,2)*thetap(yu4,yd1) + pow(gLW42,2)*thetap(yu4,yd2) - pow(smgLW33,2)*thetap(yu3,yd3)  -  2*( pow(gW3L43,2)*thetap(yu4,yu3)   )   )
    try:
        delT = (3./(8.*pi*pow(sin(thetaw)*cos(thetaw),2)))*(1/pow(g,2))*( (pow(gLW43t,2) + pow(gRW43t,2))*thetap(yu4,yd3) + 2*gLW43t*gRW43t*thetam(yu4,yd3)
                                                                          #+ pow(gLW41t,2)*thetap(yu4,yd1) + pow(gLW42t,2)*thetap(yu4,yd2)
                                                                          + (pow(gLW53t,2) + pow(gRW53t,2))*thetap(yu5,yd3) + 2*gLW53t*gRW53t*thetam(yu5,yd3)
                                                                          #+ pow(gLW51t,2)*thetap(yu5,yd1)
                                                                          + pow(gLW52t,2)*thetap(yu5,yd2) + (pow(gLW44t,2) + pow(gRW44t,2))*thetap(yu4,yd4) + 2*gLW44t*gRW44t*thetam(yu4,yd4) + (pow(gLW54t,2) + pow(gRW54t,2))*thetap(yu5,yd4) + 2*gLW54t*gRW54t*thetam(yu5,yd4) + (pow(gLW55t,2) + pow(gRW55t,2))*thetap(yu5,yd5) + 2*gLW55t*gRW55t*thetam(yu5,yd5) + (pow(gLW34t,2) + pow(gRW34t,2))*thetap(yu3,yd4) + 2*gLW34t*gRW34t*thetam(yu3,yd4) + (pow(gLW35t,2) + pow(gRW35t,2))*thetap(yu3,yd5) + 2*gLW35t*gRW35t*thetam(yu3,yd5) + (pow(gLW45t,2) + pow(gRW45t,2))*thetap(yu4,yd5) + 2*gLW45t*gRW45t*thetam(yu4,yd5) + (pow(gLW33t,2)*thetap(yu3,yd3) - pow(smgLW33,2)*thetap(yuSM3,yd3))  -  2*( (pow(gW3L43t,2) + pow(gW3R43t,2))*thetap(yu4,yu3) + 2*gW3L43t*gW3R43t*thetam(yu4,yu3) + (pow(gW3L53t,2) + pow(gW3R53t,2))*thetap(yu5,yu3) + 2*gW3L53t*gW3R53t*thetam(yu5,yu3) + (pow(gW3L54t,2) + pow(gW3R54t,2))*thetap(yu5,yu4) + 2*gW3L54t*gW3R54t*thetam(yu5,yu4)  )
     #+ pow(gLW14b,2)*thetap(yu1,yd4) + pow(gLW24b,2)*thetap(yu2,yd4) + pow(gLW15b,2)*thetap(yu1,yd5) + pow(gLW25b,2)*thetap(yu2,yd5)
     - 2*( (pow(gW3L43b,2) + pow(gW3R43b,2))*thetap(yd4,yd3) + 2*gW3L43b*gW3R43b*thetam(yd4,yd3) + (pow(gW3L53b,2) + pow(gW3R53b,2))*thetap(yd5,yd3) + 2*gW3L53b*gW3R53b*thetam(yd5,yd3) + (pow(gW3L54b,2) + pow(gW3R54b,2))*thetap(yd5,yd4) + 2*gW3L54b*gW3R54b*thetam(yd5,yd4)  )                                          )    
    except ValueError:
        continue
    # \Delta S
    try:
        delS = (3./(2.*pi))*(2/pow(g,2))*( (pow(gLW43t,2) + pow(gRW43t,2))*psip(yu4,yd3) + 2*gLW43t*gRW43t*psim(yu4,yd3) + (pow(gLW53t,2) + pow(gRW53t,2))*psip(yu5,yd3) + 2*gLW53t*gRW53t*psim(yu5,yd3) + (pow(gLW44t,2) + pow(gRW44t,2))*psip(yu4,yd4) + 2*gLW44t*gRW44t*psim(yu4,yd4) + (pow(gLW54t,2) + pow(gRW54t,2))*psip(yu5,yd4) + 2*gLW54t*gRW54t*psim(yu5,yd4) + (pow(gLW55t,2) + pow(gRW55t,2))*psip(yu5,yd5) + 2*gLW55t*gRW55t*psim(yu5,yd5) + (pow(gLW34t,2) + pow(gRW34t,2))*psip(yu3,yd4) + 2*gLW34t*gRW34t*psim(yu3,yd4) + (pow(gLW35t,2) + pow(gRW35t,2))*psip(yu3,yd5) + 2*gLW35t*gRW35t*psim(yu3,yd5) + (pow(gLW45t,2) + pow(gRW45t,2))*psip(yu4,yd5) + 2*gLW45t*gRW45t*psim(yu4,yd5)  + ( pow(gLW33t,2)*psip(yu3,yd3)  - pow(smgLW33,2)*psip(yuSM3,yd3) )   -  2*( (pow(gW3L43t,2) + pow(gW3R43t,2))*psip(yu4,yu3) + 2*gW3L43t*gW3R43t*psim(yu4,yu3) + (pow(gW3L53t,2) + pow(gW3R53t,2))*psip(yu5,yu3) + 2*gW3L53t*gW3R53t*psim(yu5,yu3) + (pow(gW3L54t,2) + pow(gW3R54t,2))*psip(yu5,yu4) + 2*gW3L54t*gW3R54t*psim(yu5,yu4)   )
            - 2*((pow(gW3L43b,2) + pow(gW3R43b,2))*psip(yd4,yd3) + 2*gW3L43b*gW3R43b*psim(yd4,yd3) + (pow(gW3L53b,2) + pow(gW3R53b,2))*psip(yd5,yd3) + 2*gW3L53b*gW3R53b*psim(yd5,yd3) + (pow(gW3L54b,2) + pow(gW3R54b,2))*psip(yd5,yd4) + 2*gW3L54b*gW3R54b*psim(yd5,yd4))                               
        )
        #delS = (1./(2.*pi))*(2/pow(g,2))*( (pow(gLW43t,2) + pow(gRW43t,2))*psip(yu4,yd3) + 2*gLW43t*gRW43t*psim(yu4,yd3) + pow(gLW41t,2)*psip(yu4,yd1) + pow(gLW42t,2)*psip(yu4,yd2) + (pow(gLW53t,2) + pow(gRW53t,2))*psip(yu5,yd3) + 2*gLW53t*gRW53t*psim(yu5,yd3) + pow(gLW51t,2)*psip(yu5,yd1) + pow(gLW52t,2)*psip(yu5,yd2)  + (pow(gLW44t,2) + pow(gRW44t,2))*psip(yu4,yd4) + 2*gLW44t*gRW44t*psim(yu4,yd4) + (pow(gLW54t,2) + pow(gRW54t,2))*psip(yu5,yd4) + 2*gLW54t*gRW54t*psim(yu5,yd4) + (pow(gLW55t,2) + pow(gRW55t,2))*psip(yu5,yd5) + 2*gLW55t*gRW55t*psim(yu5,yd5) + (pow(gLW34t,2) + pow(gRW34t,2))*psip(yu3,yd4) + 2*gLW34t*gRW34t*psim(yu3,yd4) + (pow(gLW35t,2) + pow(gRW35t,2))*psip(yu3,yd5) + 2*gLW35t*gRW35t*psim(yu3,yd5) + (pow(gLW45t,2) + pow(gRW45t,2))*psip(yu4,yd5) + 2*gLW45t*gRW45t*psim(yu4,yd5)  + ( pow(gLW33t,2)*psip(yu3,yd3)  - pow(smgLW33,2)*psip(yuSM3,yd3) )   -  2*( (pow(gW3L43t,2) + pow(gW3R43t,2))*psip(yu4,yu3) + 2*gW3L43t*gW3R43t*psim(yu4,yu3) + (pow(gW3L53t,2) + pow(gW3R53t,2))*psip(yu5,yu3) + 2*gW3L53t*gW3R53t*psim(yu5,yu3) + (pow(gW3L54t,2) + pow(gW3R54t,2))*psip(yu5,yu4) + 2*gW3L54t*gW3R54t*psim(yu5,yu4)   )  )
    except ValueError:
        continue
    # \Delta U
    try:
        delU = -(3./(2.*pi))*(2/pow(g,2))*( (pow(gLW43t,2) + pow(gRW43t,2))*chip(yu4,yd3) + 2*gLW43t*gRW43t*chim(yu4,yd3) + (pow(gLW53t,2) + pow(gRW53t,2))*chip(yu5,yd3) + 2*gLW53t*gRW53t*chim(yu5,yd3) + (pow(gLW44t,2) + pow(gRW44t,2))*chip(yu4,yd4) + 2*gLW44t*gRW44t*chim(yu4,yd4) + (pow(gLW54t,2) + pow(gRW54t,2))*chip(yu5,yd4) + 2*gLW54t*gRW54t*chim(yu5,yd4) + (pow(gLW55t,2) + pow(gRW55t,2))*chip(yu5,yd5) + 2*gLW55t*gRW55t*chim(yu5,yd5) + (pow(gLW34t,2) + pow(gRW34t,2))*chip(yu3,yd4) + 2*gLW34t*gRW34t*chim(yu3,yd4) + (pow(gLW35t,2) + pow(gRW35t,2))*chip(yu3,yd5) + 2*gLW35t*gRW35t*chim(yu3,yd5) + (pow(gLW45t,2) + pow(gRW45t,2))*chip(yu4,yd5) + 2*gLW45t*gRW45t*chim(yu4,yd5)  + ( pow(gLW33t,2)*chip(yu3,yd3) - pow(smgLW33,2)*chip(yuSM3,yd3) ) -  2*( (pow(gW3L43t,2) + pow(gW3R43t,2))*chip(yu4,yu3) + 2*gW3L43t*gW3R43t*chim(yu4,yu3) + (pow(gW3L53t,2) + pow(gW3R53t,2))*chip(yu5,yu3) + 2*gW3L53t*gW3R53t*chim(yu5,yu3) + (pow(gW3L54t,2) + pow(gW3R54t,2))*chip(yu5,yu4) + 2*gW3L54t*gW3R54t*chim(yu5,yu4)   )
     - 2*( (pow(gW3L43b,2) + pow(gW3R43b,2))*chip(yd4,yd3) + 2*gW3L43b*gW3R43b*chim(yd4,yd3) + (pow(gW3L53b,2) + pow(gW3R53b,2))*chip(yd5,yd3) + 2*gW3L53b*gW3R53b*chim(yd5,yd3) + (pow(gW3L54b,2) + pow(gW3R54b,2))*chip(yd5,yd4) + 2*gW3L54b*gW3R54b*chim(yd5,yd4)  )                                       
        )
        #delU = -(1./(2.*pi))*(2/pow(g,2))*( (pow(gLW43t,2) + pow(gRW43t,2))*chip(yu4,yd3) + 2*gLW43t*gRW43t*chim(yu4,yd3) + pow(gLW41t,2)*chip(yu4,yd1) + pow(gLW42t,2)*chip(yu4,yd2) + (pow(gLW53t,2) + pow(gRW53t,2))*chip(yu5,yd3) + 2*gLW53t*gRW53t*chim(yu5,yd3) + pow(gLW51t,2)*chip(yu5,yd1) + pow(gLW52t,2)*chip(yu5,yd2) + (pow(gLW44t,2) + pow(gRW44t,2))*chip(yu4,yd4) + 2*gLW44t*gRW44t*chim(yu4,yd4) + (pow(gLW54t,2) + pow(gRW54t,2))*chip(yu5,yd4) + 2*gLW54t*gRW54t*chim(yu5,yd4) + (pow(gLW55t,2) + pow(gRW55t,2))*chip(yu5,yd5) + 2*gLW55t*gRW55t*chim(yu5,yd5) + (pow(gLW34t,2) + pow(gRW34t,2))*chip(yu3,yd4) + 2*gLW34t*gRW34t*chim(yu3,yd4) + (pow(gLW35t,2) + pow(gRW35t,2))*chip(yu3,yd5) + 2*gLW35t*gRW35t*chim(yu3,yd5) + (pow(gLW45t,2) + pow(gRW45t,2))*chip(yu4,yd5) + 2*gLW45t*gRW45t*chim(yu4,yd5)  + ( pow(gLW33,2)*chip(yu3,yd3) - pow(smgLW33,2)*chip(yuSM3,yd3) ) -  2*( (pow(gW3L43t,2) + pow(gW3R43t,2))*chip(yu4,yu3) + 2*gW3L43t*gW3R43t*chim(yu4,yu3) + (pow(gW3L53t,2) + pow(gW3R53t,2))*chip(yu5,yu3) + 2*gW3L53t*gW3R53t*chim(yu5,yu3) + (pow(gW3L54t,2) + pow(gW3R54t,2))*chip(yu5,yu4) + 2*gW3L54t*gW3R54t*chim(yu5,yu4)   )  )
    except ValueError:
        continue
    
    



    # Now collect the data to pass the EW precision, etc
    print 'first passed'
    mylist = [mt4,tan(beta),lamHt34,mHH,(pow(ULda[1][0],2) + pow(ULda[1][1],2) + pow(URda[1][1],2))/2.0,(pow(URda[1][0],2) + pow(ULda[1][2],2) + pow(URda[1][2],2))/2.0,delS,delT,delU,lamHrt34,lamHt44,lamHrt44,gLW44t,gLW43t,gLW45t,gLW33t,gLW34t,gLW35t,gLW55t,gLW53t,gLW54t,gRW44t,gRW43t,gRW45t,gRW33t,gRW34t,gRW35t,gRW55t,gRW54t,gRW53t,gLZ44t,gLZ43t,gLZ45t,gLZ33t,gLZ34t,gLZ35t,gLZ55t,gLZ53t,gLZ54t,gRZ44t,gRZ43t,gRZ45t,gRZ33t,gRZ34t,gRZ35t,gRZ55t,gRZ53t,gRZ54t,kaT,kaQ,ka,kab,mt5,MTT,MQQ,lamHt55,lamHrt55,lamHt35,lamHrt35,lamHt45,lamHrt45,lamHb34,lamHrb34,lamHb35,lamHrb35,lamHb45,lamHrb45,lamHb44,lamHrb44,lamHb55,lamHrb55,gLZ33b,gLZ44b,gLZ55b,gLZ34b,gLZ43b,gLZ35b,gLZ53b,gLZ45b,gLZ54b,gRZ33b,gRZ44b,gRZ55b,gRZ34b,gRZ43b,gRZ35b,gRZ53b,gRZ45b,gRZ54b,gLW42t,gLW52t,gLW24b,gLW25b,gLW41t,gLW51t,gLW14b,gLW15b,mb4,mb5,(pow(ULda[2][0],2) + pow(ULda[2][1],2) + pow(URda[2][1],2))/2.0,(pow(URda[2][0],2) + pow(ULda[2][2],2) + pow(URda[2][2],2))/2.0,(pow(VLda[1][0],2) + pow(VLda[1][1],2) + pow(VRda[1][1],2))/2.0,(pow(VRda[1][0],2) + pow(VLda[1][2],2) + pow(VRda[1][2],2))/2.0,(pow(VLda[2][0],2) + pow(VLda[2][1],2) + pow(VRda[2][1],2))/2.0,(pow(VRda[2][0],2) + pow(VLda[2][2],2) + pow(VRda[2][2],2))/2.0,lamHtest,lamHpmb43,lamHpmbr43,lamHpm43,lamHpmr43,"\n"]       # write the list with mass and cross section
    fevout=open("t1ewptallhelp-s6.dat", "a+")
    #fevout=open("t1ewptallhelp-0.5Yukawa.dat", "a+")
    for item in mylist:
        fevout.write(str(item) + '\t')   # write the list by inserting tab between the elements
    fevout.close()                       # we change the numbers in mylist into 'str' since only     

    
#exe("./t1ewptallhelp.py")



"""    
    # Branching ratios
    gamt4w = (mt4/(32.*pi))*pow(mt4/MW,2)*pow(1-pow(MW/mt4,2),2)*(1 + 2*pow(MW/mt4,2))*(pow(gLW43,2))*unitstep(mt4 - MW)
    gamt4z = (mt4/(32.*pi))*pow(mt4/MZ,2)*pow(1-pow(MZ/mt4,2),2)*(1 + 2*pow(MZ/mt4,2))*(pow(gLZe24,2) + pow(gRZe24,2))*unitstep(me4 - MZ - mmu)    
    gamt4h125 = (me4/(16.*pi))*pow(1-pow(125/me4,2),2)*(pow(lamHe24/tan(beta),2) + pow(lamHre24/tan(beta),2))*unitstep(me4 - 125 - mmu)
    game4all = game4w + game4z + game4h125

    

    
    try:
        bre4w = game4w/game4all
        bre4h125 = game4h125/game4all
        bre5w = game5w/game5all
        bre5h125 = game5h125/game5all
        bre5z = game5z/game5all
        bre5e4z = game5e4z/game5all
        bre5HH = game5HH/game5all
        bre5e4HH = game5e4HH/game5all
    except ZeroDivisionError:
        continue

    # Branching Ratios of Higgs
    mb = 2.71#4.18
    alphas = 0.109683
    alphaEM = 1/127.462
    Nf = 5
    gamHtobb = ((3*GF)/(4*sqrt(2)*pi))*mHH*pow(mb*tan(beta),2)*(1 + 5.67*(alphas/pi) + (35.94 - 1.36*Nf)*pow(alphas,2)/pow(pi,2) + (164.14 - 25.77*Nf + 0.26*pow(Nf,2))*pow(alphas,3)/pow(pi,3) + (pow(alphas,2)/pow(pi,2))*(1.57 - (2/3.0)*log(pow(mHH,2)/pow(mt,2)) + (1/9.0)*pow(log(pow(mb,2)/pow(mHH,2)),2)))
    paramtau = pow(mHH/(2.*mt),2)
    fftau = pow(asin(sqrt(paramtau)),2)
    Ahalf = 2*(paramtau + (paramtau-1)*fftau)/pow(paramtau,2)
    gamHtogg = (GF*pow(alphas,2)*pow(mHH,3)/(36.*sqrt(2)*pow(pi,3)*pow(tan(beta),2)))*pow(abs(3*Ahalf/4.),2)
    gamHtoe4 = 2.0*(mHH/(16.*pi))*(lamHe24**2 + lamHre24**2)*pow((1 - pow(me4,2)/pow(mHH,2)),2)*unitstep(mHH - me4)
    gamHto2e4 = (mHH/(16.*pi))*((lamHe44**2 + lamHre44**2)*(1 - (me4**2 + me4**2)/mHH**2) - 4*lamHe44*lamHre44*me4*me4/mHH**2 )*sqrt(abs(phase(1,(me4**2)/mHH**2,(me4**2)/mHH**2)))*unitstep(mHH - 2.0*me4)
    if mHH < 2.0*125.0:
        gamHtohh = 0
    else:
        gamHtohh = (GF/(16.0*pi*sqrt(2)))*((MZ**4)/mHH)*(sqrt(1 - 4.0*(125.0/mHH)**2))*((3.0*sin(2.0*beta)*cos(2.0*beta))**2)
    mc = 0.6
    gamHtocc = ((3*GF)/(4*sqrt(2)*pi))*mHH*pow(mc/tan(beta),2)*(1 + 5.67*(alphas/pi) + (35.94 - 1.36*Nf)*pow(alphas,2)/pow(pi,2) + (164.14 - 25.77*Nf + 0.26*pow(Nf,2))*pow(alphas,3)/pow(pi,3) + (pow(alphas,2)/pow(pi,2))*(1.57 - (2/3.0)*log(pow(mHH,2)/pow(mt,2)) + (1/9.0)*pow(log(pow(mc,2)/pow(mHH,2)),2)))
    taue4 = pow(mHH/(2.*me4),2)
    taue5 = pow(mHH/(2.*me5),2)
    try:
        fftaue4 = pow(asin(sqrt(taue4)),2)
        fftaue5 = pow(asin(sqrt(taue5)),2)
    except ValueError:
        continue
    Ae4 = 2.0*(taue4 + (taue4-1)*fftaue4)/pow(taue4,2)
    Ae5 = 2.0*(taue5 + (taue5-1)*fftaue5)/pow(taue5,2)
    gamHdiph = (GF*pow(alphaEM,2)*pow(mHH,3)/(128.*sqrt(2)*pow(pi,3)))*pow(abs(-(3.0*((2.0/3.0)**2)*Ahalf)/tan(beta) + (lamHe44*v/me4)*Ae4 + (lamHe55*v/me5)*Ae5),2)
    Aone = -8.32
    lamSMhe44 = lamHe44/tan(beta)
    lamSMhe55 = lamHe55/tan(beta)
    Atopsm = 1.84
    tausme4 = pow(125.0/(2.*me4),2)
    tausme5 = pow(125.0/(2.*me5),2)
    ffsmtaue4 = pow(asin(sqrt(tausme4)),2)
    ffsmtaue5 = pow(asin(sqrt(tausme5)),2)
    ASMe4 = 2.0*(tausme4 + (tausme4-1)*ffsmtaue4)/pow(tausme4,2)
    ASMe5 = 2.0*(tausme5 + (tausme5-1)*ffsmtaue5)/pow(tausme5,2)
    gamSMe4dp = (GF*pow(alphaEM,2)*pow(125.0,3)/(128.*sqrt(2)*pow(pi,3)))*pow(abs(Aone + Atopsm + (lamSMhe44*v/me4)*ASMe4  + (lamSMhe55*v/me5)*ASMe5),2)
    gamSModp = (GF*pow(alphaEM,2)*pow(125.0,3)/(128.*sqrt(2)*pow(pi,3)))*pow(abs(Aone + Atopsm),2)
    gamSMotot = 0.00407
    brsmdpnew = gamSMe4dp/(gamSMotot - gamSModp + gamSMe4dp)
    RgamgamSM = gamSMe4dp/gamSModp
    mmtau = 1.777
    if 1.0 - 4.0*pow(mmtau/mHH,2) < 0:
        gamHtotata = 0
    else:
        gamHtotata = (GF*mHH*(mmtau**2)*((tan(beta))**2)/(4.*sqrt(2)*pi))*pow(sqrt(1.0 - 4.0*pow(mmtau/mHH,2)),3)

    gamHtot = gamHtoe4 + gamHtogg + gamHtobb + gamHtohh + gamHtocc + gamHto2e4 + gamHdiph + gamHtotata
    brHtoe4 = gamHtoe4/gamHtot
    brHtobb = gamHtobb/gamHtot
    brHtohh = gamHtohh/gamHtot
    brHtogg = gamHtogg/gamHtot
    brHtocc = gamHtocc/gamHtot
    brHtodi = gamHdiph/gamHtot
    brHto2e4 = gamHto2e4/gamHtot
    
    # GeV^-2 = 3.89 * 10^-4 b = 3.89 * 10^8 pb
    brour = brHtoe4*bre4w
    Rnu4numu = pow(gLZnu,2)/pow(g/(2.*cos(thetaw)),2)
    approxtest = (lamE*v*cos(beta))/ME
    muonerr = 2*2*errMwLHC/MwLHC

    totalsigfidem = 1 # just to macth the previous scan : dummy parameter

"""

