#!/usr/bin/python


from __future__ import division
import os

exe = os.system

import numpy

from numpy import *
from math import *
import scipy as sp
from scipy import linalg
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import interpolate
from scipy.stats import gaussian_kde
from scipy.stats import kde

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab

#from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
#from mpl_toolkits.axes_grid1.inset_locator import mark_inset
#from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

from scipy.interpolate import griddata
from scipy.ndimage.filters import gaussian_filter
import scipy.ndimage
import scipy.interpolate


from scipy.special import spence
import cmath

#import scipy.interpolate


MW = 80.385
MZ = 91.1876
g = 0.652954
thetaw = acos(MW/MZ)
mmu = 0.1056584                  #physics muon mass
GF = (sqrt(2)/8.)*pow(g/MW,2)
v = (1/sqrt(sqrt(2.0)*GF))/sqrt(2.0)             #174 GeV

mb = 2.5 # MSbar running mass around mH = 500 GeV (roughly similar for mH = 155 - 800 GeV: 2.73368 - 2.43571)
alphas = 0.109683
alphaEM = 1/127.462
Nf = 5
mt = 173.21

mbpole = 4.18
mmtau = 1.777
mc = 0.6

alphast = 0.096



    
    
####### Cross section & BRs



#def sigmaHSM13TeV(mass):
    # plotdigitizer for m_H > 1 TeV
    #xx = numpy.array([130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350,360,370,380,390,400,420,440,450,460,480,500,520,540,550,560,580,600,620,640,650,660,680,700,720,740,750,760,780,800,820,840,850,860,880,900,920,940,950,960,980,1000,1007,1126,1259,1322,1398,1500,1596,1711,1784,1927,2052,2154,2231,2294,2409])
    #yy = numpy.array([41.3585, 36.3604, 32.5065, 29.1704, 25.2013, 22.6027, 20.0772, 18.354, 17.0885, 16.004, 15.0476, 14.1357, 13.2776, 12.5443, 11.916, 11.3918, 10.9646, 10.6239, 10.3848, 10.2403, 10.2335, 10.4095, 10.6587, 10.7305, 10.6824, 10.4749, 10.1257, 9.68019, 8.63685, 7.53588, 6.99899, 6.49012, 5.54757, 4.72401, 4.0173, 3.41724, 3.15417, 2.91057, 2.48435, 2.12756, 1.82636, 1.57277, 1.46171, 1.35926, 1.1782, 1.02523, 0.896173, 0.785216, 0.735862, 0.690181, 0.609041, 0.539285, 0.47917, 0.42687, 0.403525, 0.381833, 0.342328, 0.307655, 0.277642, 0.25146, 0.239094, 0.2276, 0.206846, 0.1888, 0.1190, 0.05937, 0.030498, 0.0203318, 0.0143626, 0.00903597, 0.0056848, 0.0035765, 0.002526479, 0.00137520, 0.0008906, 0.000629133, 0.0004579, 0.00037353, 0.00021544])
    # FeynHiggs-2.11.3
    #xx = numpy.array([130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350,360,370,380,390,400,420,440,450,460,480,500,520,540,550,560,580,600,620,640,650,660,680,700,720,740,750,760,780,800,820,840,850,860,880,900,920,940,950,960,980,1000,1050,1100,1150,1200,1250,1300,1350,1400,1450,1500,1550,1600,1650,1700,1750,1800,1850,1900,1950,2000,2050,2100,2150,2200,2250,2300,2350,2400,2450,2500,2550,2600,2650,2700,2750,2800,2850,2900,2950,3000])
    #yy = numpy.array([41.3585, 36.3604, 32.5065, 29.1704, 25.2013, 22.6027, 20.0772, 18.354, 17.0885, 16.004, 15.0476, 14.1357, 13.2776, 12.5443, 11.916, 11.3918, 10.9646, 10.6239, 10.3848, 10.2403, 10.2335, 10.4095, 10.6587, 10.7305, 10.6824, 10.4749, 10.1257, 9.68019, 8.63685, 7.53588, 6.99899, 6.49012, 5.54757, 4.72401, 4.0173, 3.41724, 3.15417, 2.91057, 2.48435, 2.12756, 1.82636, 1.57277, 1.46171, 1.35926, 1.1782, 1.02523, 0.896173, 0.785216, 0.735862, 0.690181, 0.609041, 0.539285, 0.47917, 0.42687, 0.403525, 0.381833, 0.342328, 0.307655, 0.277642, 0.25146, 0.239094, 0.2276, 0.206846, 0.1888, 0.110595, 0.0818091, 0.0607165, 0.0447824, 0.0320765, 0.0203666, 0.00188701, 0.0429729, 0.0216849, 0.0151792, 0.0113362, 0.00866861, 0.00670021, 0.00520521, 0.00405203, 0.00315461, 0.00245271, 0.00190233, 0.00147045, 0.00113179, 0.000866762, 0.000659975, 0.000499283, 0.000375029, 0.000279512, 0.000206576, 0.000151299, 0.000109751, 7.880301e-05, 5.59749e-05, 3.931119e-05, 2.728183e-05, 1.869961e-05, 1.265218e-05, 8.445944e-06, 5.559838e-06, 3.607381e-06, 2.30583e-06, 1.451312e-06, 8.990629e-07])
    # SusHi
    #xx = numpy.array([130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350,360,370,380,390,400,420,440,450,460,480,500,520,540,550,560,580,600,620,640,650,660,680,700,720,740,750,760,780,800,820,840,850,860,880,900,920,940,950,960,980,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400,2500,2600,2700,2800,2900,3000,3100,3200,3300,3400,3500,3600,3700,3800,3900,4000])
    #yy = numpy.array([41.3585, 36.3604, 32.5065, 29.1704, 25.2013, 22.6027, 20.0772, 18.354, 17.0885, 16.004, 15.0476, 14.1357, 13.2776, 12.5443, 11.916, 11.3918, 10.9646, 10.6239, 10.3848, 10.2403, 10.2335, 10.4095, 10.6587, 10.7305, 10.6824, 10.4749, 10.1257, 9.68019, 8.63685, 7.53588, 6.99899, 6.49012, 5.54757, 4.72401, 4.0173, 3.41724, 3.15417, 2.91057, 2.48435, 2.12756, 1.82636, 1.57277, 1.46171, 1.35926, 1.1782, 1.02523, 0.896173, 0.785216, 0.735862, 0.690181, 0.609041, 0.539285, 0.47917, 0.42687, 0.403525, 0.381833, 0.342328, 0.307655, 0.277642, 0.25146, 0.239094, 0.2276, 0.206846, 0.1888, 6.68414860E-02, 3.83259316E-02, 2.26255386E-02, 1.37070738E-02, 8.49122598E-03, 5.36538100E-03,  3.45073385E-03, 2.25435582E-03, 1.49395395E-03, 1.00274173E-03, 6.80747476E-04, 4.67071762E-04, 3.23563915E-04, 2.26115258E-04, 1.59293560E-04, 1.13011871E-04, 8.07263356E-05, 5.80161101E-05, 4.19288011E-05, 3.04587064E-05, 2.22334423E-05, 1.63043209E-05, 1.20020123E-05, 8.86915808E-06, 6.57563447E-06, 4.89124473E-06, 3.64882181E-06, 2.72956138E-06, 2.04655525E-06, 1.53773723E-06])  
    #order = 2
    #mysig13 = InterpolatedUnivariateSpline(xx, yy, k=order)
    #return mysig13(mass)


#def sigmaHSM27TeV(mass):
    # SusHi
    #xxhe = numpy.array([130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350,360,370,380,390,400,420,440,450,460,480,500,520,540,550,560,580,600,620,640,650,660,680,700,720,740,750,760,780,800,820,840,850,860,880,900,920,940,950,960,980,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400,2500,2600,2700,2800,2900,3000,3100,3200,3300,3400,3500,3600,3700,3800,3900,4000])
    #yyhe = numpy.array([1.36164726*100, 1.20497527*100, 1.07535364*100, 9.67058931*10, 8.75707126*10, 7.98106715*10, 7.31730330*10, 6.74701341*10, 6.25524346*10, 5.82907844*10, 5.45898292*10, 5.13786527*10, 4.85965232*10, 4.61931031*10, 4.41399678*10, 4.24103026*10, 4.09894083*10, 3.98750939*10, 3.90814604*10, 3.86663020*10, 3.87350380*10, 3.95863814*10, 4.28283728*10, 4.48427311*10, 4.52596085*10, 4.46279804*10, 4.33073362*10, 4.15507769*10, 3.73885955*10, 3.30101333*10, 3.08786526*10, 2.88251560*10, 2.50130230*10, 2.16313126*10, 1.86777095*10, 1.61225858*10, 1.49813442*10, 1.39239639*10, 1.20383439*10, 1.04238586*10, 9.04159176, 7.85775897, 7.33092685, 6.84289673, 5.97139765, 5.22192380, 4.57615634, 4.01864597, 3.76888137, 3.53648480, 3.11841240, 2.75543857, 2.43959790, 2.16402562, 2.03955244, 1.92313832, 1.71223313, 1.52714401, 1.36439795, 1.22105346, 1.15584511, 1.09456656, 9.82760355E-01, 8.83724959E-01, 5.30850588E-01, 3.29359459E-01, 2.10208545E-01, 1.37531372E-01, 9.19613954E-02, 6.26936823E-02, 4.34855850E-02, 3.06370457E-02, 2.18892808E-02, 1.58394283E-02, 1.15956793E-02, 8.57969772E-03, 6.40992681E-03, 4.83238146E-03, 3.67363283E-03, 2.81439625E-03, 2.17153475E-03, 1.68672685E-03, 1.31839618E-03, 1.03652270E-03, 8.19424519E-04, 6.51143559E-04, 5.19945791E-04, 4.17075729E-04, 3.36006375E-04, 2.71807370E-04, 2.20722642E-04, 1.79899214E-04, 1.47141882E-04, 1.20750146E-04])  
    #order = 2
    #mysig27 = InterpolatedUnivariateSpline(xxhe, yyhe, k=order)
    #return mysig27(mass)



#27TeV  SusHi
xs27 = loadtxt('xsec_27TeV.txt')
hemass = numpy.array(xs27[:,0])
sigggF27 = numpy.array(xs27[:,1])
sigbbF27 = numpy.array(xs27[:,3])


#13TeV SusHi
xs13 = loadtxt('xsec_13TeV.txt')
h13mass = numpy.array(xs13[:,0])
sigggF13 = numpy.array(xs13[:,1])
sigbbF13 = numpy.array(xs13[:,3])

#14TeV SusHi
xs14 = loadtxt('xsec_14TeV.txt')
h14mass = numpy.array(xs14[:,0])
sigggF14 = numpy.array(xs14[:,1])
sigbbF14 = numpy.array(xs14[:,2])

    

def sigmaHSM13TeV(mass):
    order = 1
    mysigggF13 = InterpolatedUnivariateSpline(h13mass, sigggF13, k=order)
    return mysigggF13(mass)

def sigmabbF13(mass):
    order = 1
    mysigbbF13 = InterpolatedUnivariateSpline(h13mass, sigbbF13, k=order)
    return mysigbbF13(mass)

def sigmaHSM14TeV(mass):
    order = 1
    mysigggF14 = InterpolatedUnivariateSpline(h14mass, sigggF14, k=order)
    return mysigggF14(mass)

def sigmabbF14(mass):
    order = 1
    mysigbbF14 = InterpolatedUnivariateSpline(h14mass, sigbbF14, k=order)
    return mysigbbF14(mass)



def sigmaHSM27TeV(mass):
    order = 1
    mysigggF27 = InterpolatedUnivariateSpline(hemass, sigggF27, k=order)
    return mysigggF27(mass)
    

def sigmabbF27(mass):
    order = 1
    mysigbbF27 = InterpolatedUnivariateSpline(hemass, sigbbF27, k=order)
    return mysigbbF27(mass)
    
    


xs1 = loadtxt('singlevlqxs.dat')
#xs1 = loadtxt('singlevlqxs-b4bj.dat')
t4mass = numpy.array(xs1[:,0])
#b4mass = numpy.array(xs1[:,0])
#bjwl = numpy.array(xs1[:,1])
#bjwr = numpy.array(xs1[:,2])
xsbjsingle = numpy.array(xs1[:,-1])

xs2 = loadtxt('singlevlqxs-tj.dat')
#xs2 = loadtxt('singlevlqxs-all.dat')
#t4mass = numpy.array(xs2[:,0])
#tjwl = numpy.array(xs2[:,1])
#tjwr = numpy.array(xs2[:,2])
xstjsingle = numpy.array(xs2[:,-1])


xs3 = loadtxt('singlevlqxs-all.dat')
tryt4mass = numpy.array(xs3[:,0])
tjwl = numpy.array(xs3[:,1])
tjwr = numpy.array(xs3[:,2])
tryxstjsingle = numpy.array(xs3[:,-1])





def myt4bj(mass):
    order = 2
    mysigbj = InterpolatedUnivariateSpline(t4mass, xsbjsingle, k=order)
    return mysigbj(mass)

#def myt4tj(mass):
    #order = 2
    #mysigtj = InterpolatedUnivariateSpline(t4mass, xstjsingle, k=order)
    #return mysigtj(mass)

#def grid4d(x, y, z, w, resX=16, resY=16, resZ=16):#resX=9, resY=9):
    #"Convert 4 column data to matplotlib grid"
    #xi = numpy.linspace(min(x), max(x), resX)
    #yi = numpy.linspace(min(y), max(y), resY)
    #zi = numpy.linspace(min(z), max(z), resZ)
    #W = griddata((x, y, z), w, (xi[None,:], yi[:,None], zi[:,None]), method='linear')
    #return xi, yi, zi, W


xin = numpy.array(tryt4mass)
yin = numpy.array(tjwl)
zin = numpy.array(tjwr)
win = numpy.array(tryxstjsingle)

cartcoord = zip(xin, yin, zin)

myinterp = scipy.interpolate.LinearNDInterpolator(cartcoord, win, fill_value=0)


#print 'try =', myinterp(1200.,0.001,0.002)*1



#tryf1, tryf2, tryf3, tryf4 = grid4d(tryt4mass, tjwl, tjwr, tryxstjsingle)
    
#def tryt4tj(mass, gwl, gwr):
    #tryresult = scipy.interpolate.Rbf(tryf1, tryf2, tryf3, tryf4, function='thin_plate')
    #tryresult = scipy.interpolate.Rbf(tryt4mass, tjwl, tjwr, tryxstjsingle, function='linear')
    #return tryresult(mass, gwl, gwr)


#def testfun(x, y, z):
    #return grid4d(tryt4mass, tjwl, tjwr, tryxstjsingle)

#print testfun(1000.,0.01,0.1)
    
#tryt4tj(1000.,0.01,0.1)


######


listlep = ['em','mm','ee']



nlistmt4 = []
nlistmHH = []
nlisttanbe = []
nlisttotalxs = []


tlistmt4 = []
tlisttanbe = []
tlistlamHt34 = []
tlistmHH = []
tlistlamHrt34 = []
tlistlamHt44 = []
tlistlamHrt44 = []
tlistgLW43 = []
tlistgLW33 = []
tlistgLZ44 = []
tlistgLZ43 = []
tlistgLZ33 = []
tlistgRZ44 = []
tlistgRZ43 = []
tlistgRZ33 = []
tlistbrHtobb = []
tlistbrHtohh = []
tlistbrHtot4 = []
tlistbrHtogg = []
tlistbrHtocc = []
tlistbrHtodi = []
tlistbrHtott = []
tlistbrHtotata = []
tlistgamHtot = []
tlistgamHtotata = []
tlistgamHtobb = []
tlistgamHtot4 = []
tlistgamHtogg = []
tlistgamHtohh = []
tlistgamHtocc = []
tlistgamHtodi = []
tlistgamHtott = []
tlistbrt4w = []
tlistbrt4z = []
tlistbrt4h = []
tlistbrour = []
tlistkaT = []
tlistggHratio = []
tlisttotalratio = []
tlistprodratiobr = []
tlistlamSMt44 = []
tlistggSMratio = []
tlistHWratio = []
tlistbbFratio = []
tlistdoubfr = []
tlistsingfr = []
tlistkaQ = []
tlisttotalxs = []
tlisttotalt4t = []
tlisttotalb4b = []

tlisttryt4tj = []

tlistmyt4bj = []
tlistmyt4tj = []
tlistgRW43 = []
tlistmb4 = []
tlistbrb4w = []
tlistbrb4z = []
tlistbrb4h = []
tlistbrHtob4 = []
tlistMTT = []
tlistbrb4t4w = []
tlistbrt4b4w = []

tlistgLW44t = []
tlistgRW44t = []

tlistbrt4HH = []
tlistbrb4HH = []
tlistgamHtob4 = []

tlistlamHpmb43 = []
tlistlamHpmbr43 = []
tlistlamHb34 = []
tlistlamHbr34 = []

tlistxs2HDM1 = []
tlistxs2HDM7 = []
tlistxs2HDM50 = []


talistxs2HDM1 = []
tclistxs2HDM7 = []
tglistxs2HDM50 = []

tlistlamHb44 = []

talistmt4 = []
talisttanbe = []
talistlamHt34 = []
talistmHH = []
talistlamHrt34 = []
talistlamHt44 = []
talistlamHrt44 = []
talistgLW43 = []
talistgLW33 = []
talistgLZ44 = []
talistgLZ43 = []
talistgLZ33 = []
talistgRZ44 = []
talistgRZ43 = []
talistgRZ33 = []
talistbrHtobb = []
talistbrHtohh = []
talistbrHtot4 = []
talistbrHtogg = []
talistbrHtocc = []
talistbrHtodi = []
talistbrHtott = []
talistbrHtotata = []
talistgamHtot = []
talistgamHtotata = []
talistgamHtobb = []
talistgamHtot4 = []
talistgamHtogg = []
talistgamHtohh = []
talistgamHtocc = []
talistgamHtodi = []
talistgamHtott = []
talistbrt4w = []
talistbrt4z = []
talistbrt4h = []
talistbrour = []
talistkaT = []
talistggHratio = []
talisttotalratio = []
talistprodratiobr = []
talistlamSMt44 = []
talistggSMratio = []
talistbbFratio = []

talisttotalt4t = []
talistmyt4bj = []
talistmyt4tj = []
talistgRW43 = []

talistkaQ = []
talistka = []
talistkab = []

talistbrt4HH = []
talistbrb4HH = []



tblistmt4 = []
tblisttanbe = []
tblistlamHt34 = []
tblistmHH = []
tblistlamHrt34 = []
tblistlamHt44 = []
tblistlamHrt44 = []
tblistgLW43 = []
tblistgLW33 = []
tblistgLZ44 = []
tblistgLZ43 = []
tblistgLZ33 = []
tblistgRZ44 = []
tblistgRZ43 = []
tblistgRZ33 = []
tblistbrHtobb = []
tblistbrHtohh = []
tblistbrHtot4 = []
tblistbrHtogg = []
tblistbrHtocc = []
tblistbrHtodi = []
tblistbrHtott = []
tblistgamHtot = []
tblistgamHtotata = []
tblistgamHtobb = []
tblistgamHtot4 = []
tblistgamHtogg = []
tblistgamHtohh = []
tblistgamHtocc = []
tblistgamHtodi = []
tblistgamHtott = []
tblistbrHtotata = []
tblistbrt4w = []
tblistbrt4z = []
tblistbrt4h = []
tblistbrour = []
tblistkaT = []
tblistggHratio = []
tblisttotalratio = []
tblistprodratiobr = []
tblistlamSMt44 = []
tblistggSMratio = []
tblistbbFratio = []
tblistbrourb4 = []
tblistsigb4BRw = []
tblistsigb4BRz = []

tblistbrb4w = []
tblistbrb4z = []
tblistbrb4h = []


tclistmt4 = []
tclisttanbe = []
tclistlamHt34 = []
tclistmHH = []
tclistlamHrt34 = []
tclistlamHt44 = []
tclistlamHrt44 = []
tclistgLW43 = []
tclistgLW33 = []
tclistgLZ44 = []
tclistgLZ43 = []
tclistgLZ33 = []
tclistgRZ44 = []
tclistgRZ43 = []
tclistgRZ33 = []
tclistbrHtobb = []
tclistbrHtohh = []
tclistbrHtot4 = []
tclistbrHtogg = []
tclistbrHtocc = []
tclistbrHtodi = []
tclistbrHtott = []
tclistbrHtotata = []
tclistgamHtot = []
tclistgamHtotata = []
tclistgamHtobb = []
tclistgamHtot4 = []
tclistgamHtogg = []
tclistgamHtohh = []
tclistgamHtocc = []
tclistgamHtodi = []
tclistgamHtott = []
tclistbrt4w = []
tclistbrt4z = []
tclistbrt4h = []
tclistbrour = []
tclistkaT = []
tclistggHratio = []
tclisttotalratio = []
tclistprodratiobr = []
tclistlamSMt44 = []
tclistggSMratio = []
tclistbbFratio = []
tclistbrourb4 = []
tclistsigb4BRw = []
tclistsigb4BRz = []

tclistbrb4w = []
tclistbrb4z = []
tclistbrb4h = []

telistmt4 = []
telisttanbe = []
telistlamHt34 = []
telistmHH = []
telistlamHrt34 = []
telistlamHt44 = []
telistlamHrt44 = []
telistgLW43 = []
telistgLW33 = []
telistgLZ44 = []
telistgLZ43 = []
telistgLZ33 = []
telistgRZ44 = []
telistgRZ43 = []
telistgRZ33 = []
telistbrHtobb = []
telistbrHtohh = []
telistbrHtot4 = []
telistbrHtogg = []
telistbrHtocc = []
telistbrHtodi = []
telistbrHtott = []
telistbrHtotata = []
telistgamHtot = []
telistgamHtotata = []
telistgamHtobb = []
telistgamHtot4 = []
telistgamHtogg = []
telistgamHtohh = []
telistgamHtocc = []
telistgamHtodi = []
telistgamHtott = []
telistbrt4w = []
telistbrt4z = []
telistbrt4h = []
telistbrour = []
telistkaT = []
telistggHratio = []
telisttotalratio = []
telistprodratiobr = []
telistlamSMt44 = []
telistggSMratio = []
telistbbFratio = []
telistbrourb4 = []
telistsigb4BRw = []
telistsigb4BRz = []

telistbrb4w = []
telistbrb4z = []
telistbrb4h = []

tflistmt4 = []
tflisttanbe = []
tflistlamHt34 = []
tflistmHH = []
tflistlamHrt34 = []
tflistlamHt44 = []
tflistlamHrt44 = []
tflistgLW43 = []
tflistgLW33 = []
tflistgLZ44 = []
tflistgLZ43 = []
tflistgLZ33 = []
tflistgRZ44 = []
tflistgRZ43 = []
tflistgRZ33 = []
tflistbrHtobb = []
tflistbrHtohh = []
tflistbrHtot4 = []
tflistbrHtogg = []
tflistbrHtocc = []
tflistbrHtodi = []
tflistbrHtott = []
tflistbrHtotata = []
tflistgamHtot = []
tflistgamHtotata = []
tflistgamHtobb = []
tflistgamHtot4 = []
tflistgamHtogg = []
tflistgamHtohh = []
tflistgamHtocc = []
tflistgamHtodi = []
tflistgamHtott = []
tflistbrt4w = []
tflistbrt4z = []
tflistbrt4h = []
tflistbrour = []
tflistkaT = []
tflistggHratio = []
tflisttotalratio = []
tflistprodratiobr = []
tflistlamSMt44 = []
tflistggSMratio = []
tflistbbFratio = []
tflistbrourb4 = []
tflistsigb4BRw = []
tflistsigb4BRz = []

tflistbrb4w = []
tflistbrb4z = []
tflistbrb4h = []

tglistmt4 = []
tglisttanbe = []
tglistlamHt34 = []
tglistmHH = []
tglistlamHrt34 = []
tglistlamHt44 = []
tglistlamHrt44 = []
tglistgLW43 = []
tglistgLW33 = []
tglistgLZ44 = []
tglistgLZ43 = []
tglistgLZ33 = []
tglistgRZ44 = []
tglistgRZ43 = []
tglistgRZ33 = []
tglistbrHtobb = []
tglistbrHtohh = []
tglistbrHtot4 = []
tglistbrHtogg = []
tglistbrHtocc = []
tglistbrHtodi = []
tglistbrHtott = []
tglistbrHtotata = []
tglistgamHtot = []
tglistgamHtotata = []
tglistgamHtobb = []
tglistgamHtot4 = []
tglistgamHtogg = []
tglistgamHtohh = []
tglistgamHtocc = []
tglistgamHtodi = []
tglistgamHtott = []
tglistbrt4w = []
tglistbrt4z = []
tglistbrt4h = []
tglistbrour = []
tglistkaT = []
tglistggHratio = []
tglisttotalratio = []
tglistprodratiobr = []
tglistlamSMt44 = []
tglistggSMratio = []
tglistbbFratio = []
tglistbrourb4 = []
tglistsigb4BRw = []
tglistsigb4BRz = []

tglistbrb4w = []
tglistbrb4z = []
tglistbrb4h = []



tdlistmt4 = []
tdlisttanbe = []
tdlistlamHt34 = []
tdlistmHH = []
tdlistlamHrt34 = []
tdlistlamHt44 = []
tdlistlamHrt44 = []
tdlistgLW43 = []
tdlistgLW33 = []
tdlistgLZ44 = []
tdlistgLZ43 = []
tdlistgLZ33 = []
tdlistgRZ44 = []
tdlistgRZ43 = []
tdlistgRZ33 = []
tdlistbrHtobb = []
tdlistbrHtohh = []
tdlistbrHtot4 = []
tdlistbrHtogg = []
tdlistbrHtocc = []
tdlistbrHtodi = []
tdlistbrHtott = []
tdlistbrHtotata = []
tdlistgamHtot = []
tdlistgamHtotata = []
tdlistgamHtobb = []
tdlistgamHtot4 = []
tdlistgamHtogg = []
tdlistgamHtohh = []
tdlistgamHtocc = []
tdlistgamHtodi = []
tdlistgamHtott = []
tdlistbrt4w = []
tdlistbrt4z = []
tdlistbrt4h = []
tdlistbrour = []
tdlistkaT = []
tdlistkaQ = []
tdlistka = []
tdlistkaba = []
tdlistggHratio = []
tdlisttotalratio = []
tdlistprodratiobr = []
tdlistlamSMt44 = []
tdlistggSMratio = []
tdlistbbFratio = []

tdlistsingfr = []
tdlistlamh43 = []
tdlistlamh34 = []



tdalistmt4 = []
tdalisttanbe = []
tdalistlamHt34 = []
tdalistmHH = []
tdalistlamHrt34 = []
tdalistlamHt44 = []
tdalistlamHrt44 = []
tdalistgLW43 = []
tdalistgLW33 = []
tdalistgLZ44 = []
tdalistgLZ43 = []
tdalistgLZ33 = []
tdalistgRZ44 = []
tdalistgRZ43 = []
tdalistgRZ33 = []
tdalistbrHtobb = []
tdalistbrHtohh = []
tdalistbrHtot4 = []
tdalistbrHtogg = []
tdalistbrHtocc = []
tdalistbrHtodi = []
tdalistbrHtott = []
tdalistbrHtotata = []
tdalistgamHtot = []
tdalistgamHtotata = []
tdalistgamHtobb = []
tdalistgamHtot4 = []
tdalistgamHtogg = []
tdalistgamHtohh = []
tdalistgamHtocc = []
tdalistgamHtodi = []
tdalistgamHtott = []
tdalistbrt4w = []
tdalistbrt4z = []
tdalistbrt4h = []
tdalistbrour = []
tdalistkaT = []
tdalistkaQ = []
tdalistka = []
tdalistkaba = []
tdalistggHratio = []
tdalisttotalratio = []
tdalistprodratiobr = []
tdalistlamSMt44 = []
tdalistggSMratio = []
tdalistbbFratio = []

tdalistsingfr = []
tdalistmb4 = []
tdalistgamt4toh = []
tdalistgamt4tow = []
tdalistcheck = []
tdalistMTT = []
tdalistMQQ = []
tdalistlamh43 = []
tdalistlamh34 = []


tdalisttest1 = []
tdalisttest2 = []
tdalisttest3 = []
tdalisttest4 = []


tdblistmt4 = []
tdblisttanbe = []
tdblistlamHt34 = []
tdblistmHH = []
tdblistlamHrt34 = []
tdblistlamHt44 = []
tdblistlamHrt44 = []
tdblistgLW43 = []
tdblistgLW33 = []
tdblistgLZ44 = []
tdblistgLZ43 = []
tdblistgLZ33 = []
tdblistgRZ44 = []
tdblistgRZ43 = []
tdblistgRZ33 = []
tdblistbrHtobb = []
tdblistbrHtohh = []
tdblistbrHtot4 = []
tdblistbrHtogg = []
tdblistbrHtocc = []
tdblistbrHtodi = []
tdblistbrHtott = []
tdblistbrHtotata = []
tdblistgamHtot = []
tdblistgamHtotata = []
tdblistgamHtobb = []
tdblistgamHtot4 = []
tdblistgamHtogg = []
tdblistgamHtohh = []
tdblistgamHtocc = []
tdblistgamHtodi = []
tdblistgamHtott = []
tdblistbrt4w = []
tdblistbrt4z = []
tdblistbrt4h = []
tdblistbrour = []
tdblistkaT = []
tdblistkaQ = []
tdblistka = []
tdblistkaba = []
tdblistggHratio = []
tdblisttotalratio = []
tdblistprodratiobr = []
tdblistlamSMt44 = []
tdblistggSMratio = []
tdblistbbFratio = []
tdblistlamh43 = []
tdblistlamh34 = []



tdclistmt4 = []
tdclisttanbe = []
tdclistlamHt34 = []
tdclistmHH = []
tdclistlamHrt34 = []
tdclistlamHt44 = []
tdclistlamHrt44 = []
tdclistgLW43 = []
tdclistgLW33 = []
tdclistgLZ44 = []
tdclistgLZ43 = []
tdclistgLZ33 = []
tdclistgRZ44 = []
tdclistgRZ43 = []
tdclistgRZ33 = []
tdclistbrHtobb = []
tdclistbrHtohh = []
tdclistbrHtot4 = []
tdclistbrHtogg = []
tdclistbrHtocc = []
tdclistbrHtodi = []
tdclistbrHtott = []
tdclistbrHtotata = []
tdclistgamHtot = []
tdclistgamHtotata = []
tdclistgamHtobb = []
tdclistgamHtot4 = []
tdclistgamHtogg = []
tdclistgamHtohh = []
tdclistgamHtocc = []
tdclistgamHtodi = []
tdclistgamHtott = []
tdclistbrt4w = []
tdclistbrt4z = []
tdclistbrt4h = []
tdclistbrour = []
tdclistkaT = []
tdclistggHratio = []
tdclisttotalratio = []
tdclistprodratiobr = []
tdclistlamSMt44 = []
tdclistggSMratio = []
tdclistbbFratio = []
tdclistlamh43 = []
tdclistlamh34 = []




tddlistmt4 = []
tddlisttanbe = []
tddlistlamHt34 = []
tddlistmHH = []
tddlistlamHrt34 = []
tddlistlamHt44 = []
tddlistlamHrt44 = []
tddlistgLW43 = []
tddlistgLW33 = []
tddlistgLZ44 = []
tddlistgLZ43 = []
tddlistgLZ33 = []
tddlistgRZ44 = []
tddlistgRZ43 = []
tddlistgRZ33 = []
tddlistbrHtobb = []
tddlistbrHtohh = []
tddlistbrHtot4 = []
tddlistbrHtogg = []
tddlistbrHtocc = []
tddlistbrHtodi = []
tddlistbrHtott = []
tddlistbrHtotata = []
tddlistgamHtot = []
tddlistgamHtotata = []
tddlistgamHtobb = []
tddlistgamHtot4 = []
tddlistgamHtogg = []
tddlistgamHtohh = []
tddlistgamHtocc = []
tddlistgamHtodi = []
tddlistgamHtott = []
tddlistbrt4w = []
tddlistbrt4z = []
tddlistbrt4h = []
tddlistbrour = []
tddlistkaT = []
tddlistggHratio = []
tddlisttotalratio = []
tddlistprodratiobr = []
tddlistlamSMt44 = []
tddlistggSMratio = []
tddlistbbFratio = []






alistmt4 = []
alisttanbe = []
alistlamHt34 = []
alistmHH = []
alistlamHrt34 = []
alistlamHt44 = []
alistlamHrt44 = []
alistgLW43 = []
alistgLW33 = []
alistgLZ44 = []
alistgLZ43 = []
alistgLZ33 = []
alistgRZ44 = []
alistgRZ43 = []
alistgRZ33 = []
alistbrHtobb = []
alistbrHtohh = []
alistbrHtot4 = []
alistbrHtogg = []
alistbrHtocc = []
alistbrHtodi = []
alistbrHtott = []
alistgamHtot = []
alistgamHtotata = []
alistgamHtobb = []
alistgamHtot4 = []
alistgamHtogg = []
alistgamHtohh = []
alistgamHtocc = []
alistgamHtodi = []
alistbrHtotata = []
alistgamHtott = []
alistbrt4w = []
alistbrt4z = []
alistbrt4h = []
alistbrour = []
alistkaT = []
alistggHratio = []
alisttotalratio = []
alistprodratiobr = []
alistlamSMt44 = []
alistggSMratio = []
alistmb4 = []
alistbrb4w = []
alistbrb4z = []
alistbrb4h = [] 
alistbrHtob4 = []

alistMTT = []

alistbrb4t4w = []
alistbrt4b4w = []

alistbrt5z = []
alistbrt5w = []
alistbrt5h = []

alistbrt4HH = []
alistbrb4HH = []


blistmt4 = []
blisttanbe = []
blistlamHt34 = []
blistmHH = []
blistlamHrt34 = []
blistlamHt44 = []
blistlamHrt44 = []
blistgLW43 = []
blistgLW33 = []
blistgLZ44 = []
blistgLZ43 = []
blistgLZ33 = []
blistgRZ44 = []
blistgRZ43 = []
blistgRZ33 = []
blistbrHtobb = []
blistbrHtohh = []
blistbrHtot4 = []
blistbrHtogg = []
blistbrHtocc = []
blistbrHtodi = []
blistbrHtott = []
blistbrHtotata = []
blistgamHtot = []
blistgamHtotata = []
blistgamHtobb = []
blistgamHtot4 = []
blistgamHtogg = []
blistgamHtohh = []
blistgamHtocc = []
blistgamHtodi = []
blistgamHtott = []
blistbrt4w = []
blistbrt4z = []
blistbrt4h = []
blistbrour = []
blistkaT = []
blistggHratio = []
blisttotalratio = []
blistprodratiobr = []
blistlamSMt44 = []
blistggSMratio = []
blistmb4 = []
blistbrb4w = []
blistbrb4z = []
blistbrb4h = [] 
blistbrHtob4 = []

blistbrb4t4w = []
blistbrt4b4w = []

blistbrt5z = []
blistbrt5w = []
blistbrt5h = []

blistbrt4HH = []
blistbrb4HH = []

clistmt4 = []
clisttanbe = []
clistlamHt34 = []
clistmHH = []
clistlamHrt34 = []
clistlamHt44 = []
clistlamHrt44 = []
clistgLW43 = []
clistgLW33 = []
clistgLZ44 = []
clistgLZ43 = []
clistgLZ33 = []
clistgRZ44 = []
clistgRZ43 = []
clistgRZ33 = []
clistbrHtobb = []
clistbrHtohh = []
clistbrHtot4 = []
clistbrHtogg = []
clistbrHtocc = []
clistbrHtodi = []
clistbrHtott = []
clistgamHtot = []
clistgamHtotata = []
clistgamHtobb = []
clistgamHtot4 = []
clistgamHtogg = []
clistgamHtohh = []
clistgamHtocc = []
clistgamHtodi = []
clistgamHtott = []
clistbrHtotata = []
clistbrt4w = []
clistbrt4z = []
clistbrt4h = []
clistbrour = []
clistkaT = []
clistggHratio = []
clisttotalratio = []
clistprodratiobr = []
clistlamSMt44 = []
clistggSMratio = []
clistmb4 = []
clistbrb4w = []
clistbrb4z = []
clistbrb4h = [] 
clistbrHtob4 = []

clistbrb4t4w = []
clistbrt4b4w = []

clistbrt5z = []
clistbrt5w = []
clistbrt5h = []

clistbrt4HH = []
clistbrb4HH = []

dlistmt4 = []
dlisttanbe = []
dlistlamHt34 = []
dlistmHH = []
dlistlamHrt34 = []
dlistlamHt44 = []
dlistlamHrt44 = []
dlistgLW43 = []
dlistgLW33 = []
dlistgLZ44 = []
dlistgLZ43 = []
dlistgLZ33 = []
dlistgRZ44 = []
dlistgRZ43 = []
dlistgRZ33 = []
dlistbrHtobb = []
dlistbrHtohh = []
dlistbrHtot4 = []
dlistbrHtogg = []
dlistbrHtocc = []
dlistbrHtodi = []
dlistbrHtott = []
dlistgamHtot = []
dlistgamHtotata = []
dlistgamHtobb = []
dlistgamHtot4 = []
dlistgamHtogg = []
dlistgamHtohh = []
dlistgamHtocc = []
dlistgamHtodi = []
dlistgamHtott = []
dlistbrHtotata = []
dlistbrt4w = []
dlistbrt4z = []
dlistbrt4h = []
dlistbrour = []
dlistkaT = []
dlistggHratio = []
dlisttotalratio = []
dlistprodratiobr = []
dlistlamSMt44 = []
dlistggSMratio = []
dlistmb4 = []
dlistbrb4w = []
dlistbrb4z = []
dlistbrb4h = [] 
dlistbrHtob4 = []

dlistbrb4t4w = []
dlistbrt4b4w = []

dlistbrt5z = []
dlistbrt5w = []
dlistbrt5h = []

dlistbrt4HH = []
dlistbrb4HH = []

elisttanbe = []
elistmHH = []

alistbrb4H0 = []
blistbrb4H0 = []
clistbrb4H0 = []
dlistbrb4H0 = []


tablistmt4 = []
tablisttanbe = []
tablistlamHt34 = []
tablistmHH = []
tablistlamHrt34 = []
tablistlamHt44 = []
tablistlamHrt44 = []
tablistgLW43 = []
tablistgLW33 = []
tablistgLZ44 = []
tablistgLZ43 = []
tablistgLZ33 = []
tablistgRZ44 = []
tablistgRZ43 = []
tablistgRZ33 = []
tablistbrHtobb = []
tablistbrHtohh = []
tablistbrHtot4 = []
tablistbrHtogg = []
tablistbrHtocc = []
tablistbrHtodi = []
tablistbrHtott = []
tablistbrHtotata = []
tablistgamHtot = []
tablistgamHtotata = []
tablistgamHtobb = []
tablistgamHtot4 = []
tablistgamHtogg = []
tablistgamHtohh = []
tablistgamHtocc = []
tablistgamHtodi = []
tablistgamHtott = []
tablistbrt4w = []
tablistbrt4z = []
tablistbrt4h = []
tablistbrour = []
tablistkaT = []
tablistggHratio = []
tablisttotalratio = []
tablistprodratiobr = []
tablistlamSMt44 = []
tablistggSMratio = []

tablisttotalxs = []
tablisttotalt4t = []
tablistmyt4bj = []
tablistmyt4tj = []




taclistmt4 = []
taclisttanbe = []
taclistlamHt34 = []
taclistmHH = []
taclistlamHrt34 = []
taclistlamHt44 = []
taclistlamHrt44 = []
taclistgLW43 = []
taclistgLW33 = []
taclistgLZ44 = []
taclistgLZ43 = []
taclistgLZ33 = []
taclistgRZ44 = []
taclistgRZ43 = []
taclistgRZ33 = []
taclistbrHtobb = []
taclistbrHtohh = []
taclistbrHtot4 = []
taclistbrHtogg = []
taclistbrHtocc = []
taclistbrHtodi = []
taclistbrHtott = []
taclistgamHtot = []
taclistgamHtotata = []
taclistgamHtobb = []
taclistgamHtot4 = []
taclistgamHtogg = []
taclistgamHtohh = []
taclistgamHtocc = []
taclistgamHtodi = []
taclistgamHtott = []
taclistbrHtotata = []
taclistbrt4w = []
taclistbrt4z = []
taclistbrt4h = []
taclistbrour = []
taclistkaT = []
taclistggHratio = []
taclisttotalratio = []
taclistprodratiobr = []
taclistlamSMt44 = []
taclistggSMratio = []

taclisttotalxs = []
taclisttotalt4t = []
taclistmyt4bj = []
taclistmyt4tj = []



tadlistmt4 = []
tadlisttanbe = []
tadlistlamHt34 = []
tadlistmHH = []
tadlistlamHrt34 = []
tadlistlamHt44 = []
tadlistlamHrt44 = []
tadlistgLW43 = []
tadlistgLW33 = []
tadlistgLZ44 = []
tadlistgLZ43 = []
tadlistgLZ33 = []
tadlistgRZ44 = []
tadlistgRZ43 = []
tadlistgRZ33 = []
tadlistbrHtobb = []
tadlistbrHtohh = []
tadlistbrHtot4 = []
tadlistbrHtogg = []
tadlistbrHtocc = []
tadlistbrHtodi = []
tadlistbrHtott = []
tadlistgamHtot = []
tadlistgamHtotata = []
tadlistgamHtobb = []
tadlistgamHtot4 = []
tadlistgamHtogg = []
tadlistgamHtohh = []
tadlistgamHtocc = []
tadlistgamHtodi = []
tadlistgamHtott = []
tadlistbrHtotata = []
tadlistbrt4w = []
tadlistbrt4z = []
tadlistbrt4h = []
tadlistbrour = []
tadlistkaT = []
tadlistggHratio = []
tadlisttotalratio = []
tadlistprodratiobr = []
tadlistlamSMt44 = []
tadlistggSMratio = []

tadlisttotalxs = []
tadlisttotalt4t = []
tadlistmyt4bj = []
tadlistmyt4tj = []



tdalistgRW43t = []
tdalistgammat4toh = []
tdalistgammat4tow = []


tblistsigBRw = []
tblistsigBRz = []
tclistsigBRw = []
tclistsigBRz = []
telistsigBRw = []
telistsigBRz = []
tflistsigBRw = []
tflistsigBRz = []
tglistsigBRw = []
tglistsigBRz = []

tblistlamHpmbr43 = []
tblistlamHbr34 = []
tblistbrb4HH = []


talisttotalxs = []
tblisttotalxs = []
tclisttotalxs = []
tdlisttotalxs = []
telisttotalxs = []
tflisttotalxs = []
tglisttotalxs = []


#Different from the input in t1ewptall-HWWadd.py from
#totalratio = ldata[79]
#brHtotata = ldata[80]
#do +1 in the ldata[i] from brHtotata


nlistbrt4HH = []
nlistbrb4HH = []

mlisttanbe = []
mlistbrt4HH = []
mlistbrb4HH = []

for kk in range(0,1):
    #fnin = open("t1ewptall-em-noconstraint.dat")
    fnin = open("t1ewptall-em-s8-Hpm.dat")
    #fnin = open("t1ewptall-em-test-s8-Hpm.dat")
    nlistscan = map(str,fnin.readlines())
    fnin.close()
    for ii in range(0,len(nlistscan)-1):
        nres = nlistscan[ii]
        nldata = map(str,nres.split())
        nlmt4 = [float(nldata[0])]
        nltanbe = [float(nldata[1])]
        nlmHH = [float(nldata[3])]
        nltotalxs = [float(nldata[79])*sigmaHSM13TeV(float(nldata[3]))]
        nlbrt4HH = [float(nldata[-4])]
        nlbrb4HH = [float(nldata[-8])] # Only for scenario 3
        nlistmt4.extend(nlmt4)
        nlisttanbe.extend(nltanbe)
        nlistmHH.extend(nlmHH)
        nlisttotalxs.extend(nltotalxs)
        nlistbrt4HH.extend(nlbrt4HH)
        nlistbrb4HH.extend(nlbrb4HH)
        


fmin = open("t1ewptall-em-s9-Hpm.dat")
#fmin = open("t1ewptall-em-test-s9-Hpm.dat")
mlistscan = map(str,fmin.readlines())
fmin.close()
for k in range(0,len(mlistscan)-1):
    mres = mlistscan[k]
    mldata = map(str,mres.split())
    mltanbe = [float(mldata[1])]
    mlbrt4HH = [float(mldata[-4])]
    mlbrb4HH = [float(mldata[-8])] # Only for scenario 3
    mlisttanbe.extend(mltanbe)
    mlistbrt4HH.extend(mlbrt4HH)
    mlistbrb4HH.extend(mlbrb4HH)



olistmt4 = []
olistmHH = []
olisttotalxs = []
olisttanbe = []
olistlamHt44 = []
olistlamHb44 = []
olistmb4 = []

otlistmHH = []
otlisttest = []
otlistmt4 = []
otlistmb4 = []


#foin = open("t1ewptall-em-s4-tan1.dat")
#foin = open("t1ewptall-em-s4-mH1500.dat")
#foin = open("t1ewptall-em-s5-tan1.dat")
#foin = open("t1ewptall-em-s5-mH1500.dat")
#foin = open("t1ewptall-em-s7-tan1.dat")
foin = open("t1ewptall-em-s7-tan1-test.dat")
#foin = open("t1ewptall-em-s7-mH1500.dat")
olistscan = map(str,foin.readlines())
foin.close()
for jj in range(0,len(olistscan)-1):
    ores = olistscan[jj]
    oldata = map(str,ores.split())
    olmt4 = [float(oldata[0])]
    oltanbe = [float(oldata[1])]
    olmHH = [float(oldata[3])]
    #14TeV
    oltotalxs = [float(oldata[75])*sigmaHSM14TeV(float(oldata[3])) + ((float(oldata[1]))**2)*sigmabbF14(float(oldata[3]))]
    ollamHt44 = [float(oldata[10])]
    ollamHb44 = [float(oldata[107])]
    olmb4 = [float(oldata[137])]
    olistmt4.extend(olmt4)
    olisttanbe.extend(oltanbe)
    olistmHH.extend(olmHH)
    olisttotalxs.extend(oltotalxs)
    olistlamHt44.extend(ollamHt44)
    olistlamHb44.extend(ollamHb44)
    olistmb4.extend(olmb4)
    #if float(oldata[57]) > 0.2: #BR(H -> t4 t)
    if float(oldata[92]) > 0.5: #BR(H -> b4 b)
    #if abs(float(oldata[10])) > 0.01 and abs(float(oldata[11])) > 0.01 and abs(float(oldata[77])) < 1 and abs(float(oldata[78])) < 1: #lamHt44, lamHrt44, lamHt55, lamHrt55
    #if abs(float(oldata[107])) > 0.1 and abs(float(oldata[108])) > 0.1 and abs(float(oldata[109])) < 1 and abs(float(oldata[110])) < 1: #lamHb44, lamHrb44, lamHb55, lamHrb55
        otltest = [float(oldata[75])*sigmaHSM14TeV(float(oldata[3])) + ((float(oldata[1]))**2)*sigmabbF14(float(oldata[3]))] 
        otlmHH = [float(oldata[3])]
        otlmt4 = [float(oldata[0])]
        otlmb4 = [float(oldata[137])]
        otlisttest.extend(otltest)
        otlistmHH.extend(otlmHH)
        otlistmt4.extend(otlmt4)
        otlistmb4.extend(otlmb4)


    

plistmt4 = []
plistmHH = []
plisttotalxs = []
plisttanbe = []


ptlistmt4 = []
ptlistmb4 = []
ptlistmHH = []


#fpin = open("t1ewptall-em-s4-tan7.dat")
#fpin = open("t1ewptall-em-s4-mH2500.dat")
#fpin = open("t1ewptall-em-s5-tan7.dat")
#fpin = open("t1ewptall-em-s5-mH2500.dat")
#fpin = open("t1ewptall-em-s7-tan7.dat")
fpin = open("t1ewptall-em-s7-tan7-test.dat")
#fpin = open("t1ewptall-em-s7-mH2500.dat")
plistscan = map(str,fpin.readlines())
foin.close()
for aa in range(0,len(plistscan)-1):
    pres = plistscan[aa]
    pldata = map(str,pres.split())
    plmt4 = [float(pldata[0])]
    pltanbe = [float(pldata[1])]
    plmHH = [float(pldata[3])]
    #14TeV
    pltotalxs = [float(pldata[75])*sigmaHSM14TeV(float(pldata[3])) + ((float(pldata[1]))**2)*sigmabbF14(float(pldata[3]))]
    plistmt4.extend(plmt4)
    plisttanbe.extend(pltanbe)
    plistmHH.extend(plmHH)
    plisttotalxs.extend(pltotalxs)
    #if float(pldata[57]) > 0.2: #BR(H -> t4 t)
    if float(pldata[92]) > 0.5: #BR(H -> b4 b)
        ptlmHH = [float(pldata[3])]
        ptlmt4 = [float(pldata[0])]
        ptlmb4 = [float(pldata[137])]
        ptlistmHH.extend(ptlmHH)
        ptlistmt4.extend(ptlmt4)
        ptlistmb4.extend(ptlmb4)
       



    
qlistmt4 = []
qlistmHH = []
qlisttotalxs = []
qlisttanbe = []

qtlistmt4 = []
qtlistmb4 = []
qtlistmHH = []


#fqin = open("t1ewptall-em-s4-tan50.dat")
#fqin = open("t1ewptall-em-s4-mH4000.dat")
#fqin = open("t1ewptall-em-s5-tan50.dat")
#fqin = open("t1ewptall-em-s5-mH4000.dat")
#fqin = open("t1ewptall-em-s7-tan50.dat")
fqin = open("t1ewptall-em-s7-tan50-test.dat")
#fqin = open("t1ewptall-em-s7-mH4000.dat")
qlistscan = map(str,fqin.readlines())
fqin.close()
for bb in range(0,len(qlistscan)-1):
    qres = qlistscan[bb]
    qldata = map(str,qres.split())
    qlmt4 = [float(qldata[0])]
    qltanbe = [float(qldata[1])]   
    qlmHH = [float(qldata[3])]
    #14TeV
    qltotalxs = [float(qldata[75])*sigmaHSM14TeV(float(qldata[3])) + ((float(qldata[1]))**2)*sigmabbF14(float(qldata[3]))]
    qlistmt4.extend(qlmt4)
    qlisttanbe.extend(qltanbe)
    qlistmHH.extend(qlmHH)
    qlisttotalxs.extend(qltotalxs)
    #if float(qldata[57]) > 0.2: #BR(H -> t4 t)
    if float(qldata[92]) > 0.5: #BR(H -> b4 b)
        qtlmHH = [float(qldata[3])]
        qtlmt4 = [float(qldata[0])]
        qtlmb4 = [float(qldata[137])]
        qtlistmHH.extend(qtlmHH)
        qtlistmt4.extend(qtlmt4)
        qtlistmb4.extend(qtlmb4)


    
    
    

    
for k in range(0,1):
    #fin = open("t1ewptall-check.dat")
    #fin = open("t1ewptall-0.5Yukawa.dat")
    #fin = open("t1ewptall-0.5Yukawa-em.dat")
    fin = open("t1ewptall-em.dat")
    #fin = open("t1ewptall-test-em.dat")
    listscan = map(str,fin.readlines())
    fin.close()
    for i in range(0,len(listscan)-1):
        res = listscan[i]
        ldata = map(str,res.split())
        #if float(ldata[0]) >= 230.0:
        #if float(ldata[1]) < 5.:
        tlbrHtobb = [float(ldata[55])]
        tlbrHtohh = [float(ldata[56])]
        tlbrHtot4 = [float(ldata[57])]
        tlbrHtogg = [float(ldata[58])]
        tlbrHtocc = [float(ldata[59])]
        tlbrHtodi = [float(ldata[60])]
        tlbrHtott = [float(ldata[61])]
        tlbrHtotata = [float(ldata[80])]
        tltanbe = [float(ldata[1])]
        tlmt4 = [float(ldata[0])]
        tlmHH = [float(ldata[3])]
        tlbrt4w = [float(ldata[71])]
        tlbrt4z = [float(ldata[72])]
        tlbrt4h = [float(ldata[73])]
        tllamHt44 = [float(ldata[10])]
        #tlbrour = [float(ldata[31])*float(ldata[8])*float(ldata[49])]
        #tlbrour = [float(ldata[38])*float(ldata[22])]        
        tlbrour = [float(ldata[74])]        
        tllamHt34 = [float(ldata[2])]
        tllamHrt34 = [float(ldata[9])]
        tllamHt44 = [float(ldata[10])]
        tllamHrt44 = [float(ldata[11])]
        tlgLW43 = [float(ldata[13])]
        tlgRW43 = [float(ldata[22])]
        tlgLW33 = [float(ldata[15])]
        tlgLZ44 = [float(ldata[30])]
        tlgLZ43 = [float(ldata[31])]
        tlgLZ33 = [float(ldata[33])]
        tlgRZ44 = [float(ldata[39])]
        tlgRZ43 = [float(ldata[40])]
        tlgRZ33 = [float(ldata[42])]
        tlgamHtot = [float(ldata[62])]
        #tlgamHtotata = [float(ldata[63])]
        tlgamHtotata = [float(ldata[63])/float(ldata[3])]
        #tlgamHtobb = [float(ldata[64])]
        tlgamHtobb = [float(ldata[64])/float(ldata[3])]
        #tlgamHtot4 = [float(ldata[65])]
        tlgamHtot4 = [float(ldata[65])/float(ldata[3])]
        #tlgamHtogg = [float(ldata[66])]
        tlgamHtogg = [float(ldata[66])/float(ldata[3])]
        #tlgamHtohh = [float(ldata[67])]
        tlgamHtohh = [float(ldata[67])/float(ldata[3])]
        #tlgamHtocc = [float(ldata[68])]
        tlgamHtocc = [float(ldata[68])/float(ldata[3])]
        tlgamHtodi = [float(ldata[69])]
        #tlgamHtott = [float(ldata[70])]
        tlgamHtott = [float(ldata[70])/float(ldata[3])]
        tlkaT = [float(ldata[48])]
        tlggHratio = [float(ldata[75])]
        tltotalratio = [float(ldata[79])]
        tlprodratiobr = [float(ldata[74])*float(ldata[79])]
        tllamSMt44 = [-float(ldata[10])*float(ldata[1])]
        tlggSMratio = [float(ldata[76])]
        tlbbFratio = [float(ldata[79]) - float(ldata[75])]
        tlHWratio = [float(1)]#[sqrt((float(ldata[2]))**2 + (float(ldata[9]))**2)/sqrt((float(ldata[13]))**2 + (float(ldata[22]))**2)]
        #tldoubfr = [float(ldata[4])] #t4
        tldoubfr = [float(ldata[-16])] #b4
        tlkaQ = [float(ldata[49])]
        #tltotalxs = [float(ldata[79])*sigmaHSM13TeV(float(ldata[3]))]
        tltotalt4t = [float(ldata[79])*float(ldata[57])*sigmaHSM13TeV(float(ldata[3]))] #sigma * BR(H -> t4 t)
        #tltotalt4t = [float(ldata[75])*float(ldata[57])*sigmaHSM13TeV(float(ldata[3]))] #sigma * BR(H -> t4 t)
        tltotalb4b = [float(ldata[79])*float(ldata[92])*sigmaHSM13TeV(float(ldata[3]))] #tlmyt4bj = [myt4bj(float(ldata[0]))]        
        #14TeV
        tltotalxs = [float(ldata[75])*sigmaHSM14TeV(float(ldata[3])) + ((float(ldata[1]))**2)*sigmabbF14(float(ldata[3]))]
        tlmyt4bj = [(((float(ldata[13]))**2 + (float(ldata[22]))**2)/(2.*(0.1**2)))*myt4bj(float(ldata[0]))]
        #tlmyt4bj = [(((float(ldata[13]))**2)/(2.*0.01))*myt4bj(float(ldata[0]))]
        #tlmyt4tj = [(((float(ldata[31]))**2 + (float(ldata[40]))**2)/(2.*0.1**2))*myt4tj(float(ldata[0]))]
        #tltotalt4t = [(((float(ldata[13]))**2)/(2.*0.01))*myt4bj(float(ldata[0]))]
        #tlmyt4bj = [0.001*float(ldata[145])] #in pb unit
        tlmyt4tj = [0.001*float(ldata[146])] #in pb unit
        #tltryt4tj = [tryt4tj(float(ldata[0]),float(ldata[13]),float(ldata[22]))]
        tlsingfr = [float(ldata[5])]
        tlmb4 = [float(ldata[137])]
        tlbrb4w = [float(ldata[83])]
        tlbrb4z = [float(ldata[84])]
        tlbrb4h = [float(ldata[85])]
        tlbrHtob4 = [float(ldata[92])]
        tlMTT = [float(ldata[53])]
        tlgLW44t = [float(ldata[12])]
        tlgRW44t = [float(ldata[21])]
        tlbrt4HH = [float(ldata[-4])]
        tlbrb4HH = [float(ldata[-8])] # Only for scenario 3
        tllamHpmb43 = [float(ldata[-7])] # Only for scenario 2
        tllamHpmbr43 = [float(ldata[-6])] # Only for scenario 2
        tllamHb34 = [float(ldata[101])/float(ldata[1])] # lambda^h_{34} for now
        tllamHbr34 = [float(ldata[102])] # lambda^H_{43} 
        #14TeV
        tlxs2HDM1 = [float(ldata[-1])*sigmaHSM14TeV(float(ldata[3])) + sigmabbF14(float(ldata[3]))]
        tlxs2HDM7 = [float(ldata[-1])*sigmaHSM14TeV(float(ldata[3])) + (7**2)*sigmabbF14(float(ldata[3]))]
        tlxs2HDM50 = [float(ldata[-1])*sigmaHSM14TeV(float(ldata[3])) + (50**2)*sigmabbF14(float(ldata[3]))]
        tlgamHtob4 = [float(ldata[-4])/float(ldata[3])] # Only for the Hcascade paper
        tllamHb44 = [float(ldata[107])]
        tlbrb4t4w = [float(ldata[86])]
        #tlbrb4t4w = [float(ldata[83]) + float(ldata[84]) + float(ldata[85]) + float(ldata[86]) + float(ldata[87])]  #test
        tlbrt4b4w = [float(ldata[81])]
        tlistmt4.extend(tlmt4)
        tlisttanbe.extend(tltanbe)
        tlistlamHt34.extend(tllamHt34)
        tlistmHH.extend(tlmHH)
        tlistlamHrt34.extend(tllamHrt34)
        tlistlamHt44.extend(tllamHt44)
        tlistlamHrt44.extend(tllamHrt44)
        tlistgLW43.extend(tlgLW43)
        tlistgLW33.extend(tlgLW33)
        tlistgLZ44.extend(tlgLZ44)
        tlistgLZ43.extend(tlgLZ43)
        tlistgLZ33.extend(tlgLZ33)
        tlistgRZ44.extend(tlgRZ44)
        tlistgRZ43.extend(tlgRZ43)
        tlistgRZ33.extend(tlgRZ33)
        tlistbrHtobb.extend(tlbrHtobb)
        tlistbrHtohh.extend(tlbrHtohh)
        tlistbrHtot4.extend(tlbrHtot4)
        tlistbrHtogg.extend(tlbrHtogg)
        tlistbrHtocc.extend(tlbrHtocc)
        tlistbrHtodi.extend(tlbrHtodi)
        tlistbrHtott.extend(tlbrHtott)
        tlistgamHtot.extend(tlgamHtot)
        tlistgamHtotata.extend(tlgamHtotata)
        tlistgamHtobb.extend(tlgamHtobb)
        tlistgamHtot4.extend(tlgamHtot4)
        tlistgamHtogg.extend(tlgamHtogg)
        tlistgamHtohh.extend(tlgamHtohh)
        tlistgamHtocc.extend(tlgamHtocc)
        tlistgamHtodi.extend(tlgamHtodi)
        tlistgamHtott.extend(tlgamHtott)
        tlistbrt4w.extend(tlbrt4w)
        tlistbrt4z.extend(tlbrt4z)
        tlistbrt4h.extend(tlbrt4h)
        tlistbrour.extend(tlbrour)
        tlistkaT.extend(tlkaT)
        tlistggHratio.extend(tlggHratio)
        tlisttotalratio.extend(tltotalratio)
        tlistprodratiobr.extend(tlprodratiobr)
        tlistlamSMt44.extend(tllamSMt44)
        tlistggSMratio.extend(tlggSMratio)
        tlistbbFratio.extend(tlbbFratio)
        tlistbrHtotata.extend(tlbrHtotata)
        tlistHWratio.extend(tlHWratio)
        tlistdoubfr.extend(tldoubfr)
        tlistkaQ.extend(tlkaQ)
        tlisttotalxs.extend(tltotalxs)
        tlisttotalt4t.extend(tltotalt4t)
        tlistmyt4bj.extend(tlmyt4bj)
        tlistmyt4tj.extend(tlmyt4tj)
        tlistgRW43.extend(tlgRW43)
        tlistsingfr.extend(tlsingfr)
        tlistmb4.extend(tlmb4)
        tlistbrb4w.extend(tlbrb4w)
        tlistbrb4z.extend(tlbrb4z)
        tlistbrb4h.extend(tlbrb4h)
        tlistbrHtob4.extend(tlbrHtob4)
        tlistMTT.extend(tlMTT)
        tlisttotalb4b.extend(tltotalb4b)
        tlistgLW44t.extend(tlgLW44t)
        tlistgRW44t.extend(tlgRW44t)
        tlistbrt4HH.extend(tlbrt4HH)
        tlistbrb4HH.extend(tlbrb4HH)
        tlistlamHpmb43.extend(tllamHpmb43)
        tlistlamHpmbr43.extend(tllamHpmbr43)
        tlistlamHb34.extend(tllamHb34)
        tlistlamHbr34.extend(tllamHbr34)
        tlistxs2HDM1.extend(tlxs2HDM1)
        tlistxs2HDM7.extend(tlxs2HDM7)
        tlistxs2HDM50.extend(tlxs2HDM50)
        #tlisttryt4tj.extend(tltryt4tj)
        tlistgamHtob4.extend(tlgamHtob4)
        tlistlamHb44.extend(tllamHb44)
        tlistbrb4t4w.extend(tlbrb4t4w)
        tlistbrt4b4w.extend(tlbrt4b4w)
        if float(ldata[1]) > 0.999 and float(ldata[1]) < 1.001:# tan(beta)
            talbrHtobb = [float(ldata[55])]
            talbrHtohh = [float(ldata[56])]
            talbrHtot4 = [float(ldata[57])]
            talbrHtogg = [float(ldata[58])]
            talbrHtocc = [float(ldata[59])]
            talbrHtodi = [float(ldata[60])]
            talbrHtott = [float(ldata[61])]
            talbrHtotata = [float(ldata[80])]
            taltanbe = [float(ldata[1])]
            talmt4 = [float(ldata[0])]
            talmHH = [float(ldata[3])]
            talbrt4w = [float(ldata[71])]
            talbrt4z = [float(ldata[72])]
            talbrt4h = [float(ldata[73])]
            tallamHt44 = [float(ldata[10])]
            #talbrour = [float(ldata[31])*float(ldata[8])*float(ldata[49])]
            #talbrour = [float(ldata[38])*float(ldata[22])]        
            talbrour = [float(ldata[74])]        
            tallamHt34 = [float(ldata[2])]
            tallamHrt34 = [float(ldata[9])]
            tallamHt44 = [float(ldata[10])]
            tallamHrt44 = [float(ldata[11])]
            talgLW43 = [float(ldata[13])]
            talgLW33 = [float(ldata[15])]
            talgLZ44 = [float(ldata[30])]
            talgLZ43 = [float(ldata[31])]
            talgLZ33 = [float(ldata[33])]
            talgRZ44 = [float(ldata[39])]
            talgRZ43 = [float(ldata[40])]
            talgRZ33 = [float(ldata[42])]
            talgamHtot = [float(ldata[62])]
            talgamHtotata = [float(ldata[63])]
            talgamHtobb = [float(ldata[64])]
            talgamHtot4 = [float(ldata[65])]
            talgamHtogg = [float(ldata[66])]
            talgamHtohh = [float(ldata[67])]
            talgamHtocc = [float(ldata[68])]
            talgamHtodi = [float(ldata[69])]
            talgamHtott = [float(ldata[70])]
            talkaT = [float(ldata[48])]
            talggHratio = [float(ldata[75])]
            taltotalratio = [float(ldata[79])]
            talprodratiobr = [float(ldata[74])*float(ldata[79])]
            tallamSMt44 = [-float(ldata[10])*float(ldata[1])]
            talggSMratio = [float(ldata[76])]
            talbbFratio = [float(ldata[79]) - float(ldata[75])]
            taltotalt4t = [float(ldata[79])*float(ldata[57])*sigmaHSM13TeV(float(ldata[3]))] #sigma_H * BR(H -> t4 t)
            talmyt4bj = [(((float(ldata[13]))**2 + (float(ldata[22]))**2)/(2.*(0.1**2)))*myt4bj(float(ldata[0]))]
            #talmyt4bj = [0.001*float(ldata[145])] #in pb unit
            talmyt4tj = [0.001*float(ldata[146])] #in pb unit
            talkaQ = [abs(float(ldata[76]))]
            talka = [abs(float(ldata[77]))]
            talkab = [abs(float(ldata[78]))]
            talbrt4HH = [float(ldata[-4])]
            talbrb4HH = [float(ldata[-8])] # Only for scenario 3
            #14TeV
            taltotalxs = [float(ldata[75])*sigmaHSM14TeV(float(ldata[3])) + ((float(ldata[1]))**2)*sigmabbF14(float(ldata[3]))]
            talxs2HDM1 = [float(ldata[-1])*sigmaHSM14TeV(float(ldata[3])) + sigmabbF14(float(ldata[3]))]
            talistmt4.extend(talmt4)
            talisttanbe.extend(taltanbe)
            talistlamHt34.extend(tallamHt34)
            talistmHH.extend(talmHH)
            talistlamHrt34.extend(tallamHrt34)
            talistlamHt44.extend(tallamHt44)
            talistlamHrt44.extend(tallamHrt44)
            talistgLW43.extend(talgLW43)
            talistgLW33.extend(talgLW33)
            talistgLZ44.extend(talgLZ44)
            talistgLZ43.extend(talgLZ43)
            talistgLZ33.extend(talgLZ33)
            talistgRZ44.extend(talgRZ44)
            talistgRZ43.extend(talgRZ43)
            talistgRZ33.extend(talgRZ33)
            talistbrHtobb.extend(talbrHtobb)
            talistbrHtohh.extend(talbrHtohh)
            talistbrHtot4.extend(talbrHtot4)
            talistbrHtogg.extend(talbrHtogg)
            talistbrHtocc.extend(talbrHtocc)
            talistbrHtodi.extend(talbrHtodi)
            talistbrHtott.extend(talbrHtott)
            talistgamHtot.extend(talgamHtot)
            talistgamHtotata.extend(talgamHtotata)
            talistgamHtobb.extend(talgamHtobb)
            talistgamHtot4.extend(talgamHtot4)
            talistgamHtogg.extend(talgamHtogg)
            talistgamHtohh.extend(talgamHtohh)
            talistgamHtocc.extend(talgamHtocc)
            talistgamHtodi.extend(talgamHtodi)
            talistgamHtott.extend(talgamHtott)
            talistbrt4w.extend(talbrt4w)
            talistbrt4z.extend(talbrt4z)
            talistbrt4h.extend(talbrt4h)
            talistbrour.extend(talbrour)
            talistkaT.extend(talkaT)
            talistggHratio.extend(talggHratio)
            talisttotalratio.extend(taltotalratio)
            talistprodratiobr.extend(talprodratiobr)
            talistlamSMt44.extend(tallamSMt44)
            talistggSMratio.extend(talggSMratio)
            talistbbFratio.extend(talbbFratio)
            talistbrHtotata.extend(talbrHtotata)
            talisttotalt4t.extend(taltotalt4t)
            talistmyt4bj.extend(talmyt4bj)
            talistmyt4tj.extend(talmyt4tj)
            talistkaQ.extend(talkaQ)
            talistka.extend(talka)
            talistkab.extend(talkab)
            talistbrt4HH.extend(talbrt4HH)
            talistbrb4HH.extend(talbrb4HH)
            talisttotalxs.extend(taltotalxs)
            talistxs2HDM1.extend(talxs2HDM1)
            #if float(ldata[4]) < 0.5 and float(ldata[4]) >= 0.05:# doublet-fraction < 50%: singlet-like
            if abs(float(ldata[48])) < 0.01: # kaT < 0.05
                tablbrHtobb = [float(ldata[55])]
                tablbrHtohh = [float(ldata[56])]
                tablbrHtot4 = [float(ldata[57])]
                tablbrHtogg = [float(ldata[58])]
                tablbrHtocc = [float(ldata[59])]
                tablbrHtodi = [float(ldata[60])]
                tablbrHtott = [float(ldata[61])]
                tablbrHtotata = [float(ldata[80])]
                tabltanbe = [float(ldata[1])]
                tablmt4 = [float(ldata[0])]
                tablmHH = [float(ldata[3])]
                tablbrt4w = [float(ldata[71])]
                tablbrt4z = [float(ldata[72])]
                tablbrt4h = [float(ldata[73])]
                tabllamHt44 = [float(ldata[10])]
                tablbrour = [float(ldata[74])]        
                tabllamHt34 = [float(ldata[2])]
                tabllamHrt34 = [float(ldata[9])]
                tabllamHt44 = [float(ldata[10])]
                tabllamHrt44 = [float(ldata[11])]
                tablgLW43 = [float(ldata[13])]
                tablgLW33 = [float(ldata[15])]
                tablgLZ44 = [float(ldata[30])]
                tablgLZ43 = [float(ldata[31])]
                tablgLZ33 = [float(ldata[33])]
                tablgRZ44 = [float(ldata[39])]
                tablgRZ43 = [float(ldata[40])]
                tablgRZ33 = [float(ldata[42])]
                tablgamHtot = [float(ldata[62])]
                tablgamHtotata = [float(ldata[63])]
                tablgamHtobb = [float(ldata[64])]
                tablgamHtot4 = [float(ldata[65])]
                tablgamHtogg = [float(ldata[66])]
                tablgamHtohh = [float(ldata[67])]
                tablgamHtocc = [float(ldata[68])]
                tablgamHtodi = [float(ldata[69])]
                tablgamHtott = [float(ldata[70])]
                tablkaT = [float(ldata[48])]
                tablggHratio = [float(ldata[75])]
                tabltotalratio = [float(ldata[79])]
                tablprodratiobr = [float(ldata[74])*float(ldata[79])]
                tabllamSMt44 = [-float(ldata[10])*float(ldata[1])]
                tablggSMratio = [float(ldata[76])]
                tabltotalt4t = [float(ldata[79])*float(ldata[57])*sigmaHSM13TeV(float(ldata[3]))] #sigma_H * BR(H -> t4 t)
                tablmyt4bj = [(((float(ldata[13]))**2 + (float(ldata[22]))**2)/(2.*(0.1**2)))*myt4bj(float(ldata[0]))]
                #tablmyt4bj = [0.001*float(ldata[145])] #in pb unit
                tablmyt4tj = [0.001*float(ldata[146])] #in pb unit
                tablistmt4.extend(tablmt4)
                tablisttanbe.extend(tabltanbe)
                tablistlamHt34.extend(tabllamHt34)
                tablistmHH.extend(tablmHH)
                tablistlamHrt34.extend(tabllamHrt34)
                tablistlamHt44.extend(tabllamHt44)
                tablistlamHrt44.extend(tabllamHrt44)
                tablistgLW43.extend(tablgLW43)
                tablistgLW33.extend(tablgLW33)
                tablistgLZ44.extend(tablgLZ44)
                tablistgLZ43.extend(tablgLZ43)
                tablistgLZ33.extend(tablgLZ33)
                tablistgRZ44.extend(tablgRZ44)
                tablistgRZ43.extend(tablgRZ43)
                tablistgRZ33.extend(tablgRZ33)
                tablistbrHtobb.extend(tablbrHtobb)
                tablistbrHtohh.extend(tablbrHtohh)
                tablistbrHtot4.extend(tablbrHtot4)
                tablistbrHtogg.extend(tablbrHtogg)
                tablistbrHtocc.extend(tablbrHtocc)
                tablistbrHtodi.extend(tablbrHtodi)
                tablistbrHtott.extend(tablbrHtott)
                tablistgamHtot.extend(tablgamHtot)
                tablistgamHtotata.extend(tablgamHtotata)
                tablistgamHtobb.extend(tablgamHtobb)
                tablistgamHtot4.extend(tablgamHtot4)
                tablistgamHtogg.extend(tablgamHtogg)
                tablistgamHtohh.extend(tablgamHtohh)
                tablistgamHtocc.extend(tablgamHtocc)
                tablistgamHtodi.extend(tablgamHtodi)
                tablistgamHtott.extend(tablgamHtott)
                tablistbrt4w.extend(tablbrt4w)
                tablistbrt4z.extend(tablbrt4z)
                tablistbrt4h.extend(tablbrt4h)
                tablistbrour.extend(tablbrour)
                tablistkaT.extend(tablkaT)
                tablistggHratio.extend(tablggHratio)
                tablisttotalratio.extend(tabltotalratio)
                tablistprodratiobr.extend(tablprodratiobr)
                tablistlamSMt44.extend(tabllamSMt44)
                tablistggSMratio.extend(tablggSMratio)
                tablistbrHtotata.extend(tablbrHtotata)
                tablisttotalt4t.extend(tabltotalt4t)
                tablistmyt4bj.extend(tablmyt4bj)
                tablistmyt4tj.extend(tablmyt4tj)
            #if float(ldata[4]) >= 0.5 and float(ldata[4]) < 0.95:# doublet-fraction 50-95%: doublet-like
            if abs(float(ldata[48])) < 0.001: # kaT < 0.01
                taclbrHtobb = [float(ldata[55])]
                taclbrHtohh = [float(ldata[56])]
                taclbrHtot4 = [float(ldata[57])]
                taclbrHtogg = [float(ldata[58])]
                taclbrHtocc = [float(ldata[59])]
                taclbrHtodi = [float(ldata[60])]
                taclbrHtott = [float(ldata[61])]
                taclbrHtotata = [float(ldata[80])]
                tacltanbe = [float(ldata[1])]
                taclmt4 = [float(ldata[0])]
                taclmHH = [float(ldata[3])]
                taclbrt4w = [float(ldata[71])]
                taclbrt4z = [float(ldata[72])]
                taclbrt4h = [float(ldata[73])]
                tacllamHt44 = [float(ldata[10])]
                taclbrour = [float(ldata[74])]        
                tacllamHt34 = [float(ldata[2])]
                tacllamHrt34 = [float(ldata[9])]
                tacllamHt44 = [float(ldata[10])]
                tacllamHrt44 = [float(ldata[11])]
                taclgLW43 = [float(ldata[13])]
                taclgLW33 = [float(ldata[15])]
                taclgLZ44 = [float(ldata[30])]
                taclgLZ43 = [float(ldata[31])]
                taclgLZ33 = [float(ldata[33])]
                taclgRZ44 = [float(ldata[39])]
                taclgRZ43 = [float(ldata[40])]
                taclgRZ33 = [float(ldata[42])]
                taclgamHtot = [float(ldata[62])]
                taclgamHtotata = [float(ldata[63])]
                taclgamHtobb = [float(ldata[64])]
                taclgamHtot4 = [float(ldata[65])]
                taclgamHtogg = [float(ldata[66])]
                taclgamHtohh = [float(ldata[67])]
                taclgamHtocc = [float(ldata[68])]
                taclgamHtodi = [float(ldata[69])]
                taclgamHtott = [float(ldata[70])]
                taclkaT = [float(ldata[48])]
                taclggHratio = [float(ldata[75])]
                tacltotalratio = [float(ldata[79])]
                taclprodratiobr = [float(ldata[74])*float(ldata[79])]
                tacllamSMt44 = [-float(ldata[10])*float(ldata[1])]
                taclggSMratio = [float(ldata[76])]
                tacltotalt4t = [float(ldata[79])*float(ldata[57])*sigmaHSM13TeV(float(ldata[3]))] #sigma_H * BR(H -> t4 t)
                taclmyt4bj = [(((float(ldata[13]))**2 + (float(ldata[22]))**2)/(2.*(0.1**2)))*myt4bj(float(ldata[0]))]
                #taclmyt4bj = [0.001*float(ldata[145])] #in pb unit
                taclmyt4tj = [0.001*float(ldata[146])] #in pb unit
                taclistmt4.extend(taclmt4)
                taclisttanbe.extend(tacltanbe)
                taclistlamHt34.extend(tacllamHt34)
                taclistmHH.extend(taclmHH)
                taclistlamHrt34.extend(tacllamHrt34)
                taclistlamHt44.extend(tacllamHt44)
                taclistlamHrt44.extend(tacllamHrt44)
                taclistgLW43.extend(taclgLW43)
                taclistgLW33.extend(taclgLW33)
                taclistgLZ44.extend(taclgLZ44)
                taclistgLZ43.extend(taclgLZ43)
                taclistgLZ33.extend(taclgLZ33)
                taclistgRZ44.extend(taclgRZ44)
                taclistgRZ43.extend(taclgRZ43)
                taclistgRZ33.extend(taclgRZ33)
                taclistbrHtobb.extend(taclbrHtobb)
                taclistbrHtohh.extend(taclbrHtohh)
                taclistbrHtot4.extend(taclbrHtot4)
                taclistbrHtogg.extend(taclbrHtogg)
                taclistbrHtocc.extend(taclbrHtocc)
                taclistbrHtodi.extend(taclbrHtodi)
                taclistbrHtott.extend(taclbrHtott)
                taclistgamHtot.extend(taclgamHtot)
                taclistgamHtotata.extend(taclgamHtotata)
                taclistgamHtobb.extend(taclgamHtobb)
                taclistgamHtot4.extend(taclgamHtot4)
                taclistgamHtogg.extend(taclgamHtogg)
                taclistgamHtohh.extend(taclgamHtohh)
                taclistgamHtocc.extend(taclgamHtocc)
                taclistgamHtodi.extend(taclgamHtodi)
                taclistgamHtott.extend(taclgamHtott)
                taclistbrt4w.extend(taclbrt4w)
                taclistbrt4z.extend(taclbrt4z)
                taclistbrt4h.extend(taclbrt4h)
                taclistbrour.extend(taclbrour)
                taclistkaT.extend(taclkaT)
                taclistggHratio.extend(taclggHratio)
                taclisttotalratio.extend(tacltotalratio)
                taclistprodratiobr.extend(taclprodratiobr)
                taclistlamSMt44.extend(tacllamSMt44)
                taclistggSMratio.extend(taclggSMratio)
                taclistbrHtotata.extend(taclbrHtotata)
                taclisttotalt4t.extend(tacltotalt4t)
                taclistmyt4bj.extend(taclmyt4bj)
                taclistmyt4tj.extend(taclmyt4tj)
            #if float(ldata[4]) >= 0.95:# doublet-fraction > 95%: doublet-like
            if abs(float(ldata[48])) < 0.001 and abs(float(ldata[9])) > 0.01:
            # kaT < 0.001 and lambda^h_{t_4 t} > 0.001
                tadlbrHtobb = [float(ldata[55])]
                tadlbrHtohh = [float(ldata[56])]
                tadlbrHtot4 = [float(ldata[57])]
                tadlbrHtogg = [float(ldata[58])]
                tadlbrHtocc = [float(ldata[59])]
                tadlbrHtodi = [float(ldata[60])]
                tadlbrHtott = [float(ldata[61])]
                tadlbrHtotata = [float(ldata[80])]
                tadltanbe = [float(ldata[1])]
                tadlmt4 = [float(ldata[0])]
                tadlmHH = [float(ldata[3])]
                tadlbrt4w = [float(ldata[71])]
                tadlbrt4z = [float(ldata[72])]
                tadlbrt4h = [float(ldata[73])]
                tadllamHt44 = [float(ldata[10])]
                tadlbrour = [float(ldata[74])]        
                tadllamHt34 = [float(ldata[2])]
                tadllamHrt34 = [float(ldata[9])]
                tadllamHt44 = [float(ldata[10])]
                tadllamHrt44 = [float(ldata[11])]
                tadlgLW43 = [float(ldata[13])]
                tadlgLW33 = [float(ldata[15])]
                tadlgLZ44 = [float(ldata[30])]
                tadlgLZ43 = [float(ldata[31])]
                tadlgLZ33 = [float(ldata[33])]
                tadlgRZ44 = [float(ldata[39])]
                tadlgRZ43 = [float(ldata[40])]
                tadlgRZ33 = [float(ldata[42])]
                tadlgamHtot = [float(ldata[62])]
                tadlgamHtotata = [float(ldata[63])]
                tadlgamHtobb = [float(ldata[64])]
                tadlgamHtot4 = [float(ldata[65])]
                tadlgamHtogg = [float(ldata[66])]
                tadlgamHtohh = [float(ldata[67])]
                tadlgamHtocc = [float(ldata[68])]
                tadlgamHtodi = [float(ldata[69])]
                tadlgamHtott = [float(ldata[70])]
                tadlkaT = [float(ldata[48])]
                tadlggHratio = [float(ldata[75])]
                tadltotalratio = [float(ldata[79])]
                tadlprodratiobr = [float(ldata[74])*float(ldata[79])]
                tadllamSMt44 = [-float(ldata[10])*float(ldata[1])]
                tadlggSMratio = [float(ldata[76])]
                tadltotalt4t = [float(ldata[79])*float(ldata[57])*sigmaHSM13TeV(float(ldata[3]))] #sigma_H * BR(H -> t4 t)
                tadlmyt4bj = [(((float(ldata[13]))**2 + (float(ldata[22]))**2)/(2.*(0.1**2)))*myt4bj(float(ldata[0]))]
                #tadlmyt4bj = [0.001*float(ldata[145])] #in pb unit
                tadlmyt4tj = [0.001*float(ldata[146])] #in pb unit
                tadlistmt4.extend(tadlmt4)
                tadlisttanbe.extend(tadltanbe)
                tadlistlamHt34.extend(tadllamHt34)
                tadlistmHH.extend(tadlmHH)
                tadlistlamHrt34.extend(tadllamHrt34)
                tadlistlamHt44.extend(tadllamHt44)
                tadlistlamHrt44.extend(tadllamHrt44)
                tadlistgLW43.extend(tadlgLW43)
                tadlistgLW33.extend(tadlgLW33)
                tadlistgLZ44.extend(tadlgLZ44)
                tadlistgLZ43.extend(tadlgLZ43)
                tadlistgLZ33.extend(tadlgLZ33)
                tadlistgRZ44.extend(tadlgRZ44)
                tadlistgRZ43.extend(tadlgRZ43)
                tadlistgRZ33.extend(tadlgRZ33)
                tadlistbrHtobb.extend(tadlbrHtobb)
                tadlistbrHtohh.extend(tadlbrHtohh)
                tadlistbrHtot4.extend(tadlbrHtot4)
                tadlistbrHtogg.extend(tadlbrHtogg)
                tadlistbrHtocc.extend(tadlbrHtocc)
                tadlistbrHtodi.extend(tadlbrHtodi)
                tadlistbrHtott.extend(tadlbrHtott)
                tadlistgamHtot.extend(tadlgamHtot)
                tadlistgamHtotata.extend(tadlgamHtotata)
                tadlistgamHtobb.extend(tadlgamHtobb)
                tadlistgamHtot4.extend(tadlgamHtot4)
                tadlistgamHtogg.extend(tadlgamHtogg)
                tadlistgamHtohh.extend(tadlgamHtohh)
                tadlistgamHtocc.extend(tadlgamHtocc)
                tadlistgamHtodi.extend(tadlgamHtodi)
                tadlistgamHtott.extend(tadlgamHtott)
                tadlistbrt4w.extend(tadlbrt4w)
                tadlistbrt4z.extend(tadlbrt4z)
                tadlistbrt4h.extend(tadlbrt4h)
                tadlistbrour.extend(tadlbrour)
                tadlistkaT.extend(tadlkaT)
                tadlistggHratio.extend(tadlggHratio)
                tadlisttotalratio.extend(tadltotalratio)
                tadlistprodratiobr.extend(tadlprodratiobr)
                tadlistlamSMt44.extend(tadllamSMt44)
                tadlistggSMratio.extend(tadlggSMratio)
                tadlistbrHtotata.extend(tadlbrHtotata)
                tadlisttotalt4t.extend(tadltotalt4t)
                tadlistmyt4bj.extend(tadlmyt4bj)
                tadlistmyt4tj.extend(tadlmyt4tj)

            
        #if float(ldata[0]) > float(ldata[137]): #mt4 > mb4
        #if float(ldata[1]) < 0.501 and float(ldata[1]) > 0.499:# tan(beta) and float(ldata[0]) > 1000 and float(ldata[0]) < 1400: #tan(beta) and mt4 1.2 TeV
        #if abs(float(ldata[48])) < 0.01:# and abs(float(ldata[102])) > 0.01:# lamB & lamHr34
        if abs(float(ldata[1])) < 1:# tan(beta)
        #if abs(float(ldata[49])) < 0.01:# lamQ
        #if abs(float(ldata[49])) < 0.01:# lamQ
        #if abs(float(ldata[50])) < 0.01:# lam
        #if abs(float(ldata[51])) < 0.01:# lamba
            tblbrHtobb = [float(ldata[55])]
            tblbrHtohh = [float(ldata[56])]
            tblbrHtot4 = [float(ldata[57])]
            tblbrHtogg = [float(ldata[58])]
            tblbrHtocc = [float(ldata[59])]
            tblbrHtodi = [float(ldata[60])]
            tblbrHtott = [float(ldata[61])]
            tblbrHtotata = [float(ldata[80])]
            tbltanbe = [float(ldata[1])]
            tblmt4 = [float(ldata[0])]
            tblmHH = [float(ldata[3])]
            tblbrt4w = [float(ldata[71])]
            tblbrt4z = [float(ldata[72])]
            tblbrt4h = [float(ldata[73])]
            tbllamHt44 = [float(ldata[10])]
            #tblbrour = [float(ldata[74])]  # BR(H -> t4 t)*BR(t4 -> t h)
            #tblbrour = [float(ldata[79])*float(ldata[57])*float(ldata[73])]  # (sigma_H / sigma_H^SM)BR(H -> t4 t)*BR(t4 -> t h)
            #tblbrour = [float(ldata[79])*float(ldata[57])*float(ldata[73])]  # (sigma_H / sigma_H^SM)BR(H -> t4 t)*BR(t4 -> t h)
            #tblsigBRw = [float(ldata[79])*float(ldata[57])*float(ldata[71])]  # (sigma_H / sigma_H^SM)BR(H -> t4 t)*BR(t4 -> b W)
            #tblsigBRz = [float(ldata[79])*float(ldata[57])*float(ldata[72])]  # (sigma_H / sigma_H^SM)BR(H -> t4 t)*BR(t4 -> t Z)

            #tblbrourb4 = [float(ldata[92])*float(ldata[85])] # BR(H -> b4 b)*BR(b4 -> th)
            # 14 TeV
            tblbrour = [(float(ldata[75]) + ((float(ldata[1]))**2)*sigmabbF14(float(ldata[3]))/sigmaHSM14TeV(float(ldata[3])))*float(ldata[57])*float(ldata[73])]
            tblsigBRw = [(float(ldata[75]) + ((float(ldata[1]))**2)*sigmabbF14(float(ldata[3]))/sigmaHSM14TeV(float(ldata[3])))*float(ldata[57])*float(ldata[71])]
            tblsigBRz = [(float(ldata[75]) + ((float(ldata[1]))**2)*sigmabbF14(float(ldata[3]))/sigmaHSM14TeV(float(ldata[3])))*float(ldata[57])*float(ldata[72])]
            tblbrourb4 = [(float(ldata[75]) + ((float(ldata[1]))**2)*sigmabbF14(float(ldata[3]))/sigmaHSM14TeV(float(ldata[3])))*float(ldata[92])*float(ldata[85])]
            tblsigb4BRw = [(float(ldata[75]) + ((float(ldata[1]))**2)*sigmabbF14(float(ldata[3]))/sigmaHSM14TeV(float(ldata[3])))*float(ldata[92])*float(ldata[83])]
            tblsigb4BRz = [(float(ldata[75]) + ((float(ldata[1]))**2)*sigmabbF14(float(ldata[3]))/sigmaHSM14TeV(float(ldata[3])))*float(ldata[92])*float(ldata[84])]
            tbllamHt34 = [float(ldata[2])]
            tbllamHrt34 = [float(ldata[9])]
            tbllamHt44 = [float(ldata[10])]
            tbllamHrt44 = [float(ldata[11])]
            tblgLW43 = [float(ldata[13])]
            tblgLW33 = [float(ldata[15])]
            tblgLZ44 = [float(ldata[30])]
            tblgLZ43 = [float(ldata[31])]
            tblgLZ33 = [float(ldata[33])]
            tblgRZ44 = [float(ldata[39])]
            tblgRZ43 = [float(ldata[40])]
            tblgRZ33 = [float(ldata[42])]
            tblgamHtot = [float(ldata[62])]
            tblgamHtotata = [float(ldata[63])]
            tblgamHtobb = [float(ldata[64])]
            tblgamHtot4 = [float(ldata[65])]
            tblgamHtogg = [float(ldata[66])]
            tblgamHtohh = [float(ldata[67])]
            tblgamHtocc = [float(ldata[68])]
            tblgamHtodi = [float(ldata[69])]
            tblgamHtott = [float(ldata[70])]
            tblkaT = [float(ldata[48])]
            tblggHratio = [float(ldata[75])]
            tbltotalratio = [float(ldata[79])]
            tblprodratiobr = [float(ldata[74])*float(ldata[79])]
            tbllamSMt44 = [-float(ldata[10])*float(ldata[1])]
            tblggSMratio = [float(ldata[76])]
            tblbrb4w = [float(ldata[83])]
            tblbrb4z = [float(ldata[84])]
            tblbrb4h = [float(ldata[85])]
            tbllamHpmbr43 = [float(ldata[-6])] # Only for scenario 2
            tbllamHbr34 = [float(ldata[102])] # lambda^H_{43} 
            tblbrb4HH = [float(ldata[-8])] 
            #tbltotalxs = [float(ldata[79])*sigmaHSM13TeV(float(ldata[3]))]
            #14TeV
            tbltotalxs = [float(ldata[75])*sigmaHSM14TeV(float(ldata[3])) + ((float(ldata[1]))**2)*sigmabbF14(float(ldata[3]))]
            tblistmt4.extend(tblmt4)
            tblisttanbe.extend(tbltanbe)
            tblistlamHt34.extend(tbllamHt34)
            tblistmHH.extend(tblmHH)
            tblistlamHrt34.extend(tbllamHrt34)
            tblistlamHt44.extend(tbllamHt44)
            tblistlamHrt44.extend(tbllamHrt44)
            tblistgLW43.extend(tblgLW43)
            tblistgLW33.extend(tblgLW33)
            tblistgLZ44.extend(tblgLZ44)
            tblistgLZ43.extend(tblgLZ43)
            tblistgLZ33.extend(tblgLZ33)
            tblistgRZ44.extend(tblgRZ44)
            tblistgRZ43.extend(tblgRZ43)
            tblistgRZ33.extend(tblgRZ33)
            tblistbrHtobb.extend(tblbrHtobb)
            tblistbrHtohh.extend(tblbrHtohh)
            tblistbrHtot4.extend(tblbrHtot4)
            tblistbrHtogg.extend(tblbrHtogg)
            tblistbrHtocc.extend(tblbrHtocc)
            tblistbrHtodi.extend(tblbrHtodi)
            tblistbrHtott.extend(tblbrHtott)
            tblistgamHtot.extend(tblgamHtot)
            tblistgamHtotata.extend(tblgamHtotata)
            tblistgamHtobb.extend(tblgamHtobb)
            tblistgamHtot4.extend(tblgamHtot4)
            tblistgamHtogg.extend(tblgamHtogg)
            tblistgamHtohh.extend(tblgamHtohh)
            tblistgamHtocc.extend(tblgamHtocc)
            tblistgamHtodi.extend(tblgamHtodi)
            tblistgamHtott.extend(tblgamHtott)
            tblistbrt4w.extend(tblbrt4w)
            tblistbrt4z.extend(tblbrt4z)
            tblistbrt4h.extend(tblbrt4h)
            tblistbrour.extend(tblbrour)
            tblistkaT.extend(tblkaT)
            tblistggHratio.extend(tblggHratio)
            tblisttotalratio.extend(tbltotalratio)
            tblistprodratiobr.extend(tblprodratiobr)
            tblistlamSMt44.extend(tbllamSMt44)
            tblistggSMratio.extend(tblggSMratio)
            tblistbrHtotata.extend(tblbrHtotata)
            tblistbrourb4.extend(tblbrourb4)
            tblistsigBRw.extend(tblsigBRw)
            tblistsigBRz.extend(tblsigBRz)
            tblistbrb4w.extend(tblbrb4w)
            tblistbrb4z.extend(tblbrb4z)
            tblistbrb4h.extend(tblbrb4h)
            tblistlamHpmbr43.extend(tbllamHpmbr43)
            tblistlamHbr34.extend(tbllamHbr34)
            tblistbrb4HH.extend(tblbrb4HH)
            tblisttotalxs.extend(tbltotalxs)
            tblistsigb4BRw.extend(tblsigb4BRw)
            tblistsigb4BRz.extend(tblsigb4BRz)

        #if float(ldata[0]) < float(ldata[137]): #mt4 < mb4
        #if float(ldata[3]) < 2000: #mHH
        if float(ldata[1]) > 1. and float(ldata[1]) < 5.:# and float(ldata[0]) > 1000 and float(ldata[0]) < 1400: #tan(beta) and mt4 1.2 TeV
        #if float(ldata[1]) > 6.9 and float(ldata[1]) < 7.1:# and float(ldata[0]) > 1000 and float(ldata[0]) < 1400: #tan(beta) and mt4 1.2 TeV
            tclbrHtobb = [float(ldata[55])]
            tclbrHtohh = [float(ldata[56])]
            tclbrHtot4 = [float(ldata[57])]
            tclbrHtogg = [float(ldata[58])]
            tclbrHtocc = [float(ldata[59])]
            tclbrHtodi = [float(ldata[60])]
            tclbrHtott = [float(ldata[61])]
            tcltanbe = [float(ldata[1])]
            tclmt4 = [float(ldata[0])]
            tclmHH = [float(ldata[3])]
            tclbrt4w = [float(ldata[71])]
            tclbrt4z = [float(ldata[72])]
            tclbrt4h = [float(ldata[73])]
            tcllamHt44 = [float(ldata[10])]
            #tclbrour = [float(ldata[74])]  # BR(H -> t4 t)*BR(t4 -> th)
            #tclbrour = [float(ldata[79])*float(ldata[57])*float(ldata[73])]  # (sigma_H / sigma_H^SM)BR(H -> t4 t)*BR(t4 -> t h)
            #tclsigBRw = [float(ldata[79])*float(ldata[57])*float(ldata[71])]  # (sigma_H / sigma_H^SM)BR(H -> t4 t)*BR(t4 -> b W)
            #tclsigBRz = [float(ldata[79])*float(ldata[57])*float(ldata[72])]  # (sigma_H / sigma_H^SM)BR(H -> t4 t)*BR(t4 -> t Z)
            #tclbrourb4 = [float(ldata[92])*float(ldata[85])] # BR(H -> b4 b)*BR(b4 -> th)
            # 14 TeV
            tclbrour = [(float(ldata[75]) + ((float(ldata[1]))**2)*sigmabbF14(float(ldata[3]))/sigmaHSM14TeV(float(ldata[3])))*float(ldata[57])*float(ldata[73])]
            tclsigBRw = [(float(ldata[75]) + ((float(ldata[1]))**2)*sigmabbF14(float(ldata[3]))/sigmaHSM14TeV(float(ldata[3])))*float(ldata[57])*float(ldata[71])]
            tclsigBRz = [(float(ldata[75]) + ((float(ldata[1]))**2)*sigmabbF14(float(ldata[3]))/sigmaHSM14TeV(float(ldata[3])))*float(ldata[57])*float(ldata[72])]
            tclbrourb4 = [(float(ldata[75]) + ((float(ldata[1]))**2)*sigmabbF14(float(ldata[3]))/sigmaHSM14TeV(float(ldata[3])))*float(ldata[92])*float(ldata[85])]
            tclsigb4BRw = [(float(ldata[75]) + ((float(ldata[1]))**2)*sigmabbF14(float(ldata[3]))/sigmaHSM14TeV(float(ldata[3])))*float(ldata[92])*float(ldata[83])]
            tclsigb4BRz = [(float(ldata[75]) + ((float(ldata[1]))**2)*sigmabbF14(float(ldata[3]))/sigmaHSM14TeV(float(ldata[3])))*float(ldata[92])*float(ldata[84])]
            tcllamHt34 = [float(ldata[2])]
            tcllamHrt34 = [float(ldata[9])]
            tcllamHt44 = [float(ldata[10])]
            tcllamHrt44 = [float(ldata[11])]
            tclgLW43 = [float(ldata[13])]
            tclgLW33 = [float(ldata[15])]
            tclgLZ44 = [float(ldata[30])]
            tclgLZ43 = [float(ldata[31])]
            tclgLZ33 = [float(ldata[33])]
            tclgRZ44 = [float(ldata[39])]
            tclgRZ43 = [float(ldata[40])]
            tclgRZ33 = [float(ldata[42])]
            tclgamHtot = [float(ldata[62])]
            tclgamHtotata = [float(ldata[63])]
            tclgamHtobb = [float(ldata[64])]
            tclgamHtot4 = [float(ldata[65])]
            tclgamHtogg = [float(ldata[66])]
            tclgamHtohh = [float(ldata[67])]
            tclgamHtocc = [float(ldata[68])]
            tclgamHtodi = [float(ldata[69])]
            tclgamHtott = [float(ldata[70])]
            tclkaT = [float(ldata[48])]
            tclggHratio = [float(ldata[75])]
            tcltotalratio = [float(ldata[79])]
            tclprodratiobr = [float(ldata[74])*float(ldata[79])]
            tcllamSMt44 = [-float(ldata[10])*float(ldata[1])]
            tclggSMratio = [float(ldata[76])]
            tclbbFratio = [float(ldata[79]) - float(ldata[75])]
            tclbrHtotata = [float(ldata[80])]
            tclbrb4w = [float(ldata[83])]
            tclbrb4z = [float(ldata[84])]
            tclbrb4h = [float(ldata[85])]
            #tcltotalxs = [float(ldata[79])*sigmaHSM13TeV(float(ldata[3]))]
            #14TeV
            tcltotalxs = [float(ldata[75])*sigmaHSM14TeV(float(ldata[3])) + (7**2)*sigmabbF14(float(ldata[3]))]
            tclxs2HDM7 = [float(ldata[-1])*sigmaHSM14TeV(float(ldata[3])) + (7**2)*sigmabbF14(float(ldata[3]))]
            tclistmt4.extend(tclmt4)
            tclisttanbe.extend(tcltanbe)
            tclistlamHt34.extend(tcllamHt34)
            tclistmHH.extend(tclmHH)
            tclistlamHrt34.extend(tcllamHrt34)
            tclistlamHt44.extend(tcllamHt44)
            tclistlamHrt44.extend(tcllamHrt44)
            tclistgLW43.extend(tclgLW43)
            tclistgLW33.extend(tclgLW33)
            tclistgLZ44.extend(tclgLZ44)
            tclistgLZ43.extend(tclgLZ43)
            tclistgLZ33.extend(tclgLZ33)
            tclistgRZ44.extend(tclgRZ44)
            tclistgRZ43.extend(tclgRZ43)
            tclistgRZ33.extend(tclgRZ33)
            tclistbrHtobb.extend(tclbrHtobb)
            tclistbrHtohh.extend(tclbrHtohh)
            tclistbrHtot4.extend(tclbrHtot4)
            tclistbrHtogg.extend(tclbrHtogg)
            tclistbrHtocc.extend(tclbrHtocc)
            tclistbrHtodi.extend(tclbrHtodi)
            tclistbrHtott.extend(tclbrHtott)
            tclistgamHtot.extend(tclgamHtot)
            tclistgamHtotata.extend(tclgamHtotata)
            tclistgamHtobb.extend(tclgamHtobb)
            tclistgamHtot4.extend(tclgamHtot4)
            tclistgamHtogg.extend(tclgamHtogg)
            tclistgamHtohh.extend(tclgamHtohh)
            tclistgamHtocc.extend(tclgamHtocc)
            tclistgamHtodi.extend(tclgamHtodi)
            tclistgamHtott.extend(tclgamHtott)
            tclistbrt4w.extend(tclbrt4w)
            tclistbrt4z.extend(tclbrt4z)
            tclistbrt4h.extend(tclbrt4h)
            tclistbrour.extend(tclbrour)
            tclistkaT.extend(tclkaT)
            tclistggHratio.extend(tclggHratio)
            tclisttotalratio.extend(tcltotalratio)
            tclistprodratiobr.extend(tclprodratiobr)
            tclistlamSMt44.extend(tcllamSMt44)
            tclistggSMratio.extend(tclggSMratio)
            tclistbbFratio.extend(tclbbFratio)
            tclistbrHtotata.extend(tclbrHtotata)
            tclistbrourb4.extend(tclbrourb4)
            tclistsigBRw.extend(tclsigBRw)
            tclistsigBRz.extend(tclsigBRz)
            tclistbrb4w.extend(tclbrb4w)
            tclistbrb4z.extend(tclbrb4z)
            tclistbrb4h.extend(tclbrb4h)
            tclisttotalxs.extend(tcltotalxs)
            tclistxs2HDM7.extend(tclxs2HDM7)
            tclistsigb4BRw.extend(tclsigb4BRw)
            tclistsigb4BRz.extend(tclsigb4BRz)
        if float(ldata[1]) > 5. and float(ldata[1]) < 10.:# and float(ldata[0]) > 1000 and float(ldata[0]) < 1400: #tan(beta) and mt4 1.2 TeV
        #if float(ldata[1]) > 9.9 and float(ldata[1]) < 10.1:# and float(ldata[0]) > 1000 and float(ldata[0]) < 1400: #tan(beta) and mt4 1.2 TeV
            telbrHtobb = [float(ldata[55])]
            telbrHtohh = [float(ldata[56])]
            telbrHtot4 = [float(ldata[57])]
            telbrHtogg = [float(ldata[58])]
            telbrHtocc = [float(ldata[59])]
            telbrHtodi = [float(ldata[60])]
            telbrHtott = [float(ldata[61])]
            teltanbe = [float(ldata[1])]
            telmt4 = [float(ldata[0])]
            telmHH = [float(ldata[3])]
            telbrt4w = [float(ldata[71])]
            telbrt4z = [float(ldata[72])]
            telbrt4h = [float(ldata[73])]
            tellamHt44 = [float(ldata[10])]
            #telbrour = [float(ldata[74])]        
            #telbrour = [float(ldata[79])*float(ldata[57])*float(ldata[73])]  # (sigma_H / sigma_H^SM)BR(H -> t4 t)*BR(t4 -> t h)
            #telsigBRw = [float(ldata[79])*float(ldata[57])*float(ldata[71])]  # (sigma_H / sigma_H^SM)BR(H -> t4 t)*BR(t4 -> b W)
            #telsigBRz = [float(ldata[79])*float(ldata[57])*float(ldata[72])]  # (sigma_H / sigma_H^SM)BR(H -> t4 t)*BR(t4 -> t Z)
            #telbrourb4 = [float(ldata[92])*float(ldata[85])] # BR(H -> b4 b)*BR(b4 -> th)
            # 14 TeV
            telbrour = [(float(ldata[75]) + ((float(ldata[1]))**2)*sigmabbF14(float(ldata[3]))/sigmaHSM14TeV(float(ldata[3])))*float(ldata[57])*float(ldata[73])]
            telsigBRw = [(float(ldata[75]) + ((float(ldata[1]))**2)*sigmabbF14(float(ldata[3]))/sigmaHSM14TeV(float(ldata[3])))*float(ldata[57])*float(ldata[71])]
            telsigBRz = [(float(ldata[75]) + ((float(ldata[1]))**2)*sigmabbF14(float(ldata[3]))/sigmaHSM14TeV(float(ldata[3])))*float(ldata[57])*float(ldata[72])]
            telbrourb4 = [(float(ldata[75]) + ((float(ldata[1]))**2)*sigmabbF14(float(ldata[3]))/sigmaHSM14TeV(float(ldata[3])))*float(ldata[92])*float(ldata[85])]
            telsigb4BRw = [(float(ldata[75]) + ((float(ldata[1]))**2)*sigmabbF14(float(ldata[3]))/sigmaHSM14TeV(float(ldata[3])))*float(ldata[92])*float(ldata[83])]
            telsigb4BRz = [(float(ldata[75]) + ((float(ldata[1]))**2)*sigmabbF14(float(ldata[3]))/sigmaHSM14TeV(float(ldata[3])))*float(ldata[92])*float(ldata[84])]
            tellamHt34 = [float(ldata[2])]
            tellamHrt34 = [float(ldata[9])]
            tellamHt44 = [float(ldata[10])]
            tellamHrt44 = [float(ldata[11])]
            telgLW43 = [float(ldata[13])]
            telgLW33 = [float(ldata[15])]
            telgLZ44 = [float(ldata[30])]
            telgLZ43 = [float(ldata[31])]
            telgLZ33 = [float(ldata[33])]
            telgRZ44 = [float(ldata[39])]
            telgRZ43 = [float(ldata[40])]
            telgRZ33 = [float(ldata[42])]
            telgamHtot = [float(ldata[62])]
            telgamHtotata = [float(ldata[63])]
            telgamHtobb = [float(ldata[64])]
            telgamHtot4 = [float(ldata[65])]
            telgamHtogg = [float(ldata[66])]
            telgamHtohh = [float(ldata[67])]
            telgamHtocc = [float(ldata[68])]
            telgamHtodi = [float(ldata[69])]
            telgamHtott = [float(ldata[70])]
            telkaT = [float(ldata[48])]
            telggHratio = [float(ldata[75])]
            teltotalratio = [float(ldata[79])]
            telprodratiobr = [float(ldata[74])*float(ldata[79])]
            tellamSMt44 = [-float(ldata[10])*float(ldata[1])]
            telggSMratio = [float(ldata[76])]
            telbbFratio = [float(ldata[79]) - float(ldata[75])]
            telbrHtotata = [float(ldata[80])]
            telbrb4w = [float(ldata[83])]
            telbrb4z = [float(ldata[84])]
            telbrb4h = [float(ldata[85])]
            #teltotalxs = [float(ldata[79])*sigmaHSM13TeV(float(ldata[3]))]
            #14TeV
            teltotalxs = [float(ldata[75])*sigmaHSM14TeV(float(ldata[3])) + ((float(ldata[1]))**2)*sigmabbF14(float(ldata[3]))]
            telistmt4.extend(telmt4)
            telisttanbe.extend(teltanbe)
            telistlamHt34.extend(tellamHt34)
            telistmHH.extend(telmHH)
            telistlamHrt34.extend(tellamHrt34)
            telistlamHt44.extend(tellamHt44)
            telistlamHrt44.extend(tellamHrt44)
            telistgLW43.extend(telgLW43)
            telistgLW33.extend(telgLW33)
            telistgLZ44.extend(telgLZ44)
            telistgLZ43.extend(telgLZ43)
            telistgLZ33.extend(telgLZ33)
            telistgRZ44.extend(telgRZ44)
            telistgRZ43.extend(telgRZ43)
            telistgRZ33.extend(telgRZ33)
            telistbrHtobb.extend(telbrHtobb)
            telistbrHtohh.extend(telbrHtohh)
            telistbrHtot4.extend(telbrHtot4)
            telistbrHtogg.extend(telbrHtogg)
            telistbrHtocc.extend(telbrHtocc)
            telistbrHtodi.extend(telbrHtodi)
            telistbrHtott.extend(telbrHtott)
            telistgamHtot.extend(telgamHtot)
            telistgamHtotata.extend(telgamHtotata)
            telistgamHtobb.extend(telgamHtobb)
            telistgamHtot4.extend(telgamHtot4)
            telistgamHtogg.extend(telgamHtogg)
            telistgamHtohh.extend(telgamHtohh)
            telistgamHtocc.extend(telgamHtocc)
            telistgamHtodi.extend(telgamHtodi)
            telistgamHtott.extend(telgamHtott)
            telistbrt4w.extend(telbrt4w)
            telistbrt4z.extend(telbrt4z)
            telistbrt4h.extend(telbrt4h)
            telistbrour.extend(telbrour)
            telistkaT.extend(telkaT)
            telistggHratio.extend(telggHratio)
            telisttotalratio.extend(teltotalratio)
            telistprodratiobr.extend(telprodratiobr)
            telistlamSMt44.extend(tellamSMt44)
            telistggSMratio.extend(telggSMratio)
            telistbbFratio.extend(telbbFratio)
            telistbrHtotata.extend(telbrHtotata)
            telistbrourb4.extend(telbrourb4)
            telistsigBRw.extend(telsigBRw)
            telistsigBRz.extend(telsigBRz)
            telistbrb4w.extend(telbrb4w)
            telistbrb4z.extend(telbrb4z)
            telistbrb4h.extend(telbrb4h)
            telisttotalxs.extend(teltotalxs)
            telistsigb4BRw.extend(telsigb4BRw)
            telistsigb4BRz.extend(telsigb4BRz)
        if float(ldata[1]) > 10. and float(ldata[1]) < 20.:# and float(ldata[0]) > 1000 and float(ldata[0]) < 1400: #tan(beta) and mt4 1.2 TeV
        #if float(ldata[1]) > 19.9 and float(ldata[1]) < 20.1:# and float(ldata[0]) > 1000 and float(ldata[0]) < 1400: #tan(beta) and mt4 1.2 TeV
            tflbrHtobb = [float(ldata[55])]
            tflbrHtohh = [float(ldata[56])]
            tflbrHtot4 = [float(ldata[57])]
            tflbrHtogg = [float(ldata[58])]
            tflbrHtocc = [float(ldata[59])]
            tflbrHtodi = [float(ldata[60])]
            tflbrHtott = [float(ldata[61])]
            tfltanbe = [float(ldata[1])]
            tflmt4 = [float(ldata[0])]
            tflmHH = [float(ldata[3])]
            tflbrt4w = [float(ldata[71])]
            tflbrt4z = [float(ldata[72])]
            tflbrt4h = [float(ldata[73])]
            tfllamHt44 = [float(ldata[10])]
            #tflbrour = [float(ldata[74])]        
            #tflbrour = [float(ldata[79])*float(ldata[57])*float(ldata[73])]  # (sigma_H / sigma_H^SM)BR(H -> t4 t)*BR(t4 -> t h)
            #tflsigBRw = [float(ldata[79])*float(ldata[57])*float(ldata[71])]  # (sigma_H / sigma_H^SM)BR(H -> t4 t)*BR(t4 -> b W)
            #tflsigBRz = [float(ldata[79])*float(ldata[57])*float(ldata[72])]  # (sigma_H / sigma_H^SM)BR(H -> t4 t)*BR(t4 -> t Z)
            #tflbrourb4 = [float(ldata[92])*float(ldata[85])] # BR(H -> b4 b)*BR(b4 -> th)
            # 14 TeV
            tflbrour = [(float(ldata[75]) + ((float(ldata[1]))**2)*sigmabbF14(float(ldata[3]))/sigmaHSM14TeV(float(ldata[3])))*float(ldata[57])*float(ldata[73])]
            tflsigBRw = [(float(ldata[75]) + ((float(ldata[1]))**2)*sigmabbF14(float(ldata[3]))/sigmaHSM14TeV(float(ldata[3])))*float(ldata[57])*float(ldata[71])]
            tflsigBRz = [(float(ldata[75]) + ((float(ldata[1]))**2)*sigmabbF14(float(ldata[3]))/sigmaHSM14TeV(float(ldata[3])))*float(ldata[57])*float(ldata[72])]
            tflbrourb4 = [(float(ldata[75]) + ((float(ldata[1]))**2)*sigmabbF14(float(ldata[3]))/sigmaHSM14TeV(float(ldata[3])))*float(ldata[92])*float(ldata[85])]
            tflsigb4BRw = [(float(ldata[75]) + ((float(ldata[1]))**2)*sigmabbF14(float(ldata[3]))/sigmaHSM14TeV(float(ldata[3])))*float(ldata[92])*float(ldata[83])]
            tflsigb4BRz = [(float(ldata[75]) + ((float(ldata[1]))**2)*sigmabbF14(float(ldata[3]))/sigmaHSM14TeV(float(ldata[3])))*float(ldata[92])*float(ldata[84])]
            tfllamHt34 = [float(ldata[2])]
            tfllamHrt34 = [float(ldata[9])]
            tfllamHt44 = [float(ldata[10])]
            tfllamHrt44 = [float(ldata[11])]
            tflgLW43 = [float(ldata[13])]
            tflgLW33 = [float(ldata[15])]
            tflgLZ44 = [float(ldata[30])]
            tflgLZ43 = [float(ldata[31])]
            tflgLZ33 = [float(ldata[33])]
            tflgRZ44 = [float(ldata[39])]
            tflgRZ43 = [float(ldata[40])]
            tflgRZ33 = [float(ldata[42])]
            tflgamHtot = [float(ldata[62])]
            tflgamHtotata = [float(ldata[63])]
            tflgamHtobb = [float(ldata[64])]
            tflgamHtot4 = [float(ldata[65])]
            tflgamHtogg = [float(ldata[66])]
            tflgamHtohh = [float(ldata[67])]
            tflgamHtocc = [float(ldata[68])]
            tflgamHtodi = [float(ldata[69])]
            tflgamHtott = [float(ldata[70])]
            tflkaT = [float(ldata[48])]
            tflggHratio = [float(ldata[75])]
            tfltotalratio = [float(ldata[79])]
            tflprodratiobr = [float(ldata[74])*float(ldata[79])]
            tfllamSMt44 = [-float(ldata[10])*float(ldata[1])]
            tflggSMratio = [float(ldata[76])]
            tflbbFratio = [float(ldata[79]) - float(ldata[75])]
            tflbrHtotata = [float(ldata[80])]
            tflbrb4w = [float(ldata[83])]
            tflbrb4z = [float(ldata[84])]
            tflbrb4h = [float(ldata[85])]
            #tfltotalxs = [float(ldata[79])*sigmaHSM13TeV(float(ldata[3]))]      
            #14TeV
            tfltotalxs = [float(ldata[75])*sigmaHSM14TeV(float(ldata[3])) + ((float(ldata[1]))**2)*sigmabbF14(float(ldata[3]))]
            tflistmt4.extend(tflmt4)
            tflisttanbe.extend(tfltanbe)
            tflistlamHt34.extend(tfllamHt34)
            tflistmHH.extend(tflmHH)
            tflistlamHrt34.extend(tfllamHrt34)
            tflistlamHt44.extend(tfllamHt44)
            tflistlamHrt44.extend(tfllamHrt44)
            tflistgLW43.extend(tflgLW43)
            tflistgLW33.extend(tflgLW33)
            tflistgLZ44.extend(tflgLZ44)
            tflistgLZ43.extend(tflgLZ43)
            tflistgLZ33.extend(tflgLZ33)
            tflistgRZ44.extend(tflgRZ44)
            tflistgRZ43.extend(tflgRZ43)
            tflistgRZ33.extend(tflgRZ33)
            tflistbrHtobb.extend(tflbrHtobb)
            tflistbrHtohh.extend(tflbrHtohh)
            tflistbrHtot4.extend(tflbrHtot4)
            tflistbrHtogg.extend(tflbrHtogg)
            tflistbrHtocc.extend(tflbrHtocc)
            tflistbrHtodi.extend(tflbrHtodi)
            tflistbrHtott.extend(tflbrHtott)
            tflistgamHtot.extend(tflgamHtot)
            tflistgamHtotata.extend(tflgamHtotata)
            tflistgamHtobb.extend(tflgamHtobb)
            tflistgamHtot4.extend(tflgamHtot4)
            tflistgamHtogg.extend(tflgamHtogg)
            tflistgamHtohh.extend(tflgamHtohh)
            tflistgamHtocc.extend(tflgamHtocc)
            tflistgamHtodi.extend(tflgamHtodi)
            tflistgamHtott.extend(tflgamHtott)
            tflistbrt4w.extend(tflbrt4w)
            tflistbrt4z.extend(tflbrt4z)
            tflistbrt4h.extend(tflbrt4h)
            tflistbrour.extend(tflbrour)
            tflistkaT.extend(tflkaT)
            tflistggHratio.extend(tflggHratio)
            tflisttotalratio.extend(tfltotalratio)
            tflistprodratiobr.extend(tflprodratiobr)
            tflistlamSMt44.extend(tfllamSMt44)
            tflistggSMratio.extend(tflggSMratio)
            tflistbbFratio.extend(tflbbFratio)
            tflistbrHtotata.extend(tflbrHtotata)
            tflistbrourb4.extend(tflbrourb4)
            tflistsigBRw.extend(tflsigBRw)
            tflistsigBRz.extend(tflsigBRz)
            tflistbrb4w.extend(tflbrb4w)
            tflistbrb4z.extend(tflbrb4z)
            tflistbrb4h.extend(tflbrb4h)
            tflisttotalxs.extend(tfltotalxs)
            tflistsigb4BRw.extend(tflsigb4BRw)
            tflistsigb4BRz.extend(tflsigb4BRz)
        if float(ldata[1]) > 20.:# and float(ldata[0]) > 1000 and float(ldata[0]) < 1400: #tan(beta) and mt4 1.2 TeV
        #if float(ldata[1]) > 48 and float(ldata[1]) < 50.1:# and float(ldata[0]) > 1000 and float(ldata[0]) < 1400: #tan(beta) and mt4 1.2 TeV
            tglbrHtobb = [float(ldata[55])]
            tglbrHtohh = [float(ldata[56])]
            tglbrHtot4 = [float(ldata[57])]
            tglbrHtogg = [float(ldata[58])]
            tglbrHtocc = [float(ldata[59])]
            tglbrHtodi = [float(ldata[60])]
            tglbrHtott = [float(ldata[61])]
            tgltanbe = [float(ldata[1])]
            tglmt4 = [float(ldata[0])]
            tglmHH = [float(ldata[3])]
            tglbrt4w = [float(ldata[71])]
            tglbrt4z = [float(ldata[72])]
            tglbrt4h = [float(ldata[73])]
            tgllamHt44 = [float(ldata[10])]
            #tglbrour = [float(ldata[74])]        
            #tglbrour = [float(ldata[79])*float(ldata[57])*float(ldata[73])]  # (sigma_H / sigma_H^SM)BR(H -> t4 t)*BR(t4 -> t h)
            #tglsigBRw = [float(ldata[79])*float(ldata[57])*float(ldata[71])]  # (sigma_H / sigma_H^SM)BR(H -> t4 t)*BR(t4 -> b W)
            #tglsigBRz = [float(ldata[79])*float(ldata[57])*float(ldata[72])]  # (sigma_H / sigma_H^SM)BR(H -> t4 t)*BR(t4 -> t Z)
            #tglbrourb4 = [float(ldata[92])*float(ldata[85])] # BR(H -> b4 b)*BR(b4 -> th)
            # 14 TeV
            tglbrour = [(float(ldata[75]) + ((float(ldata[1]))**2)*sigmabbF14(float(ldata[3]))/sigmaHSM14TeV(float(ldata[3])))*float(ldata[57])*float(ldata[73])]
            tglsigBRw = [(float(ldata[75]) + ((float(ldata[1]))**2)*sigmabbF14(float(ldata[3]))/sigmaHSM14TeV(float(ldata[3])))*float(ldata[57])*float(ldata[71])]
            tglsigBRz = [(float(ldata[75]) + ((float(ldata[1]))**2)*sigmabbF14(float(ldata[3]))/sigmaHSM14TeV(float(ldata[3])))*float(ldata[57])*float(ldata[72])]
            tglbrourb4 = [(float(ldata[75]) + ((float(ldata[1]))**2)*sigmabbF14(float(ldata[3]))/sigmaHSM14TeV(float(ldata[3])))*float(ldata[92])*float(ldata[85])]
            tglsigb4BRw = [(float(ldata[75]) + ((float(ldata[1]))**2)*sigmabbF14(float(ldata[3]))/sigmaHSM14TeV(float(ldata[3])))*float(ldata[92])*float(ldata[83])]
            tglsigb4BRz = [(float(ldata[75]) + ((float(ldata[1]))**2)*sigmabbF14(float(ldata[3]))/sigmaHSM14TeV(float(ldata[3])))*float(ldata[92])*float(ldata[84])]
            tgllamHt34 = [float(ldata[2])]
            tgllamHrt34 = [float(ldata[9])]
            tgllamHt44 = [float(ldata[10])]
            tgllamHrt44 = [float(ldata[11])]
            tglgLW43 = [float(ldata[13])]
            tglgLW33 = [float(ldata[15])]
            tglgLZ44 = [float(ldata[30])]
            tglgLZ43 = [float(ldata[31])]
            tglgLZ33 = [float(ldata[33])]
            tglgRZ44 = [float(ldata[39])]
            tglgRZ43 = [float(ldata[40])]
            tglgRZ33 = [float(ldata[42])]
            tglgamHtot = [float(ldata[62])]
            tglgamHtotata = [float(ldata[63])]
            tglgamHtobb = [float(ldata[64])]
            tglgamHtot4 = [float(ldata[65])]
            tglgamHtogg = [float(ldata[66])]
            tglgamHtohh = [float(ldata[67])]
            tglgamHtocc = [float(ldata[68])]
            tglgamHtodi = [float(ldata[69])]
            tglgamHtott = [float(ldata[70])]
            tglkaT = [float(ldata[48])]
            tglggHratio = [float(ldata[75])]
            tgltotalratio = [float(ldata[79])]
            tglprodratiobr = [float(ldata[74])*float(ldata[79])]
            tgllamSMt44 = [-float(ldata[10])*float(ldata[1])]
            tglggSMratio = [float(ldata[76])]
            tglbbFratio = [float(ldata[79]) - float(ldata[75])]
            tglbrHtotata = [float(ldata[80])]
            tglbrb4w = [float(ldata[83])]
            tglbrb4z = [float(ldata[84])]
            tglbrb4h = [float(ldata[85])]
            #tgltotalxs = [float(ldata[79])*sigmaHSM13TeV(float(ldata[3]))]
            #14TeV
            tgltotalxs = [float(ldata[75])*sigmaHSM14TeV(float(ldata[3])) + ((float(ldata[1]))**2)*sigmabbF14(float(ldata[3]))]
            tglxs2HDM50 = [float(ldata[-1])*sigmaHSM14TeV(float(ldata[3])) + (50**2)*sigmabbF14(float(ldata[3]))]
            tglistmt4.extend(tglmt4)
            tglisttanbe.extend(tgltanbe)
            tglistlamHt34.extend(tgllamHt34)
            tglistmHH.extend(tglmHH)
            tglistlamHrt34.extend(tgllamHrt34)
            tglistlamHt44.extend(tgllamHt44)
            tglistlamHrt44.extend(tgllamHrt44)
            tglistgLW43.extend(tglgLW43)
            tglistgLW33.extend(tglgLW33)
            tglistgLZ44.extend(tglgLZ44)
            tglistgLZ43.extend(tglgLZ43)
            tglistgLZ33.extend(tglgLZ33)
            tglistgRZ44.extend(tglgRZ44)
            tglistgRZ43.extend(tglgRZ43)
            tglistgRZ33.extend(tglgRZ33)
            tglistbrHtobb.extend(tglbrHtobb)
            tglistbrHtohh.extend(tglbrHtohh)
            tglistbrHtot4.extend(tglbrHtot4)
            tglistbrHtogg.extend(tglbrHtogg)
            tglistbrHtocc.extend(tglbrHtocc)
            tglistbrHtodi.extend(tglbrHtodi)
            tglistbrHtott.extend(tglbrHtott)
            tglistgamHtot.extend(tglgamHtot)
            tglistgamHtotata.extend(tglgamHtotata)
            tglistgamHtobb.extend(tglgamHtobb)
            tglistgamHtot4.extend(tglgamHtot4)
            tglistgamHtogg.extend(tglgamHtogg)
            tglistgamHtohh.extend(tglgamHtohh)
            tglistgamHtocc.extend(tglgamHtocc)
            tglistgamHtodi.extend(tglgamHtodi)
            tglistgamHtott.extend(tglgamHtott)
            tglistbrt4w.extend(tglbrt4w)
            tglistbrt4z.extend(tglbrt4z)
            tglistbrt4h.extend(tglbrt4h)
            tglistbrour.extend(tglbrour)
            tglistkaT.extend(tglkaT)
            tglistggHratio.extend(tglggHratio)
            tglisttotalratio.extend(tgltotalratio)
            tglistprodratiobr.extend(tglprodratiobr)
            tglistlamSMt44.extend(tgllamSMt44)
            tglistggSMratio.extend(tglggSMratio)
            tglistbbFratio.extend(tglbbFratio)
            tglistbrHtotata.extend(tglbrHtotata)
            tglistbrourb4.extend(tglbrourb4)
            tglistsigBRw.extend(tglsigBRw)
            tglistsigBRz.extend(tglsigBRz)
            tglistbrb4w.extend(tglbrb4w)
            tglistbrb4z.extend(tglbrb4z)
            tglistbrb4h.extend(tglbrb4h)
            tglisttotalxs.extend(tgltotalxs)
            tglistxs2HDM50.extend(tglxs2HDM50)
            tglistsigb4BRw.extend(tglsigb4BRw)
            tglistsigb4BRz.extend(tglsigb4BRz)
            

            
        #float(ldata[3]) < 2050 and float(ldata[3]) > 1950: #mHH ~ 2TeV
        #if float(ldata[5]) > 0.99: #singlet fraction > 99%
        #if float(ldata[0]) < 1505. and float(ldata[0]) > 1495.: #mt4 
        #if abs(float(ldata[10])) < 1. and abs(float(ldata[11])) < 1. and abs(float(ldata[107])) < 0.001 and abs(float(ldata[108])) < 0.001:# and abs(float(ldata[77])) < 0.05 and abs(float(ldata[78])) < 0.05: #lamHt44, lamHrt44, lamHb44, lamHrb44, lamHt55, lamHrt55
        if abs(float(ldata[10])) < 0.001 and abs(float(ldata[11])) < 0.001 and abs(float(ldata[77])) < 0.05 and abs(float(ldata[78])) < 0.05: #lamHt44, lamHrt44, lamHt55, lamHrt55
        #if abs(float(ldata[10])) > 0.1 or abs(float(ldata[11])) > 0.1 or abs(float(ldata[77])) < 0.1 or abs(float(ldata[78])) < 0.1: #lamHt44, lamHrt44, lamHt55, lamHrt55
            tdlbrHtobb = [float(ldata[55])]
            tdlbrHtohh = [float(ldata[56])]
            tdlbrHtot4 = [float(ldata[57])]
            tdlbrHtogg = [float(ldata[58])]
            tdlbrHtocc = [float(ldata[59])]
            tdlbrHtodi = [float(ldata[60])]
            tdlbrHtott = [float(ldata[61])]
            tdltanbe = [float(ldata[1])]
            tdlmt4 = [float(ldata[0])]
            tdlmHH = [float(ldata[3])]
            tdlbrt4w = [float(ldata[71])]
            tdlbrt4z = [float(ldata[72])]
            tdlbrt4h = [float(ldata[73])]
            #tdllamHt44 = [float(ldata[10])]
            tdlbrour = [float(ldata[74])]        
            tdllamHt34 = [float(ldata[2])]
            tdllamHrt34 = [float(ldata[9])]
            tdllamHt44 = [float(ldata[10])]
            tdllamHrt44 = [float(ldata[11])]
            tdlgLW43 = [float(ldata[13])]
            tdlgLW33 = [float(ldata[15])]
            tdlgLZ44 = [float(ldata[30])]
            tdlgLZ43 = [float(ldata[31])]
            tdlgLZ33 = [float(ldata[33])]
            tdlgRZ44 = [float(ldata[39])]
            tdlgRZ43 = [float(ldata[40])]
            tdlgRZ33 = [float(ldata[42])]
            tdlgamHtot = [float(ldata[62])]
            tdlgamHtotata = [float(ldata[63])]
            tdlgamHtobb = [float(ldata[64])]
            tdlgamHtot4 = [float(ldata[65])]
            tdlgamHtogg = [float(ldata[66])]
            tdlgamHtohh = [float(ldata[67])]
            tdlgamHtocc = [float(ldata[68])]
            tdlgamHtodi = [float(ldata[69])]
            tdlgamHtott = [float(ldata[70])]
            tdlkaT = [float(ldata[48])]
            tdlggHratio = [float(ldata[75])]
            tdltotalratio = [float(ldata[79])]
            tdlprodratiobr = [float(ldata[74])*float(ldata[79])]
            tdllamSMt44 = [-float(ldata[10])*float(ldata[1])]
            tdlggSMratio = [float(ldata[76])]
            tdlbbFratio = [float(ldata[79]) - float(ldata[75])]
            tdlbrHtotata = [float(ldata[80])]
            tdllamh43 = [float(ldata[9])*float(ldata[1])]#\lambda_h^{t4 t}
            tdllamh34 = [float(ldata[2])*float(ldata[1])]#\lambda_h^{t t4}
            tdlkaQ = [float(ldata[49])]
            tdlka = [float(ldata[50])]
            tdlkaba = [float(ldata[51])]
            #tdltotalxs = [float(ldata[79])*sigmaHSM13TeV(float(ldata[3]))]
            #14TeV
            tdltotalxs = [float(ldata[75])*sigmaHSM14TeV(float(ldata[3])) + ((float(ldata[1]))**2)*sigmabbF14(float(ldata[3]))]
            tdlistmt4.extend(tdlmt4)
            tdlisttanbe.extend(tdltanbe)
            tdlistlamHt34.extend(tdllamHt34)
            tdlistmHH.extend(tdlmHH)
            tdlistlamHrt34.extend(tdllamHrt34)
            tdlistlamHt44.extend(tdllamHt44)
            tdlistlamHrt44.extend(tdllamHrt44)
            tdlistgLW43.extend(tdlgLW43)
            tdlistgLW33.extend(tdlgLW33)
            tdlistgLZ44.extend(tdlgLZ44)
            tdlistgLZ43.extend(tdlgLZ43)
            tdlistgLZ33.extend(tdlgLZ33)
            tdlistgRZ44.extend(tdlgRZ44)
            tdlistgRZ43.extend(tdlgRZ43)
            tdlistgRZ33.extend(tdlgRZ33)
            tdlistbrHtobb.extend(tdlbrHtobb)
            tdlistbrHtohh.extend(tdlbrHtohh)
            tdlistbrHtot4.extend(tdlbrHtot4)
            tdlistbrHtogg.extend(tdlbrHtogg)
            tdlistbrHtocc.extend(tdlbrHtocc)
            tdlistbrHtodi.extend(tdlbrHtodi)
            tdlistbrHtott.extend(tdlbrHtott)
            tdlistgamHtot.extend(tdlgamHtot)
            tdlistgamHtotata.extend(tdlgamHtotata)
            tdlistgamHtobb.extend(tdlgamHtobb)
            tdlistgamHtot4.extend(tdlgamHtot4)
            tdlistgamHtogg.extend(tdlgamHtogg)
            tdlistgamHtohh.extend(tdlgamHtohh)
            tdlistgamHtocc.extend(tdlgamHtocc)
            tdlistgamHtodi.extend(tdlgamHtodi)
            tdlistgamHtott.extend(tdlgamHtott)
            tdlistbrt4w.extend(tdlbrt4w)
            tdlistbrt4z.extend(tdlbrt4z)
            tdlistbrt4h.extend(tdlbrt4h)
            tdlistbrour.extend(tdlbrour)
            tdlistkaT.extend(tdlkaT)
            tdlistggHratio.extend(tdlggHratio)
            tdlisttotalratio.extend(tdltotalratio)
            tdlistprodratiobr.extend(tdlprodratiobr)
            tdlistlamSMt44.extend(tdllamSMt44)
            tdlistggSMratio.extend(tdlggSMratio)
            tdlistbbFratio.extend(tdlbbFratio)
            tdlistbrHtotata.extend(tdlbrHtotata)
            tdlistlamh43.extend(tdllamh43)
            tdlistlamh34.extend(tdllamh34)
            tdlistkaQ.extend(tdlkaQ)
            tdlistka.extend(tdlka)
            tdlistkaba.extend(tdlkaba)
            tdlisttotalxs.extend(tdltotalxs)
            #if float(ldata[9])*float(ldata[1]) > 0.02:
            if float(ldata[73]) > 0.3 and float(ldata[73]) < 0.5:# and float(ldata[73]) < 0.26: #BR(t4->th)
            #if float(ldata[57]) < 0.1: #BR(H->t4t)
                tdalbrHtobb = [float(ldata[55])]
                tdalbrHtohh = [float(ldata[56])]
                tdalbrHtot4 = [float(ldata[57])]
                tdalbrHtogg = [float(ldata[58])]
                tdalbrHtocc = [float(ldata[59])]
                tdalbrHtodi = [float(ldata[60])]
                tdalbrHtott = [float(ldata[61])]
                tdaltanbe = [float(ldata[1])]
                tdalmt4 = [float(ldata[0])]
                tdalmHH = [float(ldata[3])]
                tdalbrt4w = [float(ldata[71])]
                tdalbrt4z = [float(ldata[72])]
                tdalbrt4h = [float(ldata[73])]
                tdallamHt44 = [float(ldata[10])]
                tdalbrour = [float(ldata[74])]        
                tdallamHt34 = [float(ldata[2])]
                tdallamHrt34 = [float(ldata[9])]
                tdallamh43 = [float(ldata[9])*float(ldata[1])]#\lambda_h^{t4 t}
                tdallamh34 = [float(ldata[2])*float(ldata[1])]#\lambda_h^{t t4}
                tdallamHt44 = [float(ldata[10])]
                tdallamHrt44 = [float(ldata[11])]
                tdalgLW43 = [float(ldata[13])]
                tdalgLW33 = [float(ldata[15])]
                tdalgLZ44 = [float(ldata[30])]
                tdalgLZ43 = [float(ldata[31])]
                tdalgLZ33 = [float(ldata[33])]
                tdalgRZ44 = [float(ldata[39])]
                tdalgRZ43 = [float(ldata[40])]
                tdalgRZ33 = [float(ldata[42])]
                tdalgamHtot = [float(ldata[62])]
                tdalgamHtotata = [float(ldata[63])]
                tdalgamHtobb = [float(ldata[64])]
                tdalgamHtot4 = [float(ldata[65])]
                tdalgamHtogg = [float(ldata[66])]
                tdalgamHtohh = [float(ldata[67])]
                tdalgamHtocc = [float(ldata[68])]
                tdalgamHtodi = [float(ldata[69])]
                tdalgamHtott = [float(ldata[70])]
                tdalkaT = [float(ldata[48])]
                tdalggHratio = [float(ldata[75])]
                tdaltotalratio = [float(ldata[79])]
                tdalprodratiobr = [float(ldata[74])*float(ldata[79])]
                tdallamSMt44 = [-float(ldata[10])*float(ldata[1])]
                tdalggSMratio = [float(ldata[76])]
                tdalbbFratio = [float(ldata[79]) - float(ldata[75])]
                tdalbrHtotata = [float(ldata[80])]
                tdalsingfr = [float(ldata[5])]
                tdalmb4 = [float(ldata[137])]
                tdalgRW43t = [float(ldata[22])]
                tdalgamt4toh = [float(ldata[147])]
                tdalgamt4tow = [float(ldata[148])]
                tdalMTT = [float(ldata[53])]
                tdalMQQ = [float(ldata[54])]
                tdalcheck = [float(ldata[-5])]
                tdaltest1 = [float(ldata[-4])]
                tdaltest2 = [float(ldata[-3])]
                tdaltest3 = [float(ldata[-2])]
                tdaltest4 = [float(ldata[-1])]
                tdalkaQ = [float(ldata[49])]
                tdalka = [float(ldata[50])]
                tdalkaba = [float(ldata[51])]
                tdalistmt4.extend(tdalmt4)
                tdalisttanbe.extend(tdaltanbe)
                tdalistlamHt34.extend(tdallamHt34)
                tdalistmHH.extend(tdalmHH)
                tdalistlamHrt34.extend(tdallamHrt34)
                tdalistlamHt44.extend(tdallamHt44)
                tdalistlamHrt44.extend(tdallamHrt44)
                tdalistgLW43.extend(tdalgLW43)
                tdalistgLW33.extend(tdalgLW33)
                tdalistgLZ44.extend(tdalgLZ44)
                tdalistgLZ43.extend(tdalgLZ43)
                tdalistgLZ33.extend(tdalgLZ33)
                tdalistgRZ44.extend(tdalgRZ44)
                tdalistgRZ43.extend(tdalgRZ43)
                tdalistgRZ33.extend(tdalgRZ33)
                tdalistbrHtobb.extend(tdalbrHtobb)
                tdalistbrHtohh.extend(tdalbrHtohh)
                tdalistbrHtot4.extend(tdalbrHtot4)
                tdalistbrHtogg.extend(tdalbrHtogg)
                tdalistbrHtocc.extend(tdalbrHtocc)
                tdalistbrHtodi.extend(tdalbrHtodi)
                tdalistbrHtott.extend(tdalbrHtott)
                tdalistgamHtot.extend(tdalgamHtot)
                tdalistgamHtotata.extend(tdalgamHtotata)
                tdalistgamHtobb.extend(tdalgamHtobb)
                tdalistgamHtot4.extend(tdalgamHtot4)
                tdalistgamHtogg.extend(tdalgamHtogg)
                tdalistgamHtohh.extend(tdalgamHtohh)
                tdalistgamHtocc.extend(tdalgamHtocc)
                tdalistgamHtodi.extend(tdalgamHtodi)
                tdalistgamHtott.extend(tdalgamHtott)
                tdalistbrt4w.extend(tdalbrt4w)
                tdalistbrt4z.extend(tdalbrt4z)
                tdalistbrt4h.extend(tdalbrt4h)
                tdalistbrour.extend(tdalbrour)
                tdalistkaT.extend(tdalkaT)
                tdalistggHratio.extend(tdalggHratio)
                tdalisttotalratio.extend(tdaltotalratio)
                tdalistprodratiobr.extend(tdalprodratiobr)
                tdalistlamSMt44.extend(tdallamSMt44)
                tdalistggSMratio.extend(tdalggSMratio)
                tdalistbbFratio.extend(tdalbbFratio)
                tdalistbrHtotata.extend(tdalbrHtotata)
                tdalistsingfr.extend(tdalsingfr)
                tdalistmb4.extend(tdalmb4)
                tdalistgRW43t.extend(tdalgRW43t)
                tdalistgamt4toh.extend(tdalgamt4toh)
                tdalistgamt4tow.extend(tdalgamt4tow)
                tdalistMTT.extend(tdalMTT)
                tdalistMQQ.extend(tdalMQQ)
                tdalistcheck.extend(tdalcheck)
                tdalisttest1.extend(tdaltest1)
                tdalisttest2.extend(tdaltest2)
                tdalisttest3.extend(tdaltest3)
                tdalisttest4.extend(tdaltest4)
                tdalistlamh43.extend(tdallamh43)
                tdalistlamh34.extend(tdallamh34)
                tdalistkaQ.extend(tdalkaQ)
                tdalistka.extend(tdalka)
                tdalistkaba.extend(tdalkaba)
            if float(ldata[73]) > 0.5: #BR(t4 -> h t)
            #if float(ldata[57]) < 0.2 and float(ldata[57]) >= 0.1: #BR(H->t4t)
                tdblbrHtobb = [float(ldata[55])]
                tdblbrHtohh = [float(ldata[56])]
                tdblbrHtot4 = [float(ldata[57])]
                tdblbrHtogg = [float(ldata[58])]
                tdblbrHtocc = [float(ldata[59])]
                tdblbrHtodi = [float(ldata[60])]
                tdblbrHtott = [float(ldata[61])]
                tdbltanbe = [float(ldata[1])]
                tdblmt4 = [float(ldata[0])]
                tdblmHH = [float(ldata[3])]
                tdblbrt4w = [float(ldata[71])]
                tdblbrt4z = [float(ldata[72])]
                tdblbrt4h = [float(ldata[73])]
                tdbllamHt44 = [float(ldata[10])]
                tdblbrour = [float(ldata[74])]        
                tdbllamHt34 = [float(ldata[2])]
                tdbllamHrt34 = [float(ldata[9])]
                tdbllamHt44 = [float(ldata[10])]
                tdbllamHrt44 = [float(ldata[11])]
                tdblgLW43 = [float(ldata[13])]
                tdblgLW33 = [float(ldata[15])]
                tdblgLZ44 = [float(ldata[30])]
                tdblgLZ43 = [float(ldata[31])]
                tdblgLZ33 = [float(ldata[33])]
                tdblgRZ44 = [float(ldata[39])]
                tdblgRZ43 = [float(ldata[40])]
                tdblgRZ33 = [float(ldata[42])]
                tdblgamHtot = [float(ldata[62])]
                tdblgamHtotata = [float(ldata[63])]
                tdblgamHtobb = [float(ldata[64])]
                tdblgamHtot4 = [float(ldata[65])]
                tdblgamHtogg = [float(ldata[66])]
                tdblgamHtohh = [float(ldata[67])]
                tdblgamHtocc = [float(ldata[68])]
                tdblgamHtodi = [float(ldata[69])]
                tdblgamHtott = [float(ldata[70])]
                tdblkaT = [float(ldata[48])]
                tdblggHratio = [float(ldata[75])]
                tdbltotalratio = [float(ldata[79])]
                tdblprodratiobr = [float(ldata[74])*float(ldata[79])]
                tdbllamSMt44 = [-float(ldata[10])*float(ldata[1])]
                tdblggSMratio = [float(ldata[76])]
                tdblbbFratio = [float(ldata[79]) - float(ldata[75])]
                tdblbrHtotata = [float(ldata[80])]
                tdbllamh43 = [float(ldata[9])*float(ldata[1])]#\lambda_h^{t4 t}
                tdbllamh34 = [float(ldata[2])*float(ldata[1])]#\lambda_h^{t t4}
                tdblkaQ = [float(ldata[49])]
                tdblka = [float(ldata[50])]
                tdblkaba = [float(ldata[51])]
                tdblistmt4.extend(tdblmt4)
                tdblisttanbe.extend(tdbltanbe)
                tdblistlamHt34.extend(tdbllamHt34)
                tdblistmHH.extend(tdblmHH)
                tdblistlamHrt34.extend(tdbllamHrt34)
                tdblistlamHt44.extend(tdbllamHt44)
                tdblistlamHrt44.extend(tdbllamHrt44)
                tdblistgLW43.extend(tdblgLW43)
                tdblistgLW33.extend(tdblgLW33)
                tdblistgLZ44.extend(tdblgLZ44)
                tdblistgLZ43.extend(tdblgLZ43)
                tdblistgLZ33.extend(tdblgLZ33)
                tdblistgRZ44.extend(tdblgRZ44)
                tdblistgRZ43.extend(tdblgRZ43)
                tdblistgRZ33.extend(tdblgRZ33)
                tdblistbrHtobb.extend(tdblbrHtobb)
                tdblistbrHtohh.extend(tdblbrHtohh)
                tdblistbrHtot4.extend(tdblbrHtot4)
                tdblistbrHtogg.extend(tdblbrHtogg)
                tdblistbrHtocc.extend(tdblbrHtocc)
                tdblistbrHtodi.extend(tdblbrHtodi)
                tdblistbrHtott.extend(tdblbrHtott)
                tdblistgamHtot.extend(tdblgamHtot)
                tdblistgamHtotata.extend(tdblgamHtotata)
                tdblistgamHtobb.extend(tdblgamHtobb)
                tdblistgamHtot4.extend(tdblgamHtot4)
                tdblistgamHtogg.extend(tdblgamHtogg)
                tdblistgamHtohh.extend(tdblgamHtohh)
                tdblistgamHtocc.extend(tdblgamHtocc)
                tdblistgamHtodi.extend(tdblgamHtodi)
                tdblistgamHtott.extend(tdblgamHtott)
                tdblistbrt4w.extend(tdblbrt4w)
                tdblistbrt4z.extend(tdblbrt4z)
                tdblistbrt4h.extend(tdblbrt4h)
                tdblistbrour.extend(tdblbrour)
                tdblistkaT.extend(tdblkaT)
                tdblistggHratio.extend(tdblggHratio)
                tdblisttotalratio.extend(tdbltotalratio)
                tdblistprodratiobr.extend(tdblprodratiobr)
                tdblistlamSMt44.extend(tdbllamSMt44)
                tdblistggSMratio.extend(tdblggSMratio)
                tdblistbbFratio.extend(tdblbbFratio)
                tdblistbrHtotata.extend(tdblbrHtotata)
                tdblistlamh34.extend(tdbllamh34)
                tdblistlamh43.extend(tdbllamh43)
                tdblistkaQ.extend(tdblkaQ)
                tdblistka.extend(tdblka)
                tdblistkaba.extend(tdblkaba)
            #if float(ldata[57]) < 0.3 and float(ldata[57]) >= 0.2: #BR(H->t4t)
            if float(ldata[72]) > 0.5: #BR(t4 -> t z)
                tdclbrHtobb = [float(ldata[55])]
                tdclbrHtohh = [float(ldata[56])]
                tdclbrHtot4 = [float(ldata[57])]
                tdclbrHtogg = [float(ldata[58])]
                tdclbrHtocc = [float(ldata[59])]
                tdclbrHtodi = [float(ldata[60])]
                tdclbrHtott = [float(ldata[61])]
                tdcltanbe = [float(ldata[1])]
                tdclmt4 = [float(ldata[0])]
                tdclmHH = [float(ldata[3])]
                tdclbrt4w = [float(ldata[71])]
                tdclbrt4z = [float(ldata[72])]
                tdclbrt4h = [float(ldata[73])]
                tdcllamHt44 = [float(ldata[10])]
                tdclbrour = [float(ldata[74])]        
                tdcllamHt34 = [float(ldata[2])]
                tdcllamHrt34 = [float(ldata[9])]
                tdcllamHt44 = [float(ldata[10])]
                tdcllamHrt44 = [float(ldata[11])]
                tdclgLW43 = [float(ldata[13])]
                tdclgLW33 = [float(ldata[15])]
                tdclgLZ44 = [float(ldata[30])]
                tdclgLZ43 = [float(ldata[31])]
                tdclgLZ33 = [float(ldata[33])]
                tdclgRZ44 = [float(ldata[39])]
                tdclgRZ43 = [float(ldata[40])]
                tdclgRZ33 = [float(ldata[42])]
                tdclgamHtot = [float(ldata[62])]
                tdclgamHtotata = [float(ldata[63])]
                tdclgamHtobb = [float(ldata[64])]
                tdclgamHtot4 = [float(ldata[65])]
                tdclgamHtogg = [float(ldata[66])]
                tdclgamHtohh = [float(ldata[67])]
                tdclgamHtocc = [float(ldata[68])]
                tdclgamHtodi = [float(ldata[69])]
                tdclgamHtott = [float(ldata[70])]
                tdclkaT = [float(ldata[48])]
                tdclggHratio = [float(ldata[75])]
                tdcltotalratio = [float(ldata[79])]
                tdclprodratiobr = [float(ldata[74])*float(ldata[79])]
                tdcllamSMt44 = [-float(ldata[10])*float(ldata[1])]
                tdclggSMratio = [float(ldata[76])]
                tdclbbFratio = [float(ldata[79]) - float(ldata[75])]
                tdclbrHtotata = [float(ldata[80])]
                tdcllamh43 = [float(ldata[9])*float(ldata[1])]#\lambda_h^{t4 t}
                tdcllamh34 = [float(ldata[2])*float(ldata[1])]#\lambda_h^{t t4}
                tdclistmt4.extend(tdclmt4)
                tdclisttanbe.extend(tdcltanbe)
                tdclistlamHt34.extend(tdcllamHt34)
                tdclistmHH.extend(tdclmHH)
                tdclistlamHrt34.extend(tdcllamHrt34)
                tdclistlamHt44.extend(tdcllamHt44)
                tdclistlamHrt44.extend(tdcllamHrt44)
                tdclistgLW43.extend(tdclgLW43)
                tdclistgLW33.extend(tdclgLW33)
                tdclistgLZ44.extend(tdclgLZ44)
                tdclistgLZ43.extend(tdclgLZ43)
                tdclistgLZ33.extend(tdclgLZ33)
                tdclistgRZ44.extend(tdclgRZ44)
                tdclistgRZ43.extend(tdclgRZ43)
                tdclistgRZ33.extend(tdclgRZ33)
                tdclistbrHtobb.extend(tdclbrHtobb)
                tdclistbrHtohh.extend(tdclbrHtohh)
                tdclistbrHtot4.extend(tdclbrHtot4)
                tdclistbrHtogg.extend(tdclbrHtogg)
                tdclistbrHtocc.extend(tdclbrHtocc)
                tdclistbrHtodi.extend(tdclbrHtodi)
                tdclistbrHtott.extend(tdclbrHtott)
                tdclistgamHtot.extend(tdclgamHtot)
                tdclistgamHtotata.extend(tdclgamHtotata)
                tdclistgamHtobb.extend(tdclgamHtobb)
                tdclistgamHtot4.extend(tdclgamHtot4)
                tdclistgamHtogg.extend(tdclgamHtogg)
                tdclistgamHtohh.extend(tdclgamHtohh)
                tdclistgamHtocc.extend(tdclgamHtocc)
                tdclistgamHtodi.extend(tdclgamHtodi)
                tdclistgamHtott.extend(tdclgamHtott)
                tdclistbrt4w.extend(tdclbrt4w)
                tdclistbrt4z.extend(tdclbrt4z)
                tdclistbrt4h.extend(tdclbrt4h)
                tdclistbrour.extend(tdclbrour)
                tdclistkaT.extend(tdclkaT)
                tdclistggHratio.extend(tdclggHratio)
                tdclisttotalratio.extend(tdcltotalratio)
                tdclistprodratiobr.extend(tdclprodratiobr)
                tdclistlamSMt44.extend(tdcllamSMt44)
                tdclistggSMratio.extend(tdclggSMratio)
                tdclistbbFratio.extend(tdclbbFratio)
                tdclistbrHtotata.extend(tdclbrHtotata)
                tdclistlamh43.extend(tdcllamh43)
                tdclistlamh34.extend(tdcllamh34)
            if float(ldata[57]) >= 0.3: #BR(H->t4t)
                tddlbrHtobb = [float(ldata[55])]
                tddlbrHtohh = [float(ldata[56])]
                tddlbrHtot4 = [float(ldata[57])]
                tddlbrHtogg = [float(ldata[58])]
                tddlbrHtocc = [float(ldata[59])]
                tddlbrHtodi = [float(ldata[60])]
                tddlbrHtott = [float(ldata[61])]
                tddltanbe = [float(ldata[1])]
                tddlmt4 = [float(ldata[0])]
                tddlmHH = [float(ldata[3])]
                tddlbrt4w = [float(ldata[71])]
                tddlbrt4z = [float(ldata[72])]
                tddlbrt4h = [float(ldata[73])]
                tddllamHt44 = [float(ldata[10])]
                tddlbrour = [float(ldata[74])]        
                tddllamHt34 = [float(ldata[2])]
                tddllamHrt34 = [float(ldata[9])]
                tddllamHt44 = [float(ldata[10])]
                tddllamHrt44 = [float(ldata[11])]
                tddlgLW43 = [float(ldata[13])]
                tddlgLW33 = [float(ldata[15])]
                tddlgLZ44 = [float(ldata[30])]
                tddlgLZ43 = [float(ldata[31])]
                tddlgLZ33 = [float(ldata[33])]
                tddlgRZ44 = [float(ldata[39])]
                tddlgRZ43 = [float(ldata[40])]
                tddlgRZ33 = [float(ldata[42])]
                tddlgamHtot = [float(ldata[62])]
                tddlgamHtotata = [float(ldata[63])]
                tddlgamHtobb = [float(ldata[64])]
                tddlgamHtot4 = [float(ldata[65])]
                tddlgamHtogg = [float(ldata[66])]
                tddlgamHtohh = [float(ldata[67])]
                tddlgamHtocc = [float(ldata[68])]
                tddlgamHtodi = [float(ldata[69])]
                tddlgamHtott = [float(ldata[70])]
                tddlkaT = [float(ldata[48])]
                tddlggHratio = [float(ldata[75])]
                tddltotalratio = [float(ldata[79])]
                tddlprodratiobr = [float(ldata[74])*float(ldata[79])]
                tddllamSMt44 = [-float(ldata[10])*float(ldata[1])]
                tddlggSMratio = [float(ldata[76])]
                tddlbbFratio = [float(ldata[79]) - float(ldata[75])]
                tddlbrHtotata = [float(ldata[80])]
                tddlistmt4.extend(tddlmt4)
                tddlisttanbe.extend(tddltanbe)
                tddlistlamHt34.extend(tddllamHt34)
                tddlistmHH.extend(tddlmHH)
                tddlistlamHrt34.extend(tddllamHrt34)
                tddlistlamHt44.extend(tddllamHt44)
                tddlistlamHrt44.extend(tddllamHrt44)
                tddlistgLW43.extend(tddlgLW43)
                tddlistgLW33.extend(tddlgLW33)
                tddlistgLZ44.extend(tddlgLZ44)
                tddlistgLZ43.extend(tddlgLZ43)
                tddlistgLZ33.extend(tddlgLZ33)
                tddlistgRZ44.extend(tddlgRZ44)
                tddlistgRZ43.extend(tddlgRZ43)
                tddlistgRZ33.extend(tddlgRZ33)
                tddlistbrHtobb.extend(tddlbrHtobb)
                tddlistbrHtohh.extend(tddlbrHtohh)
                tddlistbrHtot4.extend(tddlbrHtot4)
                tddlistbrHtogg.extend(tddlbrHtogg)
                tddlistbrHtocc.extend(tddlbrHtocc)
                tddlistbrHtodi.extend(tddlbrHtodi)
                tddlistbrHtott.extend(tddlbrHtott)
                tddlistgamHtot.extend(tddlgamHtot)
                tddlistgamHtotata.extend(tddlgamHtotata)
                tddlistgamHtobb.extend(tddlgamHtobb)
                tddlistgamHtot4.extend(tddlgamHtot4)
                tddlistgamHtogg.extend(tddlgamHtogg)
                tddlistgamHtohh.extend(tddlgamHtohh)
                tddlistgamHtocc.extend(tddlgamHtocc)
                tddlistgamHtodi.extend(tddlgamHtodi)
                tddlistgamHtott.extend(tddlgamHtott)
                tddlistbrt4w.extend(tddlbrt4w)
                tddlistbrt4z.extend(tddlbrt4z)
                tddlistbrt4h.extend(tddlbrt4h)
                tddlistbrour.extend(tddlbrour)
                tddlistkaT.extend(tddlkaT)
                tddlistggHratio.extend(tddlggHratio)
                tddlisttotalratio.extend(tddltotalratio)
                tddlistprodratiobr.extend(tddlprodratiobr)
                tddlistlamSMt44.extend(tddllamSMt44)
                tddlistggSMratio.extend(tddlggSMratio)
                tddlistbbFratio.extend(tddlbbFratio)
                tddlistbrHtotata.extend(tddlbrHtotata)
                
        #if float(ldata[4]) < 0.01:#t4 doublet-fraction < 5%: 95% singlet-like
        #if abs(float(ldata[48])) < 0.001*abs(float(ldata[49])) and abs(float(ldata[48])) < 0.001*abs(float(ldata[50])) and abs(float(ldata[48])) < 0.001*abs(float(ldata[51])) and float(ldata[4]) < 0.01:# and float(ldata[53]) > 0.5*float(ldata[54]):# #t4 doublet-fraction < 5%: 95% singlet-like
        #if float(ldata[4]) < 0.05:# #t4 doublet-fraction < 5%: 95% singlet-like
        if float(ldata[-16]) < 0.05:#b4 doublet-fraction < 5%: 95% singlet-like
            albrHtobb = [float(ldata[55])]
            albrHtohh = [float(ldata[56])]
            albrHtot4 = [float(ldata[57])]
            albrHtogg = [float(ldata[58])]
            albrHtocc = [float(ldata[59])]
            albrHtodi = [float(ldata[60])]
            albrHtott = [float(ldata[61])]
            albrHtotata = [float(ldata[80])]
            altanbe = [float(ldata[1])]
            almt4 = [float(ldata[0])]
            almHH = [float(ldata[3])]
            #if float(ldata[71]) + float(ldata[72]) + float(ldata[73]) == 0:
            albrt4w = [float(ldata[71])]
            albrt4z = [float(ldata[72])]
            albrt4h = [float(ldata[73])]
            #else:
                #albrt4w = [float(ldata[71])/(float(ldata[71]) + float(ldata[72]) + float(ldata[73]))]
                #albrt4z = [float(ldata[72])/(float(ldata[71]) + float(ldata[72]) + float(ldata[73]))]
                #albrt4h = [float(ldata[73])/(float(ldata[71]) + float(ldata[72]) + float(ldata[73]))]
            allamHt44 = [float(ldata[10])]
            albrour = [float(ldata[74])] #t4->ht  
            #albrour = [float(ldata[74])*float(ldata[71])] #t4->wb  
            #albrour = [float(ldata[57])*float(ldata[72])] #t4->zt  
            #albrour = [float(ldata[92])*float(ldata[85])] #b4->hb       
            #albrour = [float(ldata[92])*float(ldata[84])] #b4->zb       
            #albrour = [float(ldata[92])*float(ldata[83])] #b4->wt       
            allamHt34 = [float(ldata[2])]
            allamHrt34 = [float(ldata[9])]
            allamHt44 = [float(ldata[10])]
            allamHrt44 = [float(ldata[11])]
            algLW43 = [float(ldata[13])]
            algLW33 = [float(ldata[15])]
            algLZ44 = [float(ldata[30])]
            algLZ43 = [float(ldata[31])]
            algLZ33 = [float(ldata[33])]
            algRZ44 = [float(ldata[39])]
            algRZ43 = [float(ldata[40])]
            algRZ33 = [float(ldata[42])]
            algamHtot = [float(ldata[62])]
            algamHtotata = [float(ldata[63])]
            algamHtobb = [float(ldata[64])]
            algamHtot4 = [float(ldata[65])]
            algamHtogg = [float(ldata[66])]
            algamHtohh = [float(ldata[67])]
            algamHtocc = [float(ldata[68])]
            algamHtodi = [float(ldata[69])]
            algamHtott = [float(ldata[70])]
            alkaT = [float(ldata[48])]
            alggHratio = [float(ldata[75])]
            altotalratio = [float(ldata[79])]
            alprodratiobr = [float(ldata[74])*float(ldata[79])]
            allamSMt44 = [-float(ldata[10])*float(ldata[1])]
            alggSMratio = [float(ldata[76])]
            #almb4 = [float(ldata[137])]
            almb4 = [float(ldata[-20])]
            #if float(ldata[83]) + float(ldata[84]) + float(ldata[85]) == 0:
            albrb4w = [float(ldata[83])]
            albrb4z = [float(ldata[84])]
            albrb4h = [float(ldata[85])]
            #else:
                #albrb4w = [float(ldata[83])/(float(ldata[83]) + float(ldata[84]) + float(ldata[85]))]
                #albrb4z = [float(ldata[84])/(float(ldata[83]) + float(ldata[84]) + float(ldata[85]))]
                #albrb4h = [float(ldata[85])/(float(ldata[83]) + float(ldata[84]) + float(ldata[85]))]
            albrHtob4 = [float(ldata[92])]
            alMTT = [float(ldata[53])]
            albrb4t4w = [float(ldata[86])]
            albrt4b4w = [abs(float(ldata[81]))]
            #if float(ldata[-3]) + float(ldata[-2]) + float(ldata[-1]) == 0:
            albrt5z = [float(ldata[-3])]
            albrt5w = [float(ldata[-2])]
            albrt5h = [float(ldata[-1])]
            #else:
                #albrt5z = [float(ldata[-3])/(float(ldata[-3]) + float(ldata[-2]) + float(ldata[-1]))]
                #albrt5w = [float(ldata[-2])/(float(ldata[-3]) + float(ldata[-2]) + float(ldata[-1]))]
                #albrt5h = [float(ldata[-1])/(float(ldata[-3]) + float(ldata[-2]) + float(ldata[-1]))]
            albrt4HH = [float(ldata[-4])]
            albrb4HH = [float(ldata[-8])] 
            albrb4H0 = [float(ldata[-9])] # Only for mytest-s9
            alistmt4.extend(almt4)
            alisttanbe.extend(altanbe)
            alistlamHt34.extend(allamHt34)
            alistmHH.extend(almHH)
            alistlamHrt34.extend(allamHrt34)
            alistlamHt44.extend(allamHt44)
            alistlamHrt44.extend(allamHrt44)
            alistgLW43.extend(algLW43)
            alistgLW33.extend(algLW33)
            alistgLZ44.extend(algLZ44)
            alistgLZ43.extend(algLZ43)
            alistgLZ33.extend(algLZ33)
            alistgRZ44.extend(algRZ44)
            alistgRZ43.extend(algRZ43)
            alistgRZ33.extend(algRZ33)
            alistbrHtobb.extend(albrHtobb)
            alistbrHtohh.extend(albrHtohh)
            alistbrHtot4.extend(albrHtot4)
            alistbrHtogg.extend(albrHtogg)
            alistbrHtocc.extend(albrHtocc)
            alistbrHtodi.extend(albrHtodi)
            alistbrHtott.extend(albrHtott)
            alistgamHtot.extend(algamHtot)
            alistgamHtotata.extend(algamHtotata)
            alistgamHtobb.extend(algamHtobb)
            alistgamHtot4.extend(algamHtot4)
            alistgamHtogg.extend(algamHtogg)
            alistgamHtohh.extend(algamHtohh)
            alistgamHtocc.extend(algamHtocc)
            alistgamHtodi.extend(algamHtodi)
            alistgamHtott.extend(algamHtott)
            alistbrt4w.extend(albrt4w)
            alistbrt4z.extend(albrt4z)
            alistbrt4h.extend(albrt4h)
            alistbrour.extend(albrour)
            alistkaT.extend(alkaT)
            alistggHratio.extend(alggHratio)
            alisttotalratio.extend(altotalratio)
            alistprodratiobr.extend(alprodratiobr)
            alistlamSMt44.extend(allamSMt44)
            alistggSMratio.extend(alggSMratio)
            alistbrHtotata.extend(albrHtotata)
            alistmb4.extend(almb4)
            alistbrb4w.extend(albrb4w)
            alistbrb4z.extend(albrb4z)
            alistbrb4h.extend(albrb4h)
            alistbrHtob4.extend(albrHtob4)
            alistMTT.extend(alMTT)
            alistbrb4t4w.extend(albrb4t4w)
            alistbrt4b4w.extend(albrt4b4w)
            alistbrt5z.extend(albrt5z)
            alistbrt5w.extend(albrt5w)
            alistbrt5h.extend(albrt5h)
            alistbrt4HH.extend(albrt4HH)
            alistbrb4HH.extend(albrb4HH)
            alistbrb4H0.extend(albrb4H0)
        #if abs(float(ldata[48])) < 0.002*abs(float(ldata[49])) and abs(float(ldata[48])) < 0.002*abs(float(ldata[50])) and abs(float(ldata[48])) < 0.002*abs(float(ldata[51])) and float(ldata[4]) < 0.05:# and float(ldata[53]) > 0.5*float(ldata[54]):# #t4 doublet-f        #if float(ldata[0]) > 1450 and float(ldata[0]) < 1550 and float(ldata[4]) >= 0.05 and float(ldata[4]) < 0.5:#t4 doublet-fraction < 50%: singlet-like
        #if float(ldata[4]) >= 0.05 and float(ldata[4]) < 0.5:#t4 doublet-fraction < 50%: singlet-like
        if float(ldata[-16]) < 0.5 and float(ldata[-16]) >= 0.05:#b4 doublet-fraction < 50%: singlet-like
            blbrHtobb = [float(ldata[55])]
            blbrHtohh = [float(ldata[56])]
            blbrHtot4 = [float(ldata[57])]
            blbrHtogg = [float(ldata[58])]
            blbrHtocc = [float(ldata[59])]
            blbrHtodi = [float(ldata[60])]
            blbrHtott = [float(ldata[61])]
            blbrHtotata = [float(ldata[80])]
            bltanbe = [float(ldata[1])]
            blmt4 = [float(ldata[0])]
            blmHH = [float(ldata[3])]
            #blbrt4w = [float(ldata[71])]
            #blbrt4z = [float(ldata[72])]
            #blbrt4h = [float(ldata[73])]
            #if float(ldata[71]) + float(ldata[72]) + float(ldata[73]) == 0:
            blbrt4w = [float(ldata[71])]
            blbrt4z = [float(ldata[72])]
            blbrt4h = [float(ldata[73])]
            #else:
                #blbrt4w = [float(ldata[71])/(float(ldata[71]) + float(ldata[72]) + float(ldata[73]))]
                #blbrt4z = [float(ldata[72])/(float(ldata[71]) + float(ldata[72]) + float(ldata[73]))]
                #blbrt4h = [float(ldata[73])/(float(ldata[71]) + float(ldata[72]) + float(ldata[73]))]
            bllamHt44 = [float(ldata[10])]
            blbrour = [float(ldata[74])]  #t4->ht      
            #blbrour = [float(ldata[74])*float(ldata[71])] #t4->wb  
            #blbrour = [float(ldata[57])*float(ldata[72])] #t4->zt  
            #blbrour = [float(ldata[92])*float(ldata[85])] #b4->hb       
            #blbrour = [float(ldata[92])*float(ldata[84])] #b4->zb       
            #blbrour = [float(ldata[92])*float(ldata[83])] #b4->wt       
            bllamHt34 = [float(ldata[2])]
            bllamHrt34 = [float(ldata[9])]
            bllamHt44 = [float(ldata[10])]
            bllamHrt44 = [float(ldata[11])]
            blgLW43 = [float(ldata[13])]
            blgLW33 = [float(ldata[15])]
            blgLZ44 = [float(ldata[30])]
            blgLZ43 = [float(ldata[31])]
            blgLZ33 = [float(ldata[33])]
            blgRZ44 = [float(ldata[39])]
            blgRZ43 = [float(ldata[40])]
            blgRZ33 = [float(ldata[42])]
            blgamHtot = [float(ldata[62])]
            blgamHtotata = [float(ldata[63])]
            blgamHtobb = [float(ldata[64])]
            blgamHtot4 = [float(ldata[65])]
            blgamHtogg = [float(ldata[66])]
            blgamHtohh = [float(ldata[67])]
            blgamHtocc = [float(ldata[68])]
            blgamHtodi = [float(ldata[69])]
            blgamHtott = [float(ldata[70])]
            blkaT = [float(ldata[48])]
            blggHratio = [float(ldata[75])]
            bltotalratio = [float(ldata[79])]
            blprodratiobr = [float(ldata[74])*float(ldata[79])]
            bllamSMt44 = [-float(ldata[10])*float(ldata[1])]
            blggSMratio = [float(ldata[76])]
            #blmb4 = [float(ldata[137])]
            blmb4 = [float(ldata[-20])]
            #if float(ldata[83]) + float(ldata[84]) + float(ldata[85]) == 0:
            blbrb4w = [float(ldata[83])]
            blbrb4z = [float(ldata[84])]
            blbrb4h = [float(ldata[85])]
            #else:
                #blbrb4w = [float(ldata[83])/(float(ldata[83]) + float(ldata[84]) + float(ldata[85]))]
                #blbrb4z = [float(ldata[84])/(float(ldata[83]) + float(ldata[84]) + float(ldata[85]))]
                #blbrb4h = [float(ldata[85])/(float(ldata[83]) + float(ldata[84]) + float(ldata[85]))]
            blbrHtob4 = [float(ldata[92])]
            blbrb4t4w = [float(ldata[86])]
            blbrt4b4w = [float(ldata[81])]
            #if float(ldata[-3]) + float(ldata[-2]) + float(ldata[-1]) == 0:
            blbrt5z = [float(ldata[-3])]
            blbrt5w = [float(ldata[-2])]
            blbrt5h = [float(ldata[-1])]
            #else:
                #blbrt5z = [float(ldata[-3])/(float(ldata[-3]) + float(ldata[-2]) + float(ldata[-1]))]
                #blbrt5w = [float(ldata[-2])/(float(ldata[-3]) + float(ldata[-2]) + float(ldata[-1]))]
                #blbrt5h = [float(ldata[-1])/(float(ldata[-3]) + float(ldata[-2]) + float(ldata[-1]))]
            blbrt4HH = [float(ldata[-4])]
            blbrb4HH = [float(ldata[-8])] # Only for scenario 3
            blbrb4H0 = [float(ldata[-9])] # Only for mytest-s9
            blistmt4.extend(blmt4)
            blisttanbe.extend(bltanbe)
            blistlamHt34.extend(bllamHt34)
            blistmHH.extend(blmHH)
            blistlamHrt34.extend(bllamHrt34)
            blistlamHt44.extend(bllamHt44)
            blistlamHrt44.extend(bllamHrt44)
            blistgLW43.extend(blgLW43)
            blistgLW33.extend(blgLW33)
            blistgLZ44.extend(blgLZ44)
            blistgLZ43.extend(blgLZ43)
            blistgLZ33.extend(blgLZ33)
            blistgRZ44.extend(blgRZ44)
            blistgRZ43.extend(blgRZ43)
            blistgRZ33.extend(blgRZ33)
            blistbrHtobb.extend(blbrHtobb)
            blistbrHtohh.extend(blbrHtohh)
            blistbrHtot4.extend(blbrHtot4)
            blistbrHtogg.extend(blbrHtogg)
            blistbrHtocc.extend(blbrHtocc)
            blistbrHtodi.extend(blbrHtodi)
            blistbrHtott.extend(blbrHtott)
            blistgamHtot.extend(blgamHtot)
            blistgamHtotata.extend(blgamHtotata)
            blistgamHtobb.extend(blgamHtobb)
            blistgamHtot4.extend(blgamHtot4)
            blistgamHtogg.extend(blgamHtogg)
            blistgamHtohh.extend(blgamHtohh)
            blistgamHtocc.extend(blgamHtocc)
            blistgamHtodi.extend(blgamHtodi)
            blistgamHtott.extend(blgamHtott)
            blistbrt4w.extend(blbrt4w)
            blistbrt4z.extend(blbrt4z)
            blistbrt4h.extend(blbrt4h)
            blistbrour.extend(blbrour)
            blistkaT.extend(blkaT)
            blistggHratio.extend(blggHratio)
            blisttotalratio.extend(bltotalratio)
            blistprodratiobr.extend(blprodratiobr)
            blistlamSMt44.extend(bllamSMt44)
            blistggSMratio.extend(blggSMratio)
            blistbrHtotata.extend(blbrHtotata)
            blistmb4.extend(blmb4)
            blistbrb4w.extend(blbrb4w)
            blistbrb4z.extend(blbrb4z)
            blistbrb4h.extend(blbrb4h)
            blistbrHtob4.extend(blbrHtob4)
            blistbrb4t4w.extend(blbrb4t4w)
            blistbrt4b4w.extend(blbrt4b4w)
            blistbrt5z.extend(blbrt5z)
            blistbrt5w.extend(blbrt5w)
            blistbrt5h.extend(blbrt5h)
            blistbrt4HH.extend(blbrt4HH)
            blistbrb4HH.extend(blbrb4HH)
            blistbrb4H0.extend(blbrb4H0)
        #if float(ldata[0]) > 1450 and float(ldata[0]) < 1550 and float(ldata[4]) >= 0.5 and float(ldata[4]) < 0.95:#t4 doublet-fraction 50-95%: doublet-like
        #if float(ldata[4]) >= 0.5 and float(ldata[4]) < 0.95:#t4 doublet-fraction 50-95%: doublet-like
        if float(ldata[-16]) >= 0.5 and float(ldata[-16]) < 0.95:#b4 doublet-fraction 50-95%: doublet-like
            clbrHtobb = [float(ldata[55])]
            clbrHtohh = [float(ldata[56])]
            clbrHtot4 = [float(ldata[57])]
            clbrHtogg = [float(ldata[58])]
            clbrHtocc = [float(ldata[59])]
            clbrHtodi = [float(ldata[60])]
            clbrHtott = [float(ldata[61])]
            clbrHtotata = [float(ldata[80])]
            cltanbe = [float(ldata[1])]
            clmt4 = [float(ldata[0])]
            clmHH = [float(ldata[3])]
            #if float(ldata[71]) + float(ldata[72]) + float(ldata[73]) == 0:
            clbrt4w = [float(ldata[71])]
            clbrt4z = [float(ldata[72])]
            clbrt4h = [float(ldata[73])]
            #else:
                #clbrt4w = [float(ldata[71])/(float(ldata[71]) + float(ldata[72]) + float(ldata[73]))]
                #clbrt4z = [float(ldata[72])/(float(ldata[71]) + float(ldata[72]) + float(ldata[73]))]
                #clbrt4h = [float(ldata[73])/(float(ldata[71]) + float(ldata[72]) + float(ldata[73]))]
            cllamHt44 = [float(ldata[10])]
            clbrour = [float(ldata[74])]      #t4->ht  
            #clbrour = [float(ldata[74])*float(ldata[71])] #t4->wb  
            #clbrour = [float(ldata[57])*float(ldata[72])] #t4->zt  
            #clbrour = [float(ldata[92])*float(ldata[85])] #b4->hb       
            #clbrour = [float(ldata[92])*float(ldata[84])] #b4->zb       
            #clbrour = [float(ldata[92])*float(ldata[83])] #b4->wt       
            cllamHt34 = [float(ldata[2])]
            cllamHrt34 = [float(ldata[9])]
            cllamHt44 = [float(ldata[10])]
            cllamHrt44 = [float(ldata[11])]
            clgLW43 = [float(ldata[13])]
            clgLW33 = [float(ldata[15])]
            clgLZ44 = [float(ldata[30])]
            clgLZ43 = [float(ldata[31])]
            clgLZ33 = [float(ldata[33])]
            clgRZ44 = [float(ldata[39])]
            clgRZ43 = [float(ldata[40])]
            clgRZ33 = [float(ldata[42])]
            clgamHtot = [float(ldata[62])]
            clgamHtotata = [float(ldata[63])]
            clgamHtobb = [float(ldata[64])]
            clgamHtot4 = [float(ldata[65])]
            clgamHtogg = [float(ldata[66])]
            clgamHtohh = [float(ldata[67])]
            clgamHtocc = [float(ldata[68])]
            clgamHtodi = [float(ldata[69])]
            clgamHtott = [float(ldata[70])]
            clkaT = [float(ldata[48])]
            clggHratio = [float(ldata[75])]
            cltotalratio = [float(ldata[79])]
            clprodratiobr = [float(ldata[74])*float(ldata[79])]
            cllamSMt44 = [-float(ldata[10])*float(ldata[1])]
            clggSMratio = [float(ldata[76])]
            #clmb4 = [float(ldata[137])]
            clmb4 = [float(ldata[-20])]
            #if float(ldata[83]) + float(ldata[84]) + float(ldata[85]) == 0:
            clbrb4w = [float(ldata[83])]
            clbrb4z = [float(ldata[84])]
            clbrb4h = [float(ldata[85])]
            #else:
                #clbrb4w = [float(ldata[83])/(float(ldata[83]) + float(ldata[84]) + float(ldata[85]))]
                #clbrb4z = [float(ldata[84])/(float(ldata[83]) + float(ldata[84]) + float(ldata[85]))]
                #clbrb4h = [float(ldata[85])/(float(ldata[83]) + float(ldata[84]) + float(ldata[85]))]
            clbrHtob4 = [float(ldata[92])]
            clbrb4t4w = [float(ldata[86])]
            clbrt4b4w = [float(ldata[81])]
            if float(ldata[-3]) + float(ldata[-2]) + float(ldata[-1]) == 0:
                clbrt5z = [float(ldata[-3])]
                clbrt5w = [float(ldata[-2])]
                clbrt5h = [float(ldata[-1])]
            else:
                clbrt5z = [float(ldata[-3])/(float(ldata[-3]) + float(ldata[-2]) + float(ldata[-1]))]
                clbrt5w = [float(ldata[-2])/(float(ldata[-3]) + float(ldata[-2]) + float(ldata[-1]))]
                clbrt5h = [float(ldata[-1])/(float(ldata[-3]) + float(ldata[-2]) + float(ldata[-1]))]
            clbrt4HH = [float(ldata[-4])]
            clbrb4HH = [float(ldata[-8])] 
            clbrb4H0 = [float(ldata[-9])] # Only for mytest-s9
            clistmt4.extend(clmt4)
            clisttanbe.extend(cltanbe)
            clistlamHt34.extend(cllamHt34)
            clistmHH.extend(clmHH)
            clistlamHrt34.extend(cllamHrt34)
            clistlamHt44.extend(cllamHt44)
            clistlamHrt44.extend(cllamHrt44)
            clistgLW43.extend(clgLW43)
            clistgLW33.extend(clgLW33)
            clistgLZ44.extend(clgLZ44)
            clistgLZ43.extend(clgLZ43)
            clistgLZ33.extend(clgLZ33)
            clistgRZ44.extend(clgRZ44)
            clistgRZ43.extend(clgRZ43)
            clistgRZ33.extend(clgRZ33)
            clistbrHtobb.extend(clbrHtobb)
            clistbrHtohh.extend(clbrHtohh)
            clistbrHtot4.extend(clbrHtot4)
            clistbrHtogg.extend(clbrHtogg)
            clistbrHtocc.extend(clbrHtocc)
            clistbrHtodi.extend(clbrHtodi)
            clistbrHtott.extend(clbrHtott)
            clistgamHtot.extend(clgamHtot)
            clistgamHtotata.extend(clgamHtotata)
            clistgamHtobb.extend(clgamHtobb)
            clistgamHtot4.extend(clgamHtot4)
            clistgamHtogg.extend(clgamHtogg)
            clistgamHtohh.extend(clgamHtohh)
            clistgamHtocc.extend(clgamHtocc)
            clistgamHtodi.extend(clgamHtodi)
            clistgamHtott.extend(clgamHtott)
            clistbrt4w.extend(clbrt4w)
            clistbrt4z.extend(clbrt4z)
            clistbrt4h.extend(clbrt4h)
            clistbrour.extend(clbrour)
            clistkaT.extend(clkaT)
            clistggHratio.extend(clggHratio)
            clisttotalratio.extend(cltotalratio)
            clistprodratiobr.extend(clprodratiobr)
            clistlamSMt44.extend(cllamSMt44)
            clistggSMratio.extend(clggSMratio)
            clistbrHtotata.extend(clbrHtotata)
            clistmb4.extend(clmb4)
            clistbrb4w.extend(clbrb4w)
            clistbrb4z.extend(clbrb4z)
            clistbrb4h.extend(clbrb4h) 
            clistbrHtob4.extend(clbrHtob4)
            clistbrb4t4w.extend(clbrb4t4w)
            clistbrt4b4w.extend(clbrt4b4w)
            clistbrt5z.extend(clbrt5z)
            clistbrt5w.extend(clbrt5w)
            clistbrt5h.extend(clbrt5h)
            clistbrt4HH.extend(clbrt4HH)
            clistbrb4HH.extend(clbrb4HH)
            clistbrb4H0.extend(clbrb4H0)
        #if float(ldata[54]) > 1695 and float(ldata[54]) < 1705 and float(ldata[4]) >= 0.9995:#t4 doublet-fraction > 95%: doublet-like
        #if abs(float(ldata[48])) < 1 and abs(float(ldata[50])) < 1 and abs(float(ldata[51])) < 1 and float(ldata[4]) >= 0.9995:#t4 doublet-fraction > 95%: doublet-like
        #if abs(float(ldata[49])) > 0.5*abs(float(ldata[48])) and abs(float(ldata[49])) > 0.5*abs(float(ldata[50])) and abs(float(ldata[49])) > 0.5*abs(float(ldata[51])) and float(ldata[54]) < 0.5*float(ldata[53]) and float(ldata[4]) >= 0.99:#t4 doublet-fraction > 95%: doublet-like
        #if float(ldata[4]) >= 0.95:#t4 doublet-fraction > 95%: doublet-like
        if float(ldata[-16]) >= 0.95:#b4 doublet-fraction > 95%: doublet-like
            dlbrHtobb = [float(ldata[55])]
            dlbrHtohh = [float(ldata[56])]
            dlbrHtot4 = [float(ldata[57])]
            dlbrHtogg = [float(ldata[58])]
            dlbrHtocc = [float(ldata[59])]
            dlbrHtodi = [float(ldata[60])]
            dlbrHtott = [float(ldata[61])]
            dlbrHtotata = [float(ldata[80])]
            dltanbe = [float(ldata[1])]
            dlmt4 = [float(ldata[0])]
            dlmHH = [float(ldata[3])]
            #if float(ldata[71]) + float(ldata[72]) + float(ldata[73]) == 0:
            dlbrt4w = [float(ldata[71])]
            dlbrt4z = [float(ldata[72])]
            dlbrt4h = [float(ldata[73])]
            #else:
                #dlbrt4w = [float(ldata[71])/(float(ldata[71]) + float(ldata[72]) + float(ldata[73]))]
                #dlbrt4z = [float(ldata[72])/(float(ldata[71]) + float(ldata[72]) + float(ldata[73]))]
                #dlbrt4h = [float(ldata[73])/(float(ldata[71]) + float(ldata[72]) + float(ldata[73]))]
            dllamHt44 = [float(ldata[10])]
            dlbrour = [float(ldata[74])]      #t4->ht  
            #dlbrour = [float(ldata[74])*float(ldata[71])] #t4->wb  
            #dlbrour = [float(ldata[57])*float(ldata[72])] #t4->zt  
            #dlbrour = [float(ldata[92])*float(ldata[85])] #b4->hb       
            #dlbrour = [float(ldata[92])*float(ldata[84])] #b4->zb       
            #dlbrour = [float(ldata[92])*float(ldata[83])] #b4->wt       
            dllamHt34 = [float(ldata[2])]
            dllamHrt34 = [float(ldata[9])]
            dllamHt44 = [float(ldata[10])]
            dllamHrt44 = [float(ldata[11])]
            dlgLW43 = [float(ldata[13])]
            dlgLW33 = [float(ldata[15])]
            dlgLZ44 = [float(ldata[30])]
            dlgLZ43 = [float(ldata[31])]
            dlgLZ33 = [float(ldata[33])]
            dlgRZ44 = [float(ldata[39])]
            dlgRZ43 = [float(ldata[40])]
            dlgRZ33 = [float(ldata[42])]
            dlgamHtot = [float(ldata[62])]
            dlgamHtotata = [float(ldata[63])]
            dlgamHtobb = [float(ldata[64])]
            dlgamHtot4 = [float(ldata[65])]
            dlgamHtogg = [float(ldata[66])]
            dlgamHtohh = [float(ldata[67])]
            dlgamHtocc = [float(ldata[68])]
            dlgamHtodi = [float(ldata[69])]
            dlgamHtott = [float(ldata[70])]
            dlkaT = [float(ldata[48])]
            dlggHratio = [float(ldata[75])]
            dltotalratio = [float(ldata[79])]
            dlprodratiobr = [float(ldata[74])*float(ldata[79])]
            dllamSMt44 = [-float(ldata[10])*float(ldata[1])]
            dlggSMratio = [float(ldata[76])]
            #dlmb4 = [float(ldata[137])]
            dlmb4 = [float(ldata[-20])]
            #if float(ldata[83]) + float(ldata[84]) + float(ldata[85]) == 0:
            dlbrb4w = [float(ldata[83])]
            dlbrb4z = [float(ldata[84])]
            dlbrb4h = [float(ldata[85])]
            #else:
                #dlbrb4w = [float(ldata[83])/(float(ldata[83]) + float(ldata[84]) + float(ldata[85]))]
                #dlbrb4z = [float(ldata[84])/(float(ldata[83]) + float(ldata[84]) + float(ldata[85]))]
                #dlbrb4h = [float(ldata[85])/(float(ldata[83]) + float(ldata[84]) + float(ldata[85]))]
            dlbrHtob4 = [float(ldata[92])]
            dlbrb4t4w = [float(ldata[86])]
            dlbrt4b4w = [float(ldata[81])]
            #if float(ldata[-3]) + float(ldata[-2]) + float(ldata[-1]) == 0:
            dlbrt5z = [float(ldata[-3])]
            dlbrt5w = [float(ldata[-2])]
            dlbrt5h = [float(ldata[-1])]
            #else:
                #dlbrt5z = [float(ldata[-3])/(float(ldata[-3]) + float(ldata[-2]) + float(ldata[-1]))]
                #dlbrt5w = [float(ldata[-2])/(float(ldata[-3]) + float(ldata[-2]) + float(ldata[-1]))]
                #dlbrt5h = [float(ldata[-1])/(float(ldata[-3]) + float(ldata[-2]) + float(ldata[-1]))]
            dlbrt4HH = [float(ldata[-4])]
            dlbrb4HH = [float(ldata[-8])] # Only for scenario 3
            dlbrb4H0 = [float(ldata[-9])] # Only for mytest-s9
            dlistmt4.extend(dlmt4)
            dlisttanbe.extend(dltanbe)
            dlistlamHt34.extend(dllamHt34)
            dlistmHH.extend(dlmHH)
            dlistlamHrt34.extend(dllamHrt34)
            dlistlamHt44.extend(dllamHt44)
            dlistlamHrt44.extend(dllamHrt44)
            dlistgLW43.extend(dlgLW43)
            dlistgLW33.extend(dlgLW33)
            dlistgLZ44.extend(dlgLZ44)
            dlistgLZ43.extend(dlgLZ43)
            dlistgLZ33.extend(dlgLZ33)
            dlistgRZ44.extend(dlgRZ44)
            dlistgRZ43.extend(dlgRZ43)
            dlistgRZ33.extend(dlgRZ33)
            dlistbrHtobb.extend(dlbrHtobb)
            dlistbrHtohh.extend(dlbrHtohh)
            dlistbrHtot4.extend(dlbrHtot4)
            dlistbrHtogg.extend(dlbrHtogg)
            dlistbrHtocc.extend(dlbrHtocc)
            dlistbrHtodi.extend(dlbrHtodi)
            dlistbrHtott.extend(dlbrHtott)
            dlistgamHtot.extend(dlgamHtot)
            dlistgamHtotata.extend(dlgamHtotata)
            dlistgamHtobb.extend(dlgamHtobb)
            dlistgamHtot4.extend(dlgamHtot4)
            dlistgamHtogg.extend(dlgamHtogg)
            dlistgamHtohh.extend(dlgamHtohh)
            dlistgamHtocc.extend(dlgamHtocc)
            dlistgamHtodi.extend(dlgamHtodi)
            dlistgamHtott.extend(dlgamHtott)
            dlistbrt4w.extend(dlbrt4w)
            dlistbrt4z.extend(dlbrt4z)
            dlistbrt4h.extend(dlbrt4h)
            dlistbrour.extend(dlbrour)
            dlistkaT.extend(dlkaT)
            dlistggHratio.extend(dlggHratio)
            dlisttotalratio.extend(dltotalratio)
            dlistprodratiobr.extend(dlprodratiobr)
            dlistlamSMt44.extend(dllamSMt44)
            dlistggSMratio.extend(dlggSMratio)
            dlistbrHtotata.extend(dlbrHtotata)
            dlistmb4.extend(dlmb4)
            dlistbrb4w.extend(dlbrb4w)
            dlistbrb4z.extend(dlbrb4z)
            dlistbrb4h.extend(dlbrb4h) 
            dlistbrHtob4.extend(dlbrHtob4)
            dlistbrb4t4w.extend(dlbrb4t4w)
            dlistbrt4b4w.extend(dlbrt4b4w)
            dlistbrt5z.extend(dlbrt5z)
            dlistbrt5w.extend(dlbrt5w)
            dlistbrt5h.extend(dlbrt5h)
            dlistbrt4HH.extend(dlbrt4HH)
            dlistbrb4HH.extend(dlbrb4HH)
            dlistbrb4H0.extend(dlbrb4H0)
        #if float(ldata[79])*float(ldata[57])*sigmaHSM13TeV(float(ldata[3])) > (((float(ldata[13]))**2 + (float(ldata[22]))**2)/(2.*0.1**2))*myt4bj(float(ldata[0])):#0.001*float(ldata[81]):#in pb unit:
        if tltotalt4t > tlmyt4bj:
            eltanbe = [float(ldata[1])]
            elmHH = [float(ldata[3])]
            elisttanbe.extend(eltanbe)
            elistmHH.extend(elmHH)



print 'test =', len(otlistmHH)
            
#print 'size try =', len(alistmt4), len(blistmt4), len(clistmt4), len(dlistmt4)

#print 'size of singlet fraction =', len(alistmt4)+len(blistmt4)

#print max(tlistbrour), max(tlistbrt4h), max(tlistbrHtot4)

#print min(tlistbrour), min(tlistbrt4h), min(tlistbrHtot4)

#print max(tlistlamHt34), max(tlistlamHrt34), max(tlistbrHtott), max(tlistbrHtobb), max(tlistmHH), max(tlistmt4)

#print len(clistbrb4z)

#print talistka
#print talistkab
#print talistkaQ




    



########## Figures            



"""

fig = plt.figure()  # make a new figure : separately from the above 
plt.axis([-0.01,1.02,-0.01,1.02])
#fb, = plt.plot(tblistbrt4z,tblistbrt4w,'.', color = 'red', label='constraint',zorder=1)
#fc, = plt.plot(tclistbrt4z,tclistbrt4w,'.', color = 'blue', label='constraint',zorder=2)
#fe, = plt.plot(telistbrt4z,telistbrt4w,'.', color = 'navy', label='constraint',zorder=3)
#ff, = plt.plot(tflistbrt4z,tflistbrt4w,'.', color = 'green', label='constraint',zorder=4)
#fg, = plt.plot(tglistbrt4z,tglistbrt4w,'.', color = 'purple', label='constraint',zorder=5)
fa, = plt.plot(alistbrt4z,alistbrt4w,'.', color = 'red', label='constraint',zorder=3)
fb, = plt.plot(blistbrt4z,blistbrt4w,'.', color = 'purple', label='constraint',zorder=4)
fc, = plt.plot(clistbrt4z,clistbrt4w,'.', color = 'cyan', label='constraint',zorder=2)
fd, = plt.plot(dlistbrt4z,dlistbrt4w,'.', color = 'blue', label='constraint',zorder=1)
#fa, = plt.plot(alistbrt4z,alistbrt4h,'.', color = 'red', label='constraint',zorder=4)
#fb, = plt.plot(blistbrt4z,blistbrt4h,'.', color = 'orange', label='constraint',zorder=5)
#fb, = plt.plot(blistbrt4z,blistbrt4h,'.', color = 'purple', label='constraint',zorder=3)
#fc, = plt.plot(clistbrt4z,clistbrt4h,'.', color = 'cyan', label='constraint',zorder=2)
#fd, = plt.plot(dlistbrt4z,dlistbrt4h,'.', color = 'blue', label='constraint',zorder=1)
#fa, = plt.plot(alistbrt4b4w,alistbrt4w,'.', color = 'red', label='constraint',zorder=3)
#fb, = plt.plot(blistbrt4b4w,blistbrt4w,'.', color = 'purple', label='constraint',zorder=4)
#fc, = plt.plot(clistbrt4b4w,clistbrt4w,'.', color = 'cyan', label='constraint',zorder=2)
#fd, = plt.plot(dlistbrt4b4w,dlistbrt4w,'.', color = 'blue', label='constraint',zorder=1)
#ft, = plt.plot(tlistbrt4z,tlistbrt4w,'.', color = 'red', label='constraint',zorder=3)
#ft, = plt.plot(tlistbrb4z,tlistbrb4w,'.', color = 'red', label='constraint',zorder=3)
#fa, = plt.plot(alistbrb4z,alistbrb4w,'.', color = 'red', label='constraint',zorder=3)
#fb, = plt.plot(blistbrb4z,blistbrb4w,'.', color = 'purple', label='constraint',zorder=4)
#fc, = plt.plot(clistbrb4z,clistbrb4w,'.', color = 'cyan', label='constraint',zorder=2)
#fd, = plt.plot(dlistbrb4z,dlistbrb4w,'.', color = 'blue', label='constraint',zorder=1)
#fa, = plt.plot(alistbrb4t4w,alistbrb4w,'.', color = 'red', label='constraint',zorder=3)
#fb, = plt.plot(blistbrb4t4w,blistbrb4w,'.', color = 'purple', label='constraint',zorder=4)
#fc, = plt.plot(clistbrb4t4w,clistbrb4w,'.', color = 'cyan', label='constraint',zorder=2)
#fd, = plt.plot(dlistbrb4t4w,dlistbrb4w,'.', color = 'blue', label='constraint',zorder=1)
#fb, = plt.plot(tblistbrb4z,tblistbrb4w,'.', color = 'red', label='constraint',zorder=1)
#fc, = plt.plot(tclistbrb4z,tclistbrb4w,'.', color = 'blue', label='constraint',zorder=2)
#fe, = plt.plot(telistbrb4z,telistbrb4w,'.', color = 'navy', label='constraint',zorder=3)
#ff, = plt.plot(tflistbrb4z,tflistbrb4w,'.', color = 'green', label='constraint',zorder=4)
#fg, = plt.plot(tglistbrb4z,tglistbrb4w,'.', color = 'purple', label='constraint',zorder=5)
myax = plt.gca()
#myax.set_yscale('log', nonposy='clip')
#myax.annotate(r'$m_{t_4}$', xy=(0.9, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
#myax.annotate(r'BR($b_4 \to W t_4$)', xy=(0.75, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
#myax.annotate(r'BR($b_4 \to Z b$)', xy=(0.75, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=22)
#myax.annotate(r'BR($b_4 \to W t$)', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=22, rotation = 90)
#myax.annotate(r'BR($t_4 \to W b_4$)', xy=(0.75, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
myax.annotate(r'BR($t_4 \to Z t$)', xy=(0.75, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=22)
myax.annotate(r'BR($t_4 \to W b$)', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=22, rotation = 90)
#myax.annotate(r'BR($t_4 \to h t$)', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
#myax.annotate(r'BR($t_4 \to Z t$) + BR($t_4 \to W b$) + BR($t_4 \to h t$) = 1', xy=(0.3, 0.8), ha='left', va='top', xycoords='axes fraction', fontsize=14)
#myax.annotate(r'$m_{t_4} < m_{b_4} + M_W$', xy=(0.3, 0.9), ha='left', va='top', xycoords='axes fraction', fontsize=14)
#myax.annotate(r'$\kappa_T = 0$', xy=(0.5, 0.85), ha='left', va='top', xycoords='axes fraction', fontsize=16)
#myax.annotate(r'Scenario 3', xy=(0.5, 0.7), ha='left', va='top', xycoords='axes fraction', fontsize=16)
#myax.annotate(r'$M_T = 3$ TeV, $M_Q = M_B = 100$ TeV', xy=(0.3, 0.7), ha='left', va='top', xycoords='axes fraction', fontsize=16)
#myax.annotate(r'$M_T = 3$ TeV, $M_Q = M_B = 100$ TeV', xy=(0.4, 0.7), ha='left', va='top', xycoords='axes fraction', fontsize=16)
#myax.annotate(r'BR($t_4 \to W b_4$) = 0', xy=(0.5, 0.7), ha='left', va='top', xycoords='axes fraction', fontsize=16)
#myax.annotate(r'BR($b_4 \to Z b$) + BR($b_4 \to W t$) + BR($b_4 \to h b$) = 1', xy=(0.3, 0.8), ha='left', va='top', xycoords='axes fraction', fontsize=14)
#myax.annotate(r'$m_{b_4} < m_{t_4} + M_W$', xy=(0.3, 0.9), ha='left', va='top', xycoords='axes fraction', fontsize=14)
#myax.annotate(r'$\kappa_Q = \lambda_Q = 0$', xy=(0.7, 0.6), ha='left', va='top', xycoords='axes fraction', fontsize=16)
#myax.annotate(r'Scenario 2, $y_b = y_t$', xy=(0.6, 0.6), ha='left', va='top', xycoords='axes fraction', fontsize=16)
#myax.annotate(r'$\bar \kappa = \kappa = 0$', xy=(0.7, 0.6), ha='left', va='top', xycoords='axes fraction', fontsize=16)
#myax.annotate(r'$M_Q = 1.7\,{\rm TeV}$ and 99.95% doublet', xy=(0.2, 0.7), ha='left', va='top', xycoords='axes fraction', fontsize=16)
#myax.annotate(r'$\kappa_T = \bar \kappa = \kappa = 0$', xy=(0.3, 0.7), ha='left', va='top', xycoords='axes fraction', fontsize=16)
#myax.annotate(r'$|\kappa_Q| < 0.1$, 99.95% doublet', xy=(0.3, 0.7), ha='left', va='top', xycoords='axes fraction', fontsize=16)
#myax.annotate(r'$|\kappa_T| > 0.5 \times (|\kappa_Q|, |\kappa|, |\bar \kappa|)$, $M_Q < M_T / 2$, 99% doublet', xy=(0.1, 0.7), ha='left', va='top', xycoords='axes fraction', fontsize=16)
#myax.annotate(r'orange: $|\kappa_T| < 0.001 \times (|\kappa_Q|, |\kappa|, |\bar \kappa|)$', xy=(0.4, 0.5), ha='left', va='top', xycoords='axes fraction', fontsize=16)
#myax.annotate(r'99% singlet', xy=(0.6, 0.6), ha='left', va='top', xycoords='axes fraction', fontsize=16)
#plt.plot(xt, yt, '--', color = 'black')
majorLocator   = MultipleLocator(0.2)
majorFormatter = FormatStrFormatter('%2.1f')
minorLocator   = MultipleLocator(0.1)
ymajorLocator   = MultipleLocator(0.2)
ymajorFormatter = FormatStrFormatter('%2.1f')
yminorLocator   = MultipleLocator(0.1)
myax.tick_params(size=3.5)
#myax.tick_params(axis = 'y',zorder = 10) 
myax.xaxis.set_major_locator(majorLocator)
myax.xaxis.set_major_formatter(majorFormatter)
myax.xaxis.set_minor_locator(minorLocator)
myax.yaxis.set_major_locator(ymajorLocator)
myax.yaxis.set_major_formatter(ymajorFormatter)
myax.yaxis.set_minor_locator(yminorLocator)
first_legend = plt.legend([fa, fb, fc, fd], [r'$>$ 95% singlet-like', r'$<$ 95% singlet-like', r'$<$ 95% doublet-like', r'$>$ 95% doublet-like'], frameon=True, loc=1, numpoints = 1,fontsize = 18)
#first_legend = plt.legend([fb, fc, fe, ff, fg], [r'$\tan\beta <  1$', r'$1  < \tan\beta <  5$', r'$5 < \tan\beta < 10$', r'$10 < \tan\beta < 20$', r'$20 < \tan\beta < 50$'], frameon=True, loc=1, numpoints = 1,fontsize = 15)
first_legend.set_zorder(100)
plt.gca().add_artist(first_legend)
#second_legend = plt.legend([fgt,fgta,fgtb],[r'                  ',r'                                                        ',r'               '], frameon=False, loc=1, numpoints = 1, fontsize = 12, handletextpad=8,columnspacing = 1.1,borderpad=0.5,labelspacing=0.88)
#second_legend.set_zorder(101)
#frame = second_legend.get_frame()
#frame.set_facecolor('white')
#plt.title(r'$\kappa_T = \bar \kappa = 0$', fontsize=17)
#plt.title(r'$\kappa_T = \kappa = 0$', fontsize=17)
#plt.title(r'$\kappa_T = 0$', fontsize=17)
#plt.title(r'$\kappa = \bar \kappa = 0$', fontsize=17)
plt.title(r'Couplings to $H_u$ and $H_d$', fontsize=17)#, couplings up to 0.01', fontsize=17)
#plt.title(r'Couplings to $H_d$ only', fontsize=17)
#plt.savefig('t1ewpt-brb4s.png')
#plt.savefig('t1ewpt-brb4s-t4w.png')
plt.savefig('t1ewpt-brt4s.png')
#plt.savefig('t1ewpt-brt4s-hz.png')
#plt.savefig('t1ewpt-brt4s-b4w.png')
#plt.savefig('tanbe-brt4s.png')
#plt.savefig('tanbe-brb4s.png')

"""

"""


def tline(x):
    return 1 - x

xt = numpy.arange(0.0, 1., 0.00001)
yt = tline(xt)







fig = plt.figure()  # make a new figure : separately from the above 
plt.axis([-0.01,1.02,-0.01,1.02])
fa, = plt.plot(alistbrt4w,alistbrt4b4w,'.', color = 'red', label='constraint',zorder=3)
fb, = plt.plot(blistbrt4w,blistbrt4b4w,'.', color = 'purple', label='constraint',zorder=4)
fc, = plt.plot(clistbrt4w,clistbrt4b4w,'.', color = 'cyan', label='constraint',zorder=2)
fd, = plt.plot(dlistbrt4w,dlistbrt4b4w,'.', color = 'blue', label='constraint',zorder=1)
myax = plt.gca()
#myax.set_yscale('log', nonposy='clip')
myax.annotate(r'BR($t_4 \to W b$)', xy=(0.75, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=22)
myax.annotate(r'BR($t_4 \to W b_4$)', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=22, rotation = 90)
#plt.plot(xt, yt, '--', color = 'black')
majorLocator   = MultipleLocator(0.2)
majorFormatter = FormatStrFormatter('%2.1f')
minorLocator   = MultipleLocator(0.1)
ymajorLocator   = MultipleLocator(0.2)
ymajorFormatter = FormatStrFormatter('%2.1f')
yminorLocator   = MultipleLocator(0.1)
myax.tick_params(size=3.5)
#myax.tick_params(axis = 'y',zorder = 10) 
myax.xaxis.set_major_locator(majorLocator)
myax.xaxis.set_major_formatter(majorFormatter)
myax.xaxis.set_minor_locator(minorLocator)
myax.yaxis.set_major_locator(ymajorLocator)
myax.yaxis.set_major_formatter(ymajorFormatter)
myax.yaxis.set_minor_locator(yminorLocator)
#first_legend = plt.legend([fa, fb, fc, fd], [r'$>$ 95% singlet-like', r'$<$ 95% singlet-like', r'$<$ 95% doublet-like', r'$>$ 95% doublet-like'], frameon=True, loc=1, numpoints = 1,fontsize = 18)
#first_legend = plt.legend([fb, fc, fe, ff, fg], [r'$\tan\beta <  1$', r'$1  < \tan\beta <  5$', r'$5 < \tan\beta < 10$', r'$10 < \tan\beta < 20$', r'$20 < \tan\beta < 50$'], frameon=True, loc=1, numpoints = 1,fontsize = 15)
#first_legend.set_zorder(100)
#plt.gca().add_artist(first_legend)
plt.title(r'Couplings to $H_u$ only', fontsize=17)
plt.savefig('t1ewpt-brt4w-brt4b4w.png')


#print 'size of alistbrb4h =', len(alistbrb4h)



#print len(alistbrb4z), len(blistbrb4z), len(clistbrb4z), len(dlistbrb4z), len(alistmb4)
#print 'mt4 > mb4 =', len(tblistmt4), 'mt4 < mb4 =', len(tclistmt4)






print 'special case =', len(talistbrt4HH)

fig = plt.figure()  # make a new figure : separately from the above 
#plt.axis([100.,4000.,-0.01,1.02])
plt.axis([-1.,51.,-0.01,1.02])
#f, = plt.plot(tlistmt4,tlistbrt4HH,'.', color = 'navy', label='constraint',zorder=2)
#f, = plt.plot(tlistmHH,tlistbrt4HH,'.', color = 'navy', label='constraint',zorder=2)
#f, = plt.plot(tlisttanbe,tlistbrt4HH,'.', color = 'orange', zorder=5)
#f, = plt.plot(tlisttanbe,tlistbrt4HH,'.', color = 'yellowgreen', zorder=5)
f, = plt.plot(tlisttanbe,tlistbrt4HH,'.', color = 'gray', zorder=5) #s3
#fa, = plt.plot(alisttanbe,alistbrt4HH,'.', color = 'red', zorder=2) #s3
#fb, = plt.plot(blisttanbe,blistbrt4HH,'.', color = 'purple', zorder=1) #s3
#fc, = plt.plot(clisttanbe,clistbrt4HH,'.', color = 'cyan', zorder=4) #s3
#fd, = plt.plot(dlisttanbe,dlistbrt4HH,'.', color = 'blue', zorder=3) #s3
#fta, = plt.plot(talisttanbe,talistbrt4HH,'.', color = 'yellowgreen', zorder=6)#alpha = 0.5
#fh, = plt.plot(tlisttanbe,tlistbrt4h,'.', color = 'red', label='constraint',zorder=1)
#ftn, = plt.plot(nlisttanbe,nlistbrt4HH,'.', color = 'orange', zorder=7)#alpha = 0.5
#ftm, = plt.plot(mlisttanbe,mlistbrt4HH,'.', color = 'yellowgreen', zorder=6)#alpha = 0.5
#fa, = plt.plot(alisttanbe,alistbrt4h,'.', color = 'red', zorder=4)
#fb, = plt.plot(blisttanbe,blistbrt4h,'.', color = 'purple', zorder=3)
#fc, = plt.plot(clisttanbe,clistbrt4h,'.', color = 'cyan', zorder=2)
#fc, = plt.plot(dlisttanbe,dlistbrt4h,'.', color = 'blue', zorder=1)
#fa, = plt.plot(alistmb4,alistbrb4w,'.', color = 'red', label='constraint',zorder=2)
#fb, = plt.plot(blistmb4,blistbrb4w,'.', color = 'purple', label='constraint',zorder=1)
#fc, = plt.plot(clistmb4,clistbrb4w,'.', color = 'cyan', label='constraint',zorder=4)
#fd, = plt.plot(dlistmb4,dlistbrb4w,'.', color = 'blue', label='constraint',zorder=3)
#fa, = plt.plot(alistmt4,alistbrt4w,'.', color = 'red', label='constraint',zorder=2)
#fb, = plt.plot(blistmt4,blistbrt4w,'.', color = 'purple', label='constraint',zorder=1)
#fc, = plt.plot(clistmt4,clistbrt4w,'.', color = 'cyan', label='constraint',zorder=4)
#fd, = plt.plot(dlistmt4,dlistbrt4w,'.', color = 'blue', label='constraint',zorder=3)
myax = plt.gca()
#myax.set_yscale('log', nonposy='clip')
#myax.annotate(r'$m_{b_4}$ [GeV]', xy=(0.8, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
#myax.annotate(r'BR($b_4 \to W t$)', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
#myax.annotate(r'$m_{t_4}$ [GeV]', xy=(0.8, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
#myax.annotate(r'$m_H$ [GeV]', xy=(0.8, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
#myax.annotate(r'BR($t_4 \to W b$)', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
myax.annotate(r'$\tan\beta$', xy=(0.8, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=22)
myax.annotate(r'BR($t_4 \to H t$)', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=22, rotation = 90)
#myax.annotate(r'BR($b_4 \to H b$)', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
#myax.annotate(r'BR($t_4 \to H^\pm b$)', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=22, rotation = 90)
#myax.annotate(r'BR($b_4 \to H^\pm t$)', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
#myax.annotate(r'$M_Q, M_T, M_B \in [100, 4000]$ GeV, $M_H \in [500, 4000]$ GeV', xy=(0.1, 0.93), ha='left', va='top', xycoords='axes fraction', fontsize=16)
#myax.annotate(r'Orange point: Couplings to $H_u$ only', xy=(0.3, 0.8), ha='left', va='top', xycoords='axes fraction', fontsize=17)
#myax.annotate(r'Green point: Couplings to $H_d$ only', xy=(0.3, 0.7), ha='left', va='top', xycoords='axes fraction', fontsize=17)
#myax.annotate(r'Gray point: Couplings to $H_u$ and $H_d$', xy=(0.3, 0.6), ha='left', va='top', xycoords='axes fraction', fontsize=17)
myax.annotate(r'Couplings up to 0.01', xy=(0.3, 0.6), ha='left', va='top', xycoords='axes fraction', fontsize=17)
#myax.annotate(r'$M_Q, M_T, M_B, m_H \in [500, 4000]$ GeV', xy=(0.4, 0.8), ha='left', va='top', xycoords='axes fraction', fontsize=16)
#myax.annotate(r'$m_{t_4} > m_H + m_t$', xy=(0.6, 0.7), ha='left', va='top', xycoords='axes fraction', fontsize=16)
#myax.annotate(r'$m_{t_4} > m_H + m_b$', xy=(0.6, 0.7), ha='left', va='top', xycoords='axes fraction', fontsize=16)
#myax.annotate(r'$M_H \in [500, 4000]$ GeV', xy=(0.6, 0.7), ha='left', va='top', xycoords='axes fraction', fontsize=16)
#myax.annotate(r'Red: BR($t_4 \to h t$)', xy=(0.7, 0.7), ha='left', va='top', xycoords='axes fraction', fontsize=16)
majorLocator   = MultipleLocator(500)
majorFormatter = FormatStrFormatter('%i')
minorLocator   = MultipleLocator(100)
ymajorLocator   = MultipleLocator(0.2)
ymajorFormatter = FormatStrFormatter('%2.1f')
yminorLocator   = MultipleLocator(0.1)
myax.tick_params(size=3.5)
#myax.tick_params(axis = 'y',zorder = 10) 
#myax.xaxis.set_major_locator(majorLocator)
#myax.xaxis.set_major_formatter(majorFormatter)
#myax.xaxis.set_minor_locator(minorLocator)
myax.yaxis.set_major_locator(ymajorLocator)
myax.yaxis.set_major_formatter(ymajorFormatter)
myax.yaxis.set_minor_locator(yminorLocator)
#first_legend = plt.legend([fta, ftb, ftc, ftd], [r'$\tan\beta <  1$', r'$1  < \tan\beta <  5$', r'$5  < \tan\beta <  10$', r'$\tan\beta >  10$'], frameon=True, loc=1, numpoints = 1,fontsize = 15)
#first_legend.set_zorder(100)
#plt.gca().add_artist(first_legend)
#second_legend = plt.legend([fgt,fgta,fgtb],[r'                  ',r'                                                        ',r'               '], frameon=False, loc=1, numpoints = 1, fontsize = 12, handletextpad=8,columnspacing = 1.1,borderpad=0.5,labelspacing=0.88)
#second_legend.set_zorder(101)
#frame = second_legend.get_frame()
#frame.set_facecolor('white')
#plt.title(r'Couplings to $H_u$ and $H_d$')
#plt.title(r'Couplings to $H_u$ only')
#plt.savefig('t1ewpt-mb4-brb4w.png')
#plt.savefig('t1ewpt-mt4-brt4w.png')
#plt.savefig('t1ewpt-mHH-brt4HH.png')
#plt.savefig('t1ewpt-mt4-brt4HH.png')
plt.savefig('t1ewpt-tanbe-brt4HH.png')
#plt.savefig('t1ewpt-tanbe-brb4HH.png')
#plt.savefig('t1ewpt-tanbe-brt4Hpm.png')
#plt.savefig('t1ewpt-tanbe-brb4Hpm.png')





fig = plt.figure()  # make a new figure : separately from the above 
#plt.axis([100.,4000.,-0.01,1.02])
plt.axis([-1.,51.,-0.01,1.02])
#plt.axis([-1.,51.,-0.01,0.1])
#plt.axis([-1.,51.,0.00000000001,1.])
#f, = plt.plot(tlistmt4,tlistbrb4HH,'.', color = 'navy', label='constraint',zorder=2)
#f, = plt.plot(tlistmHH,tlistbrb4HH,'.', color = 'navy', label='constraint',zorder=2)
#f, = plt.plot(tlisttanbe,tlistbrb4HH,'.', color = 'yellowgreen', zorder=5)
f, = plt.plot(tlisttanbe,tlistbrb4HH,'.', color = 'gray', zorder=5) # s3
#fa, = plt.plot(alisttanbe,alistbrb4HH,'.', color = 'red', zorder=2) #s3
#fb, = plt.plot(blisttanbe,blistbrb4HH,'.', color = 'purple', zorder=1) #s3
#fc, = plt.plot(clisttanbe,clistbrb4HH,'.', color = 'cyan', zorder=4) #s3
#fd, = plt.plot(dlisttanbe,dlistbrb4HH,'.', color = 'blue', zorder=3) #s3
ftn, = plt.plot(nlisttanbe,nlistbrb4HH,'.', color = 'orange', zorder=7)#alpha = 0.5
ftm, = plt.plot(mlisttanbe,mlistbrb4HH,'.', color = 'yellowgreen', zorder=6)#alpha = 0.5
#fta, = plt.plot(talisttanbe,talistbrb4HH,'.', color = 'yellow', zorder=6)#alpha = 0.5
#ftb, = plt.plot(tblisttanbe,tblistbrb4HH,'.', color = 'lime', zorder=7)#alpha = 0.5
#fh, = plt.plot(tlisttanbe,tlistbrb4h,'.', color = 'red', label='constraint',zorder=1)
#fa, = plt.plot(alisttanbe,alistbrb4h,'.', color = 'red', zorder=4)
#fb, = plt.plot(blisttanbe,blistbrb4h,'.', color = 'purple', zorder=3)
#fc, = plt.plot(clisttanbe,clistbrb4h,'.', color = 'cyan', zorder=2)
#fd, = plt.plot(dlisttanbe,dlistbrb4h,'.', color = 'blue', zorder=1)
myax = plt.gca()
#myax.set_yscale('log', nonposy='clip')
#myax.annotate(r'$m_{b_4}$ [GeV]', xy=(0.8, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
#myax.annotate(r'BR($b_4 \to W t$)', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
#myax.annotate(r'$m_{t_4}$ [GeV]', xy=(0.8, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
#myax.annotate(r'$m_H$ [GeV]', xy=(0.8, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
#myax.annotate(r'BR($t_4 \to W b$)', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
myax.annotate(r'$\tan\beta$', xy=(0.8, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=22)
#myax.annotate(r'BR($t_4 \to H t$)', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
#myax.annotate(r'BR($b_4 \to H b$)', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=22, rotation = 90)
#myax.annotate(r'BR($t_4 \to H^\pm b$)', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
myax.annotate(r'BR($b_4 \to H^\pm t$)', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=22, rotation = 90)
#myax.annotate(r'$M_Q, M_T, M_B \in [100, 4000]$ GeV, $M_H \in [500, 4000]$ GeV', xy=(0.1, 0.93), ha='left', va='top', xycoords='axes fraction', fontsize=16)
#myax.annotate(r'$M_Q, M_T, M_B, m_H \in [500, 4000]$ GeV', xy=(0.5, 0.8), ha='left', va='top', xycoords='axes fraction', fontsize=16)
#myax.annotate(r'$M_Q, M_T, M_B, m_H \in [500, 4000]$ GeV', xy=(0.42, 0.4), ha='left', va='top', xycoords='axes fraction', fontsize=16)
#myax.annotate(r'$m_{b_4} > m_H + m_b$', xy=(0.6, 0.3), ha='left', va='top', xycoords='axes fraction', fontsize=16)
#myax.annotate(r'$m_{b_4} > m_H + m_t$', xy=(0.6, 0.3), ha='left', va='top', xycoords='axes fraction', fontsize=16)
#myax.annotate(r'$M_H \in [500, 4000]$ GeV', xy=(0.6, 0.7), ha='left', va='top', xycoords='axes fraction', fontsize=16)
#myax.annotate(r'Red: BR($b_4 \to h b$)', xy=(0.7, 0.7), ha='left', va='top', xycoords='axes fraction', fontsize=16)
majorLocator   = MultipleLocator(500)
majorFormatter = FormatStrFormatter('%i')
minorLocator   = MultipleLocator(100)
ymajorLocator   = MultipleLocator(0.2)
ymajorFormatter = FormatStrFormatter('%2.1f')
yminorLocator   = MultipleLocator(0.1)
myax.tick_params(size=3.5)
#myax.tick_params(axis = 'y',zorder = 10) 
#myax.xaxis.set_major_locator(majorLocator)
#myax.xaxis.set_major_formatter(majorFormatter)
#myax.xaxis.set_minor_locator(minorLocator)
#myax.yaxis.set_major_locator(ymajorLocator)
#myax.yaxis.set_major_formatter(ymajorFormatter)
#myax.yaxis.set_minor_locator(yminorLocator)
#first_legend = plt.legend([fta, ftb, ftc, ftd], [r'$\tan\beta <  1$', r'$1  < \tan\beta <  5$', r'$5  < \tan\beta <  10$', r'$\tan\beta >  10$'], frameon=True, loc=1, numpoints = 1,fontsize = 15)
#first_legend.set_zorder(100)
#plt.gca().add_artist(first_legend)
#second_legend = plt.legend([fgt,fgta,fgtb],[r'                  ',r'                                                        ',r'               '], frameon=False, loc=1, numpoints = 1, fontsize = 12, handletextpad=8,columnspacing = 1.1,borderpad=0.5,labelspacing=0.88)
#second_legend.set_zorder(101)
#frame = second_legend.get_frame()
#frame.set_facecolor('white')
#plt.title(r'Couplings to $H_u$ and $H_d$')
#plt.title(r'Couplings to $H_d$ only')
#plt.savefig('t1ewpt-mb4-brb4w.png')
#plt.savefig('t1ewpt-mt4-brt4w.png')
#plt.savefig('t1ewpt-mHH-brt4HH.png')
#plt.savefig('t1ewpt-mt4-brt4HH.png')
#plt.savefig('t1ewpt-tanbe-brt4HH.png')
#plt.savefig('t1ewpt-tanbe-brb4HH.png')
#plt.savefig('t1ewpt-tanbe-brt4Hpm.png')
plt.savefig('t1ewpt-tanbe-brb4Hpm.png')


compx = numpy.arange(0.0, 1.1, 0.00001)
def brbound(x):
    return 1 - x

brbound = brbound(compx)

"""


fig = plt.figure()  # make a new figure : separately from the above 
plt.axis([950.,4050,0.00000001,1.])
#ft, = plt.plot(tlistmHH,tlisttotalxs,'.', color = 'navy', label='constraint',zorder=1)
#ftd, = plt.plot(tdlistmHH,tdlisttotalxs,'.', color = 'blue', label='constraint',zorder=1)
fto, = plt.plot(olistmHH,olisttotalxs,'.', color = 'red', label='constraint',zorder=1)
ftp, = plt.plot(plistmHH,plisttotalxs,'.', color = 'blue', label='constraint',zorder=1)
ftq, = plt.plot(qlistmHH,qlisttotalxs,'.', color = 'green', label='constraint',zorder=1)
#fttest, = plt.plot(otlistmHH,otlisttest,'.', color = 'navy', label='constraint',zorder=1)
#fta, = plt.plot(talistmHH,talisttotalxs,'.', color = 'navy', label='constraint',zorder=1)
#ftb, = plt.plot(tblistmHH,tblisttotalxs,'.', color = 'red', label='constraint',zorder=1)
#ftc, = plt.plot(tclistmHH,tclisttotalxs,'.', color = 'blue', label='constraint',zorder=1)
#fte, = plt.plot(telistmHH,telisttotalxs,'.', color = 'green', label='constraint',zorder=1)
#ftf, = plt.plot(tflistmHH,tflisttotalxs,'.', color = 'orange', label='constraint',zorder=1)
#ftg, = plt.plot(tglistmHH,tglisttotalxs,'.', color = 'purple', label='constraint',zorder=1)
#ftSMgg, = plt.plot(tlistmHH, tlistxs2HDM1, '.', color = 'orange')
#ftxs2HDM7, = plt.plot(tlistmHH,tlistxs2HDM7,'.', color = 'brown', zorder=1)
#ftxs2HDM50, = plt.plot(tlistmHH,tlistxs2HDM50,'.', color = 'black', zorder=1)
myax = plt.gca()
myax.set_yscale('log', nonposy='clip')
myax.annotate(r'$m_H$ [GeV]', xy=(0.75, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
myax.annotate(r'$\sigma_H$ [pb]', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
#myax.annotate(r'$\frac{\sigma_H}{\sigma_{H_{\rm SM}}} \,\cdot$ BR($H \to h t \bar{t}$)', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
majorLocator   = MultipleLocator(500)
majorFormatter = FormatStrFormatter('%i')
minorLocator   = MultipleLocator(100)
#ymajorLocator   = MultipleLocator(5)
#ymajorFormatter = FormatStrFormatter('%2.1f')
#yminorLocator   = MultipleLocator(1)
myax.tick_params(size=3.5)
#myax.tick_params(axis = 'y',zorder = 10) 
myax.xaxis.set_major_locator(majorLocator)
myax.xaxis.set_major_formatter(majorFormatter)
myax.xaxis.set_minor_locator(minorLocator)
#myax.yaxis.set_major_locator(ymajorLocator)
#myax.yaxis.set_major_formatter(ymajorFormatter)
#myax.yaxis.set_minor_locator(yminorLocator)
#first_legend = plt.legend([fta, ftc, fte, ftf, ftg], [r'$\tan\beta = 1$', r'$\tan\beta = 7$', r'$\tan\beta = 10$', r'$\tan\beta = 20$', r'$\tan\beta = 50$'], frameon=True, loc=1, numpoints = 1,fontsize = 15)
first_legend = plt.legend([fto, ftp, ftq], [r'$\tan\beta = 1$', r'$\tan\beta = 7$', r'$\tan\beta = 50$'], frameon=True, loc=1, numpoints = 1,fontsize = 15)
first_legend.set_zorder(100)
plt.gca().add_artist(first_legend)
#second_legend = plt.legend([fgt,fgta,fgtb],[r'                  ',r'                                                        ',r'               '], frameon=False, loc=1, numpoints = 1, fontsize = 12, handletextpad=8,columnspacing = 1.1,borderpad=0.5,labelspacing=0.88)
#second_legend.set_zorder(101)
#frame = second_legend.get_frame()
#frame.set_facecolor('white')
#myax.annotate(r'$m_H = 900$ GeV, $m_{t_4}$ = 600 GeV', xy=(0.6, 0.6), ha='left', va='top', xycoords='axes fraction', fontsize=15)
plt.title(r'Couplings to $H_u$ only', fontsize = 17)
#plt.title(r'Couplings to $H_u$ and $H_d$', fontsize = 17)
plt.savefig('t1ewpt-mHH-xs-tanbe.png')



fig = plt.figure()  # make a new figure : separately from the above 
plt.axis([-1,51,0.0000001,5.5])
ftb4, = plt.plot(tlisttanbe,tlistgamHtob4,'.', color = 'blue', label='constraint',zorder=1)
ftt4, = plt.plot(tlisttanbe,tlistgamHtot4,'.', color = 'red', label='constraint',zorder=1)
ftt, = plt.plot(tlisttanbe,tlistgamHtott,'.', color = 'orange', label='constraint',zorder=1)
ftb, = plt.plot(tlisttanbe,tlistgamHtobb,'.', color = 'green', label='constraint',zorder=1)
fttau, = plt.plot(tlisttanbe,tlistgamHtotata,'.', color = 'purple', label='constraint',zorder=1)
ftc, = plt.plot(tlisttanbe,tlistgamHtocc,'.', color = 'lime', label='constraint',zorder=1)
ftg, = plt.plot(tlisttanbe,tlistgamHtogg,'.', color = 'navy', label='constraint',zorder=1)
fth, = plt.plot(tlisttanbe,tlistgamHtohh,'.', color = 'pink', label='constraint',zorder=1)
myax = plt.gca()
myax.set_yscale('log', nonposy='clip')
myax.annotate(r'$\tan\beta$', xy=(0.8, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
myax.annotate(r'$\Gamma_H /m_H$', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
majorLocator   = MultipleLocator(5)
majorFormatter = FormatStrFormatter('%i')
minorLocator   = MultipleLocator(1)
ymajorLocator   = MultipleLocator(0.5)
ymajorFormatter = FormatStrFormatter('%2.1f')
yminorLocator   = MultipleLocator(0.1)
myax.tick_params(size=3.5)
#myax.tick_params(axis = 'y',zorder = 10) 
myax.xaxis.set_major_locator(majorLocator)
myax.xaxis.set_major_formatter(majorFormatter)
myax.xaxis.set_minor_locator(minorLocator)
#myax.yaxis.set_major_locator(ymajorLocator)
#myax.yaxis.set_major_formatter(ymajorFormatter)
#myax.yaxis.set_minor_locator(yminorLocator)
#first_legend = plt.legend([ftt4, ftt, ftb, fttau, ftc, ftg, fth], [r'$t_4 t$', r'$t \bar t$', r'$b \bar b$', r'$\tau^+ \tau^-$', r'$cc$', r'$gg$', r'$hh$'], frameon=True, loc=1, numpoints = 1,fontsize = 15)
#first_legend = plt.legend([ftb4, ftt, ftb, fttau, ftc, ftg, fth], [r'$b_4 b$', r'$t \bar t$', r'$b \bar b$', r'$\tau^+ \tau^-$', r'$cc$', r'$gg$', r'$hh$'], frameon=True, loc=1, numpoints = 1,fontsize = 15)
first_legend = plt.legend([ftt4, ftb4, ftt, ftb, fttau, ftc, ftg, fth], [r'$t_4 t$', r'$b_4 b$', r'$t \bar t$', r'$b \bar b$', r'$\tau^+ \tau^-$', r'$cc$', r'$gg$', r'$hh$'], frameon=True, loc=1, numpoints = 1,fontsize = 15)
first_legend.set_zorder(100)
plt.gca().add_artist(first_legend)
#second_legend = plt.legend([fgt,fgta,fgtb],[r'                  ',r'                                                        ',r'               '], frameon=False, loc=1, numpoints = 1, fontsize = 12, handletextpad=8,columnspacing = 1.1,borderpad=0.5,labelspacing=0.88)
#second_legend.set_zorder(101)
#frame = second_legend.get_frame()
#frame.set_facecolor('white')
#plt.title(r'Couplings to $H_d$ only', fontsize = 17)
plt.title(r'Couplings to $H_u$ and $H_d$', fontsize = 17)
plt.savefig('t1ewpt-gamH-mHscaled.png')



fig = plt.figure()  # make a new figure : separately from the above 
plt.axis([-0.02,50.2,-0.02,1.02])
fa, = plt.plot(tlisttanbe,tlistbrHtott,'.', color = 'orange', label='constraint',zorder=1)
ft2, = plt.plot(tlisttanbe,tlistbrHtob4,'.', color = 'blue', zorder=3)#color = 'blue', label='constraint',zorder=4)
ft, = plt.plot(tlisttanbe,tlistbrHtot4,'.', color = 'red', label='constraint',zorder=4)
#fb, = plt.plot(tlisttanbe,tlistbrHtohh,'.', color = 'pink', label='constraint',zorder=5)
fc, = plt.plot(tlisttanbe,tlistbrHtobb,'.', color = 'green', label='constraint',zorder=2)
fd, = plt.plot(tlisttanbe,tlistbrHtotata,'.', color = 'purple', label='constraint',zorder=4)
myax = plt.gca()
#myax.set_yscale('log', nonposy='clip')
myax.annotate(r'$\tan\beta$', xy=(0.9, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
myax.annotate(r'BR($H$)', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
#myax.annotate(r'$\frac{\sigma_H}{\sigma_{H_{\rm SM}}} \,\cdot$ BR($H \to h t \bar{t}$)', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
majorLocator   = MultipleLocator(5)
majorFormatter = FormatStrFormatter('%i')
minorLocator   = MultipleLocator(1)
ymajorLocator   = MultipleLocator(0.2)
ymajorFormatter = FormatStrFormatter('%2.1f')
yminorLocator   = MultipleLocator(0.1)
myax.tick_params(size=3.5)
#myax.tick_params(axis = 'y',zorder = 10) 
myax.xaxis.set_major_locator(majorLocator)
myax.xaxis.set_major_formatter(majorFormatter)
myax.xaxis.set_minor_locator(minorLocator)
myax.yaxis.set_major_locator(ymajorLocator)
myax.yaxis.set_major_formatter(ymajorFormatter)
myax.yaxis.set_minor_locator(yminorLocator)
#first_legend = plt.legend([ft, fa, fc, fd], [r'BR($H \to t_4 t$)', r'BR($H \to t \bar t$)', r'BR($H \to b \bar b$)', r'BR($H \to \tau^+ \tau^-$)'], frameon=True, loc=1, numpoints = 1,fontsize = 15)
#first_legend = plt.legend([ft2, fa, fc, fd], [r'BR($H \to b_4 b$)', r'BR($H \to t \bar t$)', r'BR($H \to b \bar b$)', r'BR($H \to \tau^+ \tau^-$)'], frameon=True, loc=1, numpoints = 1,fontsize = 15)
first_legend = plt.legend([ft, ft2, fa, fc, fd], [r'BR($H \to t_4 t$)', r'BR($H \to b_4 b$)', r'BR($H \to t \bar t$)', r'BR($H \to b \bar b$)', r'BR($H \to \tau^+ \tau^-$)'], frameon=True, loc=1, numpoints = 1,fontsize = 15)
#first_legend = plt.legend([ft, fa, fb, fc, fd, fe, ff], [r'BR($H \to t_4 t$)', r'BR($H \to t t$)', r'BR($H \to h h$)', r'BR($H \to b b$)', r'BR($H \to \tau \tau$)', r'BR($H \to c c$)', r'BR($H \to g g$)'], frameon=True, loc=1, numpoints = 1,fontsize = 15)
first_legend.set_zorder(100)
plt.gca().add_artist(first_legend)
#second_legend = plt.legend([fgt,fgta,fgtb],[r'                  ',r'                                                        ',r'               '], frameon=False, loc=1, numpoints = 1, fontsize = 12, handletextpad=8,columnspacing = 1.1,borderpad=0.5,labelspacing=0.88)
#second_legend.set_zorder(101)
#frame = second_legend.get_frame()
#frame.set_facecolor('white')
#myax.annotate(r'$m_H = 900$ GeV, $m_{t_4}$ = 600 GeV', xy=(0.6, 0.6), ha='left', va='top', xycoords='axes fraction', fontsize=15)
#plt.title(r'Couplings to $H_d$ only', fontsize = 17)
plt.title(r'Couplings to $H_u$ and $H_d$', fontsize = 17)
plt.savefig('t1ewpt-tanbe-brHs.png')




fig = plt.figure()  # make a new figure : separately from the above 
#plt.axis([0.,0.005,0.,0.005])
#plt.axis([900.,4100,0.000001,100.])
plt.axis([-1.5,51.5,0.00000001,1.])
#ft, = plt.plot(tlisttanbe,tlisttotalxs,'.', color = 'navy', label='constraint',zorder=1)
fo, = plt.plot(olisttanbe,olisttotalxs,'.', color = 'orange', label='constraint',zorder=1)
fp, = plt.plot(plisttanbe,plisttotalxs,'.', color = 'navy', label='constraint',zorder=1)
fq, = plt.plot(qlisttanbe,qlisttotalxs,'.', color = 'purple', label='constraint',zorder=1)
#ftd, = plt.plot(tdlisttanbe,tdlisttotalxs,'.', color = 'black', label='constraint',zorder=1)
#plt.plot(linex,linex,'-', color = 'red')
myax = plt.gca()
myax.set_yscale('log', nonposy='clip')
#myax.annotate(r'$m_H$', xy=(0.9, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
myax.annotate(r'$\tan\beta$', xy=(0.8, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
myax.annotate(r'$\sigma(p p \to H)$ [pb]', xy=(-0.12, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
majorLocator   = MultipleLocator(5)
majorFormatter = FormatStrFormatter('%i')
minorLocator   = MultipleLocator(1)
ymajorLocator   = MultipleLocator(0.5)
ymajorFormatter = FormatStrFormatter('%2.1f')
yminorLocator   = MultipleLocator(0.1)
myax.tick_params(size=3.5)
#myax.tick_params(axis = 'y',zorder = 10) 
myax.xaxis.set_major_locator(majorLocator)
myax.xaxis.set_major_formatter(majorFormatter)
myax.xaxis.set_minor_locator(minorLocator)
#myax.yaxis.set_major_locator(ymajorLocator)
#myax.yaxis.set_major_formatter(ymajorFormatter)
#myax.yaxis.set_minor_locator(yminorLocator)
first_legend = plt.legend([fo, fp, fq], [r'$m_H = 1.5$ TeV', r'$m_H = 2.5$ TeV', r'$m_H = 4$ TeV'], frameon=True, loc=4, numpoints = 1,fontsize = 15)
#first_legend.set_zorder(100)
#plt.gca().add_artist(first_legend)
#second_legend = plt.legend([fgt,fgta,fgtb],[r'                  ',r'                                                        ',r'               '], frameon=False, loc=1, numpoints = 1, fontsize = 12, handletextpad=8,columnspacing = 1.1,borderpad=0.5,labelspacing=0.88)
#second_legend.set_zorder(101)
#frame = second_legend.get_frame()
#frame.set_facecolor('white')
#plt.title(r'Couplings to $H_d$ only', fontsize = 17)
plt.title(r'Couplings to $H_u$ and $H_d$', fontsize = 17)
plt.savefig('t1ewpt-tanbe-xs.png')





fig = plt.figure()  # make a new figure : separately from the above 
plt.axis([800.,4050,800.,4050.])
fto, = plt.plot(otlistmHH,otlistmt4,'.', color = 'red', label='constraint',zorder=1)
ftp, = plt.plot(ptlistmHH,ptlistmt4,'.', color = 'blue', label='constraint',zorder=1)
ftq, = plt.plot(qtlistmHH,qtlistmt4,'.', color = 'green', label='constraint',zorder=1)
#fto, = plt.plot(otlistmHH,otlistmb4,'.', color = 'red', label='constraint',zorder=1)
#ftp, = plt.plot(ptlistmHH,ptlistmb4,'.', color = 'blue', label='constraint',zorder=1)
#ftq, = plt.plot(qtlistmHH,qtlistmb4,'.', color = 'green', label='constraint',zorder=1)
myax = plt.gca()
#myax.set_yscale('log', nonposy='clip')
myax.annotate(r'$m_H$ [GeV]', xy=(0.75, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
myax.annotate(r'$m_{t_4}$ [GeV]', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
#myax.annotate(r'$m_{b_4}$ [GeV]', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
#myax.annotate(r'BR($H \to t_4 t$) $> 20$ %', xy=(0.2, 0.7), ha='left', va='top', xycoords='axes fraction', fontsize=17)#, rotation = 90)
myax.annotate(r'BR($H \to b_4 b$) $> 50$ %', xy=(0.2, 0.7), ha='left', va='top', xycoords='axes fraction', fontsize=17)#, rotation = 90)
majorLocator   = MultipleLocator(500)
majorFormatter = FormatStrFormatter('%i')
minorLocator   = MultipleLocator(100)
#ymajorLocator   = MultipleLocator(5)
#ymajorFormatter = FormatStrFormatter('%2.1f')
#yminorLocator   = MultipleLocator(1)
myax.tick_params(size=3.5)
#myax.tick_params(axis = 'y',zorder = 10) 
myax.xaxis.set_major_locator(majorLocator)
myax.xaxis.set_major_formatter(majorFormatter)
myax.xaxis.set_minor_locator(minorLocator)
myax.yaxis.set_major_locator(majorLocator)
myax.yaxis.set_major_formatter(majorFormatter)
myax.yaxis.set_minor_locator(minorLocator)
#first_legend = plt.legend([fta, ftc, fte, ftf, ftg], [r'$\tan\beta = 1$', r'$\tan\beta = 7$', r'$\tan\beta = 10$', r'$\tan\beta = 20$', r'$\tan\beta = 50$'], frameon=True, loc=1, numpoints = 1,fontsize = 15)
first_legend = plt.legend([fto, ftp, ftq], [r'$\tan\beta = 1$', r'$\tan\beta = 7$', r'$\tan\beta = 50$'], frameon=True, loc=2, numpoints = 1,fontsize = 15)
first_legend.set_zorder(100)
plt.gca().add_artist(first_legend)
#second_legend = plt.legend([fgt,fgta,fgtb],[r'                  ',r'                                                        ',r'               '], frameon=False, loc=1, numpoints = 1, fontsize = 12, handletextpad=8,columnspacing = 1.1,borderpad=0.5,labelspacing=0.88)
#second_legend.set_zorder(101)
#frame = second_legend.get_frame()
#frame.set_facecolor('white')
#myax.annotate(r'$m_H = 900$ GeV, $m_{t_4}$ = 600 GeV', xy=(0.6, 0.6), ha='left', va='top', xycoords='axes fraction', fontsize=15)
plt.title(r'Couplings to $H_u$ only', fontsize = 17)
#plt.title(r'Couplings to $H_u$ and $H_d$', fontsize = 17)
plt.savefig('t1ewpt-mHH-mt4-tanbe.png')




fig = plt.figure()  # make a new figure : separately from the above 
plt.axis([980.,3050.,0.00001,1.02])
#ftb, = plt.plot(tblistmHH,tblistbrour,'.', color = 'red', label='constraint',zorder=1)
#ftc, = plt.plot(tclistmHH,tclistbrour,'.', color = 'blue', label='constraint',zorder=4)
#fte, = plt.plot(telistmHH,telistbrour,'.', color = 'navy', label='constraint',zorder=4)
#ftf, = plt.plot(tflistmHH,tflistbrour,'.', color = 'green', label='constraint',zorder=5)
#ftg, = plt.plot(tglistmHH,tglistbrour,'.', color = 'purple', label='constraint',zorder=2)
#ftb, = plt.plot(tblistmHH,tblistsigBRw,'.', color = 'red', label='constraint',zorder=1)
#ftc, = plt.plot(tclistmHH,tclistsigBRw,'.', color = 'blue', label='constraint',zorder=4)
#fte, = plt.plot(telistmHH,telistsigBRw,'.', color = 'navy', label='constraint',zorder=4)
#ftf, = plt.plot(tflistmHH,tflistsigBRw,'.', color = 'green', label='constraint',zorder=5)
#ftg, = plt.plot(tglistmHH,tglistsigBRw,'.', color = 'purple', label='constraint',zorder=2)
ftb, = plt.plot(tblistmHH,tblistsigBRz,'.', color = 'red', label='constraint',zorder=1)
ftc, = plt.plot(tclistmHH,tclistsigBRz,'.', color = 'blue', label='constraint',zorder=4)
fte, = plt.plot(telistmHH,telistsigBRz,'.', color = 'orange', label='constraint',zorder=4)
ftf, = plt.plot(tflistmHH,tflistsigBRz,'.', color = 'green', label='constraint',zorder=5)
ftg, = plt.plot(tglistmHH,tglistsigBRz,'.', color = 'purple', label='constraint',zorder=2)
#ftb, = plt.plot(tblistmHH,tblistsigb4BRz,'.', color = 'red', label='constraint',zorder=1)
#ftc, = plt.plot(tclistmHH,tclistsigb4BRz,'.', color = 'blue', label='constraint',zorder=4)
#fte, = plt.plot(telistmHH,telistsigb4BRz,'.', color = 'orange', label='constraint',zorder=4)
#ftf, = plt.plot(tflistmHH,tflistsigb4BRz,'.', color = 'green', label='constraint',zorder=5)
#ftg, = plt.plot(tglistmHH,tglistsigb4BRz,'.', color = 'purple', label='constraint',zorder=2)
myax = plt.gca()
#myax.set_yscale('log', nonposy='clip')
myax.annotate(r'$m_H$ [GeV]', xy=(0.9, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
#myax.annotate(r'BR($H \to h t \bar{t}$)', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
#myax.annotate(r'($\sigma_H / \sigma_H^{\rm SM}$) $\cdot$ BR($H \to t_4 t$) $\cdot$ BR($t_4 \to h t$)', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=19, rotation = 90)
#myax.annotate(r'($\sigma_H / \sigma_H^{\rm SM}$) $\cdot$ BR($H \to t_4 t$) $\cdot$ BR($t_4 \to W b$)', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=19, rotation = 90)
myax.annotate(r'($\sigma_H / \sigma_H^{\rm SM}$) $\cdot$ BR($H \to t_4 t$) $\cdot$ BR($t_4 \to Z t$)', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=19, rotation = 90)
#myax.annotate(r'($\sigma_H / \sigma_H^{\rm SM}$) $\cdot$ BR($H \to b_4 b$) $\cdot$ BR($b_4 \to Z b$)', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=19, rotation = 90)
majorLocator   = MultipleLocator(500)
majorFormatter = FormatStrFormatter('%i')
minorLocator   = MultipleLocator(100)
ymajorLocator   = MultipleLocator(0.2)
ymajorFormatter = FormatStrFormatter('%2.1f')
yminorLocator   = MultipleLocator(0.1)
myax.tick_params(size=3.5)
myax.set_yscale('log', nonposy='clip')
#myax.tick_params(axis = 'y',zorder = 10) 
myax.xaxis.set_major_locator(majorLocator)
myax.xaxis.set_major_formatter(majorFormatter)
myax.xaxis.set_minor_locator(minorLocator)
#myax.yaxis.set_major_locator(ymajorLocator)
#myax.yaxis.set_major_formatter(ymajorFormatter)
#myax.yaxis.set_minor_locator(yminorLocator)
first_legend = plt.legend([ftb, ftc, fte, ftf, ftg], [r'$\tan\beta < 1$', r'$1 < \tan\beta < 5$', r'$5 < \tan\beta < 10$', r'$10 < \tan\beta < 20$', r'$20 < \tan\beta < 50$'], frameon=True, loc=4, numpoints = 1, fontsize = 15)
first_legend.set_zorder(100)
#plt.gca().add_artist(first_legend)
#second_legend = plt.legend([fgt,fgta,fgtb],[r'                  ',r'                                                        ',r'               '], frameon=False, loc=1, numpoints = 1, fontsize = 12, handletextpad=8,columnspacing = 1.1,borderpad=0.5,labelspacing=0.88)
#second_legend.set_zorder(101)
#frame = second_legend.get_frame()
#frame.set_facecolor('white')
#myax.annotate(r'$m_H = 900$ GeV, $m_{t_4}$ = 600 GeV', xy=(0.6, 0.6), ha='left', va='top', xycoords='axes fraction', fontsize=15)
plt.title(r'Couplings to $H_u$ only', fontsize=17)
#plt.title(r'Couplings to $H_u$ and $H_d$', fontsize = 17)
#plt.savefig('t1ewpt-brours.png')
#plt.savefig('t1ewpt-sigBR.png')
#plt.savefig('t1ewpt-sigBRw.png')
plt.savefig('t1ewpt-sigBRz.png')



fig = plt.figure()  # make a new figure : separately from the above 
plt.axis([800,2000,-0.02,1.02])
#ft, = plt.plot(tlistmb4,tlistdoubfr,'.', color = 'blue', label='constraint',zorder=1)
ft, = plt.plot(tlistmt4,tlistdoubfr,'.', color = 'blue', label='constraint',zorder=1)
myax = plt.gca()
#myax.set_yscale('log', nonposy='clip')
#myax.annotate(r'$m_{b_4}$ [GeV]', xy=(0.8, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
myax.annotate(r'$m_{t_4}$ [GeV]', xy=(0.8, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
myax.annotate(r'doublet fraction', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
majorLocator   = MultipleLocator(1000)
majorFormatter = FormatStrFormatter('%i')
minorLocator   = MultipleLocator(100)
ymajorLocator   = MultipleLocator(0.5)
ymajorFormatter = FormatStrFormatter('%2.1f')
yminorLocator   = MultipleLocator(0.1)
myax.tick_params(size=3.5)
#myax.tick_params(axis = 'y',zorder = 10) 
#myax.xaxis.set_major_locator(majorLocator)
#myax.xaxis.set_major_formatter(majorFormatter)
#myax.xaxis.set_minor_locator(minorLocator)
#myax.yaxis.set_major_locator(ymajorLocator)
#myax.yaxis.set_major_formatter(ymajorFormatter)
#myax.yaxis.set_minor_locator(yminorLocator)
#first_legend = plt.legend([ft, fxs1, fxs2], [r'$p p \to H \to t_4 t$', r'$p p \to t_4 b j$ for $g_{L,R}^{W t_4 b} = 0.1$', r'$p p \to t_4 t j$ for $g_{L,R}^{Z t_4 t} = 0.1$'], frameon=True, loc=1, numpoints = 1,fontsize = 15)
#first_legend.set_zorder(100)
#plt.gca().add_artist(first_legend)
#second_legend = plt.legend([fgt,fgta,fgtb],[r'                  ',r'                                                        ',r'               '], frameon=False, loc=1, numpoints = 1, fontsize = 12, handletextpad=8,columnspacing = 1.1,borderpad=0.5,labelspacing=0.88)
#second_legend.set_zorder(101)
#frame = second_legend.get_frame()
#frame.set_facecolor('white')
plt.title(r'Couplings to $H_d$ only', fontsize=17)
plt.savefig('t1ewpt-mVLQ-doubfr.png')



fig = plt.figure()  # make a new figure : separately from the above 
plt.axis([-0.02,1.02,-0.02,1.02])
#plt.axis([-0.02,5.02,-0.02,1.02]) #test
#ft, = plt.plot(tlistbrb4t4w,tlistbrb4w,'.', color = 'gray', label='constraint',zorder=1)
#fa, = plt.plot(alistbrb4t4w,alistbrb4w,'.', color = 'red', label='constraint',zorder=1)
#fb, = plt.plot(blistbrb4t4w,blistbrb4w,'.', color = 'purple', label='constraint',zorder=1)
#fc, = plt.plot(clistbrb4t4w,clistbrb4w,'.', color = 'cyan', label='constraint',zorder=1)
#fd, = plt.plot(dlistbrb4t4w,dlistbrb4w,'.', color = 'blue', label='constraint',zorder=1)
#fd, = plt.plot(dlistbrb4t4w,dlistbrb4w,'.', color = 'navy', label='constraint',zorder=1)
ft, = plt.plot(tlistbrt4b4w,tlistbrt4w,'.', color = 'gray', label='constraint',zorder=1)
fa, = plt.plot(alistbrt4b4w,alistbrt4w,'.', color = 'red', label='constraint',zorder=1)
fb, = plt.plot(blistbrt4b4w,blistbrt4w,'.', color = 'purple', label='constraint',zorder=1)
fc, = plt.plot(clistbrt4b4w,clistbrt4w,'.', color = 'cyan', label='constraint',zorder=1)
fd, = plt.plot(dlistbrt4b4w,dlistbrt4w,'.', color = 'blue', label='constraint',zorder=1)
#fd, = plt.plot(dlistbrb4t4w,dlistbrb4w,'.', color = 'navy', label='constraint',zorder=1)
myax = plt.gca()
#myax.set_yscale('log', nonposy='clip')
#myax.annotate(r'BR($b_4 \to W t_4$)', xy=(0.8, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
#myax.annotate(r'BR($b_4 \to Wb$)', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
myax.annotate(r'BR($t_4 \to W b_4$)', xy=(0.8, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
myax.annotate(r'BR($t_4 \to Wt$)', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
majorLocator   = MultipleLocator(1000)
majorFormatter = FormatStrFormatter('%i')
minorLocator   = MultipleLocator(100)
ymajorLocator   = MultipleLocator(0.5)
ymajorFormatter = FormatStrFormatter('%2.1f')
yminorLocator   = MultipleLocator(0.1)
myax.tick_params(size=3.5)
#myax.tick_params(axis = 'y',zorder = 10) 
#myax.xaxis.set_major_locator(majorLocator)
#myax.xaxis.set_major_formatter(majorFormatter)
#myax.xaxis.set_minor_locator(minorLocator)
#myax.yaxis.set_major_locator(ymajorLocator)
#myax.yaxis.set_major_formatter(ymajorFormatter)
#myax.yaxis.set_minor_locator(yminorLocator)
#first_legend = plt.legend([ft, fxs1, fxs2], [r'$p p \to H \to t_4 t$', r'$p p \to t_4 b j$ for $g_{L,R}^{W t_4 b} = 0.1$', r'$p p \to t_4 t j$ for $g_{L,R}^{Z t_4 t} = 0.1$'], frameon=True, loc=1, numpoints = 1,fontsize = 15)
#first_legend.set_zorder(100)
#plt.gca().add_artist(first_legend)
#second_legend = plt.legend([fgt,fgta,fgtb],[r'                  ',r'                                                        ',r'               '], frameon=False, loc=1, numpoints = 1, fontsize = 12, handletextpad=8,columnspacing = 1.1,borderpad=0.5,labelspacing=0.88)
#second_legend.set_zorder(101)
#frame = second_legend.get_frame()
#frame.set_facecolor('white')
plt.title(r'Couplings to $H_d$ only', fontsize=17)
plt.savefig('t1ewpt-test1.png')
#plt.savefig('t1ewpt-test2.png')
#plt.savefig('t1ewpt-test3.png')
#plt.savefig('t1ewpt-test4.png')



"""
fig = plt.figure()  # make a new figure : separately from the above 
plt.axis([1000.,4000.,-2.,2.])
fo, = plt.plot(olistmt4,olistlamHt44,'.', color = 'orange', label='constraint',zorder=1)
ft, = plt.plot(tlistmt4,tlistlamHt44,'.', color = 'navy', label='constraint',zorder=2)
#fo, = plt.plot(olistmb4,olistlamHb44,'.', color = 'red', label='constraint',zorder=1)
#ft, = plt.plot(tlistmb4,tlistlamHb44,'.', color = 'blue', label='constraint',zorder=2)
#plt.plot(linex,linex,'-', color = 'red')
myax = plt.gca()
#myax.set_yscale('log', nonposy='clip')
myax.annotate(r'$m_{t_4}$ [GeV]', xy=(0.9, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
#myax.annotate(r'$m_{b_4}$ [GeV]', xy=(0.9, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
myax.annotate(r'$\lambda^H_{t_4 t_4}$', xy=(-0.12, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
#myax.annotate(r'$\lambda^H_{b_4 b_4}$', xy=(-0.12, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
majorLocator   = MultipleLocator(1000)
majorFormatter = FormatStrFormatter('%i')
minorLocator   = MultipleLocator(100)
ymajorLocator   = MultipleLocator(0.5)
ymajorFormatter = FormatStrFormatter('%2.1f')
yminorLocator   = MultipleLocator(0.1)
myax.tick_params(size=3.5)
#myax.tick_params(axis = 'y',zorder = 10) 
#myax.xaxis.set_major_locator(majorLocator)
#myax.xaxis.set_major_formatter(majorFormatter)
#myax.xaxis.set_minor_locator(minorLocator)
#myax.yaxis.set_major_locator(ymajorLocator)
#myax.yaxis.set_major_formatter(ymajorFormatter)
#myax.yaxis.set_minor_locator(yminorLocator)
#first_legend = plt.legend([fo, fp, fq], [r'$m_H = 1.5$ TeV', r'$m_H = 2.5$ TeV', r'$m_H = 4$ TeV'], frameon=True, loc=4, numpoints = 1,fontsize = 15)
#first_legend.set_zorder(100)
#plt.gca().add_artist(first_legend)
#second_legend = plt.legend([fgt,fgta,fgtb],[r'                  ',r'                                                        ',r'               '], frameon=False, loc=1, numpoints = 1, fontsize = 12, handletextpad=8,columnspacing = 1.1,borderpad=0.5,labelspacing=0.88)
#second_legend.set_zorder(101)
#frame = second_legend.get_frame()
#frame.set_facecolor('white')
plt.title(r'Couplings to $H_u$ only')
plt.savefig('t1ewpt-couplings-test1.png')



fig = plt.figure()  # make a new figure : separately from the above 
#plt.axis([0.0000000001,1.1,0.0000000001,1.1])
plt.axis([-0.5,0.5,-0.5,0.5])
ft, = plt.plot(tlistlamHt44,tlistlamHb44,'.', color = 'navy', label='constraint',zorder=1)
#plt.plot(compx, compx, '--', color = 'black')
myax = plt.gca()
#myax.set_xscale('log', nonposy='clip')
#myax.set_yscale('log', nonposy='clip')
myax.annotate(r'$\lambda^H_{t_4 t_4}$', xy=(0.8, -0.045), ha='left', va='top', xycoords='axes fraction', fontsize=20)
myax.annotate(r'$\lambda^H_{b_4 b_4}$', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
majorLocator   = MultipleLocator(0.2)
majorFormatter = FormatStrFormatter('%2.1f')
minorLocator   = MultipleLocator(0.1)
ymajorLocator   = MultipleLocator(0.2)
ymajorFormatter = FormatStrFormatter('%2.1f')
yminorLocator   = MultipleLocator(0.1)
myax.tick_params(size=3.5)
#myax.tick_params(axis = 'y',zorder = 10) 
#myax.xaxis.set_major_locator(majorLocator)
#myax.xaxis.set_major_formatter(majorFormatter)
#myax.xaxis.set_minor_locator(minorLocator)
#myax.yaxis.set_major_locator(ymajorLocator)
#myax.yaxis.set_major_formatter(ymajorFormatter)
#myax.yaxis.set_minor_locator(yminorLocator)
#first_legend = plt.legend([ft, fa, fb], [r'BR($t_4 \to h t$) < 0.3', r'0.3 < BR($t_4 \to h t$) < 0.5', r'BR($t_4 \to h t$) > 0.5'], frameon=True, loc=3, numpoints = 1,fontsize = 15)
#first_legend = plt.legend([fta, ftb, ftc, ftd], [r'$\tan\beta <  1$', r'$1  < \tan\beta <  5$', r'$5  < \tan\beta <  10$', r'$\tan\beta >  10$'], frameon=True, loc=1, numpoints = 1,fontsize = 15)
#first_legend.set_zorder(100)
#plt.gca().add_artist(first_legend)
#second_legend = plt.legend([fgt,fgta,fgtb],[r'                  ',r'                                                        ',r'               '], frameon=False, loc=1, numpoints = 1, fontsize = 12, handletextpad=8,columnspacing = 1.1,borderpad=0.5,labelspacing=0.88)
#second_legend.set_zorder(101)
#frame = second_legend.get_frame()
#frame.set_facecolor('white')
plt.title(r'Couplings to $H_u$ only')
plt.savefig('t1ewpt-couplings-test2.png')



fig = plt.figure()  # make a new figure : separately from the above 
plt.axis([-0.01,1.02,-0.01,1.02])
fa, = plt.plot(alistbrb4H0,alistbrb4HH,'.', color = 'red', zorder=2) #s3
fb, = plt.plot(blistbrb4H0,blistbrb4HH,'.', color = 'purple', zorder=1) #s3
fc, = plt.plot(clistbrb4H0,clistbrb4HH,'.', color = 'cyan', zorder=4) #s3
fd, = plt.plot(dlistbrb4H0,dlistbrb4HH,'.', color = 'blue', zorder=3) #s3
plt.plot(compx, brbound, '-', color = 'black')
myax = plt.gca()
#myax.set_yscale('log', nonposy='clip')
myax.annotate(r'BR($b_4 \to H b$)', xy=(0.8, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=22)
#myax.annotate(r'BR($t_4 \to H t$)', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
#myax.annotate(r'BR($b_4 \to H b$)', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=22, rotation = 90)
#myax.annotate(r'BR($t_4 \to H^\pm b$)', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
myax.annotate(r'BR($b_4 \to H^\pm t$)', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=22, rotation = 90)
#myax.annotate(r'$M_Q, M_T, M_B \in [100, 4000]$ GeV, $M_H \in [500, 4000]$ GeV', xy=(0.1, 0.93), ha='left', va='top', xycoords='axes fraction', fontsize=16)
#myax.annotate(r'$M_Q, M_T, M_B, m_H \in [500, 4000]$ GeV', xy=(0.5, 0.8), ha='left', va='top', xycoords='axes fraction', fontsize=16)
#myax.annotate(r'$M_Q, M_T, M_B, m_H \in [500, 4000]$ GeV', xy=(0.42, 0.4), ha='left', va='top', xycoords='axes fraction', fontsize=16)
#myax.annotate(r'$m_{b_4} > m_H + m_b$', xy=(0.6, 0.3), ha='left', va='top', xycoords='axes fraction', fontsize=16)
#myax.annotate(r'$m_{b_4} > m_H + m_t$', xy=(0.6, 0.3), ha='left', va='top', xycoords='axes fraction', fontsize=16)
#myax.annotate(r'$M_H \in [500, 4000]$ GeV', xy=(0.6, 0.7), ha='left', va='top', xycoords='axes fraction', fontsize=16)
#myax.annotate(r'Red: BR($b_4 \to h b$)', xy=(0.7, 0.7), ha='left', va='top', xycoords='axes fraction', fontsize=16)
majorLocator   = MultipleLocator(500)
majorFormatter = FormatStrFormatter('%i')
minorLocator   = MultipleLocator(100)
ymajorLocator   = MultipleLocator(0.2)
ymajorFormatter = FormatStrFormatter('%2.1f')
yminorLocator   = MultipleLocator(0.1)
myax.tick_params(size=3.5)
#myax.tick_params(axis = 'y',zorder = 10) 
#myax.xaxis.set_major_locator(majorLocator)
#myax.xaxis.set_major_formatter(majorFormatter)
#myax.xaxis.set_minor_locator(minorLocator)
#myax.yaxis.set_major_locator(ymajorLocator)
#myax.yaxis.set_major_formatter(ymajorFormatter)
#myax.yaxis.set_minor_locator(yminorLocator)
#first_legend = plt.legend([fta, ftb, ftc, ftd], [r'$\tan\beta <  1$', r'$1  < \tan\beta <  5$', r'$5  < \tan\beta <  10$', r'$\tan\beta >  10$'], frameon=True, loc=1, numpoints = 1,fontsize = 15)
#first_legend.set_zorder(100)
#plt.gca().add_artist(first_legend)
#second_legend = plt.legend([fgt,fgta,fgtb],[r'                  ',r'                                                        ',r'               '], frameon=False, loc=1, numpoints = 1, fontsize = 12, handletextpad=8,columnspacing = 1.1,borderpad=0.5,labelspacing=0.88)
#second_legend.set_zorder(101)
#frame = second_legend.get_frame()
#frame.set_facecolor('white')
#plt.title(r'Couplings to $H_u$ and $H_d$')
plt.title(r'Couplings to $H_d$ only')
plt.savefig('brb4HH-brb4Hpm.png')


fig = plt.figure()  # make a new figure : separately from the above 
plt.axis([-0.01,1.02,-0.01,1.02])
fa, = plt.plot(alistbrt4HH,alistbrt4h,'.', color = 'red', label='constraint',zorder=4)
fb, = plt.plot(blistbrt4HH,blistbrt4h,'.', color = 'purple', label='constraint',zorder=3)
fc, = plt.plot(clistbrt4HH,clistbrt4h,'.', color = 'cyan', label='constraint',zorder=2)
fd, = plt.plot(dlistbrt4HH,dlistbrt4h,'.', color = 'blue', label='constraint',zorder=1)
#fa, = plt.plot(alistbrb4HH,alistbrb4h,'.', color = 'red', label='constraint',zorder=3)
#fb, = plt.plot(blistbrb4HH,blistbrb4h,'.', color = 'purple', label='constraint',zorder=4)
#fc, = plt.plot(clistbrb4HH,clistbrb4h,'.', color = 'cyan', label='constraint',zorder=2)
#fd, = plt.plot(dlistbrb4HH,dlistbrb4h,'.', color = 'blue', label='constraint',zorder=1)
myax = plt.gca()
#myax.set_yscale('log', nonposy='clip')
#myax.annotate(r'BR($b_4 \to H^\pm t$)', xy=(0.73, -0.05), ha='left', va='top', xycoords='axes fraction', fontsize=19)
#myax.annotate(r'BR($b_4 \to H b$)', xy=(0.75, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
#myax.annotate(r'BR($b_4 \to h b$)', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
myax.annotate(r'BR($t_4 \to H^\pm t$)', xy=(0.73, -0.05), ha='left', va='top', xycoords='axes fraction', fontsize=20)
#myax.annotate(r'BR($t_4 \to H t$)', xy=(0.75, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
myax.annotate(r'BR($t_4 \to h t$)', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
#myax.annotate(r'$m_{t_4} < m_{b_4} + M_W$', xy=(0.3, 0.9), ha='left', va='top', xycoords='axes fraction', fontsize=14)
#plt.plot(xt, yt, '--', color = 'black')
majorLocator   = MultipleLocator(0.2)
majorFormatter = FormatStrFormatter('%2.1f')
minorLocator   = MultipleLocator(0.1)
ymajorLocator   = MultipleLocator(0.2)
ymajorFormatter = FormatStrFormatter('%2.1f')
yminorLocator   = MultipleLocator(0.1)
myax.tick_params(size=3.5)
#myax.tick_params(axis = 'y',zorder = 10) 
myax.xaxis.set_major_locator(majorLocator)
myax.xaxis.set_major_formatter(majorFormatter)
myax.xaxis.set_minor_locator(minorLocator)
myax.yaxis.set_major_locator(ymajorLocator)
myax.yaxis.set_major_formatter(ymajorFormatter)
myax.yaxis.set_minor_locator(yminorLocator)
#first_legend = plt.legend([fb, fc], [r'$\tan\beta <  1$', r'$1  < \tan\beta <  5$'], frameon=True, loc=1, numpoints = 1,fontsize = 15)
#first_legend = plt.legend([fb, fc, fe, ff, fg], [r'$\tan\beta <  1$', r'$1  < \tan\beta <  5$', r'$5 < \tan\beta < 10$', r'$10 < \tan\beta < 20$', r'$20 < \tan\beta < 50$'], frameon=True, loc=1, numpoints = 1,fontsize = 15)
#first_legend.set_zorder(100)
#plt.gca().add_artist(first_legend)
#second_legend = plt.legend([fgt,fgta,fgtb],[r'                  ',r'                                                        ',r'               '], frameon=False, loc=1, numpoints = 1, fontsize = 12, handletextpad=8,columnspacing = 1.1,borderpad=0.5,labelspacing=0.88)
#second_legend.set_zorder(101)
#frame = second_legend.get_frame()
#frame.set_facecolor('white')
plt.title(r'Scenario 1') 
#plt.savefig('t1ewpt-brb4extend.png')
plt.savefig('t1ewpt-brt4extend.png')




"""






###### couplings

#print 'talistbbFratio max = ', max(talistbbFratio), 'min = ', min(talistbbFratio)

#print 'tlistbbFratio max = ', max(tlistbbFratio), 'min = ', min(tlistbbFratio)
#print 'tclistbbFratio max = ', max(tclistbbFratio), 'min = ', min(tclistbbFratio)

















#mytx = numpy.array(tlistmt4)
#myty = numpy.array(tlistbrt4h)
#mytz = numpy.array(tlistsingfr)



#print 'max of doublet fraction of t4 = ', max(tlistdoubfr)

#print len(alisttanbe), len(blisttanbe), len(clisttanbe), len(dlisttanbe) #red, purple, cyan, blue







"""

fig = plt.figure()  # make a new figure : separately from the above 
#plt.axis([0.0000000001,1.1,0.0000000001,1.1])
plt.axis([-0.5,0.5,-0.5,0.5])
ft, = plt.plot(tlistlamHbr34,tlistlamHpmbr43,'.', color = 'navy', label='constraint',zorder=1)
ftb, = plt.plot(tblistlamHbr34,tblistlamHpmbr43,'.', color = 'red', label='constraint',zorder=1)
#ft, = plt.plot(tlistlamHb34,tlistlamHpmbr43,'.', color = 'navy', label='constraint',zorder=1)
#ft, = plt.plot(tdlistkaT,tdlistkaQ,'.', color = 'green', label='constraint',zorder=1)
#fa, = plt.plot(tdalistkaT,tdalistkaQ,'.', color = 'orange', label='constraint',zorder=2)
#fb, = plt.plot(tdblistkaT,tdblistkaQ,'.', color = 'navy', label='constraint',zorder=3)
#fa, = plt.plot(dlistkaT,dlistlamHt43,'.', color = 'blue', label='constraint',zorder=4)
#plt.plot(compx, compx, '--', color = 'black')
myax = plt.gca()
#myax.set_xscale('log', nonposy='clip')
#myax.set_yscale('log', nonposy='clip')
myax.annotate(r'$\lambda^H_{b_4 b}$', xy=(0.8, -0.045), ha='left', va='top', xycoords='axes fraction', fontsize=20)
#myax.annotate(r'$\lambda^H_{t b_4}$', xy=(0.8, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
myax.annotate(r'$\lambda^{H^\pm}_{t b_4}$', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
myax.annotate(r'$|\lambda_B| < 0.01$', xy=(0.7, 0.9), ha='left', va='top', xycoords='axes fraction', fontsize=16)
#myax.annotate(r'$\kappa_T$', xy=(0.8, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
#myax.annotate(r'$\kappa_Q$', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
majorLocator   = MultipleLocator(0.2)
majorFormatter = FormatStrFormatter('%2.1f')
minorLocator   = MultipleLocator(0.1)
ymajorLocator   = MultipleLocator(0.2)
ymajorFormatter = FormatStrFormatter('%2.1f')
yminorLocator   = MultipleLocator(0.1)
myax.tick_params(size=3.5)
#myax.tick_params(axis = 'y',zorder = 10) 
#myax.xaxis.set_major_locator(majorLocator)
#myax.xaxis.set_major_formatter(majorFormatter)
#myax.xaxis.set_minor_locator(minorLocator)
#myax.yaxis.set_major_locator(ymajorLocator)
#myax.yaxis.set_major_formatter(ymajorFormatter)
#myax.yaxis.set_minor_locator(yminorLocator)
#first_legend = plt.legend([ft, fa, fb], [r'BR($t_4 \to h t$) < 0.3', r'0.3 < BR($t_4 \to h t$) < 0.5', r'BR($t_4 \to h t$) > 0.5'], frameon=True, loc=3, numpoints = 1,fontsize = 15)
#first_legend = plt.legend([fta, ftb, ftc, ftd], [r'$\tan\beta <  1$', r'$1  < \tan\beta <  5$', r'$5  < \tan\beta <  10$', r'$\tan\beta >  10$'], frameon=True, loc=1, numpoints = 1,fontsize = 15)
#first_legend.set_zorder(100)
#plt.gca().add_artist(first_legend)
#second_legend = plt.legend([fgt,fgta,fgtb],[r'                  ',r'                                                        ',r'               '], frameon=False, loc=1, numpoints = 1, fontsize = 12, handletextpad=8,columnspacing = 1.1,borderpad=0.5,labelspacing=0.88)
#second_legend.set_zorder(101)
#frame = second_legend.get_frame()
#frame.set_facecolor('white')
#plt.title(r'99% singlet-like $t_4$, others 100TeV', fontsize=17)
#plt.title(r'99% singlet-like $t_4$, others 10TeV', fontsize=17)
plt.savefig('t1ewpt-couplings4.png')

print 'len(tlistlamHpmb43) =', len(tlistlamHpmb43)










def mb4w(x):
    return x + MW

xmb4 = numpy.arange(1000., 4000., 1)
ymass = mb4w(xmb4)



fig = plt.figure()  # make a new figure : separately from the above 
plt.axis([1000.,4000.,1000.,4000.])
#fa, = plt.plot(alistmt4,alistmb4,'.', color = 'red', label='constraint',zorder=2)
#fb, = plt.plot(blistmt4,blistmb4,'.', color = 'purple', label='constraint',zorder=1)
#fc, = plt.plot(clistmt4,clistmb4,'.', color = 'cyan', label='constraint',zorder=4)
#fd, = plt.plot(dlistmt4,dlistmb4,'.', color = 'blue', label='constraint',zorder=3)
fa, = plt.plot(alistmb4,alistmt4,'.', color = 'red', label='constraint',zorder=4)
fb, = plt.plot(blistmb4,blistmt4,'.', color = 'purple', label='constraint',zorder=3)
fc, = plt.plot(clistmb4,clistmt4,'.', color = 'cyan', label='constraint',zorder=2)
fd, = plt.plot(dlistmb4,dlistmt4,'.', color = 'blue', label='constraint',zorder=1)
#ft, = plt.plot(tlistmb4,tlistmt4,'.', color = 'navy', label='constraint',zorder=4)
#fa, = plt.plot(alistmt4,alistbrt4w,'.', color = 'red', label='constraint',zorder=4)
#fb, = plt.plot(blistmt4,blistbrt4w,'.', color = 'purple', label='constraint',zorder=3)
#fc, = plt.plot(clistmt4,clistbrt4w,'.', color = 'cyan', label='constraint',zorder=2)
#fd, = plt.plot(dlistmt4,dlistbrt4w,'.', color = 'blue', label='constraint',zorder=3)
fline, = plt.plot(xmb4, ymass, 'k', zorder = 10)
myax = plt.gca()
#myax.set_yscale('log', nonposy='clip')
#myax.annotate(r'$m_{t_4}$ [GeV]', xy=(0.8, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
#myax.annotate(r'$m_{b_4}$', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
myax.annotate(r'$m_{b_4}$ [GeV]', xy=(0.8, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
myax.annotate(r'$m_{t_4}$', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
#myax.annotate(r'$m_{t_4}$ [GeV]', xy=(0.8, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
#myax.annotate(r'BR($t_4 \to W b$)', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
myax.annotate(r'Blue: $b_4$ is doublet for $H_u$ only scenario', xy=(0.1, 0.8), ha='left', va='top', xycoords='axes fraction', fontsize=17)
majorLocator   = MultipleLocator(500)
majorFormatter = FormatStrFormatter('%i')
minorLocator   = MultipleLocator(100)
ymajorLocator   = MultipleLocator(0.2)
ymajorFormatter = FormatStrFormatter('%2.1f')
yminorLocator   = MultipleLocator(0.1)
myax.tick_params(size=3.5)
#myax.tick_params(axis = 'y',zorder = 10) 
#myax.xaxis.set_major_locator(majorLocator)
#myax.xaxis.set_major_formatter(majorFormatter)
#myax.xaxis.set_minor_locator(minorLocator)
#myax.yaxis.set_major_locator(ymajorLocator)
#myax.yaxis.set_major_formatter(ymajorFormatter)
#myax.yaxis.set_minor_locator(yminorLocator)
#first_legend = plt.legend([fta, ftb, ftc, ftd], [r'$\tan\beta <  1$', r'$1  < \tan\beta <  5$', r'$5  < \tan\beta <  10$', r'$\tan\beta >  10$'], frameon=True, loc=1, numpoints = 1,fontsize = 15)
#first_legend.set_zorder(100)
#plt.gca().add_artist(first_legend)
#second_legend = plt.legend([fgt,fgta,fgtb],[r'                  ',r'                                                        ',r'               '], frameon=False, loc=1, numpoints = 1, fontsize = 12, handletextpad=8,columnspacing = 1.1,borderpad=0.5,labelspacing=0.88)
#second_legend.set_zorder(101)
#frame = second_legend.get_frame()
#frame.set_facecolor('white')
#plt.savefig('t1ewpt-mt4-mb4.png')
plt.savefig('t1ewpt-mb4-mt4.png')




fig = plt.figure()  # make a new figure : separately from the above 
plt.axis([0.00001,1.,0.00001,1.]) 
#plt.axis([1000.,4000.,0.,100]) 
#ft, = plt.plot(tlistgamHtot4,tlistkaT,'.', color = 'navy', label='constraint',zorder=1)
ft, = plt.plot(tlistlamHt34,tlistgLW43,'.', color = 'navy', label='constraint',zorder=1)
#fa, = plt.plot(alistgLZ43,alistgLW43,'.', color = 'red', label='constraint',zorder=1)
#fb, = plt.plot(blistgLZ43,blistgLW43,'.', color = 'purple', label='constraint',zorder=2)
#fc, = plt.plot(clistgLZ43,clistgLW43,'.', color = 'cyan', label='constraint',zorder=3)
#fd, = plt.plot(dlistgLZ43,dlistgLW43,'.', color = 'blue', label='constraint',zorder=4)
#fa, = plt.plot(alistmt4,alistggSMratio,'.', color = 'red', label='constraint',zorder=1)
#fb, = plt.plot(blistmt4,blistggSMratio,'.', color = 'purple', label='constraint',zorder=2)
#fc, = plt.plot(clistmt4,clistggSMratio,'.', color = 'cyan', label='constraint',zorder=3)
#fd, = plt.plot(dlistmt4,dlistggSMratio,'.', color = 'blue', label='constraint',zorder=4)
myax = plt.gca()
#myax.set_yscale('log', nonposy='clip')
#myax.annotate(r'$\Gamma(H \to t_4 t)$', xy=(0.9, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
#myax.annotate(r'$\kappa_T$', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
#myax.annotate(r'$g_L^{Z t_4 t}$', xy=(0.9, -0.045), ha='left', va='top', xycoords='axes fraction', fontsize=20)
myax.annotate(r'$\lambda_H^{t t_4}$', xy=(0.9, -0.045), ha='left', va='top', xycoords='axes fraction', fontsize=20)
myax.annotate(r'$g_L^{W t_4 b}$', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
#myax.annotate(r'$m_{t_4}$ [GeV]', xy=(0.9, -0.045), ha='left', va='top', xycoords='axes fraction', fontsize=20)
#myax.annotate(r'ggSMratio', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
majorLocator   = MultipleLocator(0.5)
majorFormatter = FormatStrFormatter('%2.1f')
minorLocator   = MultipleLocator(0.1)
ymajorLocator   = MultipleLocator(0.5)
ymajorFormatter = FormatStrFormatter('%2.1f')
yminorLocator   = MultipleLocator(0.1)
myax.tick_params(size=3.5)
#myax.tick_params(axis = 'y',zorder = 10) 
#myax.xaxis.set_major_locator(majorLocator)
#myax.xaxis.set_major_formatter(majorFormatter)
#myax.xaxis.set_minor_locator(minorLocator)
#myax.yaxis.set_major_locator(ymajorLocator)
#myax.yaxis.set_major_formatter(ymajorFormatter)
#myax.yaxis.set_minor_locator(yminorLocator)
#first_legend = plt.legend([fta, ftb, ftc, ftd], [r'$\tan\beta <  1$', r'$1  < \tan\beta <  5$', r'$5  < \tan\beta <  10$', r'$\tan\beta >  10$'], frameon=True, loc=1, numpoints = 1,fontsize = 15)
#first_legend.set_zorder(100)
#plt.gca().add_artist(first_legend)
#second_legend = plt.legend([fgt,fgta,fgtb],[r'                  ',r'                                                        ',r'               '], frameon=False, loc=1, numpoints = 1, fontsize = 12, handletextpad=8,columnspacing = 1.1,borderpad=0.5,labelspacing=0.88)
#second_legend.set_zorder(101)
#frame = second_legend.get_frame()
#frame.set_facecolor('white')
plt.savefig('t1ewpt-check.png')



fig = plt.figure()  # make a new figure : separately from the above 
plt.axis([-0.02,50.2,-0.02,1.02])
fa, = plt.plot(tlisttanbe,tlistbrHtott,'.', color = 'red', label='constraint',zorder=1)
ft2, = plt.plot(tlisttanbe,tlistbrHtob4,'.', color = 'blue', label='constraint',zorder=4)
ft, = plt.plot(tlisttanbe,tlistbrHtot4,'.', color = 'navy', label='constraint',zorder=4)
fb, = plt.plot(tlisttanbe,tlistbrHtohh,'.', color = 'green', label='constraint',zorder=5)
fc, = plt.plot(tlisttanbe,tlistbrHtobb,'.', color = 'cyan', label='constraint',zorder=2)
fd, = plt.plot(tlisttanbe,tlistbrHtotata,'.', color = 'purple', label='constraint',zorder=3)
myax = plt.gca()
#myax.set_yscale('log', nonposy='clip')
myax.annotate(r'$\tan\beta$', xy=(0.9, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
myax.annotate(r'BR($H$)', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
#myax.annotate(r'$\frac{\sigma_H}{\sigma_{H_{\rm SM}}} \,\cdot$ BR($H \to h t \bar{t}$)', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
majorLocator   = MultipleLocator(5)
majorFormatter = FormatStrFormatter('%i')
minorLocator   = MultipleLocator(1)
ymajorLocator   = MultipleLocator(0.2)
ymajorFormatter = FormatStrFormatter('%2.1f')
yminorLocator   = MultipleLocator(0.1)
myax.tick_params(size=3.5)
#myax.tick_params(axis = 'y',zorder = 10) 
myax.xaxis.set_major_locator(majorLocator)
myax.xaxis.set_major_formatter(majorFormatter)
myax.xaxis.set_minor_locator(minorLocator)
myax.yaxis.set_major_locator(ymajorLocator)
myax.yaxis.set_major_formatter(ymajorFormatter)
myax.yaxis.set_minor_locator(yminorLocator)
#first_legend = plt.legend([ft, fa, fb, fc, fd], [r'BR($H \to t_4 t$)', r'BR($H \to t t$)', r'BR($H \to h h$)', r'BR($H \to b b$)', r'BR($H \to \tau \tau$)'], frameon=True, loc=1, numpoints = 1,fontsize = 15)
#first_legend = plt.legend([ft2, fa, fb, fc, fd], [r'BR($H \to b_4 b$)', r'BR($H \to t t$)', r'BR($H \to h h$)', r'BR($H \to b b$)', r'BR($H \to \tau \tau$)'], frameon=True, loc=1, numpoints = 1,fontsize = 15)
first_legend = plt.legend([ft, ft2, fa, fb, fc, fd], [r'BR($H \to t_4 t$)', r'BR($H \to b_4 b$)', r'BR($H \to t t$)', r'BR($H \to h h$)', r'BR($H \to b b$)', r'BR($H \to \tau \tau$)'], frameon=True, loc=1, numpoints = 1,fontsize = 15)
#first_legend = plt.legend([ft, fa, fb, fc, fd, fe, ff], [r'BR($H \to t_4 t$)', r'BR($H \to t t$)', r'BR($H \to h h$)', r'BR($H \to b b$)', r'BR($H \to \tau \tau$)', r'BR($H \to c c$)', r'BR($H \to g g$)'], frameon=True, loc=1, numpoints = 1,fontsize = 15)
first_legend.set_zorder(100)
#plt.gca().add_artist(first_legend)
#second_legend = plt.legend([fgt,fgta,fgtb],[r'                  ',r'                                                        ',r'               '], frameon=False, loc=1, numpoints = 1, fontsize = 12, handletextpad=8,columnspacing = 1.1,borderpad=0.5,labelspacing=0.88)
#second_legend.set_zorder(101)
#frame = second_legend.get_frame()
#frame.set_facecolor('white')
#myax.annotate(r'$m_H = 900$ GeV, $m_{t_4}$ = 600 GeV', xy=(0.6, 0.6), ha='left', va='top', xycoords='axes fraction', fontsize=15)
plt.title(r'Ratio of the production cross sections, Scenario 3', fontsize=17)
plt.savefig('t1ewpt-tanbe-brHs.png')




fig = plt.figure()  # make a new figure : separately from the above 
plt.axis([-0.01,1.02,-0.01,1.02])
fa, = plt.plot(alistbrt5z,alistbrt5w,'.', color = 'blue', label='constraint',zorder=1)
fb, = plt.plot(blistbrt5z,blistbrt5w,'.', color = 'cyan', label='constraint',zorder=2)
fc, = plt.plot(clistbrt5z,clistbrt5w,'.', color = 'purple', label='constraint',zorder=4)
fd, = plt.plot(dlistbrt5z,dlistbrt5w,'.', color = 'red', label='constraint',zorder=3)
myax = plt.gca()
#myax.set_yscale('log', nonposy='clip')
myax.annotate(r'BR($t_5 \to Z t$)', xy=(0.75, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
myax.annotate(r'BR($t_5 \to W b$)', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
#plt.plot(xt, yt, '--', color = 'black')
majorLocator   = MultipleLocator(0.2)
majorFormatter = FormatStrFormatter('%2.1f')
minorLocator   = MultipleLocator(0.1)
ymajorLocator   = MultipleLocator(0.2)
ymajorFormatter = FormatStrFormatter('%2.1f')
yminorLocator   = MultipleLocator(0.1)
myax.tick_params(size=3.5)
#myax.tick_params(axis = 'y',zorder = 10) 
myax.xaxis.set_major_locator(majorLocator)
myax.xaxis.set_major_formatter(majorFormatter)
myax.xaxis.set_minor_locator(minorLocator)
myax.yaxis.set_major_locator(ymajorLocator)
myax.yaxis.set_major_formatter(ymajorFormatter)
myax.yaxis.set_minor_locator(yminorLocator)
plt.title(r'scenario 1', fontsize=17)
plt.savefig('t1ewpt-brt5s.png')




fig = plt.figure()  # make a new figure : separately from the above 
plt.axis([1000.,3000.,-0.01,1.02])
#fa, = plt.plot(alistmt4,alistbrt4h,'.', color = 'red', label='constraint',zorder=2)
#fb, = plt.plot(blistmt4,blistbrt4h,'.', color = 'purple', label='constraint',zorder=1)
#fc, = plt.plot(clistmt4,clistbrt4h,'.', color = 'cyan', label='constraint',zorder=4)
#fd, = plt.plot(dlistmt4,dlistbrt4h,'.', color = 'blue', label='constraint',zorder=3)
fa, = plt.plot(alistmb4,alistbrb4h,'.', color = 'red', label='constraint',zorder=2)
fb, = plt.plot(blistmb4,blistbrb4h,'.', color = 'purple', label='constraint',zorder=1)
fc, = plt.plot(clistmb4,clistbrb4h,'.', color = 'cyan', label='constraint',zorder=4)
fd, = plt.plot(dlistmb4,dlistbrb4h,'.', color = 'blue', label='constraint',zorder=3)
myax = plt.gca()
#myax.set_yscale('log', nonposy='clip')
myax.annotate(r'$m_{b_4}$ [GeV]', xy=(0.8, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
myax.annotate(r'BR($b_4 \to h b$)', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
#myax.annotate(r'$m_{t_4}$ [GeV]', xy=(0.8, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
#myax.annotate(r'BR($t_4 \to h t$)', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
majorLocator   = MultipleLocator(500)
majorFormatter = FormatStrFormatter('%i')
minorLocator   = MultipleLocator(100)
ymajorLocator   = MultipleLocator(0.2)
ymajorFormatter = FormatStrFormatter('%2.1f')
yminorLocator   = MultipleLocator(0.1)
myax.tick_params(size=3.5)
#myax.tick_params(axis = 'y',zorder = 10) 
myax.xaxis.set_major_locator(majorLocator)
myax.xaxis.set_major_formatter(majorFormatter)
myax.xaxis.set_minor_locator(minorLocator)
myax.yaxis.set_major_locator(ymajorLocator)
myax.yaxis.set_major_formatter(ymajorFormatter)
myax.yaxis.set_minor_locator(yminorLocator)
#first_legend = plt.legend([fta, ftb, ftc, ftd], [r'$\tan\beta <  1$', r'$1  < \tan\beta <  5$', r'$5  < \tan\beta <  10$', r'$\tan\beta >  10$'], frameon=True, loc=1, numpoints = 1,fontsize = 15)
#first_legend.set_zorder(100)
#plt.gca().add_artist(first_legend)
#second_legend = plt.legend([fgt,fgta,fgtb],[r'                  ',r'                                                        ',r'               '], frameon=False, loc=1, numpoints = 1, fontsize = 12, handletextpad=8,columnspacing = 1.1,borderpad=0.5,labelspacing=0.88)
#second_legend.set_zorder(101)
#frame = second_legend.get_frame()
#frame.set_facecolor('white')
plt.savefig('t1ewpt-mb4-brb4h.png')
#plt.savefig('t1ewpt-mt4-brt4h.png')

fig = plt.figure()  # make a new figure : separately from the above 
plt.axis([1000.,3000.,-0.01,1.02])
fa, = plt.plot(alistmb4,alistbrb4z,'.', color = 'red', label='constraint',zorder=2)
fb, = plt.plot(blistmb4,blistbrb4z,'.', color = 'purple', label='constraint',zorder=1)
fc, = plt.plot(clistmb4,clistbrb4z,'.', color = 'cyan', label='constraint',zorder=4)
fd, = plt.plot(dlistmb4,dlistbrb4z,'.', color = 'blue', label='constraint',zorder=3)
#fa, = plt.plot(alistmt4,alistbrt4z,'.', color = 'red', label='constraint',zorder=2)
#fb, = plt.plot(blistmt4,blistbrt4z,'.', color = 'purple', label='constraint',zorder=1)
#fc, = plt.plot(clistmt4,clistbrt4z,'.', color = 'cyan', label='constraint',zorder=4)
#fd, = plt.plot(dlistmt4,dlistbrt4z,'.', color = 'blue', label='constraint',zorder=3)
myax = plt.gca()
#myax.set_yscale('log', nonposy='clip')
myax.annotate(r'$m_{b_4}$ [GeV]', xy=(0.8, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
myax.annotate(r'BR($b_4 \to Z b$)', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
#myax.annotate(r'$m_{t_4}$ [GeV]', xy=(0.8, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
#myax.annotate(r'BR($t_4 \to Z t$)', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
majorLocator   = MultipleLocator(500)
vmajorFormatter = FormatStrFormatter('%i')
minorLocator   = MultipleLocator(100)
ymajorLocator   = MultipleLocator(0.2)
ymajorFormatter = FormatStrFormatter('%2.1f')
yminorLocator   = MultipleLocator(0.1)
myax.tick_params(size=3.5)
#myax.tick_params(axis = 'y',zorder = 10) 
myax.xaxis.set_major_locator(majorLocator)
myax.xaxis.set_major_formatter(majorFormatter)
myax.xaxis.set_minor_locator(minorLocator)
myax.yaxis.set_major_locator(ymajorLocator)
myax.yaxis.set_major_formatter(ymajorFormatter)
myax.yaxis.set_minor_locator(yminorLocator)
#first_legend = plt.legend([fta, ftb, ftc, ftd], [r'$\tan\beta <  1$', r'$1  < \tan\beta <  5$', r'$5  < \tan\beta <  10$', r'$\tan\beta >  10$'], frameon=True, loc=1, numpoints = 1,fontsize = 15)
#first_legend.set_zorder(100)
#plt.gca().add_artist(first_legend)
#second_legend = plt.legend([fgt,fgta,fgtb],[r'                  ',r'                                                        ',r'               '], frameon=False, loc=1, numpoints = 1, fontsize = 12, handletextpad=8,columnspacing = 1.1,borderpad=0.5,labelspacing=0.88)
#second_legend.set_zorder(101)
#frame = second_legend.get_frame()
#frame.set_facecolor('white')
plt.savefig('t1ewpt-mb4-brb4z.png')
#plt.savefig('t1ewpt-mt4-brt4z.png')


## cross section compare

linex = numpy.linspace(0,0.1,1000)

fig = plt.figure()  # make a new figure : separately from the above 
#plt.axis([0.,0.005,0.,0.005])
#plt.axis([900.,4100,0.000001,100.])
plt.axis([0.,51.,0.00000001,0.01])
#ft, = plt.plot(tlistmHH,tlisttotalxs,'.', color = 'navy', label='constraint',zorder=1)
fg, = plt.plot(nlisttanbe,nlisttotalxs,'.', color = 'gray', label='constraint',zorder=1)
ft, = plt.plot(tlisttanbe,tlisttotalxs,'.', color = 'navy', label='constraint',zorder=1)
#fta, = plt.plot(talisttotalt4t,talistmyt4bj,'.', color = 'blue', label='constraint',zorder=1)
#ftab, = plt.plot(tablisttotalt4t,tablistmyt4bj,'.', color = 'red', label='constraint',zorder=1)
#ftac, = plt.plot(taclisttotalt4t,taclistmyt4bj,'.', color = 'green', label='constraint',zorder=1)
#ftad, = plt.plot(tadlisttotalt4t,tadlistmyt4bj,'.', color = 'lime', label='constraint',zorder=1)
plt.plot(linex,linex,'-', color = 'red')
myax = plt.gca()
myax.set_yscale('log', nonposy='clip')
#myax.annotate(r'$m_H$', xy=(0.9, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
myax.annotate(r'$\tan\beta$', xy=(0.8, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
#myax.annotate(r'$\sigma(p p \to H \to t_4 t)$ [pb]', xy=(0.6, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
myax.annotate(r'$\sigma(p p \to H)$ [pb]', xy=(-0.12, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
#myax.annotate(r'$\sigma(p p \to t_4 b j)$ [pb]', xy=(-0.12, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
majorLocator   = MultipleLocator(1000)
majorFormatter = FormatStrFormatter('%i')
minorLocator   = MultipleLocator(100)
ymajorLocator   = MultipleLocator(0.5)
ymajorFormatter = FormatStrFormatter('%2.1f')
yminorLocator   = MultipleLocator(0.1)
myax.tick_params(size=3.5)
#myax.tick_params(axis = 'y',zorder = 10) 
#myax.xaxis.set_major_locator(majorLocator)
#myax.xaxis.set_major_formatter(majorFormatter)
#myax.xaxis.set_minor_locator(minorLocator)
#myax.yaxis.set_major_locator(ymajorLocator)
#myax.yaxis.set_major_formatter(ymajorFormatter)
#myax.yaxis.set_minor_locator(yminorLocator)
#first_legend = plt.legend([ft, fxs1, fxs2], [r'$p p \to H \to t_4 t$', r'$p p \to t_4 b j$ for $g_{L,R}^{W t_4 b} = 0.1$', r'$p p \to t_4 t j$ for $g_{L,R}^{Z t_4 t} = 0.1$'], frameon=True, loc=1, numpoints = 1,fontsize = 15)
#first_legend.set_zorder(100)
#plt.gca().add_artist(first_legend)
#second_legend = plt.legend([fgt,fgta,fgtb],[r'                  ',r'                                                        ',r'               '], frameon=False, loc=1, numpoints = 1, fontsize = 12, handletextpad=8,columnspacing = 1.1,borderpad=0.5,labelspacing=0.88)
#second_legend.set_zorder(101)
#frame = second_legend.get_frame()
#frame.set_facecolor('white')
#plt.savefig('t1ewpt-xs-check.png')
#plt.savefig('t1ewpt-Hxs.png')
plt.title(r'$m_H = 4\,{\rm TeV}$ Scenario 3')
plt.savefig('t1ewpt-tanbe-Hxs.png')




fig = plt.figure()  # make a new figure : separately from the above 
plt.axis([980,3020,0.000001,5.2])
#ft, = plt.plot(tlistmHH,tlisttotalxs,'.', color = 'navy', label='constraint',zorder=1)
ft, = plt.plot(tlistmHH,tlisttotalb4b,'.', color = 'navy', label='constraint',zorder=1)
#ft, = plt.plot(tlistmHH,tlisttotalt4t,'.', color = 'navy', label='constraint',zorder=1)
fxs1, = plt.plot(t4mass,xsbjsingle,'-', color = 'red', label='constraint',zorder=2)
fxs2, = plt.plot(t4mass,xstjsingle,'-', color = 'green', label='constraint',zorder=3)
#fscanxs1, = plt.plot(tlistmHH,tlistmyt4bj,'-', color = 'pink', label='constraint',zorder=4)
#fscanxs2, = plt.plot(tlistmHH,tlistmyt4tj,'-', color = 'cyan', label='constraint',zorder=5)
myax = plt.gca()
myax.set_yscale('log', nonposy='clip')
#myax.annotate(r'$m_{t_4}$', xy=(0.9, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
myax.annotate(r'$m_H$ [GeV]', xy=(0.8, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
myax.annotate(r'$\sigma_{\rm production}$ [pb]', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
majorLocator   = MultipleLocator(1000)
majorFormatter = FormatStrFormatter('%i')
minorLocator   = MultipleLocator(100)
ymajorLocator   = MultipleLocator(0.5)
ymajorFormatter = FormatStrFormatter('%2.1f')
yminorLocator   = MultipleLocator(0.1)
myax.tick_params(size=3.5)
#myax.tick_params(axis = 'y',zorder = 10) 
myax.xaxis.set_major_locator(majorLocator)
myax.xaxis.set_major_formatter(majorFormatter)
myax.xaxis.set_minor_locator(minorLocator)
#myax.yaxis.set_major_locator(ymajorLocator)
#myax.yaxis.set_major_formatter(ymajorFormatter)
#myax.yaxis.set_minor_locator(yminorLocator)
#first_legend = plt.legend([ft, fxs1, fxs2], [r'$p p \to H \to t_4 t$', r'$p p \to t_4 b j$ for $g_{L,R}^{W t_4 b} = 0.1$', r'$p p \to t_4 t j$ for $g_{L,R}^{Z t_4 t} = 0.1$'], frameon=True, loc=1, numpoints = 1,fontsize = 15)
first_legend = plt.legend([ft, fxs1, fxs2], [r'$p p \to H \to b_4 b$', r'$p p \to b_4 b j$ for $g_{L,R}^{Z b_4 b} = 0.1$', r'$p p \to b_4 t j$ for $g_{L,R}^{Z b_4 t} = 0.1$'], frameon=True, loc=1, numpoints = 1,fontsize = 15)
first_legend.set_zorder(100)
plt.gca().add_artist(first_legend)
#second_legend = plt.legend([fgt,fgta,fgtb],[r'                  ',r'                                                        ',r'               '], frameon=False, loc=1, numpoints = 1, fontsize = 12, handletextpad=8,columnspacing = 1.1,borderpad=0.5,labelspacing=0.88)
#second_legend.set_zorder(101)
#frame = second_legend.get_frame()
#frame.set_facecolor('white')
#plt.savefig('t1ewpt-xscompare.png')
plt.savefig('t1ewpt-xscompare-s2.png')




fig = plt.figure()  # make a new figure : separately from the above 
plt.axis([1200.,3500.,0.00001,1.02])
ftb, = plt.plot(tblistmHH,tblistbrHtotata,'.', color = 'red', label='constraint',zorder=1)
ftc, = plt.plot(tclistmHH,tclistbrHtotata,'.', color = 'blue', label='constraint',zorder=4)
fte, = plt.plot(telistmHH,telistbrHtotata,'.', color = 'navy', label='constraint',zorder=4)
ftf, = plt.plot(tflistmHH,tflistbrHtotata,'.', color = 'green', label='constraint',zorder=5)
ftg, = plt.plot(tglistmHH,tglistbrHtotata,'.', color = 'purple', label='constraint',zorder=2)
myax = plt.gca()
#myax.set_yscale('log', nonposy='clip')
myax.annotate(r'$m_H$ [GeV]', xy=(0.9, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
myax.annotate(r'BR($H \to \tau^+ \tau^-$)', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
#myax.annotate(r'$\frac{\sigma_H}{\sigma_{H_{\rm SM}}} \,\cdot$ BR($H \to h t \bar{t}$)', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
majorLocator   = MultipleLocator(5)
majorFormatter = FormatStrFormatter('%i')
minorLocator   = MultipleLocator(1)
ymajorLocator   = MultipleLocator(0.2)
ymajorFormatter = FormatStrFormatter('%2.1f')
yminorLocator   = MultipleLocator(0.1)
myax.tick_params(size=3.5)
myax.set_yscale('log', nonposy='clip')
#myax.tick_params(axis = 'y',zorder = 10) 
#myax.xaxis.set_major_locator(majorLocator)
#myax.xaxis.set_major_formatter(majorFormatter)
#myax.xaxis.set_minor_locator(minorLocator)
#myax.yaxis.set_major_locator(ymajorLocator)
#myax.yaxis.set_major_formatter(ymajorFormatter)
#myax.yaxis.set_minor_locator(yminorLocator)
first_legend = plt.legend([ftb, ftc, fte, ftf, ftg], [r'$\tan\beta < 1$', r'$1 < \tan\beta < 5$', r'$5 < \tan\beta < 10$', r'$10 < \tan\beta < 20$', r'$20 < \tan\beta < 50$'], frameon=True, loc=1, numpoints = 1,fontsize = 15)
first_legend.set_zorder(100)
#plt.gca().add_artist(first_legend)
#second_legend = plt.legend([fgt,fgta,fgtb],[r'                  ',r'                                                        ',r'               '], frameon=False, loc=1, numpoints = 1, fontsize = 12, handletextpad=8,columnspacing = 1.1,borderpad=0.5,labelspacing=0.88)
#second_legend.set_zorder(101)
#frame = second_legend.get_frame()
#frame.set_facecolor('white')
#myax.annotate(r'$m_H = 900$ GeV, $m_{t_4}$ = 600 GeV', xy=(0.6, 0.6), ha='left', va='top', xycoords='axes fraction', fontsize=15)
#plt.title(r'BRour, Scenario 3', fontsize=17)
plt.savefig('t1ewpt-brHtotata.png')


fig = plt.figure()  # make a new figure : separately from the above 
plt.axis([1000.,3000.,0.3,50.0])
ft, = plt.plot(tlistmHH,tlisttanbe,'.', color = 'orange', label='constraint',zorder=1)
#fe, = plt.plot(elistmHH,elisttanbe,'.', color = 'blue', label='constraint',zorder=2)
myax = plt.gca()
#myax.set_yscale('log', nonposy='clip')
#myax.annotate(r'$m_{t_4}$', xy=(0.9, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
myax.annotate(r'$m_H$ [GeV]', xy=(0.8, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
myax.annotate(r'$\tan\beta$', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
majorLocator   = MultipleLocator(1000)
majorFormatter = FormatStrFormatter('%i')
minorLocator   = MultipleLocator(100)
ymajorLocator   = MultipleLocator(0.5)
ymajorFormatter = FormatStrFormatter('%2.1f')
yminorLocator   = MultipleLocator(0.1)
myax.tick_params(size=3.5)
#myax.tick_params(axis = 'y',zorder = 10) 
myax.xaxis.set_major_locator(majorLocator)
myax.xaxis.set_major_formatter(majorFormatter)
myax.xaxis.set_minor_locator(minorLocator)
#myax.yaxis.set_major_locator(ymajorLocator)
#myax.yaxis.set_major_formatter(ymajorFormatter)
#myax.yaxis.set_minor_locator(yminorLocator)
#first_legend = plt.legend([ft, fxs1, fxs2], [r'$p p \to H \to t_4 t$', r'$p p \to t_4 b j$ for $g_{L,R}^{W t_4 b} = 0.1$', r'$p p \to t_4 t j$ for $g_{L,R}^{Z t_4 t} = 0.1$'], frameon=True, loc=1, numpoints = 1,fontsize = 15)
#first_legend.set_zorder(100)
#plt.gca().add_artist(first_legend)
#second_legend = plt.legend([fgt,fgta,fgtb],[r'                  ',r'                                                        ',r'               '], frameon=False, loc=1, numpoints = 1, fontsize = 12, handletextpad=8,columnspacing = 1.1,borderpad=0.5,labelspacing=0.88)
#second_legend.set_zorder(101)
#frame = second_legend.get_frame()
#frame.set_facecolor('white')
plt.savefig('t1ewpt-mH-tanbe.png')



fig = plt.figure()  # make a new figure : separately from the above 
plt.axis([-0.01,1.02,-0.01,1.02])
fa, = plt.plot(alistbrt4z,alistbrt4h,'.', color = 'red', label='constraint',zorder=2)
fb, = plt.plot(blistbrt4z,blistbrt4h,'.', color = 'purple', label='constraint',zorder=1)
fc, = plt.plot(clistbrt4z,clistbrt4h,'.', color = 'cyan', label='constraint',zorder=3)
fd, = plt.plot(dlistbrt4z,dlistbrt4h,'.', color = 'blue', label='constraint',zorder=4)
myax = plt.gca()
#myax.set_yscale('log', nonposy='clip')
#myax.annotate(r'$m_{t_4}$', xy=(0.9, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
myax.annotate(r'BR($t_4 \to Z t$)', xy=(0.75, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
myax.annotate(r'BR($t_4 \to h t$)', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
majorLocator   = MultipleLocator(0.2)
majorFormatter = FormatStrFormatter('%2.1f')
minorLocator   = MultipleLocator(0.1)
ymajorLocator   = MultipleLocator(0.2)
ymajorFormatter = FormatStrFormatter('%2.1f')
yminorLocator   = MultipleLocator(0.1)
myax.tick_params(size=3.5)
#myax.tick_params(axis = 'y',zorder = 10) 
myax.xaxis.set_major_locator(majorLocator)
myax.xaxis.set_major_formatter(majorFormatter)
myax.xaxis.set_minor_locator(minorLocator)
myax.yaxis.set_major_locator(ymajorLocator)
myax.yaxis.set_major_formatter(ymajorFormatter)
myax.yaxis.set_minor_locator(yminorLocator)
#first_legend = plt.legend([fta, ftb, ftc, ftd], [r'$\tan\beta <  1$', r'$1  < \tan\beta <  5$', r'$5  < \tan\beta <  10$', r'$\tan\beta >  10$'], frameon=True, loc=1, numpoints = 1,fontsize = 15)
#first_legend.set_zorder(100)
#plt.gca().add_artist(first_legend)
#second_legend = plt.legend([fgt,fgta,fgtb],[r'                  ',r'                                                        ',r'               '], frameon=False, loc=1, numpoints = 1, fontsize = 12, handletextpad=8,columnspacing = 1.1,borderpad=0.5,labelspacing=0.88)
#second_legend.set_zorder(101)
#frame = second_legend.get_frame()
#frame.set_facecolor('white')
plt.savefig('t1ewpt-t4br1.png')






fig = plt.figure()  # make a new figure : separately from the above 
plt.axis([1000,3000,0.001,1.02])
#ft, = plt.plot(tlistmt4,tlistbrHtot4,'.', color = 'navy', label='constraint',zorder=1)
ft, = plt.plot(tlistmHH,tlistbrHtot4,'.', color = 'navy', label='constraint',zorder=1)
myax = plt.gca()
myax.set_yscale('log', nonposy='clip')
#myax.annotate(r'$m_{t_4}$', xy=(0.9, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
myax.annotate(r'$m_H$ [GeV]', xy=(0.8, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
myax.annotate(r'BR($H \to t_4 t$)', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
majorLocator   = MultipleLocator(500)
majorFormatter = FormatStrFormatter('%i')
minorLocator   = MultipleLocator(100)
ymajorLocator   = MultipleLocator(0.5)
ymajorFormatter = FormatStrFormatter('%2.1f')
yminorLocator   = MultipleLocator(0.1)
myax.tick_params(size=3.5)
#myax.tick_params(axis = 'y',zorder = 10) 
myax.xaxis.set_major_locator(majorLocator)
myax.xaxis.set_major_formatter(majorFormatter)
myax.xaxis.set_minor_locator(minorLocator)
#myax.yaxis.set_major_locator(ymajorLocator)
#myax.yaxis.set_major_formatter(ymajorFormatter)
#myax.yaxis.set_minor_locator(yminorLocator)
#first_legend = plt.legend([fta, ftb, ftc, ftd], [r'$\tan\beta <  1$', r'$1  < \tan\beta <  5$', r'$5  < \tan\beta <  10$', r'$\tan\beta >  10$'], frameon=True, loc=1, numpoints = 1,fontsize = 15)
#first_legend.set_zorder(100)
#plt.gca().add_artist(first_legend)
#second_legend = plt.legend([fgt,fgta,fgtb],[r'                  ',r'                                                        ',r'               '], frameon=False, loc=1, numpoints = 1, fontsize = 12, handletextpad=8,columnspacing = 1.1,borderpad=0.5,labelspacing=0.88)
#second_legend.set_zorder(101)
#frame = second_legend.get_frame()
#frame.set_facecolor('white')
plt.savefig('t1ewpt-brHtot4.png')


fig = plt.figure()  # make a new figure : separately from the above 
plt.axis([-0.01,1.02,-0.01,1.02])
ft, = plt.plot(tclistbrHtot4,tclistbrHtott,'.', color = 'navy', label='constraint',zorder=1)
fb, = plt.plot(tblistbrHtot4,tblistbrHtott,'.', color = 'green', label='constraint',zorder=2)
myax = plt.gca()
#myax.set_yscale('log', nonposy='clip')
#myax.annotate(r'$m_{t_4}$', xy=(0.9, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
myax.annotate(r'BR($H \to t_4 t$)', xy=(0.75, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
myax.annotate(r'BR($H \to t t$)', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
majorLocator   = MultipleLocator(0.2)
majorFormatter = FormatStrFormatter('%2.1f')
minorLocator   = MultipleLocator(0.1)
ymajorLocator   = MultipleLocator(0.2)
ymajorFormatter = FormatStrFormatter('%2.1f')
yminorLocator   = MultipleLocator(0.1)
myax.tick_params(size=3.5)
#myax.tick_params(axis = 'y',zorder = 10) 
myax.xaxis.set_major_locator(majorLocator)
myax.xaxis.set_major_formatter(majorFormatter)
myax.xaxis.set_minor_locator(minorLocator)
myax.yaxis.set_major_locator(ymajorLocator)
myax.yaxis.set_major_formatter(ymajorFormatter)
myax.yaxis.set_minor_locator(yminorLocator)
first_legend = plt.legend([ft, fb], [r'$m_H \, < \, 2$ TeV', r'$m_H \, > \, 2$ TeV'], frameon=True, loc=1, numpoints = 1,fontsize = 15)
first_legend.set_zorder(100)
plt.gca().add_artist(first_legend)
#second_legend = plt.legend([fgt,fgta,fgtb],[r'                  ',r'                                                        ',r'               '], frameon=False, loc=1, numpoints = 1, fontsize = 12, handletextpad=8,columnspacing = 1.1,borderpad=0.5,labelspacing=0.88)
#second_legend.set_zorder(101)
#frame = second_legend.get_frame()
#frame.set_facecolor('white')
plt.savefig('t1ewpt-br1.png')


fig = plt.figure()  # make a new figure : separately from the above 
plt.axis([-0.01,1.02,-0.01,1.02])
ft, = plt.plot(tclistbrHtot4,tclistbrHtobb,'.', color = 'navy', label='constraint',zorder=1)
fb, = plt.plot(tblistbrHtot4,tblistbrHtobb,'.', color = 'green', label='constraint',zorder=2)
myax = plt.gca()
#myax.set_yscale('log', nonposy='clip')
#myax.annotate(r'$m_{t_4}$', xy=(0.9, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
myax.annotate(r'BR($H \to t_4 t$)', xy=(0.75, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
myax.annotate(r'BR($H \to b b$)', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
majorLocator   = MultipleLocator(0.2)
majorFormatter = FormatStrFormatter('%2.1f')
minorLocator   = MultipleLocator(0.1)
ymajorLocator   = MultipleLocator(0.2)
ymajorFormatter = FormatStrFormatter('%2.1f')
yminorLocator   = MultipleLocator(0.1)
myax.tick_params(size=3.5)
#myax.tick_params(axis = 'y',zorder = 10) 
myax.xaxis.set_major_locator(majorLocator)
myax.xaxis.set_major_formatter(majorFormatter)
myax.xaxis.set_minor_locator(minorLocator)
myax.yaxis.set_major_locator(ymajorLocator)
myax.yaxis.set_major_formatter(ymajorFormatter)
myax.yaxis.set_minor_locator(yminorLocator)
first_legend = plt.legend([ft, fb], [r'$m_H \, < \, 2$ TeV', r'$m_H \, > \, 2$ TeV'], frameon=True, loc=1, numpoints = 1,fontsize = 15)
first_legend.set_zorder(100)
plt.gca().add_artist(first_legend)
#second_legend = plt.legend([fgt,fgta,fgtb],[r'                  ',r'                                                        ',r'               '], frameon=False, loc=1, numpoints = 1, fontsize = 12, handletextpad=8,columnspacing = 1.1,borderpad=0.5,labelspacing=0.88)
#second_legend.set_zorder(101)
#frame = second_legend.get_frame()
#frame.set_facecolor('white')
plt.savefig('t1ewpt-br2.png')


fig = plt.figure()  # make a new figure : separately from the above 
plt.axis([-0.01,1.02,-0.001,0.012])
ft, = plt.plot(tclistbrHtot4,tclistbrHtogg,'.', color = 'navy', label='constraint',zorder=1)
fb, = plt.plot(tblistbrHtot4,tblistbrHtogg,'.', color = 'green', label='constraint',zorder=2)
myax = plt.gca()
#myax.set_yscale('log', nonposy='clip')
#myax.annotate(r'$m_{t_4}$', xy=(0.9, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
myax.annotate(r'BR($H \to t_4 t$)', xy=(0.75, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
myax.annotate(r'BR($H \to g g$)', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
majorLocator   = MultipleLocator(0.2)
majorFormatter = FormatStrFormatter('%2.1f')
minorLocator   = MultipleLocator(0.1)
ymajorLocator   = MultipleLocator(0.02)
ymajorFormatter = FormatStrFormatter('%2.1f')
yminorLocator   = MultipleLocator(0.005)
myax.tick_params(size=3.5)
#myax.tick_params(axis = 'y',zorder = 10) 
myax.xaxis.set_major_locator(majorLocator)
myax.xaxis.set_major_formatter(majorFormatter)
myax.xaxis.set_minor_locator(minorLocator)
#myax.yaxis.set_major_locator(ymajorLocator)
#myax.yaxis.set_major_formatter(ymajorFormatter)
#myax.yaxis.set_minor_locator(yminorLocator)
first_legend = plt.legend([ft, fb], [r'$m_H \, < \, 2$ TeV', r'$m_H \, > \, 2$ TeV'], frameon=True, loc=1, numpoints = 1,fontsize = 15)
first_legend.set_zorder(100)
plt.gca().add_artist(first_legend)
#second_legend = plt.legend([fgt,fgta,fgtb],[r'                  ',r'                                                        ',r'               '], frameon=False, loc=1, numpoints = 1, fontsize = 12, handletextpad=8,columnspacing = 1.1,borderpad=0.5,labelspacing=0.88)
#second_legend.set_zorder(101)
#frame = second_legend.get_frame()
#frame.set_facecolor('white')
plt.savefig('t1ewpt-br3.png')


fig = plt.figure()  # make a new figure : separately from the above 
plt.axis([1000,3000,0.001,1.02])
ft, = plt.plot(tlistmHH,tlistprodratiobr,'.', color = 'navy', label='constraint',zorder=1)
myax = plt.gca()
myax.set_yscale('log', nonposy='clip')
#myax.annotate(r'$m_{t_4}$', xy=(0.9, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
myax.annotate(r'$m_H$ [GeV]', xy=(0.8, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
myax.annotate(r'$\frac{\sigma_H}{\sigma_{H_{\rm SM}}} \,\cdot$ BR($H \to h t \bar{t}$)', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
majorLocator   = MultipleLocator(1000)
majorFormatter = FormatStrFormatter('%i')
minorLocator   = MultipleLocator(100)
ymajorLocator   = MultipleLocator(0.5)
ymajorFormatter = FormatStrFormatter('%2.1f')
yminorLocator   = MultipleLocator(0.1)
myax.tick_params(size=3.5)
#myax.tick_params(axis = 'y',zorder = 10) 
myax.xaxis.set_major_locator(majorLocator)
myax.xaxis.set_major_formatter(majorFormatter)
myax.xaxis.set_minor_locator(minorLocator)
#myax.yaxis.set_major_locator(ymajorLocator)
#myax.yaxis.set_major_formatter(ymajorFormatter)
#myax.yaxis.set_minor_locator(yminorLocator)
#first_legend = plt.legend([fta, ftb, ftc, ftd], [r'$\tan\beta <  1$', r'$1  < \tan\beta <  5$', r'$5  < \tan\beta <  10$', r'$\tan\beta >  10$'], frameon=True, loc=1, numpoints = 1,fontsize = 15)
#first_legend.set_zorder(100)
#plt.gca().add_artist(first_legend)
#second_legend = plt.legend([fgt,fgta,fgtb],[r'                  ',r'                                                        ',r'               '], frameon=False, loc=1, numpoints = 1, fontsize = 12, handletextpad=8,columnspacing = 1.1,borderpad=0.5,labelspacing=0.88)
#second_legend.set_zorder(101)
#frame = second_legend.get_frame()
#frame.set_facecolor('white')
plt.savefig('t1ewpt-prodratiobr.png')

fig = plt.figure()  # make a new figure : separately from the above 
plt.axis([980,3020,0.01,10000.2])
ft, = plt.plot(tlistmHH,tlistgamHtot,'.', color = 'navy', label='constraint',zorder=1)
myax = plt.gca()
myax.set_yscale('log', nonposy='clip')
#myax.annotate(r'$m_{t_4}$', xy=(0.9, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
myax.annotate(r'$m_H$ [GeV]', xy=(0.8, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
myax.annotate(r'$\Gamma_H$ [GeV]', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
majorLocator   = MultipleLocator(1000)
majorFormatter = FormatStrFormatter('%i')
minorLocator   = MultipleLocator(100)
ymajorLocator   = MultipleLocator(0.5)
ymajorFormatter = FormatStrFormatter('%2.1f')
yminorLocator   = MultipleLocator(0.1)
myax.tick_params(size=3.5)
#myax.tick_params(axis = 'y',zorder = 10) 
myax.xaxis.set_major_locator(majorLocator)
myax.xaxis.set_major_formatter(majorFormatter)
myax.xaxis.set_minor_locator(minorLocator)
#myax.yaxis.set_major_locator(ymajorLocator)
#myax.yaxis.set_major_formatter(ymajorFormatter)
#myax.yaxis.set_minor_locator(yminorLocator)
#first_legend = plt.legend([fta, ftb, ftc, ftd], [r'$\tan\beta <  1$', r'$1  < \tan\beta <  5$', r'$5  < \tan\beta <  10$', r'$\tan\beta >  10$'], frameon=True, loc=1, numpoints = 1,fontsize = 15)
#first_legend.set_zorder(100)
#plt.gca().add_artist(first_legend)
#second_legend = plt.legend([fgt,fgta,fgtb],[r'                  ',r'                                                        ',r'               '], frameon=False, loc=1, numpoints = 1, fontsize = 12, handletextpad=8,columnspacing = 1.1,borderpad=0.5,labelspacing=0.88)
#second_legend.set_zorder(101)
#frame = second_legend.get_frame()
#frame.set_facecolor('white')
plt.savefig('t1ewpt-gamHtot.png')





fig = plt.figure()  # make a new figure : separately from the above 
plt.axis([-0.2,0.2,-0.2,0.2]) 
#ft, = plt.plot(tlistkaT,tlistkaQ,'.', color = 'navy', label='constraint',zorder=1)
ft, = plt.plot(tlistgLW43,tlistgRW43,'.', color = 'navy', label='constraint',zorder=1)
myax = plt.gca()
#myax.set_yscale('log', nonposy='clip')
#myax.annotate(r'$\Gamma(H \to t_4 t)$', xy=(0.9, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
#myax.annotate(r'$\kappa_T$', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
myax.annotate(r'$g_L^{W t_4 t}$', xy=(0.9, -0.045), ha='left', va='top', xycoords='axes fraction', fontsize=20)
myax.annotate(r'$g_R^{W t_4 t}$', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
majorLocator   = MultipleLocator(0.5)
majorFormatter = FormatStrFormatter('%2.1f')
minorLocator   = MultipleLocator(0.1)
ymajorLocator   = MultipleLocator(0.5)
ymajorFormatter = FormatStrFormatter('%2.1f')
yminorLocator   = MultipleLocator(0.1)
myax.tick_params(size=3.5)
#myax.tick_params(axis = 'y',zorder = 10) 
#myax.xaxis.set_major_locator(majorLocator)
#myax.xaxis.set_major_formatter(majorFormatter)
#myax.xaxis.set_minor_locator(minorLocator)
#myax.yaxis.set_major_locator(ymajorLocator)
#myax.yaxis.set_major_formatter(ymajorFormatter)
#myax.yaxis.set_minor_locator(yminorLocator)
#first_legend = plt.legend([fta, ftb, ftc, ftd], [r'$\tan\beta <  1$', r'$1  < \tan\beta <  5$', r'$5  < \tan\beta <  10$', r'$\tan\beta >  10$'], frameon=True, loc=1, numpoints = 1,fontsize = 15)
#first_legend.set_zorder(100)
#plt.gca().add_artist(first_legend)
#second_legend = plt.legend([fgt,fgta,fgtb],[r'                  ',r'                                                        ',r'               '], frameon=False, loc=1, numpoints = 1, fontsize = 12, handletextpad=8,columnspacing = 1.1,borderpad=0.5,labelspacing=0.88)
#second_legend.set_zorder(101)
#frame = second_legend.get_frame()
#frame.set_facecolor('white')
plt.savefig('t1ewpt-couplings1.png')


fig = plt.figure()  # make a new figure : separately from the above 
plt.axis([980,3020,-0.2,0.2])
ft, = plt.plot(tlistmHH,tlistgLW43,'.', color = 'navy', label='constraint',zorder=1)
myax = plt.gca()
#myax.set_yscale('log', nonposy='clip')
#myax.annotate(r'$m_{t_4}$', xy=(0.9, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
myax.annotate(r'$m_H$ [GeV]', xy=(0.8, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
myax.annotate(r'$g_L^{W t_4 t}$', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
majorLocator   = MultipleLocator(1000)
majorFormatter = FormatStrFormatter('%i')
minorLocator   = MultipleLocator(100)
ymajorLocator   = MultipleLocator(0.5)
ymajorFormatter = FormatStrFormatter('%2.1f')
yminorLocator   = MultipleLocator(0.1)
myax.tick_params(size=3.5)
#myax.tick_params(axis = 'y',zorder = 10) 
myax.xaxis.set_major_locator(majorLocator)
myax.xaxis.set_major_formatter(majorFormatter)
myax.xaxis.set_minor_locator(minorLocator)
#myax.yaxis.set_major_locator(ymajorLocator)
#myax.yaxis.set_major_formatter(ymajorFormatter)
#myax.yaxis.set_minor_locator(yminorLocator)
#first_legend = plt.legend([fta, ftb, ftc, ftd], [r'$\tan\beta <  1$', r'$1  < \tan\beta <  5$', r'$5  < \tan\beta <  10$', r'$\tan\beta >  10$'], frameon=True, loc=1, numpoints = 1,fontsize = 15)
#first_legend.set_zorder(100)
#plt.gca().add_artist(first_legend)
#second_legend = plt.legend([fgt,fgta,fgtb],[r'                  ',r'                                                        ',r'               '], frameon=False, loc=1, numpoints = 1, fontsize = 12, handletextpad=8,columnspacing = 1.1,borderpad=0.5,labelspacing=0.88)
#second_legend.set_zorder(101)
#frame = second_legend.get_frame()
#frame.set_facecolor('white')
plt.savefig('t1ewpt-couplings3.png')


fig = plt.figure()  # make a new figure : separately from the above 
plt.axis([-0.2,50.2,-0.2,1.2])
ft, = plt.plot(tlisttanbe,tlistdoubfr,'.', color = 'orange', label='constraint',zorder=1)
myax = plt.gca()
#myax.set_yscale('log', nonposy='clip')
myax.annotate(r'$\tan\beta$ [GeV]', xy=(0.8, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
myax.annotate(r'doublet fraction', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
majorLocator   = MultipleLocator(1000)
majorFormatter = FormatStrFormatter('%i')
minorLocator   = MultipleLocator(100)
ymajorLocator   = MultipleLocator(0.5)
ymajorFormatter = FormatStrFormatter('%2.1f')
yminorLocator   = MultipleLocator(0.1)
myax.tick_params(size=3.5)
#myax.tick_params(axis = 'y',zorder = 10) 
myax.xaxis.set_major_locator(majorLocator)
myax.xaxis.set_major_formatter(majorFormatter)
myax.xaxis.set_minor_locator(minorLocator)
#myax.yaxis.set_major_locator(ymajorLocator)
#myax.yaxis.set_major_formatter(ymajorFormatter)
#myax.yaxis.set_minor_locator(yminorLocator)
#first_legend = plt.legend([ft, fxs1, fxs2], [r'$p p \to H \to t_4 t$', r'$p p \to t_4 b j$ for $g_{L,R}^{W t_4 b} = 0.1$', r'$p p \to t_4 t j$ for $g_{L,R}^{Z t_4 t} = 0.1$'], frameon=True, loc=1, numpoints = 1,fontsize = 15)
#first_legend.set_zorder(100)
#plt.gca().add_artist(first_legend)
#second_legend = plt.legend([fgt,fgta,fgtb],[r'                  ',r'                                                        ',r'               '], frameon=False, loc=1, numpoints = 1, fontsize = 12, handletextpad=8,columnspacing = 1.1,borderpad=0.5,labelspacing=0.88)
#second_legend.set_zorder(101)
#frame = second_legend.get_frame()
#frame.set_facecolor('white')
plt.savefig('t1ewpt-doubfr.png')

fig = plt.figure()  # make a new figure : separately from the above 
plt.axis([-0.2,50.2,-0.5,15.])
ft, = plt.plot(tlisttanbe,tlistggHratio,'.', color = 'navy', label='constraint',zorder=2)
fa, = plt.plot(tlisttanbe,tlisttotalratio,'.', color = 'red', label='constraint',zorder=1)
#fb, = plt.plot(tlisttanbe,tlisttotalratio,'.', color = 'blue', label='constraint',zorder=1)
#ftc, = plt.plot(tclisttanbe,tclisttotalratio,'.', color = 'cyan', label='constraint',zorder=4)
#fb, = plt.plot(tlisttanbe,tlistbbFratio,'.', color = 'green', label='constraint',zorder=4)
myax = plt.gca()
#myax.set_yscale('log', nonposy='clip')
myax.annotate(r'$\tan\beta$', xy=(0.9, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
myax.annotate(r'Ratio of production cross sections', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
#myax.annotate(r'$\frac{\sigma_H}{\sigma_{H_{\rm SM}}} \,\cdot$ BR($H \to h t \bar{t}$)', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
majorLocator   = MultipleLocator(5)
majorFormatter = FormatStrFormatter('%i')
minorLocator   = MultipleLocator(1)
ymajorLocator   = MultipleLocator(5)
ymajorFormatter = FormatStrFormatter('%2.1f')
yminorLocator   = MultipleLocator(1)
myax.tick_params(size=3.5)
#myax.tick_params(axis = 'y',zorder = 10) 
myax.xaxis.set_major_locator(majorLocator)
myax.xaxis.set_major_formatter(majorFormatter)
myax.xaxis.set_minor_locator(minorLocator)
myax.yaxis.set_major_locator(ymajorLocator)
myax.yaxis.set_major_formatter(ymajorFormatter)
myax.yaxis.set_minor_locator(yminorLocator)
first_legend = plt.legend([ft, fa], [r'$\sigma(g g \to H) / \sigma (p p \to H_{\rm SM})$', r'$\sigma(p p \to H) / \sigma(p p \to H_{\rm SM})$'], frameon=True, loc=1, numpoints = 1,fontsize = 15)
#first_legend = plt.legend([ft, fa, fb], [r'$\sigma(gg \to H) / \sigma (gg \to H_{\rm SM})$', r'$\sigma(gg \to h) / \sigma(gg \to h_{\rm SM})$', r'$\sigma(pp \to H) / \sigma(gg \to H_{\rm SM})$'], frameon=True, loc=1, numpoints = 1,fontsize = 15)
#first_legend = plt.legend([ft, fa, fb, ftb], [r'$\sigma(gg \to H) / \sigma (gg \to H_{\rm SM})$', r'$\sigma(gg \to h) / \sigma(gg \to h_{\rm SM})$', r'$\sigma(pp \to H) / \sigma(gg \to H_{\rm SM})$ for $m_H < 2$ TeV', r'$\sigma(pp \to H) / \sigma(gg \to H_{\rm SM})$ for $m_H > 2$ TeV'], frameon=True, loc=1, numpoints = 1,fontsize = 15)
first_legend.set_zorder(100)
plt.gca().add_artist(first_legend)
#second_legend = plt.legend([fgt,fgta,fgtb],[r'                  ',r'                                                        ',r'               '], frameon=False, loc=1, numpoints = 1, fontsize = 12, handletextpad=8,columnspacing = 1.1,borderpad=0.5,labelspacing=0.88)
#second_legend.set_zorder(101)
#frame = second_legend.get_frame()
#frame.set_facecolor('white')
#myax.annotate(r'$m_H = 900$ GeV, $m_{t_4}$ = 600 GeV', xy=(0.6, 0.6), ha='left', va='top', xycoords='axes fraction', fontsize=15)
plt.title(r'Ratio of the production cross sections', fontsize=17)
plt.savefig('t1ewpt-tanbe-ggHratio.png')



fig = plt.figure()  # make a new figure : separately from the above 
plt.axis([1000.,3000.,-1.01,1.01])
fa, = plt.plot(alistMTT,alistlamHt34,'.', color = 'red', label='constraint',zorder=2)
myax = plt.gca()
#myax.set_yscale('log', nonposy='clip')
myax.annotate(r'$M_T$ [GeV]', xy=(0.8, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
myax.annotate(r'$\lambda_H^{t t_4}$', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
majorLocator   = MultipleLocator(500)
majorFormatter = FormatStrFormatter('%i')
minorLocator   = MultipleLocator(100)
ymajorLocator   = MultipleLocator(0.2)
ymajorFormatter = FormatStrFormatter('%2.1f')
yminorLocator   = MultipleLocator(0.1)
myax.tick_params(size=3.5)
#myax.tick_params(axis = 'y',zorder = 10) 
myax.xaxis.set_major_locator(majorLocator)
myax.xaxis.set_major_formatter(majorFormatter)
myax.xaxis.set_minor_locator(minorLocator)
myax.yaxis.set_major_locator(ymajorLocator)
myax.yaxis.set_major_formatter(ymajorFormatter)
myax.yaxis.set_minor_locator(yminorLocator)
#first_legend = plt.legend([fta, ftb, ftc, ftd], [r'$\tan\beta <  1$', r'$1  < \tan\beta <  5$', r'$5  < \tan\beta <  10$', r'$\tan\beta >  10$'], frameon=True, loc=1, numpoints = 1,fontsize = 15)
#first_legend.set_zorder(100)
#plt.gca().add_artist(first_legend)
#second_legend = plt.legend([fgt,fgta,fgtb],[r'                  ',r'                                                        ',r'               '], frameon=False, loc=1, numpoints = 1, fontsize = 12, handletextpad=8,columnspacing = 1.1,borderpad=0.5,labelspacing=0.88)
#second_legend.set_zorder(101)
#frame = second_legend.get_frame()
#frame.set_facecolor('white')
plt.savefig('t1ewpt-mt4-check.png')



fig = plt.figure()  # make a new figure : separately from the above 
plt.axis([1000.,3000.,0.3,50.0])
ft, = plt.plot(tlistmHH,tlisttanbe,'.', color = 'orange', label='constraint',zorder=1)
fe, = plt.plot(elistmHH,elisttanbe,'.', color = 'blue', label='constraint',zorder=2)
myax = plt.gca()
#myax.set_yscale('log', nonposy='clip')
#myax.annotate(r'$m_{t_4}$', xy=(0.9, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
myax.annotate(r'$m_H$ [GeV]', xy=(0.8, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
myax.annotate(r'$\tan\beta$', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
majorLocator   = MultipleLocator(1000)
majorFormatter = FormatStrFormatter('%i')
minorLocator   = MultipleLocator(100)
ymajorLocator   = MultipleLocator(0.5)
ymajorFormatter = FormatStrFormatter('%2.1f')
yminorLocator   = MultipleLocator(0.1)
myax.tick_params(size=3.5)
#myax.tick_params(axis = 'y',zorder = 10) 
myax.xaxis.set_major_locator(majorLocator)
myax.xaxis.set_major_formatter(majorFormatter)
myax.xaxis.set_minor_locator(minorLocator)
#myax.yaxis.set_major_locator(ymajorLocator)
#myax.yaxis.set_major_formatter(ymajorFormatter)
#myax.yaxis.set_minor_locator(yminorLocator)
#first_legend = plt.legend([ft, fxs1, fxs2], [r'$p p \to H \to t_4 t$', r'$p p \to t_4 b j$ for $g_{L,R}^{W t_4 b} = 0.1$', r'$p p \to t_4 t j$ for $g_{L,R}^{Z t_4 t} = 0.1$'], frameon=True, loc=1, numpoints = 1,fontsize = 15)
#first_legend.set_zorder(100)
#plt.gca().add_artist(first_legend)
#second_legend = plt.legend([fgt,fgta,fgtb],[r'                  ',r'                                                        ',r'               '], frameon=False, loc=1, numpoints = 1, fontsize = 12, handletextpad=8,columnspacing = 1.1,borderpad=0.5,labelspacing=0.88)
#second_legend.set_zorder(101)
#frame = second_legend.get_frame()
#frame.set_facecolor('white')
plt.savefig('t1ewpt-tanbe-xsdepend.png')


fig = plt.figure()  # make a new figure : separately from the above 
plt.axis([0.,0.05,0.,0.05])
ft, = plt.plot(talisttotalt4t,talistmyt4bj,'.', color = 'blue', label='constraint',zorder=1)
plt.plot(linex,linex,'-', color = 'red')
myax = plt.gca()
#myax.set_yscale('log', nonposy='clip')
#myax.annotate(r'$m_{t_4}$', xy=(0.9, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
myax.annotate(r'$\sigma(p p \to H \to t_4 t)$ [pb]', xy=(0.6, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
myax.annotate(r'$\sigma(p p \to t_4 b j)$ [pb]', xy=(-0.15, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
majorLocator   = MultipleLocator(1000)
majorFormatter = FormatStrFormatter('%i')
minorLocator   = MultipleLocator(100)
ymajorLocator   = MultipleLocator(0.5)
ymajorFormatter = FormatStrFormatter('%2.1f')
yminorLocator   = MultipleLocator(0.1)
myax.tick_params(size=3.5)
#myax.tick_params(axis = 'y',zorder = 10) 
#myax.xaxis.set_major_locator(majorLocator)
#myax.xaxis.set_major_formatter(majorFormatter)
#myax.xaxis.set_minor_locator(minorLocator)
#myax.yaxis.set_major_locator(ymajorLocator)
#myax.yaxis.set_major_formatter(ymajorFormatter)
#myax.yaxis.set_minor_locator(yminorLocator)
#first_legend = plt.legend([ft, fxs1, fxs2], [r'$p p \to H \to t_4 t$', r'$p p \to t_4 b j$ for $g_{L,R}^{W t_4 b} = 0.1$', r'$p p \to t_4 t j$ for $g_{L,R}^{Z t_4 t} = 0.1$'], frameon=True, loc=1, numpoints = 1,fontsize = 15)
#first_legend.set_zorder(100)
#plt.gca().add_artist(first_legend)
#second_legend = plt.legend([fgt,fgta,fgtb],[r'                  ',r'                                                        ',r'               '], frameon=False, loc=1, numpoints = 1, fontsize = 12, handletextpad=8,columnspacing = 1.1,borderpad=0.5,labelspacing=0.88)
#second_legend.set_zorder(101)
#frame = second_legend.get_frame()
#frame.set_facecolor('white')
plt.savefig('t1ewpt-xs-check-t4Honly.png')



fig = plt.figure()  # make a new figure : separately from the above 
plt.axis([-0.02,50.2,-0.02,1.02])
fa, = plt.plot(talisttanbe,talistbrHtott,'.', color = 'red', label='constraint',zorder=1)
ft, = plt.plot(talisttanbe,talistbrHtot4,'.', color = 'navy', label='constraint',zorder=4)
fb, = plt.plot(talisttanbe,talistbrHtohh,'.', color = 'green', label='constraint',zorder=5)
fc, = plt.plot(talisttanbe,talistbrHtobb,'.', color = 'cyan', label='constraint',zorder=2)
fd, = plt.plot(talisttanbe,talistbrHtotata,'.', color = 'purple', label='constraint',zorder=3)
myax = plt.gca()
#myax.set_yscale('log', nonposy='clip')
myax.annotate(r'$\tan\beta$', xy=(0.9, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
myax.annotate(r'BR($H$)', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
#myax.annotate(r'$\frac{\sigma_H}{\sigma_{H_{\rm SM}}} \,\cdot$ BR($H \to h t \bar{t}$)', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
majorLocator   = MultipleLocator(5)
majorFormatter = FormatStrFormatter('%i')
minorLocator   = MultipleLocator(1)
ymajorLocator   = MultipleLocator(0.2)
ymajorFormatter = FormatStrFormatter('%2.1f')
yminorLocator   = MultipleLocator(0.1)
myax.tick_params(size=3.5)
#myax.tick_params(axis = 'y',zorder = 10) 
myax.xaxis.set_major_locator(majorLocator)
myax.xaxis.set_major_formatter(majorFormatter)
myax.xaxis.set_minor_locator(minorLocator)
myax.yaxis.set_major_locator(ymajorLocator)
myax.yaxis.set_major_formatter(ymajorFormatter)
myax.yaxis.set_minor_locator(yminorLocator)
first_legend = plt.legend([ft, fa, fb, fc, fd], [r'BR($H \to t_4 t$)', r'BR($H \to t t$)', r'BR($H \to h h$)', r'BR($H \to b b$)', r'BR($H \to \tau \tau$)'], frameon=True, loc=1, numpoints = 1,fontsize = 15)
#first_legend = plt.legend([ft, fa, fb, fc, fd, fe, ff], [r'BR($H \to t_4 t$)', r'BR($H \to t t$)', r'BR($H \to h h$)', r'BR($H \to b b$)', r'BR($H \to \tau \tau$)', r'BR($H \to c c$)', r'BR($H \to g g$)'], frameon=True, loc=1, numpoints = 1,fontsize = 15)
first_legend.set_zorder(100)
#plt.gca().add_artist(first_legend)
#second_legend = plt.legend([fgt,fgta,fgtb],[r'                  ',r'                                                        ',r'               '], frameon=False, loc=1, numpoints = 1, fontsize = 12, handletextpad=8,columnspacing = 1.1,borderpad=0.5,labelspacing=0.88)
#second_legend.set_zorder(101)
#frame = second_legend.get_frame()
#frame.set_facecolor('white')
#myax.annotate(r'$m_H = 900$ GeV, $m_{t_4}$ = 600 GeV', xy=(0.6, 0.6), ha='left', va='top', xycoords='axes fraction', fontsize=15)
plt.title(r'Ratio of the production cross sections', fontsize=17)
plt.savefig('t1ewpt-tanbe-brHs-t4Honly.png')



fig = plt.figure()  # make a new figure : separately from the above 
plt.axis([1000.,3000., 0.01, 200000.])
#ft, = plt.plot(tlistmb4,tlistHWratio,'.', color = 'navy', label='constraint',zorder=2)
ft, = plt.plot(tlistmt4,tlistHWratio,'.', color = 'navy', label='constraint',zorder=2)
myax = plt.gca()
myax.set_yscale('log', nonposy='clip')
#myax.annotate(r'$m_{b_4}$ [GeV]', xy=(0.8, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
#myax.annotate(r'$H b_4 b / W b_4 t$', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
myax.annotate(r'$m_{t_4}$ [GeV]', xy=(0.8, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
myax.annotate(r'$H t_4 t / W t_4 b$', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
majorLocator   = MultipleLocator(500)
majorFormatter = FormatStrFormatter('%i')
minorLocator   = MultipleLocator(100)
ymajorLocator   = MultipleLocator(0.2)
ymajorFormatter = FormatStrFormatter('%2.1f')
yminorLocator   = MultipleLocator(0.1)
myax.tick_params(size=3.5)
#myax.tick_params(axis = 'y',zorder = 10) 
myax.xaxis.set_major_locator(majorLocator)
myax.xaxis.set_major_formatter(majorFormatter)
myax.xaxis.set_minor_locator(minorLocator)
#myax.yaxis.set_major_locator(ymajorLocator)
#myax.yaxis.set_major_formatter(ymajorFormatter)
#myax.yaxis.set_minor_locator(yminorLocator)
#first_legend = plt.legend([fta, ftb, ftc, ftd], [r'$\tan\beta <  1$', r'$1  < \tan\beta <  5$', r'$5  < \tan\beta <  10$', r'$\tan\beta >  10$'], frameon=True, loc=1, numpoints = 1,fontsize = 15)
#first_legend.set_zorder(100)
#plt.gca().add_artist(first_legend)
#second_legend = plt.legend([fgt,fgta,fgtb],[r'                  ',r'                                                        ',r'               '], frameon=False, loc=1, numpoints = 1, fontsize = 12, handletextpad=8,columnspacing = 1.1,borderpad=0.5,labelspacing=0.88)
#second_legend.set_zorder(101)
#frame = second_legend.get_frame()
#frame.set_facecolor('white')
#plt.savefig('t1ewpt-mb4-HWcouplingRatio.png')
plt.savefig('t1ewpt-mt4-HWcouplingRatio.png')


mytx = numpy.array(tlistmt4)
myty = numpy.array(tlistbrt4h)
mytz = numpy.array(tlisttanbe)

#myxt = tlistmt4
#myyt = tlistbrt4h
#myzt = tlistsingfr

#dev = 0.01
#myxt, myyt, myzt = gaussian_filter(grid(tlistmt4, tlistbrt4h, tlistsingfr), dev, order=0)

#myxt, myyt, myzt = grid(tlistmt4, tlistbrt4h, tlistsingfr)


#idx = myzt.argsort()
#myxt, myyt, myzt = myxt[idx], myyt[idx], myzt[idx]


mbins = 100
xii, yii = numpy.mgrid[mytx.min():mytx.max():mbins*1j, myty.min():myty.max():mbins*1j]
xii, zii = numpy.mgrid[mytx.min():mytx.max():mbins*1j, mytz.min():mytz.max():mbins*1j]

#print myzt[0]
#print 'brt4h > 80% size = ', len(tdalistmt4), tdalistsingfr[0], tdalistmt4[0], tdalistbrt4h[0], tdalistmb4[0], -tdalistlamHt34[0]*tdalisttanbe[0], -tdalistlamHrt34[0]*tdalisttanbe[0], tdalistgLW43[0], tdalistgRW43t[0], tdalistgLZ43[0], tdalistgRZ43[0], tdalisttanbe[0], tdalistbrt4w[0], tdalistgamt4toh[0], tdalistgamt4tow[0], -tdalistcheck[0], -tdalisttest1[0], -tdalisttest2[0], -tdalisttest3[0], -tdalisttest4[0], tdalistMTT[0], tdalistMQQ[0]

#wg = gaussian_kde(mytz)(mytz)

#fig = plt.figure()  # make a new figure : separately from the above 
fit, myax = plt.subplots()
myax.scatter(mytx, myty, c=mytz, s = 50, vmin=0., vmax = 1.)
#myax.scatter(mytx, myty, c=wg, s = 50, vmin=0., vmax = 1.)
#plt.pcolormesh(mytx, myty, mytz)
#plt.pcolormesh(xii, yii, zii)
plt.axis([1000.,3000.,-0.01,1.02])
#f, = plt.plot(my,alistbrt4h,'.', , label='constraint',zorder=2)
#plt.imshow(myzt, extent = (numpy.amin(myxt), numpy.amax(myxt), numpy.amin(myyt), numpy.amin(myyt)), cmap=cm.hot)
#plt.colorbar()
#plt.show()
#levels = [0.05,0.5,0.95,1]
#mylabels = ['0.05','0.5','0.95','1']
#CS = plt.contour(myxt, myyt, myzt, levels)
#fmt = {}
#for l,s in zip( CS.levels, mylabels):
    #fmt[l]=2
#myax = plt.gca()
#myax.set_yscale('log', nonposy='clip')
#myax.annotate(r'$m_{t_4}$ [GeV]', xy=(0.8, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
#myax.annotate(r'BR($t_4 \to h t$)', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
#majorLocator   = MultipleLocator(500)
#majorFormatter = FormatStrFormatter('%i')
#minorLocator   = MultipleLocator(100)
#ymajorLocator   = MultipleLocator(0.2)
#ymajorFormatter = FormatStrFormatter('%2.1f')
#yminorLocator   = MultipleLocator(0.1)
#myax.tick_params(size=3.5)
#myax.tick_params(axis = 'y',zorder = 10) 
#myax.xaxis.set_major_locator(majorLocator)
#myax.xaxis.set_major_formatter(majorFormatter)
#myax.xaxis.set_minor_locator(minorLocator)
#myax.yaxis.set_major_locator(ymajorLocator)
#myax.yaxis.set_major_formatter(ymajorFormatter)
#myax.yaxis.set_minor_locator(yminorLocator)
plt.savefig('t1ewpt-try-mt4-brt4h.png')


fig = plt.figure()  # make a new figure : separately from the above 
plt.axis([990.,3001.,-0.5,50.])
ft, = plt.plot(tlistmHH,tlistggHratio,'.', color = 'navy', label='constraint',zorder=2)
#fa, = plt.plot(tlisttanbe,tlisttotalratio,'.', color = 'red', label='constraint',zorder=1)
#fb, = plt.plot(tlisttanbe,tlisttotalratio,'.', color = 'blue', label='constraint',zorder=1)
#ftc, = plt.plot(tclisttanbe,tclisttotalratio,'.', color = 'cyan', label='constraint',zorder=4)
#fb, = plt.plot(tlisttanbe,tlistbbFratio,'.', color = 'green', label='constraint',zorder=4)
myax = plt.gca()
#myax.set_yscale('log', nonposy='clip')
myax.annotate(r'$m_H$', xy=(0.9, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
myax.annotate(r'Ratio of production cross sections', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
#myax.annotate(r'$\frac{\sigma_H}{\sigma_{H_{\rm SM}}} \,\cdot$ BR($H \to h t \bar{t}$)', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
majorLocator   = MultipleLocator(5)
majorFormatter = FormatStrFormatter('%i')
minorLocator   = MultipleLocator(1)
ymajorLocator   = MultipleLocator(5)
ymajorFormatter = FormatStrFormatter('%2.1f')
yminorLocator   = MultipleLocator(1)
myax.tick_params(size=3.5)
#myax.tick_params(axis = 'y',zorder = 10) 
#myax.xaxis.set_major_locator(majorLocator)
#myax.xaxis.set_major_formatter(majorFormatter)
#myax.xaxis.set_minor_locator(minorLocator)
myax.yaxis.set_major_locator(ymajorLocator)
myax.yaxis.set_major_formatter(ymajorFormatter)
myax.yaxis.set_minor_locator(yminorLocator)
#first_legend = plt.legend([ft, fa], [r'$\sigma(g g \to H) / \sigma (p p \to H_{\rm SM})$', r'$\sigma(p p \to H) / \sigma(p p \to H_{\rm SM})$'], frameon=True, loc=1, numpoints = 1,fontsize = 15)
#first_legend = plt.legend([ft, fa, fb], [r'$\sigma(gg \to H) / \sigma (gg \to H_{\rm SM})$', r'$\sigma(gg \to h) / \sigma(gg \to h_{\rm SM})$', r'$\sigma(pp \to H) / \sigma(gg \to H_{\rm SM})$'], frameon=True, loc=1, numpoints = 1,fontsize = 15)
#first_legend = plt.legend([ft, fa, fb, ftb], [r'$\sigma(gg \to H) / \sigma (gg \to H_{\rm SM})$', r'$\sigma(gg \to h) / \sigma(gg \to h_{\rm SM})$', r'$\sigma(pp \to H) / \sigma(gg \to H_{\rm SM})$ for $m_H < 2$ TeV', r'$\sigma(pp \to H) / \sigma(gg \to H_{\rm SM})$ for $m_H > 2$ TeV'], frameon=True, loc=1, numpoints = 1,fontsize = 15)
#first_legend.set_zorder(100)
#plt.gca().add_artist(first_legend)
#second_legend = plt.legend([fgt,fgta,fgtb],[r'                  ',r'                                                        ',r'               '], frameon=False, loc=1, numpoints = 1, fontsize = 12, handletextpad=8,columnspacing = 1.1,borderpad=0.5,labelspacing=0.88)
#second_legend.set_zorder(101)
#frame = second_legend.get_frame()
#frame.set_facecolor('white')
#myax.annotate(r'$m_H = 900$ GeV, $m_{t_4}$ = 600 GeV', xy=(0.6, 0.6), ha='left', va='top', xycoords='axes fraction', fontsize=15)
plt.title(r'Ratio of the production cross sections', fontsize=17)
plt.savefig('t1ewpt-mHH-ggHratio.png')


fig = plt.figure()  # make a new figure : separately from the above 
#plt.axis([990.,2010., 0., 20.3])
#plt.axis([0.00001,0.21, 0.0, 1.])
plt.axis([0.00005,1.1, 0.00005, 1.1])
#ft, = plt.plot(tddlistmt4,tddlisttanbe,'.', color = 'blue', label='constraint',zorder=4)
#fta, = plt.plot(tdalistmt4,tdalisttanbe,'.', color = 'gold', label='constraint',zorder=1)
#fta, = plt.plot(tdalistlamh43,tdalistbrt4h,'.', color = 'blue', label='constraint',zorder=1)
fta, = plt.plot(tdlistlamh43, tdlistlamh34,'.', color = 'green', label='constraint',zorder=2)
ftb, = plt.plot(tdalistlamh43, tdalistlamh34,'.', color = 'orange', label='constraint',zorder=2)
ftc, = plt.plot(tdblistlamh43, tdblistlamh34,'.', color = 'navy', label='constraint',zorder=2)
#ftb, = plt.plot(tdalistlamh43,tdalistbrt4w,'.', color = 'red', label='constraint',zorder=2)
#ftb, = plt.plot(tdblistmt4,tdblisttanbe,'.', color = 'green', label='constraint',zorder=2)
#ftc, = plt.plot(tdclistmt4,tdclisttanbe,'.', color = 'red', label='constraint',zorder=3)
plt.plot(compx, compx, '--', color = 'black')
myax = plt.gca()
myax.set_xscale('log', nonposy='clip')
myax.set_yscale('log', nonposy='clip')
#myax2 = myax.twinx()
#myax2.set_ylim(0.998668,1.00006667)
#myax.annotate(r'$m_{t_4}$ [GeV]', xy=(0.8, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
#myax.annotate(r'$\tan\beta$', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
myax.annotate(r'$\lambda_h^{t_4 t}$', xy=(0.85, -0.03), ha='left', va='top', xycoords='axes fraction', fontsize=20)
myax.annotate(r'$\lambda^h_{t t_4}$', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
majorLocator   = MultipleLocator(500)
majorFormatter = FormatStrFormatter('%i')
minorLocator   = MultipleLocator(100)
ymajorLocator   = MultipleLocator(0.2)
ymajorFormatter = FormatStrFormatter('%2.1f')
yminorLocator   = MultipleLocator(0.1)
myax.tick_params(size=4, labelsize = 16)
#myax.tick_params(axis = 'y',zorder = 10) 
#myax.xaxis.set_major_locator(majorLocator)
#myax.xaxis.set_major_formatter(majorFormatter)
#myax.xaxis.set_minor_locator(minorLocator)
#myax.yaxis.set_major_locator(ymajorLocator)
#myax.yaxis.set_major_formatter(ymajorFormatter)
#myax.yaxis.set_minor_locator(yminorLocator)
first_legend = plt.legend([fta, ftb, ftc], [r'BR($t_4 \to h t$) < 0.3', r'0.3 < BR($t_4 \to h t$) < 0.5', r'BR($t_4 \to h t$) > 0.5'], frameon=True, loc=4, numpoints = 1,fontsize = 15)
#first_legend = plt.legend([ft, ftc, ftb, fta], [r'BR($H \to t_4 t) > 30$%', r'20% $<$ BR($H \to t_4 t) < 30$%', r'10% $<$ BR($H \to t_4 t) < 20$%', r'BR($H \to t_4 t) < 10$%'], frameon=True, loc=1, numpoints = 1,fontsize = 15)
first_legend.set_zorder(100)
plt.gca().add_artist(first_legend)
#second_legend = plt.legend([fgt,fgta,fgtb],[r'                  ',r'                                                        ',r'               '], frameon=False, loc=1, numpoints = 1, fontsize = 12, handletextpad=8,columnspacing = 1.1,borderpad=0.5,labelspacing=0.88)
#second_legend.set_zorder(101)
#frame = second_legend.get_frame()
#frame.set_facecolor('white')
plt.title(r'99% singlet-like $t_4$, others 100TeV', fontsize=17)
#plt.title(r'99% singlet-like $t_4$, others 10TeV', fontsize=17)
#plt.title(r'100% singlet $t_4 = T$', fontsize=17)
plt.savefig('t1ewpt-couplings.png')
#plt.savefig('t1ewpt-mt4-tanbe.png')


def gline(x):
    return 20.*x

xglw = numpy.arange(0.0, 1.1, 0.00001)
yglw = gline(xglw)


fig = plt.figure()  # make a new figure : separately from the above 
plt.axis([0.00005,1.1,0.00005,1.1]) 
fa, = plt.plot(tdlistgLW43,tdlistlamh34,'.', color = 'green', label='constraint',zorder=1)
fb, = plt.plot(tdalistgLW43,tdalistlamh34,'.', color = 'orange', label='constraint',zorder=1)
fc, = plt.plot(tdblistgLW43,tdblistlamh34,'.', color = 'navy', label='constraint',zorder=1)
#fa, = plt.plot(alistlamHt34,alistgLW43,'.', color = 'red', label='constraint',zorder=1)
#fb, = plt.plot(blistlamHt34,blistgLW43,'.', color = 'purple', label='constraint',zorder=2)
#fc, = plt.plot(clistlamHt34,clistgLW43,'.', color = 'cyan', label='constraint',zorder=3)
#fd, = plt.plot(dlistlamHt34,dlistgLW43,'.', color = 'blue', label='constraint',zorder=4)
ft, = plt.plot(xglw, yglw, 'k')
myax = plt.gca()
myax.set_xscale('log', nonposy='clip')
myax.set_yscale('log', nonposy='clip')
#myax.annotate(r'$\Gamma(H \to t_4 t)$', xy=(0.9, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
#myax.annotate(r'$\kappa_T$', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
myax.annotate(r'$g_L^{W t_4 b}$', xy=(0.85, -0.042), ha='left', va='top', xycoords='axes fraction', fontsize=20)
myax.annotate(r'$\lambda^h_{t t_4}$', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
majorLocator   = MultipleLocator(0.5)
majorFormatter = FormatStrFormatter('%2.1f')
minorLocator   = MultipleLocator(0.1)
ymajorLocator   = MultipleLocator(0.5)
ymajorFormatter = FormatStrFormatter('%2.1f')
yminorLocator   = MultipleLocator(0.1)
myax.tick_params(size=4, labelsize = 16)
#myax.tick_params(axis = 'y',zorder = 10) 
#myax.xaxis.set_major_locator(majorLocator)
#myax.xaxis.set_major_formatter(majorFormatter)
#myax.xaxis.set_minor_locator(minorLocator)
#myax.yaxis.set_major_locator(ymajorLocator)
#myax.yaxis.set_major_formatter(ymajorFormatter)
#myax.yaxis.set_minor_locator(yminorLocator)
first_legend = plt.legend([fta, ftb, ftc, ft], [r'BR($t_4 \to h t$) < 0.3', r'0.3 < BR($t_4 \to h t$) < 0.5', r'BR($t_4 \to h t$) > 0.3', r'Approximate relation'], frameon=True, loc=4, numpoints = 1,fontsize = 15)
first_legend.set_zorder(100)
plt.gca().add_artist(first_legend)
#second_legend = plt.legend([fgt,fgta,fgtb],[r'                  ',r'                                                        ',r'               '], frameon=False, loc=1, numpoints = 1, fontsize = 12, handletextpad=8,columnspacing = 1.1,borderpad=0.5,labelspacing=0.88)
#second_legend.set_zorder(101)
#frame = second_legend.get_frame()
#frame.set_facecolor('white')
plt.title(r'99% singlet-like $t_4$, others 100TeV', fontsize=17)
#plt.title(r'99% singlet-like $t_4$, others 10TeV', fontsize=17)
#plt.title(r'100% singlet $t_4 = T$', fontsize=17)
plt.savefig('t1ewpt-couplings2.png')








fig = plt.figure()  # make a new figure : separately from the above 
plt.axis([0.00005,1.1,0.00005,1.1])
#ft, = plt.plot(tdlistkaT,tdlistlamh43,'.', color = 'green', label='constraint',zorder=1)
#fa, = plt.plot(tdalistkaT,tdalistlamh43,'.', color = 'orange', label='constraint',zorder=2)
#fb, = plt.plot(tdblistkaT,tdblistlamh43,'.', color = 'navy', label='constraint',zorder=3)
ft, = plt.plot(tdlistlamh43,tdlistkaT,'.', color = 'green', label='constraint',zorder=1)
fa, = plt.plot(tdalistlamh43,tdalistkaT,'.', color = 'orange', label='constraint',zorder=2)
fb, = plt.plot(tdblistlamh43,tdblistkaT,'.', color = 'navy', label='constraint',zorder=3)
myax = plt.gca()
myax.set_xscale('log', nonposy='clip')
myax.set_yscale('log', nonposy='clip')
myax.annotate(r'$\lambda^h_{t_4 t}$', xy=(0.8, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
myax.annotate(r'$\kappa_T$', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
majorLocator   = MultipleLocator(0.2)
majorFormatter = FormatStrFormatter('%2.1f')
minorLocator   = MultipleLocator(0.1)
ymajorLocator   = MultipleLocator(0.2)
ymajorFormatter = FormatStrFormatter('%2.1f')
yminorLocator   = MultipleLocator(0.1)
myax.tick_params(size=3.5)
#myax.tick_params(axis = 'y',zorder = 10) 
#myax.xaxis.set_major_locator(majorLocator)
#myax.xaxis.set_major_formatter(majorFormatter)
#myax.xaxis.set_minor_locator(minorLocator)
#myax.yaxis.set_major_locator(ymajorLocator)
#myax.yaxis.set_major_formatter(ymajorFormatter)
#myax.yaxis.set_minor_locator(yminorLocator)
first_legend = plt.legend([ft, fa, fb], [r'BR($t_4 \to h t$) < 0.3', r'0.3 < BR($t_4 \to h t$) < 0.5', r'BR($t_4 \to h t$) > 0.5'], frameon=True, loc=2, numpoints = 1,fontsize = 15)
#first_legend = plt.legend([fta, ftb, ftc, ftd], [r'$\tan\beta <  1$', r'$1  < \tan\beta <  5$', r'$5  < \tan\beta <  10$', r'$\tan\beta >  10$'], frameon=True, loc=1, numpoints = 1,fontsize = 15)
first_legend.set_zorder(100)
plt.gca().add_artist(first_legend)
#second_legend = plt.legend([fgt,fgta,fgtb],[r'                  ',r'                                                        ',r'               '], frameon=False, loc=1, numpoints = 1, fontsize = 12, handletextpad=8,columnspacing = 1.1,borderpad=0.5,labelspacing=0.88)
#second_legend.set_zorder(101)
#frame = second_legend.get_frame()
#frame.set_facecolor('white')
plt.title(r'99% singlet-like $t_4$, others 100TeV', fontsize=17)
#plt.title(r'99% singlet-like $t_4$, others 10TeV', fontsize=17)
plt.savefig('t1ewpt-couplings3.png')

fig = plt.figure()  # make a new figure : separately from the above 
plt.axis([0.00005,1.1,0.00005,1.1])
ft, = plt.plot(tdlistkaT,tdlistka,'.', color = 'green', label='constraint',zorder=1)
fa, = plt.plot(tdalistkaT,tdalistka,'.', color = 'orange', label='constraint',zorder=2)
fb, = plt.plot(tdblistkaT,tdblistka,'.', color = 'navy', label='constraint',zorder=3)
#fa, = plt.plot(dlistkaT,dlistlamHt43,'.', color = 'blue', label='constraint',zorder=4)
plt.plot(compx, compx, '--', color = 'black')
myax = plt.gca()
myax.set_xscale('log', nonposy='clip')
myax.set_yscale('log', nonposy='clip')
myax.annotate(r'$\kappa_T$', xy=(0.8, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
myax.annotate(r'$\kappa$', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
majorLocator   = MultipleLocator(0.2)
majorFormatter = FormatStrFormatter('%2.1f')
minorLocator   = MultipleLocator(0.1)
ymajorLocator   = MultipleLocator(0.2)
ymajorFormatter = FormatStrFormatter('%2.1f')
yminorLocator   = MultipleLocator(0.1)
myax.tick_params(size=3.5)
#myax.tick_params(axis = 'y',zorder = 10) 
#myax.xaxis.set_major_locator(majorLocator)
#myax.xaxis.set_major_formatter(majorFormatter)
#myax.xaxis.set_minor_locator(minorLocator)
#myax.yaxis.set_major_locator(ymajorLocator)
#myax.yaxis.set_major_formatter(ymajorFormatter)
#myax.yaxis.set_minor_locator(yminorLocator)
first_legend = plt.legend([ft, fa, fb], [r'BR($t_4 \to h t$) < 0.3', r'0.3 < BR($t_4 \to h t$) < 0.5', r'BR($t_4 \to h t$) > 0.5'], frameon=True, loc=3, numpoints = 1,fontsize = 15)
#first_legend = plt.legend([fta, ftb, ftc, ftd], [r'$\tan\beta <  1$', r'$1  < \tan\beta <  5$', r'$5  < \tan\beta <  10$', r'$\tan\beta >  10$'], frameon=True, loc=1, numpoints = 1,fontsize = 15)
first_legend.set_zorder(100)
plt.gca().add_artist(first_legend)
#second_legend = plt.legend([fgt,fgta,fgtb],[r'                  ',r'                                                        ',r'               '], frameon=False, loc=1, numpoints = 1, fontsize = 12, handletextpad=8,columnspacing = 1.1,borderpad=0.5,labelspacing=0.88)
#second_legend.set_zorder(101)
#frame = second_legend.get_frame()
#frame.set_facecolor('white')
#plt.title(r'99% singlet-like $t_4$, others 100TeV', fontsize=17)
plt.title(r'99% singlet-like $t_4$, others 10TeV', fontsize=17)
plt.savefig('t1ewpt-couplings5.png')



fig = plt.figure()  # make a new figure : separately from the above 
plt.axis([0.00005,1.1,0.00005,1.1])
ft, = plt.plot(tdlistkaT,tdlistkaba,'.', color = 'green', label='constraint',zorder=1)
fa, = plt.plot(tdalistkaT,tdalistkaba,'.', color = 'orange', label='constraint',zorder=2)
fb, = plt.plot(tdblistkaT,tdblistkaba,'.', color = 'navy', label='constraint',zorder=3)
#fa, = plt.plot(dlistkaT,dlistlamHt43,'.', color = 'blue', label='constraint',zorder=4)
plt.plot(compx, compx, '--', color = 'black')
myax = plt.gca()
myax.set_xscale('log', nonposy='clip')
myax.set_yscale('log', nonposy='clip')
myax.annotate(r'$\kappa_T$', xy=(0.8, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
myax.annotate(r'$\bar \kappa$', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
majorLocator   = MultipleLocator(0.2)
majorFormatter = FormatStrFormatter('%2.1f')
minorLocator   = MultipleLocator(0.1)
ymajorLocator   = MultipleLocator(0.2)
ymajorFormatter = FormatStrFormatter('%2.1f')
yminorLocator   = MultipleLocator(0.1)
myax.tick_params(size=3.5)
#myax.tick_params(axis = 'y',zorder = 10) 
#myax.xaxis.set_major_locator(majorLocator)
#myax.xaxis.set_major_formatter(majorFormatter)
#myax.xaxis.set_minor_locator(minorLocator)
#myax.yaxis.set_major_locator(ymajorLocator)
#myax.yaxis.set_major_formatter(ymajorFormatter)
#myax.yaxis.set_minor_locator(yminorLocator)
first_legend = plt.legend([ft, fa, fb], [r'BR($t_4 \to h t$) < 0.3', r'0.3 < BR($t_4 \to h t$) < 0.5', r'BR($t_4 \to h t$) > 0.5'], frameon=True, loc=3, numpoints = 1,fontsize = 15)
#first_legend = plt.legend([fta, ftb, ftc, ftd], [r'$\tan\beta <  1$', r'$1  < \tan\beta <  5$', r'$5  < \tan\beta <  10$', r'$\tan\beta >  10$'], frameon=True, loc=1, numpoints = 1,fontsize = 15)
first_legend.set_zorder(100)
plt.gca().add_artist(first_legend)
#second_legend = plt.legend([fgt,fgta,fgtb],[r'                  ',r'                                                        ',r'               '], frameon=False, loc=1, numpoints = 1, fontsize = 12, handletextpad=8,columnspacing = 1.1,borderpad=0.5,labelspacing=0.88)
#second_legend.set_zorder(101)
#frame = second_legend.get_frame()
#frame.set_facecolor('white')
#plt.title(r'99% singlet-like $t_4$, others 100TeV', fontsize=17)
plt.title(r'99% singlet-like $t_4$, others 10TeV', fontsize=17)
plt.savefig('t1ewpt-couplings6.png')



fig = plt.figure()  # make a new figure : separately from the above 
plt.axis([-1,1,-1,1])
#plt.axis([0.000000000005,1.1,0.000000000005,1.1])
#ft, = plt.plot(tdlistkaT,tdlistlamh43,'.', color = 'green', label='constraint',zorder=1)
#fa, = plt.plot(tdalistkaT,tdalistlamh43,'.', color = 'orange', label='constraint',zorder=2)
#fb, = plt.plot(tdblistkaT,tdblistlamh43,'.', color = 'navy', label='constraint',zorder=3)
ft, = plt.plot(tlistgLW44t,tlistgRW44t,'.', color = 'navy', label='constraint',zorder=1)
#ft, = plt.plot(tdlistlamh43,tdlistgRZ43,'.', color = 'green', label='constraint',zorder=1)
#fa, = plt.plot(tdalistlamh43,tdalistgRZ43,'.', color = 'orange', label='constraint',zorder=2)
#fb, = plt.plot(tdblistlamh43,tdblistgRZ43,'.', color = 'navy', label='constraint',zorder=3)
#fc, = plt.plot(tdclistlamh43,tdclistgRZ43,'.', color = 'red', label='constraint',zorder=3)
myax = plt.gca()
#myax.set_xscale('log', nonposy='clip')
#myax.set_yscale('log', nonposy='clip')
#myax.annotate(r'$\lambda^h_{t_4 t}$', xy=(0.84, -0.045), ha='left', va='top', xycoords='axes fraction', fontsize=18.5)
#myax.annotate(r'$g_R^{Z t_4 t}$', xy=(-0.15, 1), ha='left', va='top', xycoords='axes fraction', fontsize=19, rotation = 90)
myax.annotate(r'$g_L^{W t_4 b_4}$', xy=(0.84, -0.045), ha='left', va='top', xycoords='axes fraction', fontsize=18.5)
myax.annotate(r'$g_R^{W t_4 b_4}$', xy=(-0.15, 1), ha='left', va='top', xycoords='axes fraction', fontsize=19, rotation = 90)
cmajorLocator   = MultipleLocator(0.2)
majorFormatter = FormatStrFormatter('%2.1f')
minorLocator   = MultipleLocator(0.1)
ymajorLocator   = MultipleLocator(0.2)
ymajorFormatter = FormatStrFormatter('%2.1f')
yminorLocator   = MultipleLocator(0.1)
myax.tick_params(size=3.5)
#myax.tick_params(axis = 'y',zorder = 10) 
#myax.xaxis.set_major_locator(majorLocator)
#myax.xaxis.set_major_formatter(majorFormatter)
#myax.xaxis.set_minor_locator(minorLocator)
#myax.yaxis.set_major_locator(ymajorLocator)
#myax.yaxis.set_major_formatter(ymajorFormatter)
#myax.yaxis.set_minor_locator(yminorLocator)
#first_legend = plt.legend([ft, fa, fb], [r'BR($t_4 \to h t$) < 0.3', r'0.3 < BR($t_4 \to h t$) < 0.5', r'BR($t_4 \to h t$) > 0.5'], frameon=True, loc=2, numpoints = 1,fontsize = 15)
#first_legend = plt.legend([fb,fc], [r'BR($t_4 \to h t$) > 0.5', r'BR($t_4 \to Z t$) > 0.5'], frameon=True, loc=2, numpoints = 1,fontsize = 15)
#first_legend = plt.legend([fta, ftb, ftc, ftd], [r'$\tan\beta <  1$', r'$1  < \tan\beta <  5$', r'$5  < \tan\beta <  10$', r'$\tan\beta >  10$'], frameon=True, loc=1, numpoints = 1,fontsize = 15)
#first_legend.set_zorder(100)
#plt.gca().add_artist(first_legend)
#second_legend = plt.legend([fgt,fgta,fgtb],[r'                  ',r'                                                        ',r'               '], frameon=False, loc=1, numpoints = 1, fontsize = 12, handletextpad=8,columnspacing = 1.1,borderpad=0.5,labelspacing=0.88)
#second_legend.set_zorder(101)
#frame = second_legend.get_frame()
#frame.set_facecolor('white')
#plt.title(r'99% singlet-like $t_4$, others 100TeV, $\kappa_T = \bar \kappa = 0$', fontsize=17)
#plt.title(r'99% singlet-like $t_4$, others 100TeV, $\kappa_T = \kappa = 0$', fontsize=17)
#plt.title(r'99% singlet-like $t_4$, others 100TeV, $\kappa_T = 0$', fontsize=17)
#plt.title(r'99% singlet-like $t_4$, others 100TeV, $\kappa = \bar \kappa = 0$', fontsize=17)
#plt.title(r'99% singlet-like $t_4$, others 10TeV', fontsize=17)
plt.savefig('t1ewpt-couplings7.png')

#print 'new compare =', max(tdlistgRZ43), max(tdlistgLZ43), max(tdlistlamh43), max(tdlistlamh34), max(tdlistgLW43), max(tdlistbrt4w), max(tdlistbrt4z), max(tdalistlamh43), max(tdblistlamh43), max(tdclistgLZ43), max(tdclistgRZ43), max(tdclistgLW43), max(tdclistlamh43)




fig = plt.figure()  # make a new figure : separately from the above 
plt.axis([0.00005,1.1,0.00005,1.1])
ft, = plt.plot(talistka,talistkab,'.', color = 'navy', label='constraint',zorder=1)
#fa, = plt.plot(tdalistkaT,tdalistka,'.', color = 'orange', label='constraint',zorder=2)
#fb, = plt.plot(tdblistkaT,tdblistka,'.', color = 'navy', label='constraint',zorder=3)
#fa, = plt.plot(dlistkaT,dlistlamHt43,'.', color = 'blue', label='constraint',zorder=4)
plt.plot(compx, compx, '--', color = 'black')
myax = plt.gca()
myax.set_xscale('log', nonposy='clip')
myax.set_yscale('log', nonposy='clip')
myax.annotate(r'$|\kappa|$', xy=(0.8, -0.055), ha='left', va='top', xycoords='axes fraction', fontsize=20)
myax.annotate(r'$|\bar \kappa|$', xy=(-0.13, 1), ha='left', va='top', xycoords='axes fraction', fontsize=20, rotation = 90)
majorLocator   = MultipleLocator(0.2)
majorFormatter = FormatStrFormatter('%2.1f')
minorLocator   = MultipleLocator(0.1)
ymajorLocator   = MultipleLocator(0.2)
ymajorFormatter = FormatStrFormatter('%2.1f')
yminorLocator   = MultipleLocator(0.1)
myax.tick_params(size=3.5)
#myax.tick_params(axis = 'y',zorder = 10) 
#myax.xaxis.set_major_locator(majorLocator)
#myax.xaxis.set_major_formatter(majorFormatter)
#myax.xaxis.set_minor_locator(minorLocator)
#myax.yaxis.set_major_locator(ymajorLocator)
#myax.yaxis.set_major_formatter(ymajorFormatter)
#myax.yaxis.set_minor_locator(yminorLocator)
#first_legend = plt.legend([ft, fa, fb], [r'BR($t_4 \to h t$) < 0.3', r'0.3 < BR($t_4 \to h t$) < 0.5', r'BR($t_4 \to h t$) > 0.5'], frameon=True, loc=3, numpoints = 1,fontsize = 15)
#first_legend = plt.legend([fta, ftb, ftc, ftd], [r'$\tan\beta <  1$', r'$1  < \tan\beta <  5$', r'$5  < \tan\beta <  10$', r'$\tan\beta >  10$'], frameon=True, loc=1, numpoints = 1,fontsize = 15)
#first_legend.set_zorder(100)
#plt.gca().add_artist(first_legend)
#second_legend = plt.legend([fgt,fgta,fgtb],[r'                  ',r'                                                        ',r'               '], frameon=False, loc=1, numpoints = 1, fontsize = 12, handletextpad=8,columnspacing = 1.1,borderpad=0.5,labelspacing=0.88)
#second_legend.set_zorder(101)
#frame = second_legend.get_frame()
#frame.set_facecolor('white')
plt.title(r'$\kappa_T = 0$, BR$(t_4 \to Z t) > 0.9$', fontsize=17)
plt.savefig('t1ewpt-couplings8.png')




"""
    
    
