#!/usr/bin/python



import os

exe = os.system

from math import *

import random

import numpy

from numpy import ndarray
from numpy import *


from scipy.special import spence

import cmath


from scipy import linalg
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import interpolate


def unitstep(x):
    if x > 0:
        return 1
    else:
        return 0


def phase(x,y,z):
    return (x-y-z)**2 - 4*y*z
    

def alphas(mass):
    xx = numpy.array([130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350,360,370,380,390,400,420,440,450,460,480,500,520,540,550,560,580,600,620,640,650,660,680,700,720,740,750,760,780,800,820,840,850,860,880,900,920,940,950,960,980,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400,2500,2600,2700,2800,2900,3000,3100,3200,3300,3400,3500,3600,3700,3800,3900,4000])
    yy = numpy.array([0.112813, 0.111621, 0.110533, 0.109536, 0.108616, 0.107762, 0.106968, 0.106225, 0.105528, 0.104872, 0.104253, 0.103667, 0.103112, 0.102584, 0.102081, 0.101601, 0.101142, 0.100703, 0.100282, 0.099878, 0.0994894, 0.0991153, 0.0987548, 0.098407, 0.0980712, 0.0977465, 0.0974324, 0.0971282, 0.0965474, 0.0960001, 0.095738, 0.0954831, 0.0949934, 0.0945285, 0.0940862, 0.0936645, 0.0934609, 0.0932618, 0.0928765, 0.0925074, 0.0921533, 0.091813, 0.0916477, 0.0914856, 0.0911702, 0.0908661, 0.0905725, 0.0902889, 0.0901506, 0.0900145, 0.0897489, 0.0894916,0.0890193, 0.0887783, 0.0886605, 0.0885443, 0.0883169, 0.0880958, 0.0878806, 0.087671, 0.0875683, 0.0874669, 0.0872679, 0.0870738,0.0863783, 0.0855656, 0.0848317, 0.0841635, 0.083551, 0.0829862, 0.0824626, 0.0819751, 0.0815194, 0.0810917, 0.0806892, 0.0803091, 0.0799493, 0.0796079, 0.0792833, 0.0789738, 0.0786784, 0.0783958, 0.0781251, 0.0778654, 0.0776158, 0.0773757, 0.0771445, 0.0769214, 0.0767061, 0.0764981, 0.0762968, 0.0761019, 0.0759131, 0.0757299])
    order = 2
    myalphas = InterpolatedUnivariateSpline(xx, yy, k=order)
    return myalphas(mass)


#ATLAS 1707.03347: primary target t4 -> Wb lepton+jets final state
#Used hepdata https://hepdata.net/search/?q=1707.03347
#https://www.hepdata.net/record/ins1609451
#https://www.hepdata.net/record/ins1609451
#Note that white blank region (bound below 500 GeV) is just interpolated
#(conservative bounds)
atlast4t4 = loadtxt('atlast4pair.dat')
brt4wbin = atlast4t4[:,0]  # array of the 1st column of the data file
brt4htin = atlast4t4[:,1]  
mt4input = atlast4t4[:,2]
mt4limit = interpolate.interp2d(brt4wbin, brt4htin, mt4input, kind='cubic')
def intmt4upper(brt4w, brt4h):
    if mt4 > 1338.15:
        return 0.
    else:
        return mt4limit(brt4w, brt4h)[0]

atlasb4b4 = loadtxt('atlasb4pair.dat')
brb4wtin = atlasb4b4[:,0]  # array of the 1st column of the data file
brb4hbin = atlasb4b4[:,1]  
mb4input = atlasb4b4[:,2]
mb4limit = interpolate.interp2d(brb4wtin, brb4hbin, mb4input, kind='cubic')
def intmb4upper(brb4w, brb4h):
    if mb4 > 1237.11:
        return 0.
    else:
        return mb4limit(brb4w, brb4h)[0]





exe("rm t1ewptall-s8.dat")
#exe("rm t1ewptall-test-s8.dat")
#exe("rm t1ewptall-fix-test-s1.dat")
#exe("rm t1ewptall-fix-s1.dat")
#exe("rm t1ewptall-fix-s2-noconstraint.dat")
#exe("rm t1ewptall-extreme-s4.dat")
#exe("rm t1ewptall-10TeV-s4.dat")



MW = 80.385
MZ = 91.1876
g = 0.652954
thetaw = acos(MW/MZ)
mmu = 0.1056584                  #physics muon mass
GF = (sqrt(2)/8.)*pow(g/MW,2)
v = (1/sqrt(sqrt(2.0)*GF))/sqrt(2.0)             #174 GeV
gelsm = (g/cos(thetaw))*(pow(sin(thetaw),2) - 1/2.)   # g_L^Z SM : for mu or e or tau
gersm = (g/cos(thetaw))*pow(sin(thetaw),2)            # g_R^Z SM : for mu or e or tau

listfrac = ['SM','doublet','singlet']

"""
exe("rm cleaned_chargedewptallhelp.dat")


infile = "chargedewptallhelp.dat"
outfile = "cleaned_chargedewptallhelp.dat"

delete_list = ["[", "]"]
cleanfin = open(infile)
cleanfout = open(outfile, "w+")
for line in cleanfin:
    for word in delete_list:
        line = line.replace(word, "")
    cleanfout.write(line)
cleanfin.close()
cleanfout.close()
"""

#fin=open("cleaned_chargedewptallhelp.dat")
fin=open("t1ewptallhelp-s8.dat")
#fin=open("t1ewptallhelp-test-s8.dat")
#fin=open("t1ewptallhelp-fix-s1.dat")
#fin=open("t1ewptallhelp-fix-test-s3.dat")
#fin=open("t1ewptallhelp-extreme-s4.dat")
#fin=open("t1ewptallhelp-0.5Yukawa.dat")
inhelp=fin.readlines()
fin.close()



for i in range(0,len(inhelp)-1):
    lin = inhelp[i]
    param=map(str, lin.split())
    mt4 = float(param[0])
    beta = atan(float(param[1]))
    lamHt34 = float(param[2])
    mHH = float(param[3])
    doubfr = float(param[4])   # doublet fraction
    singfr = float(param[5])   # singlet fraction
    delS = float(param[6])
    delT = float(param[7])
    delU = float(param[8])
    lamHrt34 = float(param[9])
    lamHt44 = float(param[10])
    lamHrt44 = float(param[11])
    gLW44t = float(param[12])
    gLW43t = float(param[13])
    gLW45t = float(param[14])
    gLW33t = float(param[15])
    gLW34t = float(param[16])
    gLW35t = float(param[17])
    gLW55t = float(param[18])
    gLW53t = float(param[19])
    gLW54t = float(param[20])
    gRW44t = float(param[21])
    gRW43t = float(param[22])
    gRW45t = float(param[23])
    gRW33t = float(param[24])
    gRW34t = float(param[25])
    gRW35t = float(param[26])
    gRW55t = float(param[27])
    gRW54t = float(param[28])
    gRW53t = float(param[29])
    gLZ44t = float(param[30])
    gLZ43t = float(param[31])
    gLZ45t = float(param[32])
    gLZ33t = float(param[33])
    gLZ34t = float(param[34])
    gLZ35t = float(param[35])
    gLZ55t = float(param[36])
    gLZ53t = float(param[37])
    gLZ54t = float(param[38])
    gRZ44t = float(param[39])
    gRZ43t = float(param[40])
    gRZ45t = float(param[41])
    gRZ33t = float(param[42])
    gRZ34t = float(param[43])
    gRZ35t = float(param[44])
    gRZ55t = float(param[45])
    gRZ53t = float(param[46])
    gRZ54t = float(param[47])
    kaT = float(param[48])
    kaQ = float(param[49])
    ka = float(param[50])
    kab = float(param[51])
    mt5 = float(param[52])
    MTT = float(param[53])
    MQQ = float(param[54])
    lamHt55 = float(param[55])
    lamHrt55 = float(param[56])
    lamHt35 = float(param[57])
    lamHrt35 = float(param[58])
    lamHt45 = float(param[59])
    lamHrt45 = float(param[60])
    lamHb34 = float(param[61])
    lamHrb34 = float(param[62])
    lamHb35 = float(param[63])
    lamHrb35 = float(param[64])
    lamHb45 = float(param[65])
    lamHrb45 = float(param[66])
    lamHb44 = float(param[67])
    lamHrb44 = float(param[68])
    lamHb55 = float(param[69])
    lamHrb55 = float(param[70])
    gLZ33b = float(param[71])
    gLZ44b = float(param[72])
    gLZ55b = float(param[73])
    gLZ34b = float(param[74])
    gLZ43b = float(param[75])
    gLZ35b = float(param[76])
    gLZ53b = float(param[77])
    gLZ45b = float(param[78])
    gLZ54b = float(param[79])
    gRZ33b = float(param[80])
    gRZ44b = float(param[81])
    gRZ55b = float(param[82])
    gRZ34b = float(param[83])
    gRZ43b = float(param[84])
    gRZ35b = float(param[85])
    gRZ53b = float(param[86])
    gRZ45b = float(param[87])
    gRZ54b = float(param[88])
    gLW42t = float(param[89])
    gLW52t = float(param[90])
    gLW24b = float(param[91])
    gLW25b = float(param[92])
    gLW41t = float(param[93])
    gLW51t = float(param[94])
    gLW14b = float(param[95])
    gLW15b = float(param[96])
    mb4 = float(param[97])
    mb5 = float(param[98])
    doubt5fr = float(param[99])
    singt5fr = float(param[100])
    doubb4fr = float(param[101])
    singb4fr = float(param[102])
    doubb5fr = float(param[103])
    singb5fr = float(param[104])

    lamHtest = float(param[-5])
    #lamHtest1 = float(param[-4])
    #lamHtest2 = float(param[-3])
    #lamHtest3 = float(param[-2])
    #lamHtest4 = float(param[-1])
    lamHpmb43 = float(param[-4])
    lamHpmbr43 = float(param[-3])
    lamHpm43 = float(param[-2])
    lamHpmr43 = float(param[-1])
    


    # Decide whether t4 is doublet or singlet like
    lf = [abs(float(1.)),abs(float(doubfr)),abs(float(singfr))]
    mytest = numpy.array(lf)
    #maxval = mytest.max()
    nn = numpy.unravel_index(mytest.argmax(), mytest.shape)
    frac = listfrac[nn[0]]

    # Branching Ratios
    # BR(t4)
    mtpole = 173.2
    mbpole = 4.18
    gammat4toz = (mt4/(32.*pi))*((gLZ43t**2 + gRZ43t**2)*(1. + (mt4**2 - 2.*mtpole**2)/MZ**2 + (mtpole**2 - 2.*MZ**2)/mt4**2 + (mtpole**4)/(MZ*mt4)**2) - 12.*gLZ43t*gRZ43t*mtpole/mt4)*sqrt(abs(phase(1.,(MZ/mt4)**2,(mtpole/mt4)**2)))*unitstep(mt4 - MZ - mtpole)
    
    gammat4tow = (mt4/(32.*pi))*((gLW43t**2 + gRW43t**2)*(1. + (mt4**2 - 2.*mbpole**2)/MW**2 + (mbpole**2 - 2.*MW**2)/mt4**2 + (mbpole**4)/(MW*mt4)**2) - 12.*gLW43t*gRW43t*mbpole/mt4)*sqrt(abs(phase(1.,(MW/mt4)**2,(mbpole/mt4)**2)))*unitstep(mt4 - MW - mbpole)
    
    gammat4toh = ((mt4/(64.*pi))*(((-lamHt34*tan(beta))**2 + (-lamHrt34*tan(beta))**2)*(1. + (mtpole**2 -125.**2)/mt4**2) + 4.*(lamHt34*tan(beta))*(lamHrt34*tan(beta))*mtpole/mt4)*sqrt(abs(phase(1.,(125./mt4)**2,(mtpole/mt4)**2))))*unitstep(mt4 - 125. - mtpole)

    gammat4tob4w = (mt4/(32.*pi))*((gLW44t**2 + gRW44t**2)*(1. + (mt4**2 - 2.*mb4**2)/MW**2 + (mb4**2 - 2.*MW**2)/mt4**2 + (mb4**4)/(MW*mt4)**2) - 12.*gLW44t*gRW44t*mb4/mt4)*sqrt(abs(phase(1.,(MW/mt4)**2,(mb4/mt4)**2)))*unitstep(mt4 - MW - mb4)

    gammat4tob5w = (mt4/(32.*pi))*((gLW45t**2 + gRW45t**2)*(1. + (mt4**2 - 2.*mb5**2)/MW**2 + (mb5**2 - 2.*MW**2)/mt4**2 + (mb5**4)/(MW*mt4)**2) - 12.*gLW45t*gRW45t*mb5/mt4)*sqrt(abs(phase(1.,(MW/mt4)**2,(mb5/mt4)**2)))*unitstep(mt4 - MW - mb5)

    #gammat4test = (mt4/(32.*pi))*(((-lamHt34*tan(beta))**2 + (-lamHrt34*tan(beta))**2)*(1. + (mt4**2 - 2.*mbpole**2)/MW**2))*sqrt(abs(phase(1.,(MW/mt4)**2,(mbpole/mt4)**2)))*unitstep(mt4 - MW - mbpole)
    
    #Neutral CP-even Higgs H
    #if mt4 < mHH + mt:
        #continue
    gammat4toHH = ((mt4/(64.*pi))*(((-lamHt34)**2 + (-lamHrt34)**2)*(1. + (mtpole**2 -mHH**2)/mt4**2) + 4.*(lamHt34)*(lamHrt34)*mtpole/mt4)*sqrt(abs(phase(1.,(mHH/mt4)**2,(mtpole/mt4)**2))))*unitstep(mt4 - mHH - mtpole)
    #Charged Higgs case:let's assume it H for simplicity
    #gammat4toHH = ((mt4/(64.*pi))*(((lamHpm43)**2 + (lamHpmr43)**2)*(1. + (mbpole**2 - mHH**2)/mt4**2) + 4.*(lamHpm43)*(lamHpmr43)*mbpole/mt4)*sqrt(abs(phase(1.,(mHH/mt4)**2,(mbpole/mt4)**2))))*unitstep(mt4 - mHH - mbpole)

    

    #Higgs cascade paper
    if gammat4toz + gammat4tow + gammat4toh + gammat4tob4w + gammat4tob5w + gammat4toHH == 0:
        gamt4all = 1
    else:
        gamt4all = gammat4toz + gammat4tow + gammat4toh + gammat4tob4w + gammat4tob5w + gammat4toHH
    #VLQ BR paper
    #if mt4 > MW + mb4:
        #continue
    #gamt4all = gammat4toz + gammat4tow + gammat4toh
    try:
        brt4z = gammat4toz/gamt4all
        brt4w = gammat4tow/gamt4all
        brt4h = gammat4toh/gamt4all
        brt4b4w = gammat4tob4w/gamt4all
        brt4b5w = gammat4tob5w/gamt4all
        brt4HH = gammat4toHH/gamt4all
    except ZeroDivisionError:
        continue
    
    #BR(b4)
    
    gammab4toz = (mb4/(32.*pi))*((gLZ43b**2 + gRZ43b**2)*(1. + (mb4**2 - 2.*mbpole**2)/MZ**2 + (mbpole**2 - 2.*MZ**2)/mb4**2 + (mbpole**4)/(MZ*mb4)**2) - 12.*gLZ43b*gRZ43b*mbpole/mb4)*sqrt(abs(phase(1.,(MZ/mb4)**2,(mbpole/mb4)**2)))*unitstep(mb4 - MZ - mbpole)
    
    gammab4tow = (mb4/(32.*pi))*((gLW34t**2 + gRW34t**2)*(1. + (mb4**2 - 2.*mtpole**2)/MW**2 + (mtpole**2 - 2.*MW**2)/mb4**2 + (mtpole**4)/(MW*mb4)**2) - 12.*gLW34t*gRW34t*mtpole/mb4)*sqrt(abs(phase(1.,(MW/mb4)**2,(mtpole/mb4)**2)))*unitstep(mb4 - MW - mtpole)

    gammab4toh = ((mb4/(64.*pi))*(((lamHb34/tan(beta))**2 + (lamHrb34/tan(beta))**2)*(1. + (mbpole**2 -125.**2)/mb4**2) + 4.*(-lamHb34/tan(beta))*(-lamHrb34/tan(beta))*mbpole/mb4 )*sqrt(abs(phase(1.,(125./mb4)**2,(mbpole/mb4)**2))))*unitstep(mb4 - 125. - mbpole)
    
    gammab4tot4w = (mb4/(32.*pi))*((gLW44t**2 + gRW44t**2)*(1. + (mb4**2 - 2.*mt4**2)/MW**2 + (mt4**2 - 2.*MW**2)/mb4**2 + (mt4**4)/(MW*mb4)**2) - 12.*gLW44t*gRW44t*mt4/mb4)*sqrt(abs(phase(1.,(MW/mb4)**2,(mt4/mb4)**2)))*unitstep(mb4 - MW - mt4)

    gammab4tot5w = (mb4/(32.*pi))*((gLW54t**2 + gRW54t**2)*(1. + (mb4**2 - 2.*mt5**2)/MW**2 + (mt5**2 - 2.*MW**2)/mb4**2 + (mt5**4)/(MW*mb4)**2) - 12.*gLW54t*gRW54t*mt5/mb4)*sqrt(abs(phase(1.,(MW/mb4)**2,(mt5/mb4)**2)))*unitstep(mb4 - MW - mt5)

    #Charged Higgs case:let's assume it H for simplicity
    gammab4toHH = ((mb4/(64.*pi))*(((lamHpmb43)**2 + (lamHpmbr43)**2)*(1. + (mtpole**2 - mHH**2)/mb4**2) + 4.*(lamHpmb43)*(lamHpmbr43)*mtpole/mb4)*sqrt(abs(phase(1.,(mHH/mb4)**2,(mtpole/mb4)**2))))*unitstep(mb4 - mHH - mtpole)

    #Higgs cascade paper
    if gammab4toz + gammab4tow + gammab4toh + gammab4tot4w + gammab4tot5w + gammab4toHH == 0:
        gamb4all = 1.
    else:
        gamb4all = gammab4toz + gammab4tow + gammab4toh + gammab4tot4w + gammab4tot5w + gammab4toHH
    #VLQ BR paper
    #if mb4 > MW + mt4:
        #continue
    #gamb4all = gammab4toz + gammab4tow + gammab4toh
    try:
        brb4z = gammab4toz/gamb4all
        brb4w = gammab4tow/gamb4all
        brb4h = gammab4toh/gamb4all
        brb4t4w = gammab4tot4w/gamb4all
        brb4t5w = gammab4tot5w/gamb4all
        brb4HH = gammab4toHH/gamb4all
    except ZeroDivisionError:
        continue

    
    #BR(t5)
    
    gammat5toz = (mt5/(32.*pi))*((gLZ53t**2 + gRZ53t**2)*(1. + (mt5**2 - 2.*mtpole**2)/MZ**2 + (mtpole**2 - 2.*MZ**2)/mt5**2 + (mtpole**4)/(MZ*mt5)**2) - 12.*gLZ53t*gRZ53t*mtpole/mt5)*sqrt(abs(phase(1.,(MZ/mt5)**2,(mtpole/mt5)**2)))*unitstep(mt5 - MZ - mtpole)
    
    gammat5tow = (mt5/(32.*pi))*((gLW53t**2 + gRW53t**2)*(1. + (mt5**2 - 2.*mbpole**2)/MW**2 + (mbpole**2 - 2.*MW**2)/mt5**2 + (mbpole**4)/(MW*mt5)**2) - 12.*gLW53t*gRW53t*mbpole/mt5)*sqrt(abs(phase(1.,(MW/mt5)**2,(mbpole/mt5)**2)))*unitstep(mt5 - MW - mbpole)
    
    gammat5toh = ((mt5/(64.*pi))*(((-lamHt35*tan(beta))**2 + (-lamHrt35*tan(beta))**2)*(1. + (mtpole**2 -125.**2)/mt5**2) + 4.*(lamHt35*tan(beta))*(lamHrt35*tan(beta))*mtpole/mt5)*sqrt(abs(phase(1.,(125./mt5)**2,(mtpole/mt5)**2))))*unitstep(mt5 - 125. - mtpole)

    gammat5tob4w = (mt5/(32.*pi))*((gLW54t**2 + gRW54t**2)*(1. + (mt5**2 - 2.*mb4**2)/MW**2 + (mb4**2 - 2.*MW**2)/mt5**2 + (mb4**4)/(MW*mt5)**2) - 12.*gLW54t*gRW54t*mb4/mt5)*sqrt(abs(phase(1.,(MW/mt5)**2,(mb4/mt5)**2)))*unitstep(mt5 - MW - mb4)

    gammat5tob5w = (mt5/(32.*pi))*((gLW55t**2 + gRW55t**2)*(1. + (mt5**2 - 2.*mb5**2)/MW**2 + (mb5**2 - 2.*MW**2)/mt5**2 + (mb5**4)/(MW*mt5)**2) - 12.*gLW55t*gRW55t*mb5/mt5)*sqrt(abs(phase(1.,(MW/mt5)**2,(mb5/mt5)**2)))*unitstep(mt5 - MW - mb5)

    gammat5tot4z = (mt5/(32.*pi))*((gLZ54t**2 + gRZ54t**2)*(1. + (mt5**2 - 2.*mt4**2)/MZ**2 + (mt4**2 - 2.*MZ**2)/mt5**2 + (mt4**4)/(MZ*mt5)**2) - 12.*gLZ54t*gRZ54t*mt4/mt5)*sqrt(abs(phase(1.,(MZ/mt5)**2,(mt4/mt5)**2)))*unitstep(mt5 - MZ - mt4)

    gammat5tot4h = ((mt5/(64.*pi))*(((-lamHt45*tan(beta))**2 + (-lamHrt45*tan(beta))**2)*(1. + (mt4**2 -125.**2)/mt5**2) + 4.*(lamHt45*tan(beta))*(lamHrt45*tan(beta))*mt4/mt5)*sqrt(abs(phase(1.,(125./mt5)**2,(mt4/mt5)**2))))*unitstep(mt5 - 125. - mt4)

    if gammat5toz + gammat5tow + gammat5toh + gammat5tob4w + gammat5tob5w + gammat5tot4z + gammat5tot4h == 0:
        gamt5all = 1
    else:
        gamt5all = gammat5toz + gammat5tow + gammat5toh + gammat5tob4w + gammat5tob5w + gammat5tot4z + gammat5tot4h
    #VLQ BR paper
    #if mt4 > MW + mb4:
        #continue
    #gamt4all = gammat4toz + gammat4tow + gammat4toh
    try:
        brt5z = gammat5toz/gamt5all
        brt5w = gammat5tow/gamt5all
        brt5h = gammat5toh/gamt5all
        brt5b4w = gammat5tob4w/gamt5all
        brt5b5w = gammat5tob5w/gamt5all
        brt5t4z = gammat5tot4z/gamt5all
        brt5t4h = gammat5tot4h/gamt5all
    except ZeroDivisionError:
        continue

    #BR(b5)

    gammab5toz = (mb5/(32.*pi))*((gLZ53b**2 + gRZ53b**2)*(1. + (mb5**2 - 2.*mbpole**2)/MZ**2 + (mbpole**2 - 2.*MZ**2)/mb5**2 + (mbpole**4)/(MZ*mb5)**2) - 12.*gLZ53b*gRZ53b*mbpole/mb5)*sqrt(abs(phase(1.,(MZ/mb5)**2,(mbpole/mb5)**2)))*unitstep(mb5 - MZ - mbpole)
    
    gammab5tow = (mb5/(32.*pi))*((gLW35t**2 + gRW35t**2)*(1. + (mb5**2 - 2.*mtpole**2)/MW**2 + (mtpole**2 - 2.*MW**2)/mb5**2 + (mtpole**4)/(MW*mb5)**2) - 12.*gLW35t*gRW35t*mtpole/mb5)*sqrt(abs(phase(1.,(MW/mb5)**2,(mtpole/mb5)**2)))*unitstep(mb5 - MW - mtpole)

    gammab5toh = ((mb5/(64.*pi))*(((lamHb35/tan(beta))**2 + (lamHrb35/tan(beta))**2)*(1. + (mbpole**2 -125.**2)/mb5**2) + 4.*(-lamHb35/tan(beta))*(-lamHrb35/tan(beta))*mbpole/mb5 )*sqrt(abs(phase(1.,(125./mb5)**2,(mbpole/mb5)**2))))*unitstep(mb5 - 125. - mbpole)
    
    gammab5tot4w = (mb5/(32.*pi))*((gLW45t**2 + gRW45t**2)*(1. + (mb5**2 - 2.*mt4**2)/MW**2 + (mt4**2 - 2.*MW**2)/mb5**2 + (mt4**4)/(MW*mb5)**2) - 12.*gLW45t*gRW45t*mt4/mb5)*sqrt(abs(phase(1.,(MW/mb5)**2,(mt4/mb5)**2)))*unitstep(mb5 - MW - mt4)

    gammab5tot5w = (mb5/(32.*pi))*((gLW55t**2 + gRW55t**2)*(1. + (mb5**2 - 2.*mt5**2)/MW**2 + (mt5**2 - 2.*MW**2)/mb5**2 + (mt5**4)/(MW*mb5)**2) - 12.*gLW55t*gRW55t*mt5/mb5)*sqrt(abs(phase(1.,(MW/mb5)**2,(mt5/mb5)**2)))*unitstep(mb5 - MW - mt5)

    gammab5tob4z = (mb5/(32.*pi))*((gLZ54b**2 + gRZ54b**2)*(1. + (mb5**2 - 2.*mb4**2)/MZ**2 + (mb4**2 - 2.*MZ**2)/mb5**2 + (mb4**4)/(MZ*mb5)**2) - 12.*gLZ54b*gRZ54b*mb4/mb5)*sqrt(abs(phase(1.,(MZ/mb5)**2,(mb4/mb5)**2)))*unitstep(mb5 - MZ - mb4)

    gammab5tob4h = ((mb5/(64.*pi))*(((lamHb45/tan(beta))**2 + (lamHrb45/tan(beta))**2)*(1. + (mb4**2 -125.**2)/mb5**2) + 4.*(-lamHb45/tan(beta))*(-lamHrb45/tan(beta))*mb4/mb5 )*sqrt(abs(phase(1.,(125./mb5)**2,(mb4/mb5)**2))))*unitstep(mb5 - 125. - mb4)

    if gammab5toz + gammab5tow + gammab5toh + gammab5tot4w + gammab5tot5w + gammab5tob4z + gammab5tob4h == 0:
        gamb5all = 1
    else:
        gamb5all = gammab5toz + gammab5tow + gammab5toh + gammab5tot4w + gammab5tot5w + gammab5tob4z + gammab5tob4h
    #VLQ BR paper
    #if mb4 > MW + mt4:
        #continue
    #gamb4all = gammab4toz + gammab4tow + gammab4toh
    try:
        brb5z = gammab5toz/gamb5all
        brb5w = gammab5tow/gamb5all
        brb5h = gammab5toh/gamb5all
        brb5t4w = gammab5tot4w/gamb5all
        brb5t5w = gammab5tot5w/gamb5all
        brb5b4z = gammab5tob4z/gamb5all
        brb5b4h = gammab5tob4h/gamb5all
    except ZeroDivisionError:
        continue

    
    
    # BR(H)
    mb = 2.71
    #alphas = 0.109683
    #alphas = 0.1
    alphaEM = 1/128.9 #1/127.462
    Nf = 5
    mt = 173.21
    gamHtobb = ((3*GF)/(4*sqrt(2)*pi))*mHH*pow(mb*tan(beta),2)*(1 + 5.67*(alphas(mHH)/pi) + (35.94 - 1.36*Nf)*pow(alphas(mHH),2)/pow(pi,2) + (164.14 - 25.77*Nf + 0.26*pow(Nf,2))*pow(alphas(mHH),3)/pow(pi,3) + (pow(alphas(mHH),2)/pow(pi,2))*(1.57 - (2/3.0)*log(pow(mHH,2)/pow(mt,2)) + (1/9.0)*pow(log(pow(mb,2)/pow(mHH,2)),2)))
    paramtau = pow(mHH/(2.*mt),2)
    if paramtau <= 1.:
        fftau = pow(asin(sqrt(abs(paramtau))),2)
    elif paramtau > 1.:
        fftau = -(1./4.)*(cmath.log((1. + cmath.sqrt(1. - 1./paramtau))/(1. - cmath.sqrt(1. - 1./paramtau))) - 1j*pi)**2
    Ahalf = 2*(paramtau + (paramtau-1)*fftau)/pow(paramtau,2)
    
    vlqtau = pow(mHH/(2.*mt4),2)
    if vlqtau <= 1.:
        ffvlqtau = pow(asin(sqrt(abs(vlqtau))),2)
    elif vlqtau > 1.:
        ffvlqtau = -(1./4.)*(cmath.log((1. + cmath.sqrt(1. - 1./vlqtau))/(1. - cmath.sqrt(1. - 1./vlqtau))) - 1j*pi)**2
    Ahalfvlq = 2*(vlqtau + (vlqtau-1)*ffvlqtau)/pow(vlqtau,2)

    vlq5tau = pow(mHH/(2.*mt5),2)
    if vlq5tau <= 1.:
        ffvlq5tau = pow(asin(sqrt(abs(vlq5tau))),2)
    elif vlq5tau > 1.:
        ffvlq5tau = -(1./4.)*(cmath.log((1. + cmath.sqrt(1. - 1./vlq5tau))/(1. - cmath.sqrt(1. - 1./vlq5tau))) - 1j*pi)**2
    Ahalfvlq5 = 2*(vlq5tau + (vlq5tau-1)*ffvlq5tau)/pow(vlq5tau,2)
    
    vlqbtau = pow(mHH/(2.*mb4),2)
    if vlqbtau <= 1.:
        ffvlqbtau = pow(asin(sqrt(abs(vlqbtau))),2)
    elif vlqbtau > 1.:
        ffvlqbtau = -(1./4.)*(cmath.log((1. + cmath.sqrt(1. - 1./vlqbtau))/(1. - cmath.sqrt(1. - 1./vlqbtau))) - 1j*pi)**2
    Ahalfvlqb = 2*(vlqbtau + (vlqbtau-1)*ffvlqbtau)/pow(vlqbtau,2)

    vlqb5tau = pow(mHH/(2.*mb5),2)
    if vlqb5tau <= 1.:
        ffvlqb5tau = pow(asin(sqrt(abs(vlqb5tau))),2)
    elif vlqb5tau > 1.:
        ffvlqb5tau = -(1./4.)*(cmath.log((1. + cmath.sqrt(1. - 1./vlqb5tau))/(1. - cmath.sqrt(1. - 1./vlqb5tau))) - 1j*pi)**2
    Ahalfvlqb5 = 2*(vlqb5tau + (vlqb5tau-1)*ffvlqb5tau)/pow(vlqb5tau,2)


    paramtaub = pow(mHH/(2.*mbpole),2)
    fftaub = -(1./4.)*(cmath.log((1. + cmath.sqrt(1. - 1./paramtaub))/(1. - cmath.sqrt(1. - 1./paramtaub))) - 1j*pi)**2
    Ahalfb = 2*(paramtaub + (paramtaub-1)*fftaub)/pow(paramtaub,2)
    
    #gamHtogg = (GF*pow(alphas(mHH),2)*pow(mHH,3)/(36.*sqrt(2)*pow(pi,3)))*pow(abs(3*(-Ahalf/tan(beta) + Ahalfb*tan(beta) + Ahalfvlq*lamHt44/(mt/(v*sqrt(2))) + Ahalfvlq5*lamHt55/(mt/(v*sqrt(2))) + Ahalfvlqb*lamHb44/(mbpole/(v*sqrt(2))) + Ahalfvlqb5*lamHb55/(mbpole/(v*sqrt(2)))   )/4.),2)
    gamHtogg = (GF*pow(alphas(mHH),2)*pow(mHH,3)/(36.*sqrt(2)*pow(pi,3)))*pow(abs(3*(-Ahalf/tan(beta) + Ahalfb*tan(beta) + Ahalfvlq*lamHt44/(mt4/v) + Ahalfvlq5*lamHt55/(mt5/v) + Ahalfvlqb*lamHb44/(mb4/v) + Ahalfvlqb5*lamHb55/(mb5/v)   )/4.),2)


    ggHratio = ((abs(-Ahalf/tan(beta) + Ahalfb*tan(beta) + Ahalfvlq*lamHt44*(v/mt4) + Ahalfvlq5*lamHt55*(v/mt5) + Ahalfvlqb*lamHb44*(v/mb4) + Ahalfvlqb5*lamHb55*(v/mb5) ))**2)/(abs(Ahalf + Ahalfb))**2


    # Contribution by b4
    
    # ratio of the ggH productin cross section: remind y_b = -tan(beta)*y_b^{SM}
    # I campare new diagonal Yukawa couplings with top Yukawa
    #ggHratio = ((abs(-Ahalf/tan(beta) + Ahalfb*tan(beta) + (Ahalfvlq*lamHt44 + Ahalfvlq5*lamHt55 + Ahalfvlqb*lamHb44 + Ahalfvlqb5*lamHb55)/(mt/(v*sqrt(2))) ))**2)/(abs(Ahalf + Ahalfb))**2

    
    vlqsmtau = pow(125./(2.*mt4),2)
    if vlqsmtau <= 1.:
        ffvlqsmtau = pow(asin(sqrt(abs(vlqsmtau))),2)
    elif vlqsmtau > 1.:
        ffvlqsmtau = -(1./4.)*(cmath.log((1. + cmath.sqrt(1. - 1./vlqsmtau))/(1. - cmath.sqrt(1. - 1./vlqsmtau))) - 1j*pi)**2
    Ahalfsmvlq = 2*(vlqsmtau + (vlqsmtau-1)*ffvlqsmtau)/pow(vlqsmtau,2)
    
    vlq5smtau = pow(125./(2.*mt5),2)
    if vlq5smtau <= 1.:
        ffvlq5smtau = pow(asin(sqrt(abs(vlq5smtau))),2)
    elif vlq5smtau > 1.:
        ffvlq5smtau = -(1./4.)*(cmath.log((1. + cmath.sqrt(1. - 1./vlq5smtau))/(1. - cmath.sqrt(1. - 1./vlq5smtau))) - 1j*pi)**2
    Ahalfsmvlq5 = 2*(vlq5smtau + (vlq5smtau-1)*ffvlq5smtau)/pow(vlq5smtau,2)

    vlqsmtaub = pow(125./(2.*mb4),2)
    if vlqsmtaub <= 1.:
        ffvlqsmtaub = pow(asin(sqrt(abs(vlqsmtaub))),2)
    elif vlqsmtaub > 1.:
        ffvlqsmtaub = -(1./4.)*(cmath.log((1. + cmath.sqrt(1. - 1./vlqsmtaub))/(1. - cmath.sqrt(1. - 1./vlqsmtaub))) - 1j*pi)**2
    Ahalfsmvlqb = 2*(vlqsmtaub + (vlqsmtaub-1)*ffvlqsmtaub)/pow(vlqsmtaub,2)

    vlqsmtaub5 = pow(125./(2.*mb5),2)
    if vlqsmtaub5 <= 1.:
        ffvlqsmtaub5 = pow(asin(sqrt(abs(vlqsmtaub5))),2)
    elif vlqsmtaub5 > 1.:
        ffvlqsmtaub5 = -(1./4.)*(cmath.log((1. + cmath.sqrt(1. - 1./vlqsmtaub5))/(1. - cmath.sqrt(1. - 1./vlqsmtaub5))) - 1j*pi)**2
    Ahalfsmvlqb5 = 2*(vlqsmtaub5 + (vlqsmtaub5-1)*ffvlqsmtaub5)/pow(vlqsmtaub5,2)


    ggSMratio = ((abs(Ahalf + Ahalfb + Ahalfsmvlq*(-lamHt44*tan(beta))*(v/mt4) + Ahalfsmvlq5*(-lamHt55*tan(beta))*(v/mt5) + Ahalfsmvlqb*(lamHb44/tan(beta))*(v/mb4) + Ahalfsmvlqb5*(lamHb55/tan(beta))*(v/mb5)  ))**2)/(abs(Ahalf + Ahalfb))**2    
    #ggSMratio = ((abs(Ahalf + Ahalfb + (Ahalfsmvlq*(-lamHt44*tan(beta)) + Ahalfsmvlq5*(-lamHt55*tan(beta)) + Ahalfsmvlqb*(lamHb44/tan(beta)) + Ahalfsmvlqb5*(lamHb55/tan(beta)))/(mt/(v*sqrt(2)))  ))**2)/(abs(Ahalf + Ahalfb))**2
    #test
    #ggSMratio = ((abs(Ahalf + Ahalfb + Ahalfsmvlq*(-lamHt44*tan(beta)/(mt/(v*sqrt(2)))) + Ahalfsmvlq5*(-lamHt55*tan(beta)/(mt/(v*sqrt(2)))) + Ahalfsmvlqb*(lamHb44/tan(beta)/(mt/(v*sqrt(2)))) + Ahalfsmvlqb5*(lamHb55/tan(beta)/(mt/(v*sqrt(2))))  ))**2)/(abs(Ahalf + Ahalfb))**2
    #ggSMratio = 1
    
    
    if mHH > mt4 + mtpole:
        gamHtot4 = 3*2.0*(mHH/(32.*pi))*((lamHt34**2 + lamHrt34**2)*(1. - (mt4**2 + mtpole**2)/mHH**2) + 4.*lamHt34*lamHrt34*mt4*mtpole/mHH**2 )*sqrt(abs(phase(1.,(mt4/mHH)**2,(mtpole/mHH)**2))) # 3:color factor
    else:
        gamHtot4 = 0.

    if mHH > mt5 + mtpole:
        gamHtot5 = 3*2.0*(mHH/(32.*pi))*((lamHt35**2 + lamHrt35**2)*(1. - (mt5**2 + mtpole**2)/mHH**2) + 4.*lamHt35*lamHrt35*mt5*mtpole/mHH**2 )*sqrt(abs(phase(1.,(mt5/mHH)**2,(mtpole/mHH)**2))) # 3:color factor
    else:
        gamHtot5 = 0.

    if mHH > mb4 + mbpole:
        gamHtob4 = 3*2.0*(mHH/(32.*pi))*((lamHb34**2 + lamHrb34**2)*(1. - (mb4**2 + mbpole**2)/mHH**2) + 4.*lamHb34*lamHrb34*mb4*mbpole/mHH**2 )*sqrt(abs(phase(1.,(mb4/mHH)**2,(mbpole/mHH)**2))) # 3:color factor
    else:
        gamHtob4 = 0.

    if mHH > mb5 + mbpole:
        gamHtob5 = 3*2.0*(mHH/(32.*pi))*((lamHb35**2 + lamHrb35**2)*(1. - (mb5**2 + mbpole**2)/mHH**2) + 4.*lamHb35*lamHrb35*mb5*mbpole/mHH**2 )*sqrt(abs(phase(1.,(mb5/mHH)**2,(mbpole/mHH)**2))) # 3:color factor
    else:
        gamHtob5 = 0.
        
    if mHH > mt4 + mt4:
        gamHt4t4 = 3*(mHH/(16.*pi))*(lamHt44**2)*(1. - 4.*(mt4**2)/mHH**2)*sqrt(abs(phase(1.,(mt4/mHH)**2,(mt4/mHH)**2))) # 3:color factor
    else:
        gamHt4t4 = 0.

    if mHH > mt4 + mt5:
        gamHt4t5 = 3*2.0*(mHH/(32.*pi))*((lamHt45**2 + lamHrt45**2)*(1. - (mt4**2 + mt5**2)/mHH**2) + 4.*lamHt45*lamHrt45*mt4*mt5/mHH**2 )*sqrt(abs(phase(1.,(mt4/mHH)**2,(mt5/mHH)**2))) # 3:color factor
    else:
        gamHt4t5 = 0.
        
    if mHH > mt5 + mt5:
        gamHt5t5 = 3*(mHH/(16.*pi))*(lamHt55**2)*(1. - 4.*(mt5**2)/mHH**2)*sqrt(abs(phase(1.,(mt5/mHH)**2,(mt5/mHH)**2))) # 3:color factor
    else:
        gamHt5t5 = 0.

    if mHH > mb4 + mb4:
        gamHb4b4 = 3*(mHH/(16.*pi))*(lamHb44**2)*(1. - 4.*(mb4**2)/mHH**2)*sqrt(abs(phase(1.,(mb4/mHH)**2,(mb4/mHH)**2))) # 3:color factor
    else:
        gamHb4b4 = 0.

    if mHH > mb4 + mb5:
        gamHb4b5 = 3*2.0*(mHH/(32.*pi))*((lamHb45**2 + lamHrb45**2)*(1. - (mb4**2 + mb5**2)/mHH**2) + 4.*lamHb45*lamHrb45*mb4*mb5/mHH**2 )*sqrt(abs(phase(1.,(mb4/mHH)**2,(mb5/mHH)**2))) # 3:color factor
    else:
        gamHb4b5 = 0.

        
    if mHH > mb5 + mb5:
        gamHb5b5 = 3*(mHH/(16.*pi))*(lamHb55**2)*(1. - 4.*(mb5**2)/mHH**2)*sqrt(abs(phase(1.,(mb5/mHH)**2,(mb5/mHH)**2))) # 3:color factor
    else:
        gamHb5b5 = 0.
        
        
    if mHH < 2.0*125.0:
        gamHtohh = 0
    else:
        gamHtohh = (GF/(16.0*pi*sqrt(2)))*((MZ**4)/mHH)*(sqrt(1 - 4.0*(125.0/mHH)**2))*((3.0*sin(2.0*beta)*cos(2.0*beta))**2)
        
    mc = 0.6
    gamHtocc = ((3*GF)/(4*sqrt(2)*pi))*mHH*pow(mc/tan(beta),2)*(1 + 5.67*(alphas(mHH)/pi) + (35.94 - 1.36*Nf)*pow(alphas(mHH),2)/pow(pi,2) + (164.14 - 25.77*Nf + 0.26*pow(Nf,2))*pow(alphas(mHH),3)/pow(pi,3) + (pow(alphas(mHH),2)/pow(pi,2))*(1.57 - (2/3.0)*log(pow(mHH,2)/pow(mt,2)) + (1/9.0)*pow(log(pow(mc,2)/pow(mHH,2)),2)))    
    #gamHdiph = (GF*pow(alphaEM,2)*pow(mHH,3)/(128.*sqrt(2)*pow(pi,3)))*pow(abs(3.0*(-((2.0/3.0)**2)*Ahalf/tan(beta) + ((1.0/3.0)**2)*Ahalfb*tan(beta) + (lamHt44*v/mt4)*Ahalfvlq + (lamHt55*v/mt5)*Ahalfvlq5 + (lamHb44*v/mb4)*Ahalfvlqb + (lamHb55*v/mb5)*Ahalfvlqb5 ) ),2)
    gamHdiph = (GF*pow(alphaEM,2)*pow(mHH,3)/(128.*sqrt(2)*pow(pi,3)))*pow(abs(3.0*(-((2.0/3.0)**2)*Ahalf/tan(beta) + ((1.0/3.0)**2)*Ahalfb*tan(beta) + ((2.0/3.0)**2)*(lamHt44*v/mt4)*Ahalfvlq + ((2.0/3.0)**2)*(lamHt55*v/mt5)*Ahalfvlq5 + ((1.0/3.0)**2)*(lamHb44*v/mb4)*Ahalfvlqb + ((1.0/3.0)**2)*(lamHb55*v/mb5)*Ahalfvlqb5 ) ),2)
    Aone = -8.32
    lamSMht44 = -lamHt44*tan(beta)    
    lamSMht55 = -lamHt55*tan(beta)    
    lamSMhb44 = lamHb44/tan(beta)    
    lamSMhb55 = lamHb55/tan(beta)    
    Atopsm = 1.84
    tausmt4 = pow(125.0/(2.*mt4),2)
    ffsmtaut4 = pow(asin(sqrt(abs(tausmt4))),2)
    ASMt4 = 2.0*(tausmt4 + (tausmt4-1)*ffsmtaut4)/pow(tausmt4,2)

    tausmt5 = pow(125.0/(2.*mt5),2)
    ffsmtaut5 = pow(asin(sqrt(abs(tausmt5))),2)
    ASMt5 = 2.0*(tausmt5 + (tausmt5-1)*ffsmtaut5)/pow(tausmt5,2)
    
    tausmb4 = pow(125.0/(2.*mb4),2)
    ffsmtaub4 = pow(asin(sqrt(abs(tausmb4))),2)
    ASMb4 = 2.0*(tausmb4 + (tausmb4-1)*ffsmtaub4)/pow(tausmb4,2)

    tausmb5 = pow(125.0/(2.*mb5),2)
    ffsmtaub5 = pow(asin(sqrt(abs(tausmb5))),2)
    ASMb5 = 2.0*(tausmb5 + (tausmb5-1)*ffsmtaub5)/pow(tausmb5,2)

    #gamSMt4dp = (GF*pow(alphaEM,2)*pow(125.0,3)/(128.*sqrt(2)*pow(pi,3)))*pow(abs(Aone + Atopsm + (lamSMht44*v/mt4)*ASMt4 ),2)
    gamSMvlqdp = (GF*pow(alphaEM,2)*pow(125.0,3)/(128.*sqrt(2)*pow(pi,3)))*pow(abs(Aone + Atopsm + (4./3.)*(lamSMht44*v/mt4)*ASMt4 + (4./3.)*(lamSMht55*v/mt5)*ASMt5 + (1./3.)*(lamSMhb44*v/mb4)*ASMb4 + (1./3.)*(lamSMhb55*v/mb5)*ASMb5  ),2)

    
    gamSModp = (GF*pow(alphaEM,2)*pow(125.0,3)/(128.*sqrt(2)*pow(pi,3)))*pow(abs(Aone + Atopsm),2)
    gamSMotot = 0.00407
    brsmdpnew = gamSMvlqdp/(gamSMotot - gamSModp + gamSMvlqdp)
    RgamgamSM = gamSMvlqdp/gamSModp
    
    mmtau = 1.777
    if 1.0 - 4.0*pow(mmtau/mHH,2) < 0:
        gamHtotata = 0
    else:
        gamHtotata = (GF*mHH*(mmtau**2)*((tan(beta))**2)/(4.*sqrt(2)*pi))*pow(sqrt(1.0 - 4.0*pow(mmtau/mHH,2)),3)
        
    #alphast = 0.096
    betat = sqrt(1. - 4.*(mt/mHH)**2)
    xbetat = ( 1. - betat)/(1. +  betat)
    #Spence function Li2(x) = spence(1-x)
    Li2 = spence(1.-betat)
    mLi2 = spence(1.+betat)
    Afunction = (1. + betat**2)*(4.*Li2 + 2.*mLi2 + 3.*log(xbetat)*log(2./(1.+betat)) + 2.*log(xbetat)*log(betat)) - 3.*betat*log((4.*betat**(4./3.))/(1. - betat**2))
    deltaHt = (1./betat)*Afunction + (1./(16.*betat**3))*(3. + 34.*betat**2 - 13.*betat**4)*log((1. + betat)/(1. - betat)) + (3./(8.*betat**2))*(7.*betat**2 - 1.)
    if mHH > 2.0*mtpole:
        gamHtott = ((3*GF)/(4*sqrt(2)*pi))*mHH*((mt/tan(beta))**2)*(betat**3)*(1. + (4./3.)*(alphas(mHH)/pi)*deltaHt)
    else:
        gamHtott = 0.


        
    gamHtot = gamHtot4 + gamHtogg + gamHtobb + gamHtohh + gamHtocc + gamHdiph + gamHtotata + gamHtott + gamHtot5 + gamHt4t4 + gamHt4t5 + gamHt5t5 + gamHtob4 + gamHtob5 + gamHb4b4 + gamHb4b5 + gamHb5b5
    brHtot4 = gamHtot4/gamHtot
    brHtobb = gamHtobb/gamHtot
    brHtohh = gamHtohh/gamHtot
    brHtogg = gamHtogg/gamHtot
    brHtocc = gamHtocc/gamHtot
    brHtodi = gamHdiph/gamHtot
    brHtott = gamHtott/gamHtot
    brHtotata = gamHtotata/gamHtot
    brHtot5 = gamHtot5/gamHtot
    brHt4t4 = gamHt4t4/gamHtot
    brHt4t5 = gamHt4t5/gamHtot
    brHt5t5 = gamHt5t5/gamHtot
    brHtob4 = gamHtob4/gamHtot
    brHtob5 = gamHtob5/gamHtot
    brHb4b4 = gamHb4b4/gamHtot
    brHb4b5 = gamHb4b5/gamHtot
    brHb5b5 = gamHb5b5/gamHtot
    #brHto2e4 = gamHto2e4/gamHtot
    # GeV^-2 = 3.89 * 10^-4 b = 3.89 * 10^8 pb
    brour = brHtot4*brt4h

    def f1(x,y):
        return y - x + 3.*log(y/x)
    def f2(x,y):
        return -x -y + (2.*x*y/(y-x))*log(y/x)
    
    smZbbL = (g/cos(thetaw))*((-1./2. + (1./3.)*(sin(thetaw))**2))
    smZbbR = (g/cos(thetaw))*(1./3.)*(sin(thetaw))**2
    clt = gLZ44t/(g/(2.*cos(thetaw)))
    slt = gLW43t/(g/sqrt(2))
    exx = (mtpole/MW)**2
    exxp = (mt4/MW)**2

    cltt5 = gLZ55t/(g/(2.*cos(thetaw)))
    sltt5 = gLW53t/(g/sqrt(2))
    #cltb4 = gLZ44b/(g/(2.*cos(thetaw)))
    #sltb4 = gLW43b/(g/sqrt(2))
    #cltb5 = gLZ55b/(g/(2.*cos(thetaw)))
    #sltb5 = gLW53b/(g/sqrt(2))
    
    exxp5 = (mt5/MW)**2
    #exxpb = (mb4/MW)**2
    #exxpb5 = (mb5/MW)**2

    #Naively converting the equation in 1703.06134
    #deltaXbbL = 0.006 \pm 0.002, deltaXbbR = 0.034 \pm 0.016 following 1703.06134
    deltaXbbL = (gLZ33b - smZbbL)/(g/(2.*cos(thetaw))) + ((g**2)/(32.*pi**2))*((slt**2)*(f1(exx,exxp) + (clt**2)*f2(exx,exxp)) + (sltt5**2)*(f1(exx,exxp5) + (cltt5**2)*f2(exx,exxp5)) )
    deltaXbbR = (gRZ33b - smZbbR)/(g/(2.*cos(thetaw)))

    
    # LHC results
    if doubfr >= 0.5:
        lhct4 = 1050
    else:
        lhct4 = 870

    if doubt5fr >= 0.5:
        lhct5 = 1050
    else:
        lhct5 = 870

    if doubb4fr >= 0.5:
        lhcb4 = 1050
    else:
        lhcb4 = 870

    if doubb5fr >= 0.5:
        lhcb5 = 1050
    else:
        lhcb5 = 870
        
        
        
    # EW precision conditions : oblique corrections
    #old h->\gamma \gamma  Table 10 & Fig. 24 of 1407.0558 (CMS)
    #which is ggSMratio*RgamgamSM < 1.12 + 2*0.37 and ggSMratio*RgamgamSM > 1.12 - 2*0.32
    #new h->\gamma \gamma ATLAS-CONF-2017-045
    #if delS < -0.03 + 2*0.1 and delS > -0.03 - 2*0.1 and delT < 0.01 + 2*0.12 and delT > 0.01 - 2*0.12 and delU < 0.05 + 2*0.1 and delU > 0.05 - 2*0.1 and ggSMratio*RgamgamSM < 0.99 + 2*0.14 and ggSMratio*RgamgamSM > 0.99 - 2*0.14 and deltaXbbL < -0.0002 + 2*0.0012 and deltaXbbL > -0.0002 - 2*0.0012 and deltaXbbR < 0.008 + 2*0.006 and deltaXbbR > 0.008 - 2*0.006 and mt4 > intmt4upper(brt4w, brt4h) and mb4 > intmb4upper(brb4w, brb4h):
    # test
    if delS < -0.03 + 2*0.1 and delS > -0.03 - 2*0.1 and delT < 0.01 + 2*0.12 and delT > 0.01 - 2*0.12 and delU < 0.05 + 2*0.1 and delU > 0.05 - 2*0.1 and ggSMratio*RgamgamSM < 0.99 + 2*0.14 and ggSMratio*RgamgamSM > 0.99 - 2*0.14 and deltaXbbL < -0.0002 + 2*0.0012 and deltaXbbL > -0.0002 - 2*0.0012 and deltaXbbR < 0.008 + 2*0.006 and deltaXbbR > 0.008 - 2*0.006 and ggSMratio*44.175*0.0267 < 1.13 + 2*0.13 and ggSMratio*44.175*0.0267 > 1.13 - 2*0.13:# and mt4 > intmt4upper(brt4w, brt4h) and mb4 > intmb4upper(brb4w, brb4h):
    #if 1 > 0:#delS < -0.03 + 2*0.1 and delS > -0.03 - 2*0.1 and delT < 0.01 + 2*0.12 and delT > 0.01 - 2*0.12 and delU < 0.05 + 2*0.1 and delU > 0.05 - 2*0.1:# and
    #if ggSMratio*RgamgamSM < 0.99 + 2*0.14 and ggSMratio*RgamgamSM > 0.99 - 2*0.14:# and
    #if deltaXbbL < -0.0002 + 2*0.0012 and deltaXbbL > -0.0002 - 2*0.0012 and deltaXbbR < 0.008 + 2*0.006 and deltaXbbR > 0.008 - 2*0.006:# and
    #if mt4 > mt4limit(brt4w, brt4h)[0] and mb4 > mb4limit(brb4w, brb4h)[0]:# 
        #print mb4limit(brb4w, brb4h)[0]
        print 'second passed'
        print gamt4all, mt4, mHH, tan(beta)
        mylist = [mt4,tan(beta),lamHt34,mHH,doubfr,singfr,delS,delT,delU,lamHrt34,lamHt44,lamHrt44,gLW44t,gLW43t,gLW45t,gLW33t,gLW34t,gLW35t,gLW55t,gLW53t,gLW54t,gRW44t,gRW43t,gRW45t,gRW33t,gRW34t,gRW35t,gRW55t,gRW54t,gRW53t,gLZ44t,gLZ43t,gLZ45t,gLZ33t,gLZ34t,gLZ35t,gLZ55t,gLZ53t,gLZ54t,gRZ44t,gRZ43t,gRZ45t,gRZ33t,gRZ34t,gRZ35t,gRZ55t,gRZ53t,gRZ54t,kaT,kaQ,ka,kab,mt5,MTT,MQQ,brHtobb,brHtohh,brHtot4,brHtogg,brHtocc,brHtodi,brHtott,gamHtot,gamHtotata,gamHtobb,gamHtot4,gamHtogg,gamHtohh,gamHtocc,gamHdiph,gamHtott,brt4w,brt4z,brt4h,brour,ggHratio,ggSMratio,lamHt55,lamHrt55,brHtotata,brt4b4w,brt4b5w,brb4w,brb4z,brb4h,brb4t4w,brb4t5w,brHtot5,brHt4t4,brHt4t5,brHt5t5,brHtob4,brHtob5,brHb4b4,brHb4b5,brHb5b5,lamHt35,lamHrt35,lamHt45,lamHrt45,lamHb34,lamHrb34,lamHb35,lamHrb35,lamHb45,lamHrb45,lamHb44,lamHrb44,lamHb55,lamHrb55,gLZ33b,gLZ44b,gLZ55b,gLZ34b,gLZ43b,gLZ35b,gLZ53b,gLZ45b,gLZ54b,gRZ33b,gRZ44b,gRZ55b,gRZ34b,gRZ43b,gRZ35b,gRZ53b,gRZ45b,gRZ54b,gLW42t,gLW52t,gLW24b,gLW25b,gLW41t,gLW51t,gLW14b,gLW15b,mb4,mb5,doubt5fr,singt5fr,doubb4fr,singb4fr,doubb5fr,singb5fr,gammat4toh,gammat4tow,brt5z,brt5w,brt5h,brt5b4w,brt5b5w,brt5t4z,brt5t4h,brb5z,brb5w,brb5h,brb5t4w,brb5t5w,brb5b4z,brb5b4h,brb4HH,lamHpmb43,lamHpmr43,lamHpm43,brt4HH,"\n",]#lamHtest4,"\n"]         
        fevout=open("t1ewptall-s8.dat", "a+")
        #fevout=open("t1ewptall-test-s8.dat", "a+")
        #fevout=open("t1ewptall-fix-test-s1.dat", "a+")
        #fevout=open("t1ewptall-fix-s1.dat", "a+")
        #fevout=open("t1ewptall-fix-s2-noconstraint.dat", "a+")
        #fevout=open("t1ewptall-extreme-s4.dat", "a+")
        #fevout=open("t1ewptall-10TeV-s4.dat", "a+")
        for item in mylist:
            fevout.write(str(item) + '\t')   # write the list by inserting tab between the elements
        fevout.close()                       # we change the numbers in mylist into 'str' since only     

#exe("./t1ewptall-HWWadd.py")






