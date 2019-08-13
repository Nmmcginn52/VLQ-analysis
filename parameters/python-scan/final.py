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

import array as arr

#exe("rm t1ewptall-test-em.dat")
#exe("rm t1ewptall-test-mm.dat")
exe("rm t1ewptall-em.dat")
#exe("rm t1ewptall-mm.dat")
#exe("rm t1ewptall-em-noconstraint.dat")


#exe("rm t1ewptall-0.5Yukawa-em.dat")
#exe("rm t1ewptall-0.5Yukawa-mm.dat")


listlep = ['em','mm','ee']





#diphotonCMS = loadtxt('cms-Hgg.txt')
diphotonCMS = loadtxt('cms13combine8-Hdiph.txt') #CMS 1609.02507, 13TeV+8TeV
dpmasses = numpy.array(diphotonCMS[:,0])
dpbounds0 = numpy.array(diphotonCMS[:,1])
dpbounds = [x*0.001 for x in dpbounds0] # bounds in [pb] unit

#13TeV
#bbF = loadtxt('bbFresults.dat')
#hhmass = numpy.array(bbF[:,0])
#sigbbF = numpy.array(bbF[:,1])

ditauATLAS = loadtxt('atlasHtautau.dat') #ATLAS 1709.07242, 13TeV, 36.1fb^{-1}
ditaumasses = numpy.array(ditauATLAS[:,0])
ditaubounds = numpy.array(ditauATLAS[:,1]) # in [pb] unit



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



def cmsdpbounds(mass):
    order = 2
    mybound = InterpolatedUnivariateSpline(dpmasses, dpbounds, k=order)
    return mybound(mass)

def atlastau(mass):
    order = 1
    taubound = InterpolatedUnivariateSpline(ditaumasses, ditaubounds, k=order)
    return taubound(mass)


def sigmaHSM8TeV(mass):
    x = numpy.array([130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350,360,370,380,390,400,420,440,450,460,480,500,520,540,550,560,580,600,620,640,650,660,680,700,720,740,750,760,780,800,820,840,850,860,880,900,920,940,950,960,980,1000])
    y = numpy.array([17.85,15.42,13.55,11.96,10.17,8.980,7.858,7.081,6.500,6.003,5.567,5.159,4.783,4.461,4.184,3.950,3.755,3.594,3.472,3.383,3.341,3.359,3.401,3.385,3.332,3.231,3.089,2.921,2.550,2.178,2.002,1.837,1.538,1.283,1.069,0.8913,0.8144,0.7442,0.6228,0.5230,0.4403,0.3719,0.3424,0.3153,0.2682,0.2290,0.1964,0.1689,0.1568,0.1457,0.1262,0.1097,0.0957,0.0837,0.0784,0.0735,0.0647,0.0571,0.0506,0.0450,0.0424,0.0400,0.0357,0.0320])
    order = 2
    mysig = InterpolatedUnivariateSpline(x, y, k=order)
    return mysig(mass)


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




#ATLAS 1707.04147 hepdata
def Hdiph(mass):
    xx = []
    for ii in range(0, 126):
        lxx = [200 + ii*20]
        xx.extend(lxx)
    # fb unit
    yy = [11.407, 5.64994, 5.19967, 4.68341, 2.67697, 5.04916, 1.30593, 3.23652, 1.9595, 2.24354, 1.60107, 1.50023, 1.23842, 1.02943, 1.68588, 0.99073, 0.836514, 0.868932, 0.984042, 0.796768, 1.15545, 1.00058, 0.499915, 0.515815, 0.679789, 0.906402, 1.01382, 0.513754, 0.309358, 0.856392, 0.360337, 0.550993, 0.662905, 0.417831, 0.346713, 0.21898, 0.28413, 0.215852, 0.169689, 0.318455, 0.264796, 0.255809, 0.353345, 0.245548, 0.195013, 0.23357, 0.256903, 0.219736, 0.175468, 0.162258, 0.163396, 0.210818, 0.294156, 0.195389, 0.133453, 0.169198, 0.208083, 0.236701, 0.214321, 0.162797, 0.121828, 0.19599, 0.181979, 0.168458, 0.136914, 0.0965764, 0.091175, 0.0977804, 0.141909, 0.26073, 0.31084, 0.232987, 0.114289, 0.0913504, 0.096068, 0.130986, 0.120802, 0.093682, 0.0992048, 0.133297, 0.124113, 0.0946934, 0.0973914, 0.131687, 0.131892, 0.0986928, 0.0857793, 0.0828517, 0.0819784, 0.0815417, 0.0812246, 0.0809407, 0.080857, 0.081828, 0.0921072, 0.129343, 0.13807, 0.124641, 0.0955243, 0.0884228, 0.109399, 0.137515, 0.146656, 0.175101, 0.165691, 0.151378, 0.167159, 0.167911, 0.148868, 0.140492, 0.126726, 0.0969376, 0.0833605, 0.0786849, 0.0774643, 0.0789789, 0.0921697, 0.129467, 0.137485, 0.138187, 0.132759, 0.114831, 0.0914881, 0.0809344, 0.0765409, 0.0750105]
    order = 1
    myHdiph = InterpolatedUnivariateSpline(xx, yy, k=order)
    return myHdiph(mass)



#fin = open("t1ewptall-test.dat")
fin = open("t1ewptall.dat")
#fin = open("t1ewptall-noconstraint.dat")
#fin = open("t1ewptall-mytest-s7.dat")
listscan = map(str,fin.readlines())
fin.close()


for i in range(0,len(listscan)-1):
    lin = listscan[i]
    param=map(str, lin.split())
    mt4 = float(param[0])
    beta = atan(float(param[1]))
    lamHt34 = float(param[2])
    mHH = float(param[3])
    doubfr = float(param[4])
    singfr =  float(param[5])  
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
    brHtobb = float(param[55])
    brHtohh = float(param[56])
    brHtot4 = float(param[57])
    brHtogg = float(param[58])
    brHtocc = float(param[59])
    brHtodi = float(param[60])
    brHtott = float(param[61])
    gamHtot = float(param[62])
    gamHtotata = float(param[63])
    gamHtobb = float(param[64])
    gamHtot4 = float(param[65])
    gamHtogg = float(param[66])
    gamHtohh = float(param[67])
    gamHtocc = float(param[68])
    gamHdiph = float(param[69])
    gamHtott = float(param[70])
    brt4w = float(param[71])
    brt4z = float(param[72])
    brt4h = float(param[73])
    brour = float(param[74])
    ggHratio = float(param[75])
    ggSMratio = float(param[76])
    lamHt55 = float(param[77])
    lamHrt55 = float(param[78])
    totalratio = ggHratio + ((tan(beta))**2)*sigmabbF13(mHH)/sigmaHSM13TeV(mHH)
    brHtotata = float(param[79])
    brt4b4w = float(param[80])
    brt4b5w = float(param[81])
    brb4w = float(param[82])
    brb4z = float(param[83])
    brb4h = float(param[84])
    brb4t4w = float(param[85])
    brb4t5w = float(param[86])
    brHtot5 = float(param[87])
    brHt4t4 = float(param[88])
    brHt4t5 = float(param[89])
    brHt5t5 = float(param[90])
    brHtob4 = float(param[91])
    brHtob5 = float(param[92])
    brHb4b4 = float(param[93])
    brHb4b5 = float(param[94])
    brHb5b5 = float(param[95])
    lamHt35 = float(param[96])
    lamHrt35 = float(param[97])
    lamHt45 = float(param[98])
    lamHrt45 = float(param[99])
    lamHb34 = float(param[100])
    lamHrb34 = float(param[101])
    lamHb35 = float(param[102])
    lamHrb35 = float(param[103])
    lamHb45 = float(param[104])
    lamHrb45 = float(param[105])
    lamHb44 = float(param[106])
    lamHrb44 = float(param[107])
    lamHb55 = float(param[108])
    lamHrb55 = float(param[109])
    gLZ33b = float(param[110])
    gLZ44b = float(param[111])
    gLZ55b = float(param[112])
    gLZ34b = float(param[113])
    gLZ43b = float(param[114])
    gLZ35b = float(param[115])
    gLZ53b = float(param[116])
    gLZ45b = float(param[117])
    gLZ54b = float(param[118])
    gRZ33b = float(param[119])
    gRZ44b = float(param[120])
    gRZ55b = float(param[121])
    gRZ34b = float(param[122])
    gRZ43b = float(param[123])
    gRZ35b = float(param[124])
    gRZ53b = float(param[125])
    gRZ45b = float(param[126])
    gRZ54b = float(param[127])
    gLW42t = float(param[128])
    gLW52t = float(param[129])
    gLW24b = float(param[130])
    gLW25b = float(param[131])
    gLW41t = float(param[132])
    gLW51t = float(param[133])
    gLW14b = float(param[134])
    gLW15b = float(param[135])
    mb4 = float(param[136])
    #mb4 = 2000.
    mb5 = float(param[137])
    doubt5fr = float(param[138])
    singt5fr = float(param[139])
    #doubb4fr = float(param[140])
    doubb4fr = float(param[-26])
    singb4fr = float(param[141])
    doubb5fr = float(param[142])
    singb5fr = float(param[143])
    gammat4toh = float(param[144])
    gammat4tow = float(param[145])
    #gammat4test = float(param[-1])
    brt5z = float(param[146])
    brt5w = float(param[147])
    brt5h = float(param[148])
    lamHtest = float(param[-5])
    lamHtest1 = float(param[-4])
    lamHtest2 = float(param[-3])
    lamHtest3 = float(param[-2])
    lamHtest4 = float(param[-1])
    
    xst4bj = 1000*(gLW43t**2 + gRW43t**2)*exp(17.104388867707353 - 3.904938026897961*mt4**(1./4.) + 0.3346814438843097*sqrt(mt4) - 0.0027757248618906615*mt4)
    xst4tj = 1000*(gLZ43t**2 + gRZ43t**2)*exp(11.139645954916125 - 2.7075399784571768*mt4**(1./4.) + 0.23190744108226*sqrt(mt4) - 0.0026329113099221374*mt4)

    if brt4w == 0:
        brt4wn = 0.00000000000000000001
    else:
        brt4wn = brt4w
    if brt4h == 0:
        brt4hn = 0.00000000000000000001
    else:
        brt4hn = brt4h
    if brt4z == 0:
        brt4zn = 0.00000000000000000001
    else:
        brt4zn = brt4z

#Direct search at CMS    B2G-17-015
    def cmszpt4t(mHH, mt4):
        if mHH < 1500 + 15 and mHH > 1500 - 15 and mt4 < 700 + 7 and mt4 > 700 - 7:
            #return min(0.73/brt4wn, 4./brt4hn, 3.1/brt4zn) #1703.06352
            #return arr.array('f', [0.73, 4., 3.1]) #1703.06352
            return arr.array('f', [1.218, 0.165, 0.458]) #B2G-17-015
        if mHH < 1500 + 15 and mHH > 1500 - 15 and mt4 < 900 + 9 and mt4 > 900 - 9:
            #return min(1.5/brt4wn, 3.2/brt4hn, 2.8/brt4zn) #1703.06352
            #return arr.array('f', [1.5, 3.2, 2.8]) #1703.06352
            return arr.array('f', [1.014, 0.130, 0.415]) #B2G-17-015
        if mHH < 1500 + 15 and mHH > 1500 - 15 and mt4 < 1200 + 12 and mt4 > 1200 - 12:
            #return min(8.6/brt4wn, 9.4/brt4hn, 3.4/brt4zn) #1703.06352
            #return arr.array('f', [8.6, 9.4, 3.4]) #1703.06352
            return arr.array('f', [1.035, 0.131, 0.416]) #B2G-17-015
        if mHH < 2000 + 20 and mHH > 2000 - 20 and mt4 < 900 + 9 and mt4 > 900 - 9:
            #return min(0.19/brt4wn, 0.53/brt4hn, 0.37/brt4zn) #1703.06352
            #return arr.array('f', [0.19, 0.53, 0.37]) #1703.06352
            return arr.array('f', [0.418, 0.057, 0.141]) #B2G-17-015
        if mHH < 2000 + 20 and mHH > 2000 - 20 and mt4 < 1200 + 12 and mt4 > 1200 - 12:
            #return min(0.27/brt4wn, 0.53/brt4hn, 0.30/brt4zn) #1703.06352
            #return arr.array('f', [0.27, 0.53, 0.30]) #1703.06352
            return arr.array('f', [0.410, 0.059, 0.145]) #B2G-17-015
        if mHH < 2000 + 20 and mHH > 2000 - 20 and mt4 < 1500 + 15 and mt4 > 1500 - 15:
            #return min(0.96/brt4wn, 0.60/brt4hn, 0.32/brt4zn) #1703.06352
            #return arr.array('f', [0.96, 0.60, 0.32]) #1703.06352
            return arr.array('f', [0.404, 0.075, 0.174]) #B2G-17-015
        if mHH < 2500 + 25 and mHH > 2500 - 25 and mt4 < 1200 + 12 and mt4 > 1200 - 12:
            #return min(0.29/brt4wn, 0.24/brt4hn, 0.16/brt4zn) #1703.06352
            #return arr.array('f', [0.29, 0.24, 0.16]) #1703.06352
            return arr.array('f', [0.264, 0.052, 0.095]) #B2G-17-015
        if mHH < 2500 + 25 and mHH > 2500 - 25 and mt4 < 1500 + 15 and mt4 > 1500 - 15:
            #return min(0.30/brt4wn, 0.23/brt4hn, 0.13/brt4zn) #1703.06352
            #return arr.array('f', [0.30, 0.23, 0.13]) #1703.06352
            return arr.array('f', [0.245, 0.051, 0.081]) #B2G-17-015
        else:
            #return 10000000
            return arr.array('f', [10000000, 10000000, 10000000])

    
    
    for k in range(0,1):
        #previous = loadtxt('heavye4musignpresult-%s0.05.dat' %listlep[k])
        previous = loadtxt('new-heavye4musignpresult-%s0.05.dat' %listlep[k])
        masses = previous[:,0:2]
        accNP = numpy.array(previous[:,2])
        limitsignp = numpy.array(previous[:,11])
        #limitCMS = limitfid/accNP
        limitCMS = limitsignp
        varmHH = numpy.array(masses[:,0])
        varmnv = numpy.array(masses[:,1])
        #def funclim(mH, mn4):
            #mylim = interpolate.interp2d(varmHH.flatten(), varmnv.flatten(), limitCMS.flatten(), kind='cubic')
            #return 0.001*mylim(mH, mn4)[0]    # [pb] unit
        # H -> tau tau & H -> gamma gamma & Z' -> t4 t by CMS
        if (mHH > 504. and mHH <= 2250. and totalratio*sigmaHSM13TeV(mHH)*brHtodi < cmsdpbounds(mHH) and totalratio*sigmaHSM13TeV(mHH)*brHtodi < 0.001*Hdiph(mHH) and totalratio*sigmaHSM13TeV(mHH)*brHtotata < atlastau(mHH) and totalratio*sigmaHSM13TeV(mHH)*brHtot4*brt4w < cmszpt4t(mHH,mt4)[0] and totalratio*sigmaHSM13TeV(mHH)*brHtot4*brt4h < cmszpt4t(mHH,mt4)[1] and totalratio*sigmaHSM13TeV(mHH)*brHtot4*brt4z < cmszpt4t(mHH,mt4)[2] ) or (mHH > 2250. and mHH <= 2700. and totalratio*sigmaHSM13TeV(mHH)*brHtodi < cmsdpbounds(mHH) and totalratio*sigmaHSM13TeV(mHH)*brHtodi < 0.001*Hdiph(mHH) and totalratio*sigmaHSM13TeV(mHH)*brHtot4*brt4w < cmszpt4t(mHH,mt4)[0] and totalratio*sigmaHSM13TeV(mHH)*brHtot4*brt4h < cmszpt4t(mHH,mt4)[1] and totalratio*sigmaHSM13TeV(mHH)*brHtot4*brt4z < cmszpt4t(mHH,mt4)[2] ) or (mHH > 2700. and totalratio*sigmaHSM13TeV(mHH)*brHtodi < cmsdpbounds(mHH) and totalratio*sigmaHSM13TeV(mHH)*brHtot4*brt4w < cmszpt4t(mHH,mt4)[0] and totalratio*sigmaHSM13TeV(mHH)*brHtot4*brt4h < cmszpt4t(mHH,mt4)[1] and totalratio*sigmaHSM13TeV(mHH)*brHtot4*brt4z < cmszpt4t(mHH,mt4)[2] ) or (mHH <= 504. and totalratio*sigmaHSM13TeV(mHH)*brHtodi < 0.001*Hdiph(mHH) and totalratio*sigmaHSM13TeV(mHH)*brHtotata < atlastau(mHH) and totalratio*sigmaHSM13TeV(mHH)*brHtot4*brt4w < cmszpt4t(mHH,mt4)[0] and totalratio*sigmaHSM13TeV(mHH)*brHtot4*brt4h < cmszpt4t(mHH,mt4)[1] and totalratio*sigmaHSM13TeV(mHH)*brHtot4*brt4z < cmszpt4t(mHH,mt4)[2] ):# and tan(beta) > 1:
        # GBE  no constraint related with mH
        #if 1 > 0:
        #if (mHH > 504. and mHH <= 2250. and totalratio*sigmaHSM13TeV(mHH)*brHtodi < cmsdpbounds(mHH) ) or mHH > 2250:
        #if (mHH > 504. and mHH <= 2250. and totalratio*sigmaHSM13TeV(mHH)*brHtodi < 0.001*Hdiph(mHH) ) or mHH > 2250:
        #if (mHH > 504. and mHH <= 2250. and totalratio*sigmaHSM13TeV(mHH)*brHtodi < cmsdpbounds(mHH) and totalratio*sigmaHSM13TeV(mHH)*brHtodi < 0.001*Hdiph(mHH) ) or (mHH > 2250. and mHH <= 2700. and totalratio*sigmaHSM13TeV(mHH)*brHtodi < cmsdpbounds(mHH) and totalratio*sigmaHSM13TeV(mHH)*brHtodi < 0.001*Hdiph(mHH) ) or (mHH > 2700. and totalratio*sigmaHSM13TeV(mHH)*brHtodi < cmsdpbounds(mHH) ) or (mHH <= 504. and totalratio*sigmaHSM13TeV(mHH)*brHtodi < 0.001*Hdiph(mHH) and totalratio*sigmaHSM13TeV(mHH)*brHtotata < atlastau(mHH) ):
        #if totalratio*sigmaHSM13TeV(mHH)*brHtot4*brt4w < cmszpt4t(mHH,mt4)[0] and totalratio*sigmaHSM13TeV(mHH)*brHtot4*brt4h < cmszpt4t(mHH,mt4)[1] and totalratio*sigmaHSM13TeV(mHH)*brHtot4*brt4z < cmszpt4t(mHH,mt4)[2]:
        #if totalratio*sigmaHSM13TeV(mHH)*brHtotata < atlastau(mHH) or mHH > 2250:#totalratio*sigmaHSM13TeV(mHH)*brHtodi < 0.001*Hdiph(mHH) and totalratio*sigmaHSM13TeV(mHH)*brHtotata < atlastau(mHH):# and totalratio*sigmaHSM13TeV(mHH)*brour < cmszpt4t(mHH,mt4):
        #if totalratio*sigmaHSM13TeV(mHH)*brHtodi < cmsdpbounds(mHH) and totalratio*sigmaHSM13TeV(mHH)*brHtodi < 0.001*Hdiph(mHH) and totalratio*sigmaHSM13TeV(mHH)*brHtotata < atlastau(mHH) and totalratio*sigmaHSM13TeV(mHH)*brour < cmszpt4t(mHH,mt4):#ggHratio*sigmaHSM8TeV(mHH)*brHtodi/((tan(beta))**2) < cmsdpbounds(mHH):# and sigmaHSM8TeV(mHH)*brHtoe4*bre4w*0.106/((tan(beta))**2) < funclim(mHH, me4):
        #No CMS Z' -> t4 t
        #if (mHH > 504. and mHH <= 2250. and totalratio*sigmaHSM13TeV(mHH)*brHtodi < cmsdpbounds(mHH) and totalratio*sigmaHSM13TeV(mHH)*brHtodi < 0.001*Hdiph(mHH) and totalratio*sigmaHSM13TeV(mHH)*brHtotata < atlastau(mHH) ) or (mHH > 2250. and mHH <= 2700. and totalratio*sigmaHSM13TeV(mHH)*brHtodi < cmsdpbounds(mHH) and totalratio*sigmaHSM13TeV(mHH)*brHtodi < 0.001*Hdiph(mHH) ) or (mHH > 2700. and totalratio*sigmaHSM13TeV(mHH)*brHtodi < cmsdpbounds(mHH) ) or (mHH <= 504. and totalratio*sigmaHSM13TeV(mHH)*brHtodi < 0.001*Hdiph(mHH) and totalratio*sigmaHSM13TeV(mHH)*brHtotata < atlastau(mHH) ):# and tan(beta) > 1:
        #test
        #if 1 > 0:# totalratio*sigmaHSM13TeV(mHH)*brour < cmszpt4t(mHH,mt4):# sigmaHSM8TeV(mHH)*brHtoe4*bre4w*0.106*totalratio < funclim(mHH, me4):
        #if totalratio*sigmaHSM13TeV(mHH)*brHtodi < cmsdpbounds(mHH) and totalratio*sigmaHSM13TeV(mHH)*brHtodi < 0.001*Hdiph(mHH) and totalratio*sigmaHSM13TeV(mHH)*brHtotata < atlastau(mHH):
            print 't1ewptall-HWWHdp passed'
            mylist = [mt4,tan(beta),lamHt34,mHH,doubfr,singfr,delS,delT,delU,lamHrt34,lamHt44,lamHrt44,gLW44t,gLW43t,gLW45t,gLW33t,gLW34t,gLW35t,gLW55t,gLW53t,gLW54t,gRW44t,gRW43t,gRW45t,gRW33t,gRW34t,gRW35t,gRW55t,gRW54t,gRW53t,gLZ44t,gLZ43t,gLZ45t,gLZ33t,gLZ34t,gLZ35t,gLZ55t,gLZ53t,gLZ54t,gRZ44t,gRZ43t,gRZ45t,gRZ33t,gRZ34t,gRZ35t,gRZ55t,gRZ53t,gRZ54t,kaT,kaQ,ka,kab,mt5,MTT,MQQ,brHtobb,brHtohh,brHtot4,brHtogg,brHtocc,brHtodi,brHtott,gamHtot,gamHtotata,gamHtobb,gamHtot4,gamHtogg,gamHtohh,gamHtocc,gamHdiph,gamHtott,brt4w,brt4z,brt4h,brour,ggHratio,ggSMratio,lamHt55,lamHrt55,totalratio,brHtotata,brt4b4w,brt4b5w,brb4w,brb4z,brb4h,brb4t4w,brb4t5w,brHtot5,brHt4t4,brHt4t5,brHt5t5,brHtob4,brHtob5,brHb4b4,brHb4b5,brHb5b5,lamHt35,lamHrt35,lamHt45,lamHrt45,lamHb34,lamHrb34,lamHb35,lamHrb35,lamHb45,lamHrb45,lamHb44,lamHrb44,lamHb55,lamHrb55,gLZ33b,gLZ44b,gLZ55b,gLZ34b,gLZ43b,gLZ35b,gLZ53b,gLZ45b,gLZ54b,gRZ33b,gRZ44b,gRZ55b,gRZ34b,gRZ43b,gRZ35b,gRZ53b,gRZ45b,gRZ54b,gLW42t,gLW52t,gLW24b,gLW25b,gLW41t,gLW51t,gLW14b,gLW15b,mb4,mb5,doubt5fr,singt5fr,doubb4fr,singb4fr,doubb5fr,singb5fr,xst4bj,xst4tj,gammat4toh,gammat4tow,lamHtest,lamHtest1,lamHtest2,lamHtest3,lamHtest4,brt5z,brt5w,brt5h,"\n"]  
            #fevout=open("t1ewptall-test-%s.dat" %listlep[k], "a+")
            fevout=open("t1ewptall-%s.dat" %listlep[k], "a+")
            #fevout=open("t1ewptall-%s-noconstraint.dat" %listlep[k], "a+")
            #fevout=open("t1ewptall-%s-s7-tan1-test.dat" %listlep[k], "a+")
            for item in mylist:
                fevout.write(str(item) + '\t')   
            fevout.close()                           

            #print sigmabbF13(mHH)/sigmaHSM13TeV(mHH), mHH, tan(beta), totalratio



        #if brt4b4w > 0.:
            #print 'BR(t4 -> W b4) =', brt4bw4

            
