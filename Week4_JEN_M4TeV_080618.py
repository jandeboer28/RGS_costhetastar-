#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 14:07:01 2018

@author: jan
"""

# Softdrop Jet mass Bulk vs GS  -> Monte Carlo sample(?)
# M = 2 TeV and M = 4 TeV
# by Jan de Boer @ CERN
# date 4.7.2018

from __future__ import division, print_function
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Dropout
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.models import load_model

import ROOT as rt
import numpy as np
import pandas as pd
#from sklearn import preprocessing
#from sklearn.preprocessing import MinMaxScaler

#from ROOT import * # not supported in Ipython

#import os
#os.chdir('/home/jan/Documents/CERN/Project/Week27')


# constructed in week 29.
def CalculateAngles( VPx,  VPy,  VPz,  VE,
                     D1Px, D1Py, D1Pz, D1E,
                     D2Px, D2Py, D2Pz, D2E) :
    # making lorentz vectors :
    p4V = rt.TLorentzVector(VPx, VPy, VPz, VE)
    p4D1 = rt.TLorentzVector(D1Px, D1Py, D1Pz, D1E)
    p4D2 = rt.TLorentzVector(D2Px, D2Py, D2Pz, D2E)
    
    # boosting back to rest frame of V :
    boostV = -(p4V.BoostVector())
    p4D1inVFrame = p4D1
    p4D2inVFrame = p4D2
    p4D1inVFrame.Boost(boostV)
    p4D2inVFrame.Boost(boostV)
    # unit vector of V in rf
    UnitVx = rt.TVector3(VPx, VPy, VPz)
    normVx = 1/(UnitVx.Mag())
    UnitVx *= normVx
    # costheta cos of theta, theta = polar angle
    p3D1inVFrame = rt.TVector3(p4D1inVFrame.X(),p4D1inVFrame.Y(),p4D1inVFrame.Z())
    p3D2inVFrame = rt.TVector3(p4D2inVFrame.X(),p4D2inVFrame.Y(),p4D2inVFrame.Z())
    # create z and y axes
    Unitz = p3D1inVFrame.Cross(p3D2inVFrame)
    normz = 1/(Unitz.Mag())
    Unitz *= normz
    Unity = Unitz.Cross(UnitVx)
    # Dau1
    # calculate theta 1
    UnitD1 = p3D1inVFrame.Unit()
    x_D1 = UnitD1.Dot(UnitVx)
    y_D1 = UnitD1.Dot(Unity)
    z_D1 = UnitD1.Dot(Unitz)
    # jennifers method :
    p3D1_inVFrame_unit = rt.TVector3(y_D1,z_D1,x_D1)
    costheta1 = p3D1_inVFrame_unit.Theta()
    # my method : 
#    theta = np.arccos(x_D1/p3D1_inVFrame_unit.Mag())
    # Dau1
    # calculate theta 2
    UnitD2 = p3D2inVFrame.Unit()
    x_D2 = UnitD2.Dot(UnitVx)
    y_D2 = UnitD2.Dot(Unity)
    z_D2 = UnitD2.Dot(Unitz)
    # jennifers method :
    p3D2_inVFrame_unit = rt.TVector3(y_D2,z_D2,x_D2)
    costheta2 = p3D2_inVFrame_unit.Theta()    

    
    # calculated phi1, phi2, costhetastar :
#    phi1 = p3D1inVFrame.Phi()
#    phi2 = p3D2inVFrame.Phi()
 #   costhetastar = p3D1inVFrame.CosTheta()
    
    return costheta1, costheta2


def GetWeights ( hist, costheta1list) :
    
    Nbins = hist.GetNbinsX()
    width = hist.GetBinWidth(1)
    center = hist.GetBinCenter(1)
    xmin = center - width/2
    xmax = xmin + Nbins*width
    
    binarray1 = np.arange(1,Nbins+2, 1)
    binedge = []
    
    for i in binarray1 :
        edge = hist.GetBinLowEdge(i)
        binedge.append(edge)
    
    wht = []
    entries = hist.GetEntries()
    ave = entries/Nbins
    binarray2 = np.arange(1, Nbins+1, 1)
    
    for i in binarray2 :
        number = h_theta_cm1_RS.GetBinContent(i)
        w = ave/number
        wht.append(w)
        
    weight = []
    for entry in costheta1list :
        for i, value in enumerate(binedge) :
            if (entry > value) and (entry < binedge[i+1]):
                weight.append(wht[i])
    
    return weight

def SDCondition (pt1, eta1, phi1, pt2, eta2, phi2) :
    
    ptm = min(pt1, pt2)
    p = ptm/(pt1+pt2)
    
    dr = np.sqrt((eta1-eta2)**2+(phi1-phi2)**2)
    r0 = 0.8
    zcut = 0.1
    beta = 0
    q = zcut*(dr/r0)**beta
    
    if p > q :
        condition = 1
    elif p < q :
        condition = 0
    
    return condition

## Calculate ROC curve from two histograms:
def calc_ROC(hist1, hist2):
    
    y_sig, x_edges, _ = hist1 
    y_bkg, x2_edges, _ = hist2
    
    # Check that the two histograms have the same number of bins and same range:
    if (x_edges == x2_edges).any():
        
        x_centers = 0.5*(x_edges[1:] + x_edges[:-1])
        
        N = len(y_sig)
        
        integral_sig = y_sig.sum()
        integral_bkg = y_bkg.sum()
    
        FPR = np.zeros(N) # False positive rate ()
        TPR = np.zeros(N) # True positive rate (sensitivity)
        
        # https://upload.wikimedia.org/wikipedia/commons/4/4f/ROC_curves.svg
        for i, x in enumerate(x_centers): 
            cut = (x_centers < x)
            TN = np.sum(y_bkg[cut]) / integral_bkg      # True negatives (background)
            FP = np.sum(y_bkg[~cut]) / integral_bkg     # False positives (signal)
                
            FN = np.sum(y_sig[cut]) / integral_sig     # False negatives
            TP = np.sum(y_sig[~cut]) / integral_sig    # True positives
            
            FPR[i] = FP / ( FP + TN )
            TPR[i] = TP / ( TP + FN )
        
        return FPR, TPR
    
    else:
        print("ERROR: Signal and Background histograms have different bins and ranges")



#=============================================================================#
#-----------------------------------------------------------------------------#
#     
#                         Loading data
#     
#-----------------------------------------------------------------------------#
#=============================================================================#



#f = rt.TFile("BulkGravToWW_narrow_M-4000_13TeV-madgraph.root","READ")
#f = rt.TFile("BulkGravToWW_narrow_M-2000_13TeV-madgraph.root","READ")
f = rt.TFile("BulkGravToWW_narrow_M-2000_13TeV-madgraph163.root", "READ")
t = f.Get("tree")
#t.Print("tree")
tlen = t.GetEntriesFast()

#f2 = rt.TFile("RSGravToWW_kMpl01_M-4000_TuneCUETP8M1_13TeV-pythia8.root","READ")
#f2 = rt.TFile("RSGravToWW_width0p1_M-2000_TuneCUETP8M1_13TeV-madgraph-pythia8.root", "READ")
#f2 = rt.TFile("WJetsToQQ_HT-600ToInf_TuneCUETP8M1_13TeV-madgraphMLM-pythia8.root", "READ")
f2 = rt.TFile("RSGravToWW_width0p1_M-2000_TuneCUETP8M1_13TeV-madgraph-pythia857.root", "READ")
t2 = f2.Get("tree")
#t2.Print("tree")
t2len = t2.GetEntriesFast()

#    TH1F (name root tree, title, nbins, lower, upper)




#=============================================================================#
#-----------------------------------------------------------------------------#
#     
#               Declaring all the plots, fig and hist to Fill :
#     
#-----------------------------------------------------------------------------#
#=============================================================================#


# soft drop mass
h_B = rt.TH1F("B sd jet mass", "Soft Drop jet mass",100,-10,300)
h_RS = rt.TH1F("RS sd jet mass", "Soft Drop jet mass",100,-10,300)

# invariant mass of the system
h_invM_B = rt.TH1F("B inv mass", "Invariant mass of quark pair", 100, 60, 100)
h_invM_RS = rt.TH1F("RS inv mass", "Invariant mass of quark pair", 100, 60, 100)


# theta (angle of emission) in V rest frame
h_theta_cm1_B = rt.TH1F("B theta CM", "W rest frame emission angle", 100, -1, 1)
h_theta_cm2_B = rt.TH1F("B theta CM", "W rest frame emission angle", 100, -1, 1)
h_dtheta_cm_B = rt.TH1F("B dtheta CM", "#theta_{1}+#theta_{2} in V frame", 100, -1, 1)


h_theta_cm1_RS = rt.TH1F("RS theta CM", "W rest frame emission angle", 100, -1, 1)
h_theta_cm2_RS = rt.TH1F("RS theta CM", "W rest frame emission angle", 100, -1, 1)
h_dtheta_cm_RS = rt.TH1F("RS dtheta CM", "#theta_{1}+#theta_{2} in V frame", 100, -1, 1)

                         
# scatter plots of theta vs variables

                         
                         
                         
# I. vs SD mass
scatSDTHistB = rt.TH2D("Legend", "Bulk costheta* vs SD mass", 100, -1, 1, 100, 0, 300)
scatSDTHistRS = rt.TH2D("Legend", "RS costheta* vs SD mass", 100, -1, 1, 100, 0, 300)

# I. vs SD mass
scatSDInvMHistB = rt.TH2D("Legend", "Bulk Invariant vs SD mass", 100, 60, 100, 100, 0, 300)
scatSDInvMHistRS = rt.TH2D("Legend", "RS Invariant vs SD mass", 100, 60, 100, 100, 0, 300)

# pass/fail vs SD mass
scatPFHistB = rt.TH2D("Legend", "Bulk costheta* vs SD condition", 100, -1, 1, 2, -0.5, 1.5)
scatPFHistRS = rt.TH2D("Legend", "RS costheta* vs SD condition", 100, -1, 1, 2, -0.5, 1.5)

# ----------------------------------------------------------------------------#
# tau21 plots :   SD mass, costheta, Inv mass and Pt
# I SD mass
scatTSDHistB = rt.TH2D("Legend", "Bulk tau21 vs SD mass", 100, 0, 1, 100, 0, 300)
scatTSDHistRS = rt.TH2D("Legend", "RS tau21 vs SD mass", 100, 0, 1, 100, 0, 300)
# II Inv mass
scatTInvMHistB = rt.TH2D("Legend", "Bulk tau21 vs Inv mass", 100, 0, 1, 100, 60, 100)
scatTInvMHistRS = rt.TH2D("Legend", "RS tau21 vs Inv mass", 100, 0, 1, 100, 60, 100)
# III Pt
scatTPtHistB = rt.TH2D("Legend", "Bulk tau21 vs Pt", 100, 0, 1, 100, 0, 600)
scatTPtHistRS = rt.TH2D("Legend", "RS tau21 vs Pt", 100, 0, 1, 100, 0, 600)
# IV costheta
scatTHistB = rt.TH2D("Legend", "Bulk costheta* vs tau21", 100, -1, 1, 100, 0, 1)
scatTHistRS = rt.TH2D("Legend", "RS costheta* vs tau21", 100, -1, 1, 100, 0, 1)
#
#-----------------------------------------------------------------------------#


# II. vs Pt                  
scatPtHistB = rt.TH2D("Legend", "Bulk costheta* vs Pt Dau1", 100, -1, 1, 100, 0, 600)
scatPtHistRS = rt.TH2D("Legend", "RS costheta* vs Pt Dau1", 100, -1, 1, 100, 0, 600)
# for V
scatPtHistBV = rt.TH2D("Legend", "Bulk costheta* vs Pt V", 100, -1, 1, 100, 0, 2000)
scatPtHistRSV = rt.TH2D("Legend", "RS costheta* vs Pt V", 100, -1, 1, 100, 0, 2000)


# III. vs Eta                  
scatEtaHistB = rt.TH2D("Legend", "Bulk costheta* vs Eta Dau1", 100, -1, 1, 100, -3.15, 3.15)
scatEtaHistRS = rt.TH2D("Legend", "RS costheta* vs Eta Dau1", 100, -1, 1, 100, -3.15, 3.15)
# for V
scatEtaHistBV = rt.TH2D("Legend", "Bulk costheta* vs Eta V", 100, -1, 1, 100, -3.15, 3.15)
scatEtaHistRSV = rt.TH2D("Legend", "RS costheta* vs Eta V", 100, -1, 1, 100, -3.15, 3.15)



# IV. vs E                  
scatEHistB = rt.TH2D("Legend", "Bulk costheta* vs E Dau1", 100, -1, 1, 100, 0, 4400)
scatEHistRS = rt.TH2D("Legend", "RS costheta* vs E Dau1", 100, -1, 1, 100, 0, 4400)
# for V
scatEHistBV = rt.TH2D("Legend", "Bulk costheta* vs E V", 100, -1, 1, 100, 0, 4400)
scatEHistRSV = rt.TH2D("Legend", "RS costheta* vs E V", 100, -1, 1, 100, 0, 4400)



   
#=============================================================================#
#-----------------------------------------------------------------------------#
#     
#               Selecting the data / variables 
#     
#-----------------------------------------------------------------------------#
#=============================================================================#



# Keras list RS :

VE_B   = []
VPt_B  = []
VEta_B = []
VPhi_B = []
VPx_B  = []
VPy_B  = []
VPz_B  = []
DE_B   = []
DPt_B  = []
DEta_B = []
DPhi_B = []
DPx_B  = []
DPy_B  = []
DPz_B  = []
DE2_B   = []
DPt2_B  = []
DEta2_B = []
DPhi2_B = []
DPx2_B  = []
DPy2_B  = []
DPz2_B  = []
DCT_B  = []
DCT2_B = []
DT_B   = []  
B_B    = []  
Minv_B = []
Tau21_B = []
                        
# Keras list RS :

VE   = []
VPt  = []
VEta = []
VPhi = []
VPx  = []
VPy  = []
VPz  = []
DE   = []
DPt  = []
DEta = []
DPhi = []
DPx  = []
DPy  = []
DPz  = []
DE2   = []
DPt2  = []
DEta2 = []
DPhi2 = []
DPx2  = []
DPy2  = []
DPz2  = []
DCT  = []
DCT2 = []
DT   = []  
B    = []                     
Minv_A = []   
Tau21_A = []                      
                         
                         
                         
theta_cm1_list = []
theta_cm2_list = []
Prest_list1 = []

NScat = 10000
count = 0
s     = 0

pfc = True 

for event in t:
    # soft drop mass:
    h_B.Fill(t.fj_corr_sdmass)
    # Loading in all variables Capital : W, 1 & 2 denote daughter 1 & 2 resp.
    E = t.genV_e
    PT = t.genV_pt
    ETA = t.genV_eta
    PHI = t.genV_phi
    e1 = t.genDaus_e[0]
    pt1 = t.genDaus_pt[0]
    eta1 = t.genDaus_eta[0]
    phi1 = t.genDaus_phi[0]
    e2 = t.genDaus_e[1]
    pt2 = t.genDaus_pt[1]
    eta2 = t.genDaus_eta[1]
    phi2 = t.genDaus_phi[1]
    # invariant mass is calculated by the formula below (massless limit!) :
    Minv = rt.TMath.Sqrt(2*pt1*pt2*(rt.TMath.CosH(eta1-eta2)-rt.TMath.Cos(phi1-phi2)))
    h_invM_B.Fill(Minv)
    # Soft Drop Condition
    pf = SDCondition(pt1, eta1, phi1, pt2, eta2, phi2)
    # tau21
    tau21 = t.jtau21 
    # Next :
    #   -> to calculate theta in V rest frame <-
    #
    # I. compute momentum in cartesian coordinates
    # II. project momentum of dau_1 on momentum of W
    # III. boost back to W rest frame
    # IV. compute thetaw distribution
    #
    # I :
    PX = PT*np.cos(PHI)
    PY = PT*np.sin(PHI)
    PZ = PT*np.sinh(ETA)
    P = PT*np.cosh(ETA)
    px1 = pt1*np.cos(phi1)
    py1 = pt1*np.sin(phi1)
    pz1 = pt1*np.sinh(eta1)
    p1 = pt1*np.cosh(eta1)
    px2 = pt2*np.cos(phi2)
    py2 = pt2*np.sin(phi2)
    pz2 = pt2*np.sinh(eta2)
    p2 = pt2*np.cosh(eta2)
    #
    # II :
    thetaVD1 = np.arccos((px1*PX+py1*PY+pz1*PZ)/(p1*P))
    mom_par = p1*np.cos(thetaVD1)
    mom_tra = p1*np.sin(thetaVD1)
    #
    # III :
    gamma = E/Minv
    beta = P/E
    mom_par_w = gamma*(mom_par-beta*e1)
    # IV :
    Mcostheta = np.arccos(mom_par_w/np.sqrt(mom_par_w**2+mom_tra**2))
#    h_theta_cm1_B.Fill(Mcostheta*180/np.pi)

    # -> using ROOT language 
    costheta1, costheta2 = CalculateAngles(PX, PY, PZ, E,
                                           px1, py1, pz1, e1,
                                           px2, py2, pz2, e2)
       
    
    
   
    
    # sanity check on W rest frame!
    Prest = gamma*(P-beta*E)
    Prest_list1.append(Prest)
    
    # -> making numpy arrays for Keras :
    if (pf == 1 or pfc):
        h_theta_cm1_B.Fill(np.cos(costheta1))
        h_theta_cm2_B.Fill(np.cos(costheta2))
        h_dtheta_cm_B.Fill(np.cos(costheta1+costheta2))


        VE_B.append(E)
        VPt_B.append(PT)
        VEta_B.append(ETA)
        VPhi_B.append(PHI)
        VPx_B.append(PX)
        VPy_B.append(PY)
        VPz_B.append(PZ)
        DE_B.append(e1)
        DPt_B.append(pt1)
        DEta_B.append(eta1)
        DPhi_B.append(phi1)
        DPx_B.append(px1)
        DPy_B.append(py1)
        DPz_B.append(pz1)
        DE2_B.append(e2)
        DPt2_B.append(pt2)
        DEta2_B.append(eta2)
        DPhi2_B.append(phi2)
        DPx2_B.append(px2)
        DPy2_B.append(py2)
        DPz2_B.append(pz2)
        DCT_B.append(np.cos(costheta1))
        DCT2_B.append(np.cos(costheta2))
        DT_B.append(costheta1)
        B_B.append(beta)
        Minv_B.append(Minv)
        Tau21_B.append(tau21)



    # filling scatterplot
    if s < NScat :
        scatSDTHistB.Fill(np.cos(costheta1),t.fj_corr_sdmass)
        scatSDInvMHistB.Fill(Minv, t.fj_corr_sdmass)
        scatPtHistB.Fill(np.cos(costheta1), pt1)  
        scatEtaHistB.Fill(np.cos(costheta1), eta1)
        scatEHistB.Fill(np.cos(costheta1), e1)
        scatPtHistBV.Fill(np.cos(costheta1), PT)  
        scatEtaHistBV.Fill(np.cos(costheta1), ETA)
        scatEHistBV.Fill(np.cos(costheta1), E)
        scatPFHistB.Fill(np.cos(costheta1), pf)
        # tau21 plots
        if pf == 1 :
            scatTSDHistB.Fill(tau21, t.fj_corr_sdmass)
            scatTInvMHistB.Fill(tau21, Minv)        
            scatTPtHistB.Fill(tau21, pt1)
            scatTHistB.Fill(np.cos(costheta1), tau21)
        
        
    s += 1

Prest_list1 = np.array(Prest_list1)
print(Prest_list1.mean(), Prest_list1.std())
print("t done")

Prest_list2 = []  
count = 0
s     = 0
for event in t2 :
    # soft drop mass:
    h_RS.Fill(t2.fj_corr_sdmass)   
    # Loading in all variables Capital : W, 1 & 2 denote daughter 1 & 2 resp.
    E = t2.genV_e
    PT = t2.genV_pt
    ETA = t2.genV_eta
    PHI = t2.genV_phi
    e1 = t2.genDaus_e[0]
    pt1 = t2.genDaus_pt[0]
    eta1 = t2.genDaus_eta[0]
    phi1 = t2.genDaus_phi[0]
    e2 = t2.genDaus_e[1]
    pt2 = t2.genDaus_pt[1]
    eta2 = t2.genDaus_eta[1]
    phi2 = t2.genDaus_phi[1]
    # invariant mass is calculated by the formula below (massless limit!) :
    Minv = rt.TMath.Sqrt(2*pt1*pt2*(rt.TMath.CosH(eta1-eta2)-rt.TMath.Cos(phi1-phi2)))
    h_invM_RS.Fill(Minv)
    # Soft Drop Condition
    pf = SDCondition(pt1, eta1, phi1, pt2, eta2, phi2)
    # tau21
    tau21 = t2.jtau21 
    # Next :
    #   -> to calculate theta in V rest frame <-
    #
    # I. compute momentum in cartesian coordinates
    # II. project momentum of dau_1 on momentum of W
    # III. boost back to W rest frame
    # IV. compute thetaw distribution
    #
    # I :
    PX = PT*np.cos(PHI)
    PY = PT*np.sin(PHI)
    PZ = PT*np.sinh(ETA)
    P = PT*np.cosh(ETA)
    px1 = pt1*np.cos(phi1)
    py1 = pt1*np.sin(phi1)
    pz1 = pt1*np.sinh(eta1)
    p1 = pt1*np.cosh(eta1)
    px2 = pt2*np.cos(phi2)
    py2 = pt2*np.sin(phi2)
    pz2 = pt2*np.sinh(eta2)
    p2 = pt2*np.cosh(eta2)
    #
    # II :
    thetaVD1 = np.arccos((px1*PX+py1*PY+pz1*PZ)/(p1*P))
    mom_par = p1*np.cos(thetaVD1)
    mom_tra = p1*np.sin(thetaVD1)
    #
    # III :
    gamma = E/Minv
    beta = P/E
    mom_par_w = gamma*(mom_par-beta*e1)
    # IV :
    Mcostheta = np.arccos(mom_par_w/np.sqrt(mom_par_w**2+mom_tra**2))
#    h_theta_cm1_RS.Fill(Mcostheta*180/np.pi)   #*180/np.pi 
    if count < 180000 :

        # -> using ROOT language 
        costheta1, costheta2 = CalculateAngles(PX, PY, PZ, E,
                                               px1, py1, pz1, e1,
                                               px2, py2, pz2, e2)
    


    
    # -> making numpy arrays for Keras :
        if (pf == 1 or pfc):

            h_theta_cm1_RS.Fill(np.cos(costheta1))
            h_theta_cm2_RS.Fill(np.cos(costheta2))
            h_dtheta_cm_RS.Fill(np.cos(costheta1+costheta2))
    

            VE.append(E)
            VPt.append(PT)
            VEta.append(ETA)
            VPhi.append(PHI)
            VPx.append(PX)
            VPy.append(PY)
            VPz.append(PZ)
            DE.append(e1)
            DPt.append(pt1)
            DEta.append(eta1)
            DPhi.append(phi1)
            DPx.append(px1)
            DPy.append(py1)
            DPz.append(pz1)
            DE2.append(e2)
            DPt2.append(pt2)
            DEta2.append(eta2)
            DPhi2.append(phi2)
            DPx2.append(px2)
            DPy2.append(py2)
            DPz2.append(pz2)
            DCT.append(np.cos(costheta1))
            DCT2.append(np.cos(costheta2))
            DT.append(costheta1)
            B.append(beta)
            Minv_A.append(Minv)
            Tau21_A.append(tau21)
            
            
    count += 1   
    
    # filling scatterplot
    if s < NScat :
        scatSDTHistRS.Fill(np.cos(costheta1),t2.fj_corr_sdmass)
        scatSDInvMHistRS.Fill(Minv, t2.fj_corr_sdmass)
        scatPtHistRS.Fill(np.cos(costheta1), pt1)
        scatEtaHistRS.Fill(np.cos(costheta1), eta1)
        scatEHistRS.Fill(np.cos(costheta1), e1)
        scatPtHistRSV.Fill(np.cos(costheta1), PT)
        scatEtaHistRSV.Fill(np.cos(costheta1), ETA)
        scatEHistRSV.Fill(np.cos(costheta1), E)
        scatPFHistRS.Fill(np.cos(costheta1), pf)
        # tau21 plots
        if pf == 1 :
            scatTSDHistRS.Fill(tau21, t2.fj_corr_sdmass)
            scatTInvMHistRS.Fill(tau21, Minv)        
            scatTPtHistRS.Fill(tau21, pt1)
            scatTHistRS.Fill(np.cos(costheta1), tau21)
    s += 1
   
    
    
    
    # sanity check on W rest frame!
    Prest = gamma*(P-beta*E)
    Prest_list2.append(Prest)

Prest_list2 = np.array(Prest_list2)
print(Prest_list2.mean(), Prest_list2.std()) 
print("t2 done.")

VE   = np.array(VE)
VPt  = np.array(VPt)
VEta = np.array(VEta)
VPhi = np.array(VPhi)
VPx  = np.array(VPx)
VPy  = np.array(VPy)
VPz  = np.array(VPz)
DE   = np.array(DE)
DPt  = np.array(DPt)
DEta = np.array(DEta)
DPhi = np.array(DPhi)
DPx  = np.array(DPx)
DPy  = np.array(DPy)
DPz  = np.array(DPz)
DE2   = np.array(DE2)
DPt2  = np.array(DPt2)
DEta2 = np.array(DEta2)
DPhi2 = np.array(DPhi2)
DPx2  = np.array(DPx2)
DPy2  = np.array(DPy2)
DPz2  = np.array(DPz2)
DCT  = np.array(DCT)
DCT2 = np.array(DCT2)
DT   = np.array(DT)
B    = np.array(B)
Minv_A = np.array(Minv_A)
Tau21_A = np.array(Tau21_A)


#=============================================================================#
#-----------------------------------------------------------------------------#
#     
#                         Canvas section
#     
#-----------------------------------------------------------------------------#
#=============================================================================#

saveplot = False
draw     = False
canvas   = False

if canvas :

    # canvas for soft drop mass
    c = rt.TCanvas("c1","c1",800,600)
    h_B.GetYaxis().SetTitle("Entries/bin");h_B.GetXaxis().SetTitle("Mass [GeV/c^{2}]")
    h_B.SetLineColor(rt.kRed)  
    h_B.Draw("HIST")
    h_RS.SetLineColor(rt.kBlue)
    h_RS.Draw("HIST same") 
    # adding a legend
    leg = rt.TLegend( 0.6449843,0.65038,0.8377743,0.8838219)#,NULL,"brNDC")
    leg.SetTextSize(0.03)
    leg.SetLineColor(rt.kWhite)
    leg.AddEntry(h_B,"Bulk, Nent = 40107","l")
    leg.AddEntry(h_RS,"RS, Nent = 38355","l")
    leg.Draw()  
    
    rt.gStyle.SetOptStat(0)
    if draw :
        rt.gPad.Draw()
    if saveplot :
        c.SaveAs("sdjetmass.png")
            
            # canvas for inv mass
    c1 = rt.TCanvas("c1","c1",800,600)
    h_invM_B.GetYaxis().SetTitle("Entries/bin");h_invM_B.GetXaxis().SetTitle("Mass [GeV/c^{2}]")
    h_invM_B.SetLineColor(rt.kRed)  
    h_invM_B.Draw("HIST")
    h_invM_RS.SetLineColor(rt.kBlue)
    h_invM_RS.Draw("HIST same") 
    # adding a legend
    leg = rt.TLegend( 0.6449843,0.65038,0.8377743,0.8838219)#,NULL,"brNDC")
    leg.SetTextSize(0.03)
    leg.SetLineColor(rt.kWhite)
    leg.AddEntry(h_invM_B,"Bulk, Nent = 40107","l")
    leg.AddEntry(h_invM_RS,"RS, Nent = 38355","l")
    leg.Draw()  
    
    rt.gStyle.SetOptStat(0)
    if draw :
        rt.gPad.Draw()
    if saveplot :
        c1.SaveAs("invmass.png")

    #canvas for W rest frame emission angle
    c2 = rt.TCanvas("c2","c2",800,600)
    h_theta_cm1_B.GetYaxis().SetTitle("Entries / 1 deg");h_theta_cm1_B.GetXaxis().SetTitle("Cos #theta^{*}")
    h_theta_cm1_B.GetXaxis().CenterTitle()
    h_theta_cm1_B.SetLineColor(rt.kRed)  
    h_theta_cm1_B.Draw("HIST")
    h_theta_cm1_RS.SetLineColor(rt.kBlue)
    h_theta_cm1_RS.Draw("HIST same") 
    
    


    # adding a legend
    leg = rt.TLegend( 0.7449843,0.79038,0.9333,0.8838219)#,NULL,"brNDC")
    leg.SetTextSize(0.03)
    leg.SetLineColor(rt.kWhite)
    leg.SetFillStyle(0)
    leg.AddEntry(h_theta_cm1_B,"Bulk Dau1","l") #Nent = 40107
    leg.AddEntry(h_theta_cm1_RS,"RS Dau1","l")  #Nent = 38255
    leg.Draw()  

    rt.gStyle.SetOptStat(0)
    if draw :
        rt.gPad.Draw()
    if saveplot :
        c2.SaveAs("thetastar1_W30_v0.png")

    # Creating a weighted distribution, date 01.08.2018
    #c2a = rt.TCanvas("c2a", "c2a", 800, 600)
    
    # Get bin content, can be used for the weighted distribution.
    # weighted distribution
    
    # tlen    -> Bulk
    # t2len   -> RS
    


    #ratio = []
    #binContent = []
    #binCenter = []
    #Nbin = 100
    #binarray = np.arange(1,Nbin+1,1)
    #total = 0
    #for i in binarray :
    #    number = h_theta_cm1_real.GetBinContent(i)
    #    number2 = h_theta_cm1_pred.GetBinContent(i)
    #    center = h_theta_cm1_real.GetBinCenter(i)
    #    binContent.append(number)
    #    binCenter.append(center)
    #    total += number
    #    ratio.append(number2/number)
    


    c3 = rt.TCanvas("c3","c3",800,600)
    h_theta_cm2_B.GetYaxis().SetTitle("Entries / 1 deg");h_theta_cm2_B.GetXaxis().SetTitle("Cos #theta^{*}")
    h_theta_cm2_B.GetXaxis().CenterTitle()
    h_theta_cm2_B.SetLineColor(rt.kRed)  
    h_theta_cm2_B.Draw("HIST")
    h_theta_cm2_RS.SetLineColor(rt.kBlue)
    h_theta_cm2_RS.Draw("HIST same") 


    # adding a legend
    leg = rt.TLegend( 0.7449843,0.79038,0.9333,0.8838219)#,NULL,"brNDC")
    leg.SetTextSize(0.03)
    leg.SetLineColor(rt.kWhite)
    leg.SetFillStyle(0)
    leg.AddEntry(h_theta_cm2_B,"Bulk Dau1","l") #Nent = 40107
    leg.AddEntry(h_theta_cm2_RS,"RS Dau1","l")  #Nent = 38255
    leg.Draw()  

    rt.gStyle.SetOptStat(0)
    if draw :
        rt.gPad.Draw()
    if saveplot :
        c3.SaveAs("thetastar2_W30_v0.png")

    c4 = rt.TCanvas("c4","c4",800,600)
    h_dtheta_cm_B.GetYaxis().SetTitle("Entries / 1 deg");h_dtheta_cm_B.GetXaxis().SetTitle("Cos #theta^{*}")
    h_dtheta_cm_B.GetXaxis().CenterTitle()
    h_dtheta_cm_B.SetLineColor(rt.kRed) 
    h_dtheta_cm_B.Draw("HIST") 
    h_dtheta_cm_RS.SetLineColor(rt.kBlue)
    h_dtheta_cm_RS.Draw("HIST same") 
    

    # adding a legend
    leg = rt.TLegend( 0.7449843,0.79038,0.9333,0.8838219)#,NULL,"brNDC")
    leg.SetTextSize(0.03)
    leg.SetLineColor(rt.kWhite)
    leg.SetFillStyle(0)
    leg.AddEntry(h_dtheta_cm_B,"Bulk Dau1","l") #Nent = 40107
    leg.AddEntry(h_dtheta_cm_RS,"RS Dau1","l")  #Nent = 38255
    leg.Draw()  

    rt.gStyle.SetOptStat(0)
    if draw :
        rt.gPad.Draw()
    if saveplot :
        c4.SaveAs("sumthetastar_W30_v0.png")


    #-----------------------------------------------------------------------------#
    #       Costheta SCATTER plots for various variables
    #              -> is there correlation??? <-
    #

    # SD mass plot :
    
    c5 = rt.TCanvas("c5", "c5", 800, 400)   # 800 wide, 400 tall
    c5.Divide(2,1)
    c5.cd(1)
    #rt.gStyle.SetPalette()
    scatSDTHistB.GetYaxis().SetTitle("SD mass [GeV/c^{2}]")
    scatSDTHistB.GetXaxis().SetTitle("Cos #theta^{*}")
    scatSDTHistB.GetXaxis().CenterTitle()
    scatSDTHistB.GetXaxis().SetTitleSize(16)
    scatSDTHistB.GetXaxis().SetTitleFont(43)
    scatSDTHistB.GetXaxis().SetLabelSize(16)
    scatSDTHistB.GetXaxis().SetLabelFont(43)
    scatSDTHistB.GetYaxis().SetTitleOffset(1.45)
    scatSDTHistB.GetYaxis().SetTitleSize(16)
    scatSDTHistB.GetYaxis().SetTitleFont(43)
    scatSDTHistB.GetYaxis().SetLabelSize(16)
    scatSDTHistB.GetYaxis().SetLabelFont(43)
    scatSDTHistB.SetMarkerColor(rt.kRed)
    scatSDTHistB.Draw("COLZ")
    rt.gStyle.SetOptStat(0)
    
    c5.cd(2)
    scatSDTHistRS.GetYaxis().SetTitle("SD mass [GeV/c^{2}]")
    scatSDTHistRS.GetXaxis().SetTitle("Cos #theta^{*}")
    scatSDTHistRS.GetXaxis().CenterTitle()
    scatSDTHistRS.GetXaxis().SetTitleSize(16)
    scatSDTHistRS.GetXaxis().SetTitleFont(43)
    scatSDTHistRS.GetXaxis().SetLabelSize(16)
    scatSDTHistRS.GetXaxis().SetLabelFont(43)
    scatSDTHistRS.GetYaxis().SetTitleOffset(1.45)
    scatSDTHistRS.GetYaxis().SetTitleSize(16)
    scatSDTHistRS.GetYaxis().SetTitleFont(43)
    scatSDTHistRS.GetYaxis().SetLabelSize(16)
    scatSDTHistRS.GetYaxis().SetLabelFont(43)
    scatSDTHistRS.SetMarkerColor(rt.kBlue) #Alpha(4, 0.35)
    scatSDTHistRS.Draw("COLZ")
    rt.gStyle.SetOptStat(0)
    
    #c5.cd()
    #rt.gPad.Draw()
    if draw :
        c5.cd()
        rt.gPad.Draw()
    if saveplot :
        c5.SaveAs("thetavssdmass_W31_v1.png")
 
    # Soft Drop vs Invariant Mass
    c5A = rt.TCanvas("c5A", "c5A", 800, 400)   # 800 wide, 400 tall
    c5A.Divide(2,1)
    c5A.cd(1)
    scatSDInvMHistB.GetYaxis().SetTitle("SD mass [GeV/c^{2}]")
    scatSDInvMHistB.GetXaxis().SetTitle("Inv mass [GeV/c^{2}]")
    scatSDInvMHistB.GetXaxis().CenterTitle(0)
    scatSDInvMHistB.GetXaxis().SetTitleSize(16)
    scatSDInvMHistB.GetXaxis().SetTitleFont(43)
    scatSDInvMHistB.GetXaxis().SetLabelSize(16)
    scatSDInvMHistB.GetXaxis().SetLabelFont(43)
    scatSDInvMHistB.GetYaxis().SetTitleOffset(1.45)
    scatSDInvMHistB.GetYaxis().SetTitleSize(16)
    scatSDInvMHistB.GetYaxis().SetTitleFont(43)
    scatSDInvMHistB.GetYaxis().SetLabelSize(16)
    scatSDInvMHistB.GetYaxis().SetLabelFont(43)
    scatSDInvMHistB.SetMarkerColor(rt.kRed)
    scatSDInvMHistB.Draw("COLZ")
    rt.gStyle.SetOptStat(0)
    
    c5A.cd(2)
    scatSDInvMHistRS.GetYaxis().SetTitle("SD mass [GeV/c^{2}]")
    scatSDInvMHistRS.GetXaxis().SetTitle("Inv mass [GeV/c^{2}]")
    scatSDInvMHistRS.GetXaxis().CenterTitle(0)
    scatSDInvMHistRS.GetXaxis().SetTitleSize(16)
    scatSDInvMHistRS.GetXaxis().SetTitleFont(43)
    scatSDInvMHistRS.GetXaxis().SetLabelSize(16)
    scatSDInvMHistRS.GetXaxis().SetLabelFont(43)
    scatSDInvMHistRS.GetYaxis().SetTitleOffset(1.45)
    scatSDInvMHistRS.GetYaxis().SetTitleSize(16)
    scatSDInvMHistRS.GetYaxis().SetTitleFont(43)
    scatSDInvMHistRS.GetYaxis().SetLabelSize(16)
    scatSDInvMHistRS.GetYaxis().SetLabelFont(43)
    scatSDInvMHistRS.SetMarkerColor(rt.kBlue) #Alpha(4, 0.35)
    scatSDInvMHistRS.Draw("COLZ")
    rt.gStyle.SetOptStat(0)
    
    #c5A.cd()
    #rt.gPad.Draw()
    if draw :
        c5A.cd()
        rt.gPad.Draw()
    if saveplot :
        c5A.SaveAs("InvMvssdmass_W31_v1.png")
        
    # Costheta vs SD condition
    c5B = rt.TCanvas("c5B", "c5B", 800, 400)   # 800 wide, 400 tall
    c5B.Divide(2,1)
    c5B.cd(1)
    scatPFHistB.GetYaxis().SetTitle("Soft drop condition")
    scatPFHistB.GetXaxis().SetTitle("Inv mass [GeV/c^{2}]")
    scatPFHistB.GetXaxis().CenterTitle(0)
    scatPFHistB.GetXaxis().SetTitleSize(16)
    scatPFHistB.GetXaxis().SetTitleFont(43)
    scatPFHistB.GetXaxis().SetLabelSize(16)
    scatPFHistB.GetXaxis().SetLabelFont(43)
    scatPFHistB.GetYaxis().SetTitleOffset(1.45)
    scatPFHistB.GetYaxis().SetTitleSize(16)
    scatPFHistB.GetYaxis().SetTitleFont(43)
    scatPFHistB.GetYaxis().SetLabelSize(16)
    scatPFHistB.GetYaxis().SetLabelFont(43)
    scatPFHistB.SetMarkerColor(rt.kRed)
    scatPFHistB.Draw("COLZ")
    rt.gStyle.SetOptStat(0)
    
    c5B.cd(2)
    scatPFHistRS.GetYaxis().SetTitle("Soft drop condition")
    scatPFHistRS.GetXaxis().SetTitle("Inv mass [GeV/c^{2}]")
    scatPFHistRS.GetXaxis().CenterTitle(0)
    scatPFHistRS.GetXaxis().SetTitleSize(16)
    scatPFHistRS.GetXaxis().SetTitleFont(43)
    scatPFHistRS.GetXaxis().SetLabelSize(16)
    scatPFHistRS.GetXaxis().SetLabelFont(43)
    scatPFHistRS.GetYaxis().SetTitleOffset(1.45)
    scatPFHistRS.GetYaxis().SetTitleSize(16)
    scatPFHistRS.GetYaxis().SetTitleFont(43)
    scatPFHistRS.GetYaxis().SetLabelSize(16)
    scatPFHistRS.GetYaxis().SetLabelFont(43)
    scatPFHistRS.SetMarkerColor(rt.kBlue) #Alpha(4, 0.35)
    scatPFHistRS.Draw("COLZ")
    rt.gStyle.SetOptStat(0)
    
    c5B.cd()
    rt.gPad.Draw()
    if draw :
        c5B.cd()
        rt.gPad.Draw()
    if saveplot :
        c5B.SaveAs("thetavsSDcond_W32_v0.png")






    
    # Pt plot :
    
    c5a = rt.TCanvas("c5a", "c5a", 1200, 600)
    
    c5a.Divide(2,2)


    c5a.cd(1)
    scatPtHistB.SetTitleSize(20)
    scatPtHistB.SetTitleFont(43)
    scatPtHistB.GetYaxis().SetTitle("Pt [GeV/c]")
    scatPtHistB.GetXaxis().SetTitle("Cos #theta^{*}")
    scatPtHistB.GetXaxis().CenterTitle()
    scatPtHistB.GetXaxis().SetTitleOffset(1.95)
    scatPtHistB.GetXaxis().SetTitleSize(16)
    scatPtHistB.GetXaxis().SetTitleFont(43)
    scatPtHistB.GetXaxis().SetLabelSize(16)
    scatPtHistB.GetXaxis().SetLabelFont(43)
    scatPtHistB.GetYaxis().SetTitleOffset(1.95)
    scatPtHistB.GetYaxis().SetTitleSize(16)
    scatPtHistB.GetYaxis().SetTitleFont(43)
    scatPtHistB.GetYaxis().SetLabelSize(16)
    scatPtHistB.GetYaxis().SetLabelFont(43)
    scatPtHistB.SetMarkerColor(rt.kRed)
    scatPtHistB.Draw("COLZ")
    rt.gStyle.SetOptStat(0)
    

    c5a.cd(2)
    scatPtHistRS.GetYaxis().SetTitle("Pt [GeV/c]")
    scatPtHistRS.GetXaxis().SetTitle("Cos #theta^{*}")
    scatPtHistRS.GetXaxis().CenterTitle()
    scatPtHistRS.GetXaxis().SetTitleOffset(1.95)
    scatPtHistRS.GetXaxis().SetTitleSize(16)
    scatPtHistRS.GetXaxis().SetTitleFont(43)
    scatPtHistRS.GetXaxis().SetLabelSize(16)
    scatPtHistRS.GetXaxis().SetLabelFont(43)
    scatPtHistRS.GetYaxis().SetTitleOffset(1.95)
    scatPtHistRS.GetYaxis().SetTitleSize(16)
    scatPtHistRS.GetYaxis().SetTitleFont(43)
    scatPtHistRS.GetYaxis().SetLabelSize(16)
    scatPtHistRS.GetYaxis().SetLabelFont(43)
    scatPtHistRS.SetMarkerColor(rt.kBlue) #Alpha(4, 0.35)
    scatPtHistRS.Draw("COLZ")
    rt.gStyle.SetOptStat(0)
    
    c5a.cd(3)
    scatPtHistBV.GetYaxis().SetTitle("Pt [GeV/c]")
    scatPtHistBV.GetXaxis().SetTitle("Cos #theta^{*}")
    scatPtHistBV.GetXaxis().CenterTitle()
    scatPtHistBV.GetXaxis().SetTitleOffset(1.95)
    scatPtHistBV.GetXaxis().SetTitleSize(16)
    scatPtHistBV.GetXaxis().SetTitleFont(43)
    scatPtHistBV.GetXaxis().SetLabelSize(16)
    scatPtHistBV.GetXaxis().SetLabelFont(43)
    scatPtHistBV.GetYaxis().SetTitleOffset(1.95)
    scatPtHistBV.GetYaxis().SetTitleSize(16)
    scatPtHistBV.GetYaxis().SetTitleFont(43)
    scatPtHistBV.GetYaxis().SetLabelSize(16)
    scatPtHistBV.GetYaxis().SetLabelFont(43)
    scatPtHistBV.SetMarkerColor(rt.kRed-3)
    scatPtHistBV.Draw("COLZ")
    rt.gStyle.SetOptStat(0)
    

    c5a.cd(4)
    scatPtHistRSV.GetYaxis().SetTitle("Pt [GeV/c]")
    scatPtHistRSV.GetXaxis().SetTitle("Cos #theta^{*}")
    scatPtHistRSV.GetXaxis().CenterTitle()
    scatPtHistRSV.GetXaxis().SetTitleOffset(1.95)
    scatPtHistRSV.GetXaxis().SetTitleSize(16)
    scatPtHistRSV.GetXaxis().SetTitleFont(43)
    scatPtHistRSV.GetXaxis().SetLabelSize(16)
    scatPtHistRSV.GetXaxis().SetLabelFont(43)
    scatPtHistRSV.GetYaxis().SetTitleOffset(1.95)
    scatPtHistRSV.GetYaxis().SetTitleSize(16)
    scatPtHistRSV.GetYaxis().SetTitleFont(43)
    scatPtHistRSV.GetYaxis().SetLabelSize(16)
    scatPtHistRSV.GetYaxis().SetLabelFont(43)
    scatPtHistRSV.SetMarkerColor(rt.kBlue-3) #Alpha(4, 0.35)
    scatPtHistRSV.Draw("COLZ")
    rt.gStyle.SetOptStat(0)

    #c5a.cd()
    #rt.gPad.Draw()
    if draw :
        rt.gPad.Draw()
    if saveplot :
        c5a.SaveAs("thetavspt_W31_v1_Dau1V.png")

    # Eta plot
    
    c5b = rt.TCanvas("c5b", "c5b", 1200, 600)
    c5b.Divide(2,2)
    
    c5b.cd(1)
    scatEtaHistB.GetYaxis().SetTitle("#eta")
    scatEtaHistB.GetXaxis().SetTitle("Cos #theta^{*}")
    scatEtaHistB.GetXaxis().CenterTitle()
    scatEtaHistB.GetXaxis().SetTitleOffset(1.95)
    scatEtaHistB.GetXaxis().SetTitleSize(16)
    scatEtaHistB.GetXaxis().SetTitleFont(43)
    scatEtaHistB.GetXaxis().SetLabelSize(16)
    scatEtaHistB.GetXaxis().SetLabelFont(43)
    scatEtaHistB.GetYaxis().SetTitleOffset(.95)
    scatEtaHistB.GetYaxis().SetTitleSize(16)
    scatEtaHistB.GetYaxis().SetTitleFont(43)
    scatEtaHistB.GetYaxis().SetLabelSize(16)
    scatEtaHistB.GetYaxis().SetLabelFont(43)
    scatEtaHistB.SetMarkerColor(rt.kRed)
    scatEtaHistB.Draw("COLZ")
    rt.gStyle.SetOptStat(0)
    

    c5b.cd(2)
    scatEtaHistRS.GetYaxis().SetTitle("#eta")
    scatEtaHistRS.GetXaxis().SetTitle("Cos #theta^{*}")
    scatEtaHistRS.GetXaxis().CenterTitle()
    scatEtaHistRS.GetXaxis().SetTitleOffset(1.95)
    scatEtaHistRS.GetXaxis().SetTitleSize(16)
    scatEtaHistRS.GetXaxis().SetTitleFont(43)
    scatEtaHistRS.GetXaxis().SetLabelSize(16)
    scatEtaHistRS.GetXaxis().SetLabelFont(43)
    scatEtaHistRS.GetYaxis().SetTitleOffset(.95)
    scatEtaHistRS.GetYaxis().SetTitleSize(16)
    scatEtaHistRS.GetYaxis().SetTitleFont(43)
    scatEtaHistRS.GetYaxis().SetLabelSize(16)
    scatEtaHistRS.GetYaxis().SetLabelFont(43)
    scatEtaHistRS.SetMarkerColor(rt.kBlue) #Alpha(4, 0.35)
    scatEtaHistRS.Draw("COLZ")
    rt.gStyle.SetOptStat(0)
    
    c5b.cd(3)
    scatEtaHistBV.GetYaxis().SetTitle("#eta")
    scatEtaHistBV.GetXaxis().SetTitle("Cos #theta^{*}")
    scatEtaHistBV.GetXaxis().CenterTitle()
    scatEtaHistBV.GetXaxis().SetTitleOffset(1.95)
    scatEtaHistBV.GetXaxis().SetTitleSize(16)
    scatEtaHistBV.GetXaxis().SetTitleFont(43)
    scatEtaHistBV.GetXaxis().SetLabelSize(16)
    scatEtaHistBV.GetXaxis().SetLabelFont(43)
    scatEtaHistBV.GetYaxis().SetTitleOffset(.95)
    scatEtaHistBV.GetYaxis().SetTitleSize(16)
    scatEtaHistBV.GetYaxis().SetTitleFont(43)
    scatEtaHistBV.GetYaxis().SetLabelSize(16)
    scatEtaHistBV.GetYaxis().SetLabelFont(43)
    scatEtaHistBV.SetMarkerColor(rt.kRed-3)
    scatEtaHistBV.Draw("COLZ")
    rt.gStyle.SetOptStat(0)
    

    c5b.cd(4)
    scatEtaHistRSV.GetYaxis().SetTitle("#eta")
    scatEtaHistRSV.GetXaxis().SetTitle("Cos #theta^{*}")
    scatEtaHistRSV.GetXaxis().CenterTitle()
    scatEtaHistRSV.GetXaxis().SetTitleOffset(1.95)
    scatEtaHistRSV.GetXaxis().SetTitleSize(16)
    scatEtaHistRSV.GetXaxis().SetTitleFont(43)
    scatEtaHistRSV.GetXaxis().SetLabelSize(16)
    scatEtaHistRSV.GetXaxis().SetLabelFont(43)
    scatEtaHistRSV.GetYaxis().SetTitleOffset(.95)
    scatEtaHistRSV.GetYaxis().SetTitleSize(16)
    scatEtaHistRSV.GetYaxis().SetTitleFont(43)
    scatEtaHistRSV.GetYaxis().SetLabelSize(16)
    scatEtaHistRSV.GetYaxis().SetLabelFont(43)
    scatEtaHistRSV.SetMarkerColor(rt.kBlue-3) #Alpha(4, 0.35)
    scatEtaHistRSV.Draw("COLZ")
    rt.gStyle.SetOptStat(0)

    #c5b.cd()
    #rt.gPad.Draw()
    if draw :
        rt.gPad.Draw()
    if saveplot :
        c5b.SaveAs("thetavseta_W31_v1_Dau1V.png")

    # Energy plot
    c5c = rt.TCanvas("c5c", "c5c", 1200, 600)
    c5c.Divide(2,2)
    
    c5c.cd(1)
    scatEHistB.GetYaxis().SetTitle("Energy [GeV]")
    scatEHistB.GetXaxis().SetTitle("Cos #theta^{*}")
    scatEHistB.GetXaxis().CenterTitle()
    scatEHistB.GetXaxis().SetTitleOffset(1.95)
    scatEHistB.GetXaxis().SetTitleSize(16)
    scatEHistB.GetXaxis().SetTitleFont(43)
    scatEHistB.GetXaxis().SetLabelSize(16)
    scatEHistB.GetXaxis().SetLabelFont(43)
    scatEHistB.GetYaxis().SetTitleOffset(1.95)
    scatEHistB.GetYaxis().SetTitleSize(16)
    scatEHistB.GetYaxis().SetTitleFont(43)
    scatEHistB.GetYaxis().SetLabelSize(16)
    scatEHistB.GetYaxis().SetLabelFont(43)
    scatEHistB.SetMarkerColor(rt.kRed)
    scatEHistB.Draw("COLZ")
    rt.gStyle.SetOptStat(0)


    c5c.cd(2)
    scatEHistRS.GetYaxis().SetTitle("Energy [GeV]")
    scatEHistRS.GetXaxis().SetTitle("Cos #theta^{*}")
    scatEHistRS.GetXaxis().CenterTitle()
    scatEHistRS.GetXaxis().SetTitleOffset(1.95)
    scatEHistRS.GetXaxis().SetTitleSize(16)
    scatEHistRS.GetXaxis().SetTitleFont(43)
    scatEHistRS.GetXaxis().SetLabelSize(16)
    scatEHistRS.GetXaxis().SetLabelFont(43)
    scatEHistRS.GetYaxis().SetTitleOffset(1.95)
    scatEHistRS.GetYaxis().SetTitleSize(16)
    scatEHistRS.GetYaxis().SetTitleFont(43)
    scatEHistRS.GetYaxis().SetLabelSize(16)
    scatEHistRS.GetYaxis().SetLabelFont(43)
    scatEHistRS.SetMarkerColor(rt.kBlue) #Alpha(4, 0.35)
    scatEHistRS.Draw("COLZ")
    rt.gStyle.SetOptStat(0)
    
    c5c.cd(3)
    scatEHistBV.GetYaxis().SetTitle("Energy [GeV]")
    scatEHistBV.GetXaxis().SetTitle("Cos #theta^{*}")
    scatEHistBV.GetXaxis().CenterTitle()
    scatEHistBV.GetXaxis().SetTitleOffset(1.95)
    scatEHistBV.GetXaxis().SetTitleSize(16)
    scatEHistBV.GetXaxis().SetTitleFont(43)
    scatEHistBV.GetXaxis().SetLabelSize(16)
    scatEHistBV.GetXaxis().SetLabelFont(43)
    scatEHistBV.GetYaxis().SetTitleOffset(1.95)
    scatEHistBV.GetYaxis().SetTitleSize(16)
    scatEHistBV.GetYaxis().SetTitleFont(43)
    scatEHistBV.GetYaxis().SetLabelSize(16)
    scatEHistBV.GetYaxis().SetLabelFont(43)
    scatEHistBV.SetMarkerColor(rt.kRed-3)
    scatEHistBV.Draw("COLZ")
    rt.gStyle.SetOptStat(0)


    c5c.cd(4)
    scatEHistRSV.GetYaxis().SetTitle("Energy [GeV]")
    scatEHistRSV.GetXaxis().SetTitle("Cos #theta^{*}")
    scatEHistRSV.GetXaxis().CenterTitle()
    scatEHistRSV.GetXaxis().SetTitleOffset(1.95)
    scatEHistRSV.GetXaxis().SetTitleSize(16)
    scatEHistRSV.GetXaxis().SetTitleFont(43)
    scatEHistRSV.GetXaxis().SetLabelSize(16)
    scatEHistRSV.GetXaxis().SetLabelFont(43)
    scatEHistRSV.GetYaxis().SetTitleOffset(1.95)
    scatEHistRSV.GetYaxis().SetTitleSize(16)
    scatEHistRSV.GetYaxis().SetTitleFont(43)
    scatEHistRSV.GetYaxis().SetLabelSize(16)
    scatEHistRSV.GetYaxis().SetLabelFont(43)
    scatEHistRSV.SetMarkerColor(rt.kBlue-3) #Alpha(4, 0.35)
    scatEHistRSV.Draw("COLZ")
    rt.gStyle.SetOptStat(0)

    #c5c.cd()
    #rt.gPad.Draw()
    if draw :
        rt.gPad.Draw()
    if saveplot :
        c5c.SaveAs("thetavsenergy_W31_v1_Dau1V.png")


    #-----------------------------------------------------------------------------#
    # tau21 plots :



    c6 = rt.TCanvas("c6", "c6", 800, 400)   # 800 wide, 400 tall
    c6.Divide(2,1)
    c6.cd(1)
    #rt.gStyle.SetPalette()
    scatTSDHistB.GetYaxis().SetTitle("SD mass [GeV/c^{2}]")
    scatTSDHistB.GetXaxis().SetTitle("tau21")
    scatTSDHistB.GetXaxis().CenterTitle()
    scatTSDHistB.GetXaxis().SetTitleSize(16)
    scatTSDHistB.GetXaxis().SetTitleFont(43)
    scatTSDHistB.GetXaxis().SetLabelSize(16)
    scatTSDHistB.GetXaxis().SetLabelFont(43)
    scatTSDHistB.GetYaxis().SetTitleOffset(1.45)
    scatTSDHistB.GetYaxis().SetTitleSize(16)
    scatTSDHistB.GetYaxis().SetTitleFont(43)
    scatTSDHistB.GetYaxis().SetLabelSize(16)
    scatTSDHistB.GetYaxis().SetLabelFont(43)
    scatTSDHistB.SetMarkerColor(rt.kRed)
    scatTSDHistB.Draw("COLZ")
    rt.gStyle.SetOptStat(0)
    
    c6.cd(2)
    scatTSDHistRS.GetYaxis().SetTitle("SD mass [GeV/c^{2}]")
    scatTSDHistRS.GetXaxis().SetTitle("tau21")
    scatTSDHistRS.GetXaxis().CenterTitle()
    scatTSDHistRS.GetXaxis().SetTitleSize(16)
    scatTSDHistRS.GetXaxis().SetTitleFont(43)
    scatTSDHistRS.GetXaxis().SetLabelSize(16)
    scatTSDHistRS.GetXaxis().SetLabelFont(43)
    scatTSDHistRS.GetYaxis().SetTitleOffset(1.45)
    scatTSDHistRS.GetYaxis().SetTitleSize(16)
    scatTSDHistRS.GetYaxis().SetTitleFont(43)
    scatTSDHistRS.GetYaxis().SetLabelSize(16)
    scatTSDHistRS.GetYaxis().SetLabelFont(43)
    scatTSDHistRS.SetMarkerColor(rt.kBlue) #Alpha(4, 0.35)
    scatTSDHistRS.Draw("COLZ")
    rt.gStyle.SetOptStat(0)
    
    #c6.cd()
    #rt.gPad.Draw()
    if draw :
        c6.cd()
        rt.gPad.Draw()
    if saveplot :
        c6.cd()
        c6.SaveAs("tau21vssdmass_W32_v0.png")


    # Invariant Mass
    c6A = rt.TCanvas("c6A", "c6A", 800, 400)   # 800 wide, 400 tall
    c6A.Divide(2,1)
    c6A.cd(1)
    #rt.gStyle.SetPalette()
    scatTInvMHistB.GetYaxis().SetTitle("Inv mass [GeV/c^{2}]")
    scatTInvMHistB.GetXaxis().SetTitle("tau21")
    scatTInvMHistB.GetXaxis().CenterTitle()
    scatTInvMHistB.GetXaxis().SetTitleSize(16)
    scatTInvMHistB.GetXaxis().SetTitleFont(43)
    scatTInvMHistB.GetXaxis().SetLabelSize(16)
    scatTInvMHistB.GetXaxis().SetLabelFont(43)
    scatTInvMHistB.GetYaxis().SetTitleOffset(1.45)
    scatTInvMHistB.GetYaxis().SetTitleSize(16)
    scatTInvMHistB.GetYaxis().SetTitleFont(43)
    scatTInvMHistB.GetYaxis().SetLabelSize(16)
    scatTInvMHistB.GetYaxis().SetLabelFont(43)
    scatTInvMHistB.SetMarkerColor(rt.kRed)
    scatTInvMHistB.Draw("COLZ")
    rt.gStyle.SetOptStat(0)
    
    c6A.cd(2)
    scatTInvMHistRS.GetYaxis().SetTitle("Inv mass [GeV/c^{2}]")
    scatTInvMHistRS.GetXaxis().SetTitle("tau21")
    scatTInvMHistRS.GetXaxis().CenterTitle()
    scatTInvMHistRS.GetXaxis().SetTitleSize(16)
    scatTInvMHistRS.GetXaxis().SetTitleFont(43)
    scatTInvMHistRS.GetXaxis().SetLabelSize(16)
    scatTInvMHistRS.GetXaxis().SetLabelFont(43)
    scatTInvMHistRS.GetYaxis().SetTitleOffset(1.45)
    scatTInvMHistRS.GetYaxis().SetTitleSize(16)
    scatTInvMHistRS.GetYaxis().SetTitleFont(43)
    scatTInvMHistRS.GetYaxis().SetLabelSize(16)
    scatTInvMHistRS.GetYaxis().SetLabelFont(43)
    scatTInvMHistRS.SetMarkerColor(rt.kBlue) #Alpha(4, 0.35)
    scatTInvMHistRS.Draw("COLZ")
    rt.gStyle.SetOptStat(0)
    
    #c6A.cd()
    #rt.gPad.Draw()
    if draw :
        c6A.cd()
        rt.gPad.Draw()
    if saveplot :
        c6A.cd()
        c6A.SaveAs("tau21vsinvmass_W32_v0.png")


    # Pt
    c6B = rt.TCanvas("c6B", "c6B", 800, 400)   # 800 wide, 400 tall
    c6B.Divide(2,1)
    c6B.cd(1)
    #rt.gStyle.SetPalette()
    scatTPtHistB.GetYaxis().SetTitle("Pt [GeV/c]")
    scatTPtHistB.GetXaxis().SetTitle("tau21")
    scatTPtHistB.GetXaxis().CenterTitle()
    scatTPtHistB.GetXaxis().SetTitleSize(16)
    scatTPtHistB.GetXaxis().SetTitleFont(43)
    scatTPtHistB.GetXaxis().SetLabelSize(16)
    scatTPtHistB.GetXaxis().SetLabelFont(43)
    scatTPtHistB.GetYaxis().SetTitleOffset(1.45)
    scatTPtHistB.GetYaxis().SetTitleSize(16)
    scatTPtHistB.GetYaxis().SetTitleFont(43)
    scatTPtHistB.GetYaxis().SetLabelSize(16)
    scatTPtHistB.GetYaxis().SetLabelFont(43)
    scatTPtHistB.SetMarkerColor(rt.kRed)
    scatTPtHistB.Draw("COLZ")
    rt.gStyle.SetOptStat(0)
    
    c6B.cd(2)
    scatTPtHistRS.GetYaxis().SetTitle("Pt [GeV/c]")
    scatTPtHistRS.GetXaxis().SetTitle("tau21")
    scatTPtHistRS.GetXaxis().CenterTitle()
    scatTPtHistRS.GetXaxis().SetTitleSize(16)
    scatTPtHistRS.GetXaxis().SetTitleFont(43)
    scatTPtHistRS.GetXaxis().SetLabelSize(16)
    scatTPtHistRS.GetXaxis().SetLabelFont(43)
    scatTPtHistRS.GetYaxis().SetTitleOffset(1.45)
    scatTPtHistRS.GetYaxis().SetTitleSize(16)
    scatTPtHistRS.GetYaxis().SetTitleFont(43)
    scatTPtHistRS.GetYaxis().SetLabelSize(16)
    scatTPtHistRS.GetYaxis().SetLabelFont(43)
    scatTPtHistRS.SetMarkerColor(rt.kBlue) #Alpha(4, 0.35)
    scatTPtHistRS.Draw("COLZ")
    rt.gStyle.SetOptStat(0)
    
    #c6B.cd()
    #rt.gPad.Draw()
    if draw :
        c6B.cd()
        rt.gPad.Draw()
    if saveplot :
        c6B.cd()
        c6B.SaveAs("tau21vspt_W32_v0.png")


    # costheta
    c6C = rt.TCanvas("c6C", "c6C", 800, 400)   # 800 wide, 400 tall
    c6C.Divide(2,1)
    c6C.cd(1)
    #rt.gStyle.SetPalette()
    scatTHistB.GetYaxis().SetTitle("tau21")
    scatTHistB.GetXaxis().SetTitle("Cos #theta^{*}")
    scatTHistB.GetXaxis().CenterTitle()
    scatTHistB.GetXaxis().SetTitleSize(16)
    scatTHistB.GetXaxis().SetTitleFont(43)
    scatTHistB.GetXaxis().SetLabelSize(16)
    scatTHistB.GetXaxis().SetLabelFont(43)
    scatTHistB.GetYaxis().SetTitleOffset(1.45)
    scatTHistB.GetYaxis().SetTitleSize(16)
    scatTHistB.GetYaxis().SetTitleFont(43)
    scatTHistB.GetYaxis().SetLabelSize(16)
    scatTHistB.GetYaxis().SetLabelFont(43)
    scatTHistB.SetMarkerColor(rt.kRed)
    scatTHistB.Draw("COLZ")
    rt.gStyle.SetOptStat(0)

    c6C.cd(2)
    scatTHistRS.GetYaxis().SetTitle("tau21")
    scatTHistRS.GetXaxis().SetTitle("Cos #theta^{*}")
    scatTHistRS.GetXaxis().CenterTitle()
    scatTHistRS.GetXaxis().SetTitleSize(16)
    scatTHistRS.GetXaxis().SetTitleFont(43)
    scatTHistRS.GetXaxis().SetLabelSize(16)
    scatTHistRS.GetXaxis().SetLabelFont(43)
    scatTHistRS.GetYaxis().SetTitleOffset(1.45)
    scatTHistRS.GetYaxis().SetTitleSize(16)
    scatTHistRS.GetYaxis().SetTitleFont(43)
    scatTHistRS.GetYaxis().SetLabelSize(16)
    scatTHistRS.GetYaxis().SetLabelFont(43)
    scatTHistRS.SetMarkerColor(rt.kBlue) #Alpha(4, 0.35)
    scatTHistRS.Draw("COLZ")
    rt.gStyle.SetOptStat(0)
    
    #c6C.cd()
    #rt.gPad.Draw()
    if draw :
        c6C.cd()
        rt.gPad.Draw()
    if saveplot :
        c6C.cd()
        c6C.SaveAs("thetavstau21_W32_v0.png")


# started 25.07.2018

#=============================================================================#
#-----------------------------------------------------------------------------#
#     
#                         ML toy Neural Network
#     
#-----------------------------------------------------------------------------#
#=============================================================================#

# 1D = dau1 features, 1D2 = dau2 features, 2D is both at same time.

# starting with pandas 
# creating dataframe from numpyarrays
# variables : VE, VPt, VEta, VPhi, VPx, VPy, VPz, DE, DPt, DEta, DPhi, DPx, DPy, DPz, DCT

# FOR BULK_sample :

W_B  = GetWeights(h_theta_cm1_B, DCT_B)
W2_B = GetWeights(h_theta_cm2_B, DCT2_B)

df_B = pd.DataFrame({'VE_B':VE_B, 'VPt_B':VPt_B, 'VEta_B':VEta_B, 'VPhi_B':VPhi_B, 'VPx_B':VPx_B, 'VPy_B' :VPy_B,   'VPz_B':VPz_B, 'DE_B':DE_B, 'DPt_B':DPt_B, 'DEta_B':DEta_B, 'DPhi_B':DPhi_B, 'DPx_B':DPx_B, 'DPy_B' :DPy_B,   'DPz_B':DPz_B,'DE2_B':DE2_B, 'DPt2_B':DPt2_B, 'DEta2_B':DEta2_B, 'DPhi2_B':DPhi2_B, 'DPx2_B':DPx2_B, 'DPy2_B' :DPy2_B,   'DPz2_B':DPz2_B, 'DCT_B':DCT_B, 'DCT2_B':DCT2_B, 'DT_B':DT_B, 'B_B':B_B, 'W_B':W_B, 'W2_B':W2_B, 'Tau21_B': Tau21_B})

par_detector_1D_B  = ['VE_B', 'VPt_B', 'VEta_B', 'VPhi_B', 'DE_B', 'DPt_B', 'DEta_B', 'DPhi_B']#, 'B_B']
par_cartesian_1D_B = ['VE_B', 'VPx_B', 'VPy_B',  'VPz_B',  'DE_B', 'DPx_B', 'DPy_B',  'DPz_B']#, 'B_B' ]
par_detector_1D2_B  = ['VE_B', 'VPt_B', 'VEta_B', 'VPhi_B', 'DE2_B', 'DPt2_B', 'DEta2_B', 'DPhi2_B']#, 'B_B']
par_cartesian_1D2_B = ['VE_B', 'VPx_B', 'VPy_B',  'VPz_B',  'DE2_B', 'DPx2_B', 'DPy2_B',  'DPz2_B']#, 'B_B' ]


par_detector_2D_B  = ['VE_B', 'VPt_B', 'VEta_B', 'VPhi_B', 'DE_B', 'DPt_B', 'DEta_B', 'DPhi_B', 'DE2_B', 'DPt2_B', 'DEta2_B', 'DPhi2_B', 'B_B', 'DCT_B', 'DCT2_B', 'Tau21_B']
par_cartesian_2D_B = ['VE_B', 'VPx_B', 'VPy_B',  'VPz_B',  'DE_B', 'DPx_B', 'DPy_B',  'DPz_B',  'DE2_B', 'DPx2_B', 'DPy2_B',  'DPz2_B', 'B_B', 'DCT_B', 'DCT2_B', 'Tau21_B' ]

par_Y1D_B         = ['DCT_B', 'W_B']
par_Y1D2_B        = ['DCT2_B', 'W2_B']

dfdet1D_B = df_B[par_detector_1D_B]
dfcar1D_B = df_B[par_cartesian_1D_B]
dfY1D_B   = df_B[par_Y1D_B]
dfdet1D2_B = df_B[par_detector_1D2_B]
dfcar1D2_B = df_B[par_cartesian_1D2_B]
dfY1D2_B  = df_B[par_Y1D2_B]

# for classification :

Label_B = np.ones(len(W_B))
Label2_B = np.ones(len(W2_B))

dfCLSdet2D_B = df_B[par_detector_2D_B]
dfCLScar2D_B = df_B[par_cartesian_2D_B]

#----------------
# FOR RS_sample :

W  = GetWeights(h_theta_cm1_RS, DCT)
W2 = GetWeights(h_theta_cm2_RS, DCT2)

df = pd.DataFrame({'VE':VE, 'VPt':VPt, 'VEta':VEta, 'VPhi':VPhi, 'VPx':VPx, 'VPy' :VPy,   'VPz':VPz, 'DE':DE, 'DPt':DPt, 'DEta': DEta, 'DPhi': DPhi, 'DPx':DPx, 'DPy' :DPy,   'DPz':DPz,'DE2':DE2, 'DPt2':DPt2, 'DEta2':DEta2, 'DPhi2':DPhi2, 'DPx2':DPx2, 'DPy2' :DPy2,   'DPz2':DPz2, 'DCT':DCT, 'DCT2':DCT2, 'DT':DT, 'B':B, 'W':W, 'W2':W2, 'Tau21_A':Tau21_A})

par_detector_1D  = ['VE', 'VPt', 'VEta', 'VPhi', 'DE', 'DPt', 'DEta', 'DPhi']#, 'B']
par_cartesian_1D = ['VE', 'VPx', 'VPy',  'VPz',  'DE', 'DPx', 'DPy',  'DPz']#, 'B' ]
par_detector_1D2  = ['VE', 'VPt', 'VEta', 'VPhi', 'DE2', 'DPt2', 'DEta2', 'DPhi2']#, 'B']
par_cartesian_1D2 = ['VE', 'VPx', 'VPy',  'VPz',  'DE2', 'DPx2', 'DPy2',  'DPz2']#, 'B' ]


par_detector_2D  = ['VE', 'VPt', 'VEta', 'VPhi', 'DE', 'DPt', 'DEta', 'DPhi', 'DE2', 'DPt2', 'DEta2', 'DPhi2', 'B', 'DCT', 'DCT2', 'Tau21_A']
par_cartesian_2D = ['VE', 'VPx', 'VPy',  'VPz',  'DE', 'DPx', 'DPy',  'DPz',  'DE2', 'DPx2', 'DPy2',  'DPz2', 'B' , 'DCT', 'DCT2', 'Tau21_A']

par_Y1D         = ['DCT', 'W']
par_Y1D2        = ['DCT2', 'W2']

dfdet1D = df[par_detector_1D]
dfcar1D = df[par_cartesian_1D]
dfY1D   = df[par_Y1D]
dfdet1D2 = df[par_detector_1D2]
dfcar1D2 = df[par_cartesian_1D2]
dfY1D2  = df[par_Y1D2]


# for classification :

Label = np.zeros(len(dfCLSdet2D_B))
Label2 = np.zeros(len(dfCLSdet2D_B))

dfCLSdet2D = df[par_detector_2D].loc[1:len(dfCLSdet2D_B),:]
dfCLScar2D = df[par_cartesian_2D].loc[1:len(dfCLScar2D_B), :]



#  only 1 dau info
#X_raw = dfcar1D.values
#Y = dfY1D.values

BULK = True
RS   = False
RGS  = True
CLS  = False

# FOR BULK_sample : both dau 1 & 2 data
if BULK :
    X_raw = np.concatenate((dfcar1D_B, dfcar1D2_B), axis = 0)
    Y = np.concatenate((dfY1D_B, dfY1D2_B), axis = 0)
    DIM = 8
# both dau 1 & 2 data
if RS :
    X_raw = np.concatenate((dfcar1D, dfcar1D2), axis = 0)
    Y = np.concatenate((dfY1D, dfY1D2), axis = 0)
    DIM = 8

if CLS :
    X_raw = np.concatenate((dfCLScar2D_B, dfCLScar2D), axis = 0)
    Y = np.concatenate((Label_B,Label), axis = 0)
    DIM = 16

# ----------------------------------------------------------------------------#
#                    reshaping variables for training


scaler = MinMaxScaler(feature_range=(0,1))
X = scaler.fit_transform((X_raw).reshape(-1,DIM))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.5)

if RGS :
    W = Y_train[:,1]    # sample_weights


# ----------------------------------------------------------------------------#
#                   building as regression model 

if RGS :
    Ne = 20     # set the number of epochs

    model = Sequential([
            Dense(8, input_shape=(DIM,), kernel_initializer='normal', activation= 'tanh'),
            #       Dropout(rate = 0.2),
            Dense(32, kernel_initializer='normal', activation = 'relu'),
            #       Dropout(rate = 0.2),
            Dense(64, kernel_initializer='normal', activation = 'relu'),
            Dense(128, kernel_initializer='normal', activation = 'relu'),
#            Dense(256, kernel_initializer='normal', activation = 'relu'),
            Dense(1,  kernel_initializer='normal')]) #, activation = 'linear')])       
    model.summary()
    #model.compile(loss='mean_squared_error', optimizer='adam')
    model.compile(loss='mean_squared_error', optimizer='adam')  # Adam the method used the minimize the loss, lr = learning rate is the step it search on your hyperparameters

# ----------------------------------------------------------------------------#
#                   training of the model

    history = model.fit(X_train, Y_train[:,0], sample_weight = W, batch_size = 20, validation_split = 0.1, shuffle = True, epochs = Ne, verbose = 2) #, batch_size = 10

    predictions = model.predict(X_test, verbose=0)#, batch_size = 5)  #batch_size=10,
#predictions = model.predict(Xt, batch_size=10, verbose=0)


# ----------------------------------------------------------------------------#
#                 plotting the prediction in a histogram

    
    if BULK :
        h_theta_cm1_pred= rt.TH1F("Bulk theta CM", "ML predicted cos #theta^{*}", 100, -1, 1)
        h_theta_cm1_real= rt.TH1F("Bulk theta CM", "ML predicted cos#theta^{2}", 100, -1, 1)

    if RS :
        h_theta_cm1_pred= rt.TH1F("RS theta CM", "ML predicted cos #theta^{*}", 100, -1, 1)
        h_theta_cm1_real= rt.TH1F("RS theta CM", "ML predicted cos#theta^{2}", 100, -1, 1)


    for i in Y_test[:,0] :
        h_theta_cm1_real.Fill(i)
    
    for i in predictions :
        h_theta_cm1_pred.Fill(i)
        #   h_theta_cm1_pred.Fill(np.cos(i))

    # Get bin content, can be used for the weighted distribution.
    # weighted distribution
    
    #ratio = []
    #binContent = []
    #binCenter = []
    #Nbin = 100
    #binarray = np.arange(1,Nbin+1,1)
    #total = 0
    #for i in binarray :
    #    number = h_theta_cm1_real.GetBinContent(i)
    #    number2 = h_theta_cm1_pred.GetBinContent(i)
    #    center = h_theta_cm1_real.GetBinCenter(i)
    #    binContent.append(number)
    #    binCenter.append(center)
    #    total += number
    #    ratio.append(number2/number)
    
    #print(np.array(ratio)-1)
    
    #gr = rt.TGraph(100, binCenter, ratio)
  
    
    
    
    #canvas for W rest frame emission angle
    # c7
    c7 = rt.TCanvas("c7","c7",800,600)
    
    # upper plot will be in pad1    
    pad1 = rt.TPad("pad1","pad1", 0, 0.3, 1, 1.0)
    pad1.SetBottomMargin(0)
    pad1.SetGridx()
    pad1.Draw()
    pad1.cd()
    h_theta_cm1_pred.SetStats(0)
    h_theta_cm1_pred.Draw("HIST")
    h_theta_cm1_real.Draw("HIST same")
    
    # add legend in pad1
    leg = rt.TLegend( 0.7449843,0.79038,0.8333,0.8838219)#,NULL,"brNDC")
    leg.SetTextSize(0.04)
    leg.SetLineColor(rt.kWhite)
    leg.SetFillStyle(0)
    leg.AddEntry(h_theta_cm1_pred,"Keras RS","l") #Nent = 40107
    leg.AddEntry(h_theta_cm1_real,"Real RS","l")  #Nent = 38255
    leg.Draw() 
    

    ## TGaxis
    #h_theta_cm1_pred.GetYaxis().SetLabelSize(0.)
    #axis = rt.TGaxis(-5, 20, -5, 220, 20, 220, 510, "")
    #axis.SetLabelFont(43)
    #axis.SetLabelSize(28)
    #axis.Draw()
    

    # used the following tutorial
    # https://root.cern.ch/root/html/tutorials/hist/ratioplot.C.html
    # Finished 01.08.2018
    

    # lower plot will be in pad
    c7.cd()
    pad2 = rt.TPad("pad2", "pad2", 0, 0.05, 1, 0.3)
    pad2.SetTopMargin(0)
    pad2.SetBottomMargin(0.2)
    pad2.SetGridx()
    pad2.Draw()
    pad2.cd()
    

    # define ratio plot
    h3 = h_theta_cm1_pred.Clone("h3")
    h3.SetLineColor(rt.kBlack)
    h3.SetMinimum(0.8)
    h3.SetMaximum(1.2)
    h3.Sumw2()
    h3.SetStats(0)
    h3.Divide(h_theta_cm1_real)
    h3.SetMarkerStyle(21)
    h3.Draw("ep")
    
    #pad2.cd()
    line = rt.TLine(-1, 1, 1, 1)
    line.SetLineColor(rt.kRed)
    line.Draw()
    

    # settings for histograms
    h_theta_cm1_pred.GetYaxis().SetTitle("Entries / 1 deg")
    h_theta_cm1_pred.GetYaxis().SetTitleSize(18)
    h_theta_cm1_pred.GetYaxis().SetTitleFont(43)
    h_theta_cm1_pred.GetYaxis().SetLabelSize(18)
    h_theta_cm1_pred.GetYaxis().SetLabelFont(43)
    #h_theta_cm1_pred.GetYaxis().SetRange(0,900)
    
    h_theta_cm1_pred.GetXaxis().SetTitle("Cos #theta^{*}")
    h_theta_cm1_pred.GetXaxis().CenterTitle()
    h_theta_cm1_pred.GetXaxis().SetTitleSize(18)
    h_theta_cm1_pred.GetXaxis().SetTitleFont(43)
    h_theta_cm1_pred.SetLineColor(rt.kRed)  
    h_theta_cm1_real.SetLineColor(rt.kBlue)
    
    h3.SetTitle("")
    h3.GetYaxis().SetTitle("ratio pred/real")
    h3.GetYaxis().SetTitleSize(18)
    h3.GetYaxis().SetTitleFont(43)
    h3.GetYaxis().SetLabelSize(14)
    h3.GetYaxis().SetLabelFont(43)
    h3.GetXaxis().SetTitle("Cos #theta^{*}")
    h3.GetXaxis().CenterTitle()           
    h3.GetXaxis().SetTitleSize(18)
    h3.GetXaxis().SetTitleFont(43)
    h3.GetXaxis().SetTitleOffset(3.55)
    h3.GetXaxis().SetLabelSize(18)
    h3.GetXaxis().SetLabelFont(43)

    c7.cd()
    rt.gPad.Draw()
    
    if saveplot :
        c7.cd()
        c7.SaveAs("ML_thetastar1_W31_car_v6_D2_ratio_Bulk.png")



    # loss function
    c8 = rt.TCanvas("c8","c8", 800, 600)
    
    x = np.arange(1.0,Ne+1)
    y = np.array(history.history['loss'])
    gr = rt.TGraph(Ne, x, y)
    gr.SetTitle("Model Loss")
    gr.GetYaxis().SetTitleOffset(1.55)
    gr.GetXaxis().SetTitle( 'Epoch' )
    #gr.GetXaxis().CenterTitle()
    gr.GetYaxis().SetTitle( 'Loss (MSE)' )
    #gr.GetYaxis().CenterTitle()
    gr.SetLineColor(4)
    gr.SetMarkerColor(4)
    
    y1 = np.array(history.history['val_loss'])
    gr2 = rt.TGraph(Ne,x, y1)
    gr2.SetLineColor(6)
    gr2.SetMarkerColor(6)
    
    # adding a legend
    leg = rt.TLegend( 0.6449843,0.79038,0.8333,0.8838219)#,NULL,"brNDC")
    leg.SetTextSize(0.03)
    leg.SetLineColor(rt.kWhite)
    leg.SetFillStyle(0)
    leg.AddEntry(gr,"loss","l") #Nent = 40107
    leg.AddEntry(gr2,"val_loss","l")  #Nent = 38255
    


    gr.Draw("AC*")
    gr2.Draw("CP")
    leg.Draw() 
    rt.gPad.Draw()
    if saveplot :
        c8.cd()
        c8.SaveAs("ML_loss_W31_car_v6_D2_ratio_Bulk.png")


#-----------------------------------------------------------------------------#
#                   building as classification model 

if CLS :
    Ne = 10     # set the number of epochs

    model = Sequential([
            Dense(8, input_shape=(DIM,), activation= 'relu'),
            Dense(32, activation = 'relu'),
            Dense(64, activation = 'relu'),
            Dense(64, activation = 'relu'),
            Dense(2,  activation = 'softmax')]) #, activation = 'linear')])       
    model.summary()
    
    model.compile(Adam(lr=.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])  # Adam the method used the minimize the loss, lr = learning rate is the step it search on your hyperparameters

# ----------------------------------------------------------------------------#
#                   training of the model
   
    
    
    history = model.fit(X_train, Y_train, batch_size = 10, validation_split = 0.1, shuffle = True, epochs = Ne, verbose = 2) #, batch_size = 10
    predictions = model.predict(X_test, verbose=0)  #batch_size=10,

 
# ----------------------------------------------------------------------------#
#                   plotting the outcome
    
    VBulk = rt.TH1F("Bulk", "Keras value", 100, 0, 1) 
    VRS = rt.TH1F("RS", "Keras value", 100, 0, 1)     
    VBulk_list = []
    VRS_list = []
    for i, entry in enumerate(Y_test) :
        if entry == 1 :
            VBulk.Fill(predictions[i, 1])
            VBulk_list.append(predictions[i, 1])
        elif entry == 0 :
            VRS.Fill(predictions[i, 1])
            VRS_list.append(predictions[i, 1])
 

# ROC-curve and AUC :
    fig, ax = plt.subplots(figsize=(10,6))


    VBulk_hist = ax.hist(VBulk_list, bins = 100, range=(0,1))
    VRS_hist = ax.hist(VRS_list, bins = 100, range=(0,1)) 
    FPR, TPR = calc_ROC(VBulk_hist, VRS_hist)      #
    AUC = metrics.auc(FPR, TPR) 

           
    # Plotting the results
    
    c8 = rt.TCanvas("c8", "c8", 800, 400)   # 800 wide, 400 tall
    c8.Divide(2,1)
    c8.cd(1)
    #rt.gStyle.SetPalette()
    VBulk.GetYaxis().SetTitle("Entries")
    VBulk.GetXaxis().SetTitle("Value")
    VBulk.GetXaxis().CenterTitle(0)
    VBulk.GetXaxis().SetTitleSize(16)
    VBulk.GetXaxis().SetTitleFont(43)
    VBulk.GetXaxis().SetLabelSize(16)
    VBulk.GetXaxis().SetLabelFont(43)
    VBulk.GetYaxis().SetTitleOffset(1.45)
    VBulk.GetYaxis().SetTitleSize(16)
    VBulk.GetYaxis().SetTitleFont(43)
    VBulk.GetYaxis().SetLabelSize(16)
    VBulk.GetYaxis().SetLabelFont(43)
    VBulk.SetLineColor(rt.kRed)
    VBulk.Draw("HIST")
    VRS.SetLineColor(rt.kBlue)
    c8.cd(1)
    VRS.Draw("HIST same")
    rt.gStyle.SetOptStat(0)
    
    leg = rt.TLegend( 0.1449843,0.79038,0.8333,0.8838219)#,NULL,"brNDC")
    leg.SetTextSize(0.04)
    leg.SetLineColor(rt.kWhite)
    leg.SetFillStyle(0)
    leg.AddEntry(VBulk,"Bulk (sgn)","l") #Nent = 40107
    leg.AddEntry(VRS,"RS (bkg)","l")  #Nent = 38255
    leg.Draw() 
    
    c8.cd(2)

    gr = rt.TGraph(100, TPR, FPR )
    gr.SetTitle("ROC currve")
    gr.GetYaxis().SetTitleOffset(1.55)
    gr.GetXaxis().SetTitle( 'Signal efficiency' )
    gr.GetXaxis().SetLimits(0.,1.)
    gr.GetHistogram().SetMaximum(1)
    gr.GetHistogram().SetMinimum(0)
    #gr.GetXaxis().CenterTitle()
    gr.GetYaxis().SetTitle( 'Background acceptance' )
    #gr.GetYaxis().CenterTitle()
    gr.SetLineColor(4)
    gr.SetMarkerColor(4)
    gr.Draw("AC")
    
    #xrandom = np.array([0, 1])
    #yrandom = np.array([0, 1])
    #gr2 = rt.TGraph(2, xrandom, yrandom)
    #gr2.SetLineColor(6)
    #gr2.SetMarkerColor(6)
    #gr2.Draw("AC")
    #leg = rt.TLegend( 0.6449843,0.79038,0.8333,0.8838219)#,NULL,"brNDC")
    #leg.SetTextSize(0.03)
    #leg.SetLineColor(rt.kWhite)
    #leg.SetFillStyle(0)
    #leg.AddEntry(gr,"ROC Curve AUC = ","l")
    
    

    c8.cd()
    rt.gPad.Draw()
    if draw :
        c8.cd()
        rt.gPad.Draw()
    if saveplot :
        c8.cd()
        c8.SaveAs("Keras_clas_W32_v0.png")
 


raw_input('.....Press enter to exit ....')
