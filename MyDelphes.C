#define MyDelphes_cxx
#include "MyDelphes.h"
#include <TRef.h>
#include <math.h>
#include <TRefArray.h>
#include <TROOT.h>
#include <TLorentzVector.h>
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <iostream>

// modified to accept a file_name

void MyDelphes::Loop(TString file_name)
{
//   In a ROOT session, you can do:
//      root> .L MyDelphes.C
//      root> MyDelphes t
//      root> t.GetEntry(12); // Fill t data members with entry number 12
//      root> t.Show();       // Show values of entry 12
//      root> t.Show(16);     // Read and show values of entry 16
//      root> t.Loop();       // Loop on all entries
//

//     This is the loop skeleton where:
//    jentry is the global entry number in the chain
//    ientry is the entry number in the current Tree
//  Note that the argument to GetEntry must be:
//    jentry for TChain::GetEntry
//    ientry for TTree::GetEntry and TBranch::GetEntry
//
//       To read only selected branches, Insert statements like:
// METHOD1:
//    fChain->SetBranchStatus("*",0);  // disable all branches
//    fChain->SetBranchStatus("branchname",1);  // activate branchname
// METHOD2: replace line
//    fChain->GetEntry(jentry);       //read all branches
//by  b_branchname->GetEntry(ientry); //read only this branch
   if (fChain == 0) return;

    // nentries is the number of events
    
    float weight =0;
    
   Long64_t nentries = fChain->GetEntriesFast();
    
    
    //   open file for writing
    //   std::ofstream os("prova.txt");
    //   os << nentries << endl;
    
    
    // Create the output file that will contain all the histograms
    
    TString output_name = "histograms/hist_";
    output_name+=file_name;
    output_name+=".root";
    TFile out (output_name,"RECREATE");
    
    // Create the histograms
    
    //   TH1 *ptmu1 = new TH1F ("ptmu1", "ptmu1", 100, 0, 200); ptmu1->Sumw2 ();
    //   TH1 *ptmu2 = new TH1F ("ptmu2", "ptmu2", 100, 0, 200); ptmu2->Sumw2 ();
    //   TH1 *ptmu3 = new TH1F ("ptmu3", "ptmu3", 100, 0, 200); ptmu3->Sumw2 ();
    //   TH1 *ptmu4 = new TH1F ("ptmu4", "ptmu4", 100, 0, 20); ptmu4->Sumw2 ();
    TH1 *mW = new TH1F ("mW", "mW", 100, 0, 2000); mW->Sumw2 ();//2
    TH1 *mt = new TH1F ("mt", "mt", 100, 0, 2000); mt->Sumw2 ();//3
    TH1 *mH = new TH1F ("mH", "mH", 100, 0, 2000); mH->Sumw2 ();//6
    TH1 *mt4 = new TH1F ("mt4", "mt4", 100, 0, 2000); mt4->Sumw2 ();//9
    //   TH1 *deltar = new TH1F ("deltar", "DeltaR", 100, 0, 1); deltar->Sumw2 ();
    //   TH1 *mass4 = new TH1F ("mass4", "mass4", 100, 0, 200); mass4->Sumw2 ();
    //   TH1 *mass3 = new TH1F ("mass3", "mass3", 100, 0, 200); mass3->Sumw2 ();
    

   Long64_t nbytes = 0, nb = 0;
    
// Loops over events
   for (Long64_t jentry=0; jentry<nentries;jentry++) {
      Long64_t ientry = LoadTree(jentry);
      if (ientry < 0) break;
      nb = fChain->GetEntry(jentry);   nbytes += nb;
      // if (Cut(ientry) < 0) continue;
       
       weight += Event_Weight[0];
       
       //////////////////////
       // Here begins my code
       //////////////////////
       
       int *njets =  &Jet_size;
       int njetsvalue ;
       njetsvalue=*njets;
       //  cout << njetsvalue << endl;
       
       
       TLorentzVector *jets = new TLorentzVector[njetsvalue];
       
       //  std::vector<ROOT::Math::TLorentzVector<ROOT::Math::PxPyPzE4D<double>>> jets;
       //    TLorentzVector *j;
       for (int iter = 0; iter < njetsvalue; ++ iter)
       {
           jets[iter].SetPtEtaPhiM(Jet_PT[0],Jet_Eta[0],Jet_Phi[0],Jet_Mass[0]);
           //       jets->push_back (j);
       };
       
       if (njetsvalue > 7)
       {
           mW->Fill ((jets[0]+jets[1]).M() , Event_Weight[0]);
           mt->Fill ((jets[0]+jets[1]+jets[2]).M() , Event_Weight[0]);
           mH->Fill ((jets[0]+jets[1]+jets[2]+jets[3]+jets[4]+jets[5]).M() , Event_Weight[0]);
           mt4->Fill ((jets[0]+jets[1]+jets[2]+jets[3]+jets[4]+jets[5]+jets[6]+jets[7]+jets[8]).M() , Event_Weight[0]);
       }
       
       
       
       
       ////////////////////
       // Here ends my code
       ////////////////////
       
       
   }
    
    
    //   cout << "Integrate Weight is: " << weight << endl;
    
    
    // write to the output file
    out.Write ();
}
