void macro (TString file_name = "tag_1_delphes")
{
	gROOT->LoadMacro ("MyDelphes.C+");
    TString complete_file_name = "/Users/navinmcginnis/physics_programs/analysis/t4_test/Events/run_01/"; complete_file_name+=file_name;     
    complete_file_name+="_events.root";
   TFile *file = new TFile (complete_file_name);
	TTree *tree = (TTree*) file->Get ("Delphes");
	if (!tree)
	{
		std::cout << "couldn't find tree Delphes in file " << file_name << std::endl;
		return;
	};
	MyDelphes t (tree);
 	t.Loop(file_name);
};


