// Reproduction/extension driver for AblationGA (the original ablation.cpp was not available).
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <cmath>
#include "TSPInstance.h"
#include "AblationGA.h"
using namespace std;
static vector<string> splitCsv(const string& s){vector<string> o;stringstream ss(s);string it;while(getline(ss,it,','))if(!it.empty())o.push_back(it);return o;}
int main(int argc,char**argv){
  vector<string> ds={"eil51"}, conds={"EM_LS"}; int runs=20,gens=1000; double theta=0.5; string repl="ri",tag="",out="out/runs.csv",dataDir="data"; int seed0=1;
  for(int i=1;i<argc;++i){string a=argv[i];auto nx=[&]{return i+1<argc?string(argv[++i]):string();};
    if(a=="--datasets")ds=splitCsv(nx());else if(a=="--conds")conds=splitCsv(nx());else if(a=="--runs")runs=stoi(nx());
    else if(a=="--gens")gens=stoi(nx());else if(a=="--theta")theta=stod(nx());else if(a=="--repl")repl=nx();
    else if(a=="--tag")tag=nx();else if(a=="--out")out=nx();else if(a=="--data-dir")dataDir=nx();else if(a=="--seed0")seed0=stoi(nx());
    else{cerr<<"unknown "<<a<<"\n";return 1;}}
  g_seReplacement = (repl=="db")?1:0;
  bool newf=!filesystem::exists(out); ofstream f(out,ios::app); if(newf) f<<"condition,instance,run,best,elapsed_ms,best_generation\n";
  for(auto&d:ds){TSPInstance inst; inst.load(dataDir+"/"+d+".tsp");
    for(auto&c:conds){
      for(int r=0;r<runs;++r){
        AblationGAConfig cfg; cfg.generations=gens; cfg.seed=seed0+r; cfg.similarityThreshold=theta;
        cfg.useEssentialMutation=cfg.useLocalSearch=cfg.useSEbeforeLS=cfg.useSEafterLS=false;
        if(c=="None"){}
        else if(c=="EM_only"){cfg.useEssentialMutation=true;}
        else if(c=="LS_only"){cfg.useLocalSearch=true;}
        else if(c=="EM_LS"){cfg.useEssentialMutation=cfg.useLocalSearch=true;}
        else if(c=="EM_SE_LS"){cfg.useEssentialMutation=cfg.useLocalSearch=cfg.useSEbeforeLS=true;}
        else if(c=="EM_LS_SE_both"){cfg.useEssentialMutation=cfg.useLocalSearch=cfg.useSEbeforeLS=cfg.useSEafterLS=true;}
        else if(c=="EM_LS_SE_after"){cfg.useEssentialMutation=cfg.useLocalSearch=cfg.useSEafterLS=true;}
        else if(c=="LS_SE_both"){cfg.useLocalSearch=cfg.useSEbeforeLS=cfg.useSEafterLS=true;}
        else if(c=="LS_SE_after"){cfg.useLocalSearch=cfg.useSEafterLS=true;}
        else {cerr<<"bad cond "<<c<<"\n";return 1;}
        AblationGA ga(inst,cfg); GAResult res=ga.run();
        f<<c<<tag<<","<<d<<","<<r<<","<<res.best<<","<<res.elapsedMs<<","<<res.bestGeneration<<"\n"; f.flush();
        cout<<c<<tag<<" "<<d<<" run "<<r+1<<"/"<<runs<<" best="<<res.best<<" t="<<res.elapsedMs/1000<<"s gen="<<res.bestGeneration<<endl;
      }}}
  return 0;}
