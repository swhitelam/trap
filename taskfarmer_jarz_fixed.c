
# include <unistd.h>
# include <stdio.h>                             /* C standard input and output */
# include <stdlib.h>                            /* standard library */
# include <math.h>                              /* math library */
# include <iostream>
# include <iomanip>
# include <time.h>


using namespace std;
using std::cout;
using std::endl;
using std::cin;

#include <fstream>
using std::ifstream;
#include <cstdlib>

char st[256];
long int rn_seed;
long ltime = time(NULL);
long int rn_seed2 = ltime;

const int n_traj=50;

//functions void
void setup(void);
void jobcomplete(int i);
void make_outputs(void);
void make_output(const char *varname);


//functions int
int q_jobs_done(void);

//functions bool
bool q_file_exist(const char *fileName);


int main(void){

//seed RN generator
srand48(rn_seed2);

//setup
setup();

//monitor
while(q_jobs_done()==0){sleep(30);make_outputs();}

return 0;

}

void setup(void){

int n;

long int rn_seed;
 
//loop through parameter cases
for(n=0;n<n_traj;n++){

//directory
snprintf(st,sizeof(st),"mkdir iteration_%d",n);
system(st);

//parameters for etna
snprintf(st,sizeof(st),"swarm_%d.sh",n);
ofstream soutput(st,ios::out);

soutput << "#!/usr/bin/env bash" << endl;
snprintf(st,sizeof(st),"#SBATCH --job-name=swarm_%d",n);
soutput << st << endl;
soutput << "#SBATCH --partition=etna-shared" << endl;
//soutput << "#SBATCH --partition=etna" << endl;
soutput << "#SBATCH --account=nano" << endl;
soutput << "#SBATCH --qos=normal" << endl;
snprintf(st,sizeof(st),"#SBATCH --nodes=%d",1);
soutput << st << endl;
soutput << "#SBATCH --time=48:00:00" << endl;
//soutput << "#SBATCH --mem=1000mb" << endl;
soutput << "#SBATCH --ntasks=1" << endl;
snprintf(st,sizeof(st),"./swarm_%d",n);
soutput << st << endl;
soutput.close();
 
//copy executable
snprintf(st,sizeof(st),"mv swarm_%d.sh iteration_%d",n,n);
system(st);

rn_seed = (int) (drand48()*2000000);

ofstream output("input_parameters.dat",ios::out);
output << rn_seed << " " << 0 << " " << n << endl;

snprintf(st,sizeof(st),"cp input_parameters.dat iteration_%d",n);
system(st);

//copy code
snprintf(st,sizeof(st),"cp jarz.c iteration_%d",n);
system(st);

//copy file
jobcomplete(0);
snprintf(st,sizeof(st),"mv jobcomplete.dat iteration_%d",n);
system(st);

//compile code
snprintf(st,sizeof(st),"cd iteration_%d; g++ -Wall -mcmodel=medium -o swarm_%d jarz.c -lm -O",n,n); system(st);
cout << st << endl;
 
snprintf(st,sizeof(st),"cd iteration_%d; sbatch swarm_%d.sh",n,n); system(st);
 

}}



void jobcomplete(int i){

 //snprintf(st,sizeof(st),"rm jobcomplete.dat");
 //system(st);
 
 snprintf(st,sizeof(st),"jobcomplete.dat");
 ofstream output_job(st,ios::out);
 output_job << i << endl;
 output_job.close();

}


int q_jobs_done(void){

int i;
int q1=0;
int flag=0;
int counter=0;

for(i=0;i<n_traj;i++){

snprintf(st,sizeof(st),"iteration_%d/jobcomplete.dat",i);
ifstream infile(st, ios::in);
infile >> flag;

if(flag==1){counter++;}
flag=0;

}

cout << " counter " << counter << endl;

if(counter==n_traj){q1=1;}
else{q1=0;}

return (q1);

}


bool q_file_exist(const char *fileName){
  
  std::ifstream infile(fileName);
  return infile.good();
 
 }

void make_output(const char *varname){

int i;

double tf;
double quantity[n_traj];

//output fil
snprintf(st,sizeof(st),"output_%s.dat",varname);
ofstream out1(st,ios::out);

//read phi values
for(i=0;i<n_traj;i++){

//read and write data
snprintf(st,sizeof(st),"iteration_%d/report_%s.dat",i,varname);
if(q_file_exist(st)==1){
ifstream infile(st, ios::in);
while (!infile.eof()){infile >> tf >> quantity[i];}
out1 << tf << " " << quantity[i] << endl;

}}

}

void make_outputs(void){

make_output("work");
make_output("work_variance");
make_output("omega");
make_output("omega_variance");
make_output("heat");
make_output("jarz");
make_output("jarz_block_variance");
make_output("jarz2");
make_output("jarz2_block_variance");

}


