
#include <unistd.h>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <time.h>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <string>
#include <string.h>

using namespace std;

char st[256];
long seed;

//mmm, pi
const double pi=4.0*atan(1.0);

//flag
int q_net=0;
int q_forward=1;
int q_vary_beta=1;
int q_production_run=1;

//training
const int number_of_trajectories=1000000;

//time parameters (fixed)
const int number_of_report_steps=1000;

//time parameters (set in initialize)
double trajectory_time;
double timestep=1e-3;

int report_step;
int trajectory_steps;

//for averaging
double wd[number_of_trajectories];
double om[number_of_trajectories];
double he[number_of_trajectories];
double pos[number_of_trajectories];
double jarz[number_of_trajectories];
double jarz2[number_of_trajectories];


//plot potential
const int potential_pics=10;

int traj_number=0;
int potential_pic_counter=0;

double potential_plot_increment=5;
double pos_time[number_of_trajectories][potential_pics+2];

//potential parameters
double c0;
double kay=1.0;
double beta=1.0;
double beta_prime=1.0;
double delta_eff=0.0;

//boundary values
double c0_final=5.0;
double c0_initial=0.0;

double kay_final=3.0;
double kay_initial=1.0;

//model parameters
double position;

//net parameters
const int depth=4; //stem of mushroom net, >=1 (so two-layer net at least)
const int width=4;
const int width_final=10;

const int number_of_inputs=1;
const int number_of_outputs=2;
const int number_of_net_parameters=number_of_inputs*width+width*width*(depth-1)+width*width_final+width_final*number_of_outputs+depth*width+width_final+number_of_outputs;

double inputs[number_of_inputs];
double hidden_node[width][depth];
double outputs[number_of_outputs];
double hidden_node_final[width_final];

//GA hyperparms
double sigma_mutate=0.01;

double np;
double phi;

//registers
const int n_points=2500;
double initial_position_prob[n_points];
double initial_position_value[n_points];
double mutation[number_of_net_parameters];
double net_parameters[number_of_net_parameters];
double net_parameters_holder[number_of_net_parameters];

int initial_state;
int number_state_one;
double work_time[number_of_report_steps+1][2]; //starting states P,M
double energy_time[number_of_report_steps+1][2];
double position_time[number_of_report_steps+1][2];
double work_time_variance[number_of_report_steps+1][2];
double energy_time_variance[number_of_report_steps+1][2];
double position_time_variance[number_of_report_steps+1][2];

//vars int
int iteration;
int generation=0;
int record_trajectory=1;

double tau;
double heat;
double work;
double energy;
double omega;
double min_var;
double mean_eff;
double mean_heat;
double mean_work;
double mean_jarz;
double mean_jarz2;
double mean_omega;
double histo_width;
double work_variance;
double work_skewness;
double omega_variance;
double jarz_block_variance;
double jarz2_block_variance;

//functions void
void ga(void);
void run_net(void);
void read_net(void);
void averaging(void);
void store_net(void);
void initialize(void);
void output_net(void);
void mutate_net(void);
void restore_net(void);
void equilibrate(void);
void jobcomplete(int i);
void langevin_step(void);
void initialize_net(void);
void run_trajectory(void);
void reset_registers(void);
void output_histogram(void);
void run_fixed_protocol(void);
void run_trajectory_average(void);
void output_histogram_omega(void);
void record_position(int step_number);
void output_potential(int step_number);
void update_potential(int step_number);
void output_trajectory_average_data(void);
void output_trajectory_data(int step_number);
void output_histogram_position(int time_slice);
void record_trajectory_averages(int step_number);
void plot_function(const char *varname, const char *xlabel, double x_min, double x_max, const char *ylabel, double y_min, double y_max);
void plot_individual_function(const char *varname, const char *xlabel, int count1, double x_min, double x_max, const char *ylabel, double y_min, double y_max);

//functions double
double test_phi(void);
double potential(void);
double gauss_rv(double sigma);

int main(void){
  
//RN generator
initialize();

//GA
if(q_net==1){ga();}
else{run_fixed_protocol();}

//exit
return 0;

}

void initialize(void){

//clean up
snprintf(st,sizeof(st),"rm report_*");
cout << st << endl;
cout << endl;
system(st);

ifstream infile0("input_parameters.dat", ios::in);
while (!infile0.eof ()){infile0 >> seed >> generation >> iteration;}

//seed RN generator
srand48(seed);

//clean up
if(generation>1){
snprintf(st,sizeof(st),"rm net_*_gen_%d.dat",generation-2);
cout << st << endl;
cout << endl;
system(st);
}

//set trajectory length
if(q_net==0){

if(q_vary_beta==0){beta_prime=1.0;trajectory_time=1.0+iteration;}
if(q_vary_beta==1){beta_prime=1.5-1.4*(iteration/49.0);trajectory_time=10.0;}

}
else{trajectory_time=10.0;}

trajectory_steps=trajectory_time/timestep;
report_step=trajectory_steps/number_of_report_steps;
if(report_step==0){report_step=1;}

cout << " beta prime " << beta_prime << " tf " << trajectory_time << endl;

//free-energy difference
delta_eff=0.5*log(kay_final/kay_initial);
cout << " delta eff " << delta_eff << endl;

//initialize net
if(generation==0){initialize_net();}
else{read_net();}

//optimal protocol
snprintf(st,sizeof(st),"jumps.dat");
ofstream output_jumps(st,ios::out);
output_jumps << 1-q_forward << " " << 0 << endl;
output_jumps << 1-q_forward << " " << c0_final/(trajectory_time+2.0) << endl;
output_jumps << q_forward << " " << c0_final*(trajectory_time+1.0)/(trajectory_time+2.0) << endl;
output_jumps << q_forward << " " << c0_final << endl;
output_jumps.close();

//optimal value
snprintf(st,sizeof(st),"opt.dat");
ofstream output_opt(st,ios::out);
output_opt << (2.0*q_forward-1.0)*c0_final*c0_final/(trajectory_time + 2.0) << " " << 0 << endl;
output_opt << (2.0*q_forward-1.0)*c0_final*c0_final/(trajectory_time + 2.0) << " " << 1 << endl;
output_opt.close();

//zero
snprintf(st,sizeof(st),"zero.dat");
ofstream output_zero(st,ios::out);
output_zero << delta_eff << " " << 0 << endl;
output_zero << delta_eff << " " << 1 << endl;
output_zero.close();

}


double gauss_rv(double sigma){

double g1;
double r1,r2;
double two_pi = 2.0*pi;

r1=drand48();
r2=drand48();

g1=sqrt(-2.0*log(r1))*sigma*cos(two_pi*r2);

return (g1);

}


void update_potential(int step_number){

double e1,e2;

//initial energy
e1=potential();

//protocol
if(q_net==1){run_net();}
else{

//lambda
if(q_forward==1){c0=c0_final*(tau*trajectory_time+1.0)/(trajectory_time+2.0);}
if(q_forward==0){c0=c0_final*((1.0-tau)*trajectory_time+1.0)/(trajectory_time+2.0);}

//kay
if(q_forward==1){kay=kay_initial*(1.0-tau)+kay_final*tau;}
if(q_forward==0){kay=kay_initial*tau+kay_final*(1.0-tau);}

//beta
beta=beta_prime;

}

//final-time potential form
if(step_number==trajectory_steps){

if(q_forward==1){c0=c0_final;kay=kay_final;}
if(q_forward==0){c0=c0_initial;kay=kay_initial;}

beta=1.0;

}

//final energy
e2=potential();

//work
work+=e2-e1;
omega+=e2-e1;

//log energy
energy=e2;

}

void run_trajectory(void){

int i;

//initial position
equilibrate();

//run traj
for(i=0;i<=trajectory_steps;i++){

//output data
if(i<trajectory_steps){output_trajectory_data(i);output_potential(i);}

//update potential (and work)
update_potential(i);

//update position (and heat)
langevin_step();

//record position
record_position(i);

//record trajectory averages
record_trajectory_averages(i);

//update time
if(i!=trajectory_steps){
tau+=1.0/(1.0*trajectory_steps);
}}

//final-time data
output_potential(trajectory_steps);
output_trajectory_data(trajectory_steps);

//increment trajectory counter
traj_number++;

}

void langevin_step(void){

double e1;
double e2;
double grad;

//initial energy
e1=potential();

//gradient term
grad=kay*(position-c0);

double a1=-1.0*grad*timestep;
double a2=sqrt(2.0*timestep/beta);
position+=a1+a2*gauss_rv(1.0);

//final energy
e2=potential();

//heat increment
heat+=e2-e1;
omega+=(1.0-beta)*(e2-e1);

}


void output_trajectory_data(int step_number){

if(record_trajectory==1){
if((step_number % report_step==0) || (step_number==trajectory_steps)){

snprintf(st,sizeof(st),"report_position_gen_%d.dat",generation);
ofstream out1(st,ios::app);

snprintf(st,sizeof(st),"report_work_gen_%d.dat",generation);
ofstream out2(st,ios::app);

snprintf(st,sizeof(st),"report_heat_gen_%d.dat",generation);
ofstream out3(st,ios::app);

snprintf(st,sizeof(st),"report_omega_gen_%d.dat",generation);
ofstream out4(st,ios::app);

snprintf(st,sizeof(st),"report_beta_gen_%d.dat",generation);
ofstream out5(st,ios::app);

snprintf(st,sizeof(st),"report_c0_gen_%d.dat",generation);
ofstream out6(st,ios::app);

snprintf(st,sizeof(st),"report_kay_gen_%d.dat",generation);
ofstream out8(st,ios::app);

snprintf(st,sizeof(st),"report_energy_gen_%d.dat",generation);
ofstream out7(st,ios::app);

out1 << tau << " " << position << endl;
out2 << tau << " " << work << endl;
out3 << tau << " " << heat << endl;
out4 << tau << " " << omega << endl;
out5 << tau << " " << beta << endl;
out6 << tau << " " << c0 << endl;
out7 << tau << " " << potential() << endl;
out8 << tau << " " << kay << endl;


}}

}


void plot_function(const char *varname, const char *xlabel, double x_min, double x_max, const char *ylabel, double y_min, double y_max){

const char *varname1="cee";
const char *varname1b="c0";
const char *varname1c="position";
const char *varname1d="beta";

const char *varname2="potential";
const char *varname2b="pos_time";
const char *varname2c="boltz";

const char *varname3="wd";

const char *varname4="om";

 //output file
 snprintf(st,sizeof(st),"report_%s.asy",varname);
 ofstream output_interface_asy(st,ios::out);
 
 //write output file
 output_interface_asy << "import graph;" << endl;
 output_interface_asy << "import stats;"<< endl;
 
 output_interface_asy << "from \"graph_routines.asy\" access *;"<< endl;
 
 output_interface_asy << "picture p2;"<< endl;
 output_interface_asy << "defaultpen(1.5);"<< endl;
 
 output_interface_asy << "real ymin=" << y_min << ";"<< endl;
 output_interface_asy << "real ymax=" << y_max << ";"<< endl;
 
 output_interface_asy << "real xmin=" << x_min << ";"<< endl;
 output_interface_asy << "real xmax=" << x_max << ";"<< endl;
 
 output_interface_asy << "size(p2,400,400,IgnoreAspect);"<< endl;
 output_interface_asy << "scale(p2,Linear(true),Linear(true));"<< endl;
 
 
 //void simplot_symbol(picture p, string filename,string name,pen pn,int poly,real a1,real a2,real a3,real s1)

if((varname!=varname1) && (varname!=varname2)){
snprintf(st,sizeof(st),"simplot_simple(p2,\"report_%s_gen_%d.dat\",\"\\Large $$\",pen1,%g,%g,%g,%g);",varname,generation,0.2,0.8,0.2,1.25);
output_interface_asy << st << endl;
}

if(varname==varname1){

snprintf(st,sizeof(st),"simplot_simple(p2,\"report_%s_gen_%d.dat\",\"\\Large $$\",pen1,%g,%g,%g,%g);",varname1b,generation,0.5,0.5,0.5,1.2);
output_interface_asy << st << endl;

snprintf(st,sizeof(st),"simplot_simple_dashed(p2,\"jumps.dat\",\"\\Large $$\",pen1,%g,%g,%g,%g);",0.0,0.0,0.0,1.0);
output_interface_asy << st << endl;

snprintf(st,sizeof(st),"simplot_simple(p2,\"report_%s_gen_%d.dat\",\"\\Large $$\",pen1,%g,%g,%g,%g);",varname1c,generation,0.1,0.8,0.1,1.2);
output_interface_asy << st << endl;

snprintf(st,sizeof(st),"simplot_simple(p2,\"report_%s_gen_%d.dat\",\"\\Large $$\",pen1,%g,%g,%g,%g);",varname1d,generation,0.7,0.2,0.2,1.2);
output_interface_asy << st << endl;

}

if(varname==varname2){
for(int i=0;i<potential_pics+2;i++){

snprintf(st,sizeof(st),"simplot_simple(p2,\"report_%s_pic_%d_gen_%d.dat\",\"\\Large $$\",pen1,%g,%g,%g,%g);",varname,i,generation,0.0,0.0,0.0,0.75);
output_interface_asy << st << endl;

//snprintf(st,sizeof(st),"simplot_simple(p2,\"report_%s_pic2_%d_gen_%d.dat\",\"\\Large $$\",pen1,%g,%g,%g,%g);",varname,i,generation,0.1,0.1,0.8,1.1);
//output_interface_asy << st << endl;

snprintf(st,sizeof(st),"simplot_simple(p2,\"report_%s_%d_gen_%d.dat\",\"\\Large $$\",pen1,%g,%g,%g,%g);",varname2b,i,generation,0.2,0.8,0.2,1.1);
output_interface_asy << st << endl;

snprintf(st,sizeof(st),"simplot_simple_dashed(p2,\"report_%s_pic_%d_gen_%d.dat\",\"\\Large $$\",pen1,%g,%g,%g,%g);",varname2c,i,generation,0.0,0.0,0.0,0.5);
output_interface_asy << st << endl;

}}

if((varname==varname3) || (varname==varname4)){

snprintf(st,sizeof(st),"simplot_simple_dashed(p2,\"zero.dat\",\"\\Large $$\",pen1,%g,%g,%g,%g);",0.1,0.1,0.1,1.0);
output_interface_asy << st << endl;

snprintf(st,sizeof(st),"simplot_simple_dashed(p2,\"opt.dat\",\"\\Large $$\",pen1,%g,%g,%g,%g);",0.0,0.0,0.0,1.0);
output_interface_asy << st << endl;

if(q_forward==1){
snprintf(st,sizeof(st),"simplot_simple_dashed(p2,\"report_%s_weighted_gen_%d.dat\",\"\\Large $$\",pen1,%g,%g,%g,%g);",varname,generation,0.2,0.2,0.8,0.8);
output_interface_asy << st << endl;
}

}

 output_interface_asy << "xlimits(p2, xmin, xmax, true);"<< endl;
 output_interface_asy << "ylimits(p2, ymin, ymax, true);"<< endl;
 
 snprintf(st,sizeof(st),"%s",xlabel);
 output_interface_asy << "xaxis(p2,Label(\"$"<<st<<"$\",fontsize(30pt)),BottomTop,LeftTicks(Label(fontsize(30)),new real[]{" << x_min <<"," << x_max <<"})); "<< endl;
 snprintf(st,sizeof(st),"%s",ylabel);
 output_interface_asy << "yaxis(p2,Label(\"$"<<st<<"$\",fontsize(30pt)),LeftRight,RightTicks(Label(fontsize(30)),new real[]{"<< y_min << "," << y_max <<"})); "<< endl;
 
 
 output_interface_asy << "scale(Linear(true),Linear(true)); "<< endl;
 
 if(varname==varname2){output_interface_asy << "add(p2.fit(600,200),(0,0),S);"<< endl;}
 else{output_interface_asy << "add(p2.fit(300,250),(0,0),S);"<< endl;}

if(q_production_run==0){
snprintf(st,sizeof(st),"/usr/local/texlive/2017/bin/x86_64-darwin/asy report_%s.asy",varname);
system(st);

snprintf(st,sizeof(st),"open report_%s.eps",varname);
system(st);
}
 
 
}


void plot_individual_function(const char *varname, const char *xlabel, int count1, double x_min, double x_max, const char *ylabel, double y_min, double y_max){


const char *varname1="potential_pic_individual";
const char *varname1a="pos_time_individual";
const char *varname1b="boltz_pic_individual";

 //output file
 snprintf(st,sizeof(st),"report_%s_%d.asy",varname,count1);
 ofstream output_interface_asy(st,ios::out);
 
 //write output file
 output_interface_asy << "import graph;" << endl;
 output_interface_asy << "import stats;"<< endl;
 
 output_interface_asy << "from \"graph_routines.asy\" access *;"<< endl;
 
 output_interface_asy << "picture p2;"<< endl;
 output_interface_asy << "defaultpen(1.2);"<< endl;
 
 output_interface_asy << "real ymin=" << y_min << ";"<< endl;
 output_interface_asy << "real ymax=" << y_max << ";"<< endl;
 
 output_interface_asy << "real xmin=" << x_min << ";"<< endl;
 output_interface_asy << "real xmax=" << x_max << ";"<< endl;
 
 output_interface_asy << "size(p2,400,400,IgnoreAspect);"<< endl;
 output_interface_asy << "scale(p2,Linear(true),Linear(true));"<< endl;
 
 
if(varname==varname1){

snprintf(st,sizeof(st),"simplot_simple(p2,\"report_%s_%d_gen_%d.dat\",\"\\Large $$\",pen1,%g,%g,%g,%g);",varname1,count1,generation,0.1,0.1,0.1,1.0);
output_interface_asy << st << endl;

snprintf(st,sizeof(st),"simplot_simple_scale_vertical(p2,\"report_%s_%d_gen_%d.dat\",\"\\Large $$\",pen1,%g,%g,%g,%g,%g);",varname1a,count1,generation,0.2,0.8,0.2,1.2,2.0);
output_interface_asy << st << endl;

snprintf(st,sizeof(st),"simplot_simple_dashed_scale_vertical(p2,\"report_%s_%d_gen_%d.dat\",\"\\Large $$\",pen1,%g,%g,%g,%g,%g);",varname1b,count1,generation,0.0,0.0,0.0,0.6,2.0);
output_interface_asy << st << endl;

}

 output_interface_asy << "xlimits(p2, xmin, xmax, true);"<< endl;
 output_interface_asy << "ylimits(p2, ymin, ymax, true);"<< endl;
 
 snprintf(st,sizeof(st),"%s",xlabel);
 output_interface_asy << "xaxis(p2,Label(\"$"<<st<<"$\",fontsize(30pt)),BottomTop,LeftTicks(Label(fontsize(30)),new real[]{" << x_min <<", 0, " << x_max <<"})); "<< endl;
 snprintf(st,sizeof(st),"%s",ylabel);
 output_interface_asy << "yaxis(p2,Label(\"$"<<st<<"$\",fontsize(30pt)),LeftRight,RightTicks(Label(fontsize(30)),new real[]{"<< y_min << "," << y_max <<"})); "<< endl;
 
 
 output_interface_asy << "scale(Linear(true),Linear(true)); "<< endl;
 output_interface_asy << "add(p2.fit(250,225),(0,0),S);"<< endl;

if(q_production_run==0){

snprintf(st,sizeof(st),"/usr/local/texlive/2017/bin/x86_64-darwin/asy report_%s_%d.asy",varname,count1);
system(st);

snprintf(st,sizeof(st),"epstopdf report_%s_%d.eps",varname,count1);
system(st);

//snprintf(st,sizeof(st),"open report_%s_%d.eps",varname,count1);
//system(st);

if(count1==potential_pics){

char st2[1024] = "gs -q -dNOPAUSE -dBATCH -sDEVICE=pdfwrite -sOutputFile=report_combined.pdf";

    for(int i = 0; i < potential_pics+1; i++) {
        char temp[128];
        snprintf(temp, sizeof(temp), " report_potential_pic_individual_%d.pdf", i);
        strcat(st2, temp);
    }

    printf("%s\n", st2);
    system(st2);
    
   snprintf(st, sizeof(st), "open report_combined.pdf");
   system(st);
   
}}
 
}

 

void read_net(void){

int i;

snprintf(st,sizeof(st),"net_in_gen_%d.dat",generation);
ifstream infile(st, ios::in);

for(i=0;i<number_of_net_parameters;i++){infile >> net_parameters[i];}

}

void output_net(void){

int i;

//parameter file
snprintf(st,sizeof(st),"net_out_gen_%d.dat",generation);
ofstream out_net(st,ios::out);

for(i=0;i<number_of_net_parameters;i++){out_net << net_parameters[i] << " ";}

}


void store_net(void){

int i;
for(i=0;i<number_of_net_parameters;i++){net_parameters_holder[i]=net_parameters[i];}

}

void mutate_net(void){

int i;
for(i=0;i<number_of_net_parameters;i++){mutation[i]=gauss_rv(sigma_mutate);}
for(i=0;i<number_of_net_parameters;i++){net_parameters[i]+=mutation[i];}

}


void restore_net(void){

int i;
for(i=0;i<number_of_net_parameters;i++){net_parameters[i]=net_parameters_holder[i];}

}

void initialize_net(void){

int i;
for(i=0;i<number_of_net_parameters;i++){net_parameters[i]=gauss_rv(0.0);}

}


void run_net(void){

int pid=0;

int i,j,k;

double mu=0.0;
double sigma=0.0;
double delta=1e-4;

//inputs
if(q_forward==1){inputs[0]=tau;}
if(q_forward==0){inputs[0]=1.0-tau;}

//surface layer
for(i=0;i<width;i++){
hidden_node[i][0]=net_parameters[pid];pid++;
for(j=0;j<number_of_inputs;j++){hidden_node[i][0]+=inputs[j]*net_parameters[pid];pid++;}
}

//layer norm (pre-activation)
mu=0.0;sigma=0.0;
for(i=0;i<width;i++){mu+=hidden_node[i][0];sigma+=hidden_node[i][0]*hidden_node[i][0];}
mu=mu/width;sigma=sigma/width;
sigma=sqrt(sigma-mu*mu)+delta;
for(i=0;i<width;i++){hidden_node[i][0]=(hidden_node[i][0]-mu)/sigma;}

//activation
for(i=0;i<width;i++){hidden_node[i][0]=tanh(hidden_node[i][0]);}


//stem layers
for(j=1;j<depth;j++){
for(i=0;i<width;i++){
hidden_node[i][j]=net_parameters[pid];pid++;
for(k=0;k<width;k++){hidden_node[i][j]+=hidden_node[k][j-1]*net_parameters[pid];pid++;}
}

//layer norm (pre-activation)
mu=0.0;sigma=0.0;
for(i=0;i<width;i++){mu+=hidden_node[i][j];sigma+=hidden_node[i][j]*hidden_node[i][j];}
mu=mu/width;sigma=sigma/width;
sigma=sqrt(sigma-mu*mu)+delta;
for(i=0;i<width;i++){hidden_node[i][j]=(hidden_node[i][j]-mu)/sigma;}

//activation
for(i=0;i<width;i++){hidden_node[i][j]=tanh(hidden_node[i][j]);}

}

//final layer
for(i=0;i<width_final;i++){
hidden_node_final[i]=net_parameters[pid];pid++;
for(j=0;j<width;j++){hidden_node_final[i]+=hidden_node[j][depth-1]*net_parameters[pid];pid++;}
}

//layer norm (pre-activation)
mu=0.0;sigma=0.0;
for(i=0;i<width_final;i++){mu+=hidden_node_final[i];sigma+=hidden_node_final[i]*hidden_node_final[i];}
mu=mu/width_final;sigma=sigma/width_final;
sigma=sqrt(sigma-mu*mu)+delta;
for(i=0;i<width_final;i++){hidden_node_final[i]=(hidden_node_final[i]-mu)/sigma;}


//activation
for(i=0;i<width_final;i++){hidden_node_final[i]=tanh(hidden_node_final[i]);}

//outputs
for(i=0;i<number_of_outputs;i++){
outputs[i]=net_parameters[pid];pid++;
for(j=0;j<width_final;j++){outputs[i]+=hidden_node_final[j]*net_parameters[pid];pid++;}
}

//c0
if(q_forward==1){c0=c0_final*(tau*trajectory_time+1.0)/(trajectory_time+2.0);}
if(q_forward==0){c0=c0_final*((1.0-tau)*trajectory_time+1.0)/(trajectory_time+2.0);}
c0+=outputs[0];

//beta
beta=1.0+outputs[1];
if(beta<0.1){beta=0.1;}

}


void jobcomplete(int i){

 //snprintf(st,sizeof(st),"rm jobcomplete.dat");
 //system(st);
 
 snprintf(st,sizeof(st),"jobcomplete.dat");
 ofstream output_job(st,ios::out);
 output_job << i << endl;
 output_job.close();

}



void run_trajectory_average(void){

int i;

//reset registers
reset_registers();

for(i=0;i<number_of_trajectories;i++){

run_trajectory();

wd[i]=work;
he[i]=heat;
om[i]=omega;
pos[i]=position;
jarz[i]=exp(-wd[i]);
jarz2[i]=exp(-om[i]);

}

//averaging (sets phi)
averaging();

//outputs
output_trajectory_average_data();
traj_number=0;

}



void output_histogram(void){

snprintf(st,sizeof(st),"report_wd_gen_%d.dat",generation);
ofstream output_wd(st,ios::out);

snprintf(st,sizeof(st),"report_wd_weighted_gen_%d.dat",generation);
ofstream output_wd_weighted(st,ios::out);

int i;
int nbin=0;
const int bins=50;

double histo[bins];
double maxwork=0.0;
double minwork=0.0;

//work recorded elswehere
for(i=0;i<bins;i++){histo[i]=0.0;}

for(i=0;i<number_of_trajectories;i++){

if(i==0){maxwork=wd[i];minwork=wd[i];}
else{

if(wd[i]>maxwork){maxwork=wd[i];}
if(wd[i]<minwork){minwork=wd[i];}
}}

//record width
histo_width=(maxwork-minwork)/(1.0*bins);

//safeguard
if((fabs(maxwork)<1e-6) && (fabs(minwork)<1e-6)){

histo_width=0.1;
maxwork=1.0;
minwork=0.0;

}

for(i=0;i<number_of_trajectories;i++){

nbin=(int) (1.0*bins*(wd[i]-minwork)/(maxwork-minwork));
if(nbin>=bins){nbin=bins-1;}

histo[nbin]+=1.0/(1.0*number_of_trajectories);

}

//output
double w1;
for(i=0;i<bins;i++){

if(histo[nbin]>0.5/(1.0*number_of_trajectories)){
w1=maxwork*i/(1.0*bins)+minwork*(1.0-i/(1.0*bins))+0.5*(maxwork-minwork)/(1.0*bins);

output_wd << (2.0*q_forward-1.0)*w1 << " " << histo[i]/histo_width << endl;
output_wd_weighted << w1 << " " << exp(delta_eff-w1)*histo[i]/histo_width << endl;


}
}

//plot histogram
if(q_forward==1){plot_function("wd","W",minwork,maxwork,"P(W)",0,5.0/(histo_width*bins));}
if(q_forward==0){plot_function("wd","W",-maxwork,maxwork,"P(W)",0,5.0/(histo_width*bins));}
}


void output_histogram_omega(void){

snprintf(st,sizeof(st),"report_om_gen_%d.dat",generation);
ofstream output_wd(st,ios::out);

snprintf(st,sizeof(st),"report_om_weighted_gen_%d.dat",generation);
ofstream output_wd_weighted(st,ios::out);

int i;
int nbin=0;
const int bins=50;

double histo[bins];
double maxwork=0.0;
double minwork=0.0;

//work recorded elswehere
for(i=0;i<bins;i++){histo[i]=0.0;}

for(i=0;i<number_of_trajectories;i++){

if(i==0){maxwork=om[i];minwork=om[i];}
else{

if(om[i]>maxwork){maxwork=om[i];}
if(om[i]<minwork){minwork=om[i];}
}}

//record width
histo_width=(maxwork-minwork)/(1.0*bins);

//safeguard
if((fabs(maxwork)<1e-6) && (fabs(minwork)<1e-6)){

histo_width=0.1;
maxwork=1.0;
minwork=0.0;

}

for(i=0;i<number_of_trajectories;i++){

nbin=(int) (1.0*bins*(om[i]-minwork)/(maxwork-minwork));
if(nbin>=bins){nbin=bins-1;}

histo[nbin]+=1.0/(1.0*number_of_trajectories);

}

//output
double w1;
for(i=0;i<bins;i++){

if(histo[nbin]>0.5/(1.0*number_of_trajectories)){
w1=maxwork*i/(1.0*bins)+minwork*(1.0-i/(1.0*bins))+0.5*(maxwork-minwork)/(1.0*bins);

output_wd << (2.0*q_forward-1.0)*w1 << " " << histo[i]/histo_width << endl;
output_wd_weighted << w1 << " " << exp(delta_eff-w1)*histo[i]/histo_width << endl;


}
}

//plot histogram
if(q_forward==1){plot_function("om","\\Sigma",minwork,maxwork,"P(\\Sigma)",0,5.0/(histo_width*bins));}
if(q_forward==0){plot_function("om","\\Sigma",-maxwork,maxwork,"P(\\Sigma)",0,5.0/(histo_width*bins));}

}


void ga(void){

record_trajectory=0;

//mutate net
mutate_net();

//calculate order parameter
run_trajectory_average();

//output evolutionary order parameter
snprintf(st,sizeof(st),"report_phi_gen_%d.dat",generation);
ofstream output_phi(st,ios::out);
output_phi << np << endl;

//output other order parameters
snprintf(st,sizeof(st),"report_order_parameters_gen_%d.dat",generation);
ofstream output_op(st,ios::out);
output_op << mean_work << " " << work_variance << " ";
output_op << mean_omega << " " << omega_variance << " ";
output_op << mean_heat << " ";
output_op << -log(mean_jarz) << " " << jarz_block_variance << " ";
output_op << -log(mean_jarz2) << " " << jarz2_block_variance << endl;

snprintf(st,sizeof(st),"report_work.dat");
ofstream out1(st,ios::out);

snprintf(st,sizeof(st),"report_work_variance.dat");
ofstream out2(st,ios::out);

snprintf(st,sizeof(st),"report_omega.dat");
ofstream out3(st,ios::out);

snprintf(st,sizeof(st),"report_omega_variance.dat");
ofstream out4(st,ios::out);

snprintf(st,sizeof(st),"report_heat.dat");
ofstream out5(st,ios::out);

snprintf(st,sizeof(st),"report_jarz.dat");
ofstream out6(st,ios::out);

snprintf(st,sizeof(st),"report_jarz_block_variance.dat");
ofstream out7(st,ios::out);

snprintf(st,sizeof(st),"report_jarz2.dat");
ofstream out8(st,ios::out);

snprintf(st,sizeof(st),"report_jarz2_block_variance.dat");
ofstream out9(st,ios::out);

double q1=trajectory_time;

out1 << q1 << " " << mean_work << endl;
out2 << q1 << " " << work_variance << endl;
out3 << q1 << " " << mean_omega << endl;
out4 << q1 << " " << omega_variance << endl;
out5 << q1 << " " << mean_heat << endl;
out6 << q1 << " " << -log(mean_jarz) << endl;
out7 << q1 << " " << jarz_block_variance << endl;
out8 << q1 << " " << -log(mean_jarz2) << endl;
out9 << q1 << " " << jarz2_block_variance << endl;

//output histograms
output_histogram();
output_histogram_omega();
for(int i=0;i<potential_pics+2;i++){output_histogram_position(i);}

//output net
output_net();

//trajectory data
record_trajectory=1;
run_trajectory();

//jobcomplete
jobcomplete(1);

if(q_production_run==0){

//plots
plot_function("cee","t",-0.25,1.1*tau,"c",-3,1.2*c0_final);
for(int i=0;i<potential_pics+1;i++){plot_individual_function("potential_pic_individual","x",i,-2.0,7.0,"",0,kay);}

}

}

void run_fixed_protocol(void){

double q1;

record_trajectory=0;

//calculate order parameter
run_trajectory_average();

//output evolutionary order parameter
snprintf(st,sizeof(st),"report_phi.dat");
ofstream output_phi(st,ios::out);
output_phi << np << endl;

//output other order parameters
snprintf(st,sizeof(st),"report_order_parameters.dat");
ofstream output_op(st,ios::out);
output_op << mean_work << " " << work_variance << " ";
output_op << mean_omega << " " << omega_variance << " ";
output_op << mean_heat << " ";
output_op << -log(mean_jarz) << " " << jarz_block_variance << " ";
output_op << -log(mean_jarz2) << " " << jarz2_block_variance << endl;

snprintf(st,sizeof(st),"report_work.dat");
ofstream out1(st,ios::out);

snprintf(st,sizeof(st),"report_work_variance.dat");
ofstream out2(st,ios::out);

snprintf(st,sizeof(st),"report_omega.dat");
ofstream out3(st,ios::out);

snprintf(st,sizeof(st),"report_omega_variance.dat");
ofstream out4(st,ios::out);

snprintf(st,sizeof(st),"report_heat.dat");
ofstream out5(st,ios::out);

snprintf(st,sizeof(st),"report_jarz.dat");
ofstream out6(st,ios::out);

snprintf(st,sizeof(st),"report_jarz_block_variance.dat");
ofstream out7(st,ios::out);

snprintf(st,sizeof(st),"report_jarz2.dat");
ofstream out8(st,ios::out);

snprintf(st,sizeof(st),"report_jarz2_block_variance.dat");
ofstream out9(st,ios::out);

if(q_vary_beta==0){q1=trajectory_time;}
if(q_vary_beta==1){q1=beta_prime;}

out1 << q1 << " " << mean_work << endl;
out2 << q1 << " " << work_variance << endl;
out3 << q1 << " " << mean_omega << endl;
out4 << q1 << " " << omega_variance << endl;
out5 << q1 << " " << mean_heat << endl;
out6 << q1 << " " << -log(mean_jarz) << endl;
out7 << q1 << " " << jarz_block_variance << endl;
out8 << q1 << " " << -log(mean_jarz2) << endl;
out9 << q1 << " " << jarz2_block_variance << endl;

//output histograms
output_histogram();
output_histogram_omega();
for(int i=0;i<potential_pics+2;i++){output_histogram_position(i);}

//output net
output_net();

//trajectory data
record_trajectory=1;
run_trajectory();

//jobcomplete
jobcomplete(1);

if(q_production_run==0){

//plots
plot_function("cee","t",-0.25,1.1*tau,"c",-3,1.2*c0_final);
for(int i=0;i<potential_pics+1;i++){plot_individual_function("potential_pic_individual","x",i,-2.0,7.0,"",0,kay);}

}

}

void averaging(void){

int i,j;
int block=100;
int number_of_blocks=number_of_trajectories/block;

double jarz_accumulator=0.0;
double jarz2_accumulator=0.0;
double jarz_block_average=0.0;
double jarz2_block_average=0.0;
jarz_block_variance=0.0;
jarz2_block_variance=0.0;


//normalization
double n1=1.0/(1.0*number_of_trajectories);
double en[2]={1.0/(1.0*(number_of_trajectories-number_state_one)),1.0/(1.0*number_state_one)};

//reset counters
mean_eff=0.0;
mean_work=0.0;
mean_heat=0.0;
mean_jarz=0.0;
mean_jarz2=0.0;
mean_omega=0.0;
work_variance=0.0;
omega_variance=0.0;

for(i=0;i<number_of_trajectories;i++){

mean_work+=wd[i]*n1;
mean_heat+=he[i]*n1;
mean_omega+=om[i]*n1;
mean_jarz+=jarz[i]*n1;
mean_jarz2+=jarz2[i]*n1;
work_variance+=wd[i]*wd[i]*n1;
omega_variance+=om[i]*om[i]*n1;

//block averaging
jarz_accumulator+=jarz[i]/(1.0*block);
jarz2_accumulator+=jarz2[i]/(1.0*block);

if((i+1) % block == 0){

jarz_block_average+=jarz_accumulator/(1.0*number_of_blocks);
jarz_block_variance+=jarz_accumulator*jarz_accumulator/(1.0*number_of_blocks);
jarz_accumulator=0.0;

jarz2_block_average+=jarz2_accumulator/(1.0*number_of_blocks);
jarz2_block_variance+=jarz2_accumulator*jarz2_accumulator/(1.0*number_of_blocks);
jarz2_accumulator=0.0;

}
}

//variances
work_variance-=mean_work*mean_work;
omega_variance-=mean_omega*mean_omega;

jarz_block_variance-=jarz_block_average*jarz_block_average;
jarz2_block_variance-=jarz2_block_average*jarz2_block_average;

//time-dependent averages
for(i=0;i<number_of_report_steps+1;i++){
for(j=0;j<2;j++){

work_time[i][j]*=en[j];
energy_time[i][j]*=en[j];
position_time[i][j]*=en[j];

work_time_variance[i][j]*=en[j];
energy_time_variance[i][j]*=en[j];
position_time_variance[i][j]*=en[j];

}}

for(i=0;i<number_of_report_steps+1;i++){
for(j=0;j<2;j++){

work_time_variance[i][j]=work_time_variance[i][j]-work_time[i][j]*work_time[i][j];
energy_time_variance[i][j]=energy_time_variance[i][j]-energy_time[i][j]*energy_time[i][j];
position_time_variance[i][j]=position_time_variance[i][j]-position_time[i][j]*position_time[i][j];

}}

//new phi
np=mean_omega;

}

double potential(void){

double p1=position-c0;
double q1=0.5*kay*p1*p1;

return (q1);

}


void equilibrate(void){

//reset counters
tau=0.0;
work=0.0;
heat=0.0;
omega=0;

//reset potential
beta=1.0;
if(q_forward==1){c0=c0_initial;kay=kay_initial;}
if(q_forward==0){c0=c0_final;kay=kay_final;}

//set position
initial_state=q_forward;
number_state_one+=initial_state;
position=c0+gauss_rv(1.0);

//set energy
energy=potential();

//picture counter
potential_pic_counter=0;

}

void output_potential(int step_number){
if(record_trajectory==1){

int ok=0;
if(step_number % (trajectory_steps/potential_pics) == 0){ok=1;}
if(step_number==0){ok=1;}
if(step_number==trajectory_steps){ok=1;}

if(ok==1){

snprintf(st,sizeof(st),"report_potential_pic_%d_gen_%d.dat",potential_pic_counter,generation);
ofstream out_pic(st,ios::out);

snprintf(st,sizeof(st),"report_boltz_pic_%d_gen_%d.dat",potential_pic_counter,generation);
ofstream out_boltz(st,ios::out);

snprintf(st,sizeof(st),"report_potential_pic2_%d_gen_%d.dat",potential_pic_counter,generation);
ofstream out_pic2(st,ios::out);

snprintf(st,sizeof(st),"report_boltz_pic_individual_%d_gen_%d.dat",potential_pic_counter,generation);
ofstream out_boltz_individual(st,ios::out);

snprintf(st,sizeof(st),"report_potential_pic_individual_%d_gen_%d.dat",potential_pic_counter,generation);
ofstream out_pic_individial(st,ios::out);

int i;
double e1;
double e_min=0;
double e_values[n_points];

double x1=-3.0;
double x2=7.0;
double zed=0.0;
double delta_x=0.0;


//record position
double position_holder=position;

for(i=0;i<n_points;i++){

position=x1+(x2-x1)*i/(1.0*n_points-1.0);
e1=potential();

out_pic << potential_plot_increment*potential_pic_counter+position << " " << e1 << endl;
out_pic_individial << position << " " << e1 << endl;

}

//boltzmann weight

//log energies; compute minimum
//flag: won't draw pictures properly if limits too tight
x1=-5.0;x2=10.0;delta_x=(x2-x1)/(1.0*n_points);
for(i=0;i<n_points;i++){

position=x1+(x2-x1)*i/(1.0*n_points-1.0);
e1=potential();

if(i==0){e_min=e1;}
else{if(e1<e_min){e_min=e1;}}

e_values[i]=e1;

}

//calculate Z
for(i=0;i<n_points;i++){

e_values[i]-=e_min;
zed+=delta_x*exp(-beta*e_values[i]);

}

//plot point
for(i=0;i<n_points;i++){

position=x1+(x2-x1)*i/(1.0*n_points-1.0);
//if((position>-1.5) && (position<1.5)){
out_boltz << potential_plot_increment*potential_pic_counter+position << " " << exp(-beta*e_values[i])/zed << endl;
out_boltz_individual << position << " " << exp(-beta*e_values[i])/zed << endl;

//}

}


//reset position
position=position_holder;

//label position of bead
e1=potential();
out_pic2 << potential_plot_increment*potential_pic_counter+position << " " << e1+0.02 << endl;
out_pic2 << potential_plot_increment*potential_pic_counter+position << " " << e1-0.02 << endl;

potential_pic_counter++;

}}}



void output_histogram_position(int time_slice){

snprintf(st,sizeof(st),"report_pos_time_%d_gen_%d.dat",time_slice,generation);
ofstream output_pos_time(st,ios::out);

snprintf(st,sizeof(st),"report_pos_time_individual_%d_gen_%d.dat",time_slice,generation);
ofstream output_pos_time_individual(st,ios::out);

int i;
int nbin=0;
const int bins=50;

double histo[bins];
double maxpos=0.0;
double minpos=0.0;

//pos recorded elswehere
for(i=0;i<bins;i++){histo[i]=0.0;}

for(i=0;i<number_of_trajectories;i++){

if(i==0){maxpos=pos_time[i][time_slice];minpos=pos_time[i][time_slice];}
else{

if(pos_time[i][time_slice]>maxpos){maxpos=pos_time[i][time_slice];}
if(pos_time[i][time_slice]<minpos){minpos=pos_time[i][time_slice];}
}}

//record width
histo_width=(maxpos-minpos)/(1.0*bins);

//safeguard
if((fabs(maxpos)<1e-6) && (fabs(minpos)<1e-6)){

histo_width=0.1;
maxpos=1.0;
minpos=0.0;

}

for(i=0;i<number_of_trajectories;i++){

nbin=(int) (1.0*bins*(pos_time[i][time_slice]-minpos)/(maxpos-minpos));
if(nbin==bins){nbin--;}

histo[nbin]+=1.0/(1.0*number_of_trajectories);

}

//output
double x1;
for(i=0;i<bins;i++){

if(histo[nbin]>0.5/(1.0*number_of_trajectories)){
x1=maxpos*i/(1.0*bins)+minpos*(1.0-i/(1.0*bins))+0.5*(maxpos-minpos)/(1.0*bins);

output_pos_time << x1 + potential_plot_increment*time_slice << " " << histo[i]/histo_width << endl;
output_pos_time_individual << x1 << " " << histo[i]/histo_width << endl;


}
}

//histogram plotted from position histogram

}


void record_position(int step_number){

int ok=0;
int entry=0;
int dt=trajectory_steps/potential_pics;

if(step_number==0){ok=1;}
if(step_number==trajectory_steps){ok=1;entry=potential_pics+1;}
if(step_number % dt == 0){ok=1;entry=step_number/dt;}

if(ok==1){pos_time[traj_number][entry]=position;}

}


void reset_registers(void){

int i,j;

traj_number=0;
number_state_one=0;

for(i=0;i<number_of_report_steps+1;i++){
for(j=0;j<2;j++){

work_time[i][j]=0.0;
energy_time[i][j]=0.0;
position_time[i][j]=0.0;

work_time_variance[i][j]=0.0;
energy_time_variance[i][j]=0.0;
position_time_variance[i][j]=0.0;

}}

}



void record_trajectory_averages(int step_number){

int s1;

if(step_number % report_step==0){

s1=step_number/report_step;
if(s1<=number_of_report_steps){

work_time[s1][initial_state]+=work;
energy_time[s1][initial_state]+=energy;
position_time[s1][initial_state]+=position;

work_time_variance[s1][initial_state]+=work*work;
energy_time_variance[s1][initial_state]+=energy*energy;
position_time_variance[s1][initial_state]+=position*position;

}}

}

void output_trajectory_average_data(void){

int i;
double t1;

snprintf(st,sizeof(st),"report_work_average_state_0_gen_%d.dat",generation);
ofstream out1(st,ios::app);

snprintf(st,sizeof(st),"report_work_average_state_1_gen_%d.dat",generation);
ofstream out2(st,ios::app);

snprintf(st,sizeof(st),"report_work_variance_state_0_gen_%d.dat",generation);
ofstream out3(st,ios::app);

snprintf(st,sizeof(st),"report_work_variance_state_1_gen_%d.dat",generation);
ofstream out4(st,ios::app);

snprintf(st,sizeof(st),"report_energy_average_state_0_gen_%d.dat",generation);
ofstream out5(st,ios::app);

snprintf(st,sizeof(st),"report_energy_average_state_1_gen_%d.dat",generation);
ofstream out6(st,ios::app);

snprintf(st,sizeof(st),"report_energy_variance_state_0_gen_%d.dat",generation);
ofstream out7(st,ios::app);

snprintf(st,sizeof(st),"report_energy_variance_state_1_gen_%d.dat",generation);
ofstream out8(st,ios::app);

snprintf(st,sizeof(st),"report_position_average_state_0_gen_%d.dat",generation);
ofstream out9(st,ios::app);

snprintf(st,sizeof(st),"report_position_average_state_1_gen_%d.dat",generation);
ofstream out10(st,ios::app);

snprintf(st,sizeof(st),"report_position_variance_state_0_gen_%d.dat",generation);
ofstream out11(st,ios::app);

snprintf(st,sizeof(st),"report_position_variance_state_1_gen_%d.dat",generation);
ofstream out12(st,ios::app);

for(i=0;i<number_of_report_steps+1;i++){

t1=timestep*i*report_step;

out1 << t1 << " " << work_time[i][0] << endl;
out2 << t1 << " " << work_time[i][1] << endl;
out3 << t1 << " " << work_time_variance[i][0] << endl;
out4 << t1 << " " << work_time_variance[i][1] << endl;

out5 << t1 << " " << energy_time[i][0] << endl;
out6 << t1 << " " << energy_time[i][1] << endl;
out7 << t1 << " " << energy_time_variance[i][0] << endl;
out8 << t1 << " " << energy_time_variance[i][1] << endl;

out9 << t1 << " " << position_time[i][0] << endl;
out10 << t1 << " " << position_time[i][1] << endl;
out11 << t1 << " " << position_time_variance[i][0] << endl;
out12 << t1 << " " << position_time_variance[i][1] << endl;

}

}

