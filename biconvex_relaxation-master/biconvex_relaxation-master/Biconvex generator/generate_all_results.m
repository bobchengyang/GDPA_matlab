function [ output_args ] = generate_all_results
close all;clear;
%matlabpool close force;
diary('matlabOutputsSDPT3_parallel');
diary on;
N=250;
noOfRepitions=10;
MM = 100:100:1500;
useSDPT3 = false;

for i=1:length(MM)
    do_all_steps(MM(i),N,noOfRepitions,useSDPT3);
end
diary off

end
