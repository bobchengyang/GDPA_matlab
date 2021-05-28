function do_all_steps(M,N,noOfRepitions,useSDPT3)

rng(0);


phmaxOutput_mean.matError = 0;
phmaxOutput_mean.t =0;
phmaxOutput_mean.iterations = 0;
phmaxOutput_mean.estimateMatError=0;
phmaxOutput_mean.absObjectiveError = 0;

gdrmOutput_mean.matError = 0;
gdrmOutput_mean.t =0;
gdrmOutput_mean.iterations = 0;
gdrmOutput_mean.estimateMatError=0;
gdrmOutput_mean.absObjectiveError = 0;

SDPT3Output_mean.matError = 0;
SDPT3Output_mean.t =0;
SDPT3Output_mean.iterations = 0;
SDPT3Output_mean.estimateMatError=0;
SDPT3Output_mean.absObjectiveError = 0;

if useSDPT3
    for i=1:noOfRepitions
        [SDPT3Output{i}] = wrapper_general_SDPT3(M,N,i);
        SDPT3Output_mean.matError = SDPT3Output_mean.matError + abs(SDPT3Output{i}.matError);
        SDPT3Output_mean.t =SDPT3Output_mean.t+SDPT3Output{i}.t;
        SDPT3Output_mean.estimateMatError=SDPT3Output_mean.estimateMatError+abs(SDPT3Output{i}.estimateMatError);
        SDPT3Output_mean.absObjectiveError = SDPT3Output_mean.absObjectiveError+abs(SDPT3Output{i}.absObjectiveError);
        
    end
    
    SDPT3Output_mean.matError = SDPT3Output_mean.matError/noOfRepitions;
    SDPT3Output_mean.t =SDPT3Output_mean.t/noOfRepitions;
    SDPT3Output_mean.iterations = SDPT3Output_mean.iterations/noOfRepitions;
    SDPT3Output_mean.estimateMatError=SDPT3Output_mean.estimateMatError/noOfRepitions;
    SDPT3Output_mean.absObjectiveError = SDPT3Output_mean.absObjectiveError/noOfRepitions;
    
    save(['SDPT3Output_new_1itr_' num2str(M) num2str(N)],'SDPT3Output','SDPT3Output_mean');
    
else
    for i=1:noOfRepitions
        [phmaxOutput{i}, gdrmOutput{i}] = wrapper_general(M,N,0,0);
         phmaxOutput_mean.matError = phmaxOutput_mean.matError + abs(phmaxOutput{i}.matError);
         phmaxOutput_mean.t =phmaxOutput_mean.t+phmaxOutput{i}.t;
         phmaxOutput_mean.iterations = phmaxOutput_mean.iterations+phmaxOutput{i}.iterations;
         phmaxOutput_mean.estimateMatError=phmaxOutput_mean.estimateMatError+abs(phmaxOutput{i}.estimateMatError);
         phmaxOutput_mean.absObjectiveError = phmaxOutput_mean.absObjectiveError+abs(phmaxOutput{i}.absObjectiveError);
%         
         gdrmOutput_mean.matError = gdrmOutput_mean.matError + abs(gdrmOutput{i}.matError);
         gdrmOutput_mean.t =gdrmOutput_mean.t+gdrmOutput{i}.t;
         gdrmOutput_mean.iterations = gdrmOutput_mean.iterations+gdrmOutput{i}.iterations;
         gdrmOutput_mean.estimateMatError=gdrmOutput_mean.estimateMatError+abs(gdrmOutput{i}.estimateMatError);
         gdrmOutput_mean.absObjectiveError = gdrmOutput_mean.absObjectiveError+abs(gdrmOutput{i}.absObjectiveError);
        
    end
    phmaxOutput_mean.matError = phmaxOutput_mean.matError/noOfRepitions;
    phmaxOutput_mean.t =phmaxOutput_mean.t/noOfRepitions;
    phmaxOutput_mean.iterations = phmaxOutput_mean.iterations/noOfRepitions;
    phmaxOutput_mean.estimateMatError=phmaxOutput_mean.estimateMatError/noOfRepitions;
    phmaxOutput_mean.absObjectiveError = phmaxOutput_mean.absObjectiveError/noOfRepitions;
    
    gdrmOutput_mean.matError = gdrmOutput_mean.matError/noOfRepitions;
    gdrmOutput_mean.t =gdrmOutput_mean.t/noOfRepitions;
    gdrmOutput_mean.iterations = gdrmOutput_mean.iterations/noOfRepitions;
    gdrmOutput_mean.estimateMatError=gdrmOutput_mean.estimateMatError/noOfRepitions;
    gdrmOutput_mean.absObjectiveError = gdrmOutput_mean.absObjectiveError/noOfRepitions;
    
     save(['phmaxOutput_3_only_10_rept_' num2str(M) num2str(N)],'phmaxOutput','gdrmOutput','phmaxOutput_mean','gdrmOutput_mean');
end
end

