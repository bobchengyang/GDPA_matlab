clear;
clc;
close all;
%% std
results_std_mean=zeros(17,62);
results_std_std=zeros(17,62);
for dataset_i=1:17
    if dataset_i~=5 && dataset_i~=7 && dataset_i~=17
        n_to_be_avg=50;
    elseif dataset_i==5
        n_to_be_avg=40;
    elseif dataset_i==7
        n_to_be_avg=20;  
    elseif dataset_i==17
        n_to_be_avg=10;        
    end
    [dataset_str] = get_dataset_name(dataset_i);
    result_str=['results_' dataset_str '_traintest_std_noise.mat'];
    load(result_str);
    
    results_std_mean(dataset_i,:)=mean(results(1:n_to_be_avg,:));
    results_std_std(dataset_i,:)=std(results(1:n_to_be_avg,:));
end

load results_std_for_plot_noise.mat

results_std_for_plot_reorder=results_std_for_plot_noise(:,[22 16 20 ...
    6 8 4]);

datasize_order=[139	137	153	172	250	61	135	117	29	110	153	36	86	114	42	400	62];
[datasize_order_value,datasize_order_idx]=sort(datasize_order);
datasize_order_idx=[datasize_order_idx 18];

results_std_for_plot_reorder=results_std_for_plot_reorder(datasize_order_idx,:);

method_name=["(8) MOSEK" '(8) CDCS'...
    '(12) SDcut'...
    '(21) CDCS' '(21) SDcut'...
    '\color{black}\bf(22) GDPA'];
names = {'australian'; 'breast-cancer'; 'diabetes';...
    'fourclass'; 'german'; 'haberman';...
    'heart'; 'ILPD'; 'liver-disorders';...
    'monk1'; 'pima'; 'planning';...
    'voting'; 'WDBC'; 'sonar';...
    'madelon'; 'colon-cancer'; '\color{black}\bfavg.'};
names=names(datasize_order_idx);

ncolors = distinguishable_colors(6);
figure(1);hold on;
for i=1:size(results_std_for_plot_reorder,2) % number of methods
    if i<=2
        plot(results_std_for_plot_reorder(:,i),...
            'LineStyle','none',...
            'LineWidth',1,...
            'Marker','+',...
            'color',ncolors(i,:),'DisplayName',num2str(method_name(i)));
    elseif i>2 && i<=3 
        plot(results_std_for_plot_reorder(:,i),...
            'LineStyle','none',...
            'LineWidth',1,...
            'Marker','s','color',ncolors(i,:),'DisplayName',num2str(method_name(i)));
    elseif i>3 && i~=6
        plot(results_std_for_plot_reorder(:,i),...
            'LineStyle','none',...
            'LineWidth',1,...
            'Marker','p','color',ncolors(i,:),'DisplayName',num2str(method_name(i)));
    else
        plot(results_std_for_plot_reorder(:,i),...
            'LineStyle','none',...
            'LineWidth',1,...
            'Marker','o','color','blue','DisplayName',num2str(method_name(i)));
    end
end

% xlabel('dataset no.', 'FontSize', 12);
ylabel('error rate (%)', 'FontSize', 12);
set(gca,'fontname','times', 'FontSize', 12)  % Set it to times
xlim([1 18]);
set(gca,'xtick',(1:18),'xticklabel',names);xtickangle(90);
ylim([min(vec(results_std_for_plot_reorder)) max(vec(results_std_for_plot_reorder))]);
grid on;
legend;


results_std_for_plot_reorder_time=results_std_for_plot_noise(:,[22 16 20 ...
    6 8 4]-1);
results_std_for_plot_reorder_time=results_std_for_plot_reorder_time(datasize_order_idx,:);

method_name=["(8) MOSEK" '(8) CDCS'...
    '(12) SDcut'...
    '(21) CDCS' '(21) SDcut'...
    '\color{black}\bf(22) GDPA'];
names = {'australian'; 'breast-cancer'; 'diabetes';...
    'fourclass'; 'german'; 'haberman';...
    'heart'; 'ILPD'; 'liver-disorders';...
    'monk1'; 'pima'; 'planning';...
    'voting'; 'WDBC'; 'sonar';...
    'madelon'; 'colon-cancer'; '\color{black}\bfavg.'};
names=names(datasize_order_idx);
% names(end)=[];
% results_std_for_plot_reorder_time=results_std_for_plot_reorder_time(1:end-1,:);
figure(2);hold on;
for i=1:size(results_std_for_plot_reorder_time,2) % number of methods
    if i<=2
        plot(results_std_for_plot_reorder_time(:,i),...
            'LineStyle','none',...
            'LineWidth',1,...
            'Marker','+',...
            'color',ncolors(i,:),'DisplayName',num2str(method_name(i)));
    elseif i>2 && i<=3
        plot(results_std_for_plot_reorder_time(:,i),...
            'LineStyle','none',...
            'LineWidth',1,...
            'Marker','s','color',ncolors(i,:),'DisplayName',num2str(method_name(i)));
    elseif i>3 && i~=6
        plot(results_std_for_plot_reorder_time(:,i),...
            'LineStyle','none',...
            'LineWidth',1,...
            'Marker','p','color',ncolors(i,:),'DisplayName',num2str(method_name(i)));
    else
        plot(results_std_for_plot_reorder_time(:,i),...
            'LineStyle','none',...
            'LineWidth',1,...
            'Marker','o','color','blue','DisplayName',num2str(method_name(i)));
    end
end

% xlabel('dataset no.', 'FontSize', 12);
ylabel('runtime (ms)', 'FontSize', 12);
set(gca,'fontname','times', 'FontSize', 12)  % Set it to times
xlim([1 18]);
set(gca,'xtick',(1:18),'xticklabel',names);xtickangle(90);
ylim([min(vec(results_std_for_plot_reorder_time)) max(vec(results_std_for_plot_reorder_time))]);
grid on;
set(gca, 'YScale', 'log')
legend;


%% std





