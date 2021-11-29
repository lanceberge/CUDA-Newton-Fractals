% create a plot based on the data in ../data with name in it
% step is an optional argument that allows us to step through
% the newton iteration
function create_plot(name, step)
    clf;
    % read data
    if ~exist('step', 'var')
        df = csvread("../data/"+name+"Data.csv", 1);
  
    else
        df = csvread("../data/"+name+"Data-"+step+".csv", 1);
    end
    
     solns = csvread("../data/"+name+"Solns.csv", 1);

    % rgb triplets for our color scheme
    color_map = [0 0.4470 0.7410; 0.8500 0.3250 0.0980; ... 
        0.4940 0.1840 0.5560; 0.4660 0.6740 0.1880; ...
         0.6350 0.0780 0.1840; 0.3010 0.7450 0.9330; ...
         0.9290 0.6940 0.1250; 1 0 0; .69 .61 .85; ...
         1 .75 .8; 0 0.6 0.3; 0 .5 .5];
     
    pointsize = 25;

    
    if ~(strcmp('bigTest3', name) | strcmp('bigTest3L1', name))
        % is a vector from 1:3 for 3 solutions
        soln_vals = (1:length(solns))';
    
        % map indices to colors - indices are in df(:,3)
        soln_colors = color_map(soln_vals, :);
        df_colors = color_map(df(:,3)+1, :);
        
        % create the plot, df(:, 1) is the Re value, df(:,2) is Im
        scatter(df(:,1), df(:,2), pointsize, df_colors, 'filled');
        hold on;
        
        scatter(solns(:,1), solns(:,2), pointsize+10, soln_colors, ...
        'filled', 'MarkerEdgeColor', 'k');
    
    % bigTest3 has 30 roots, so we need a different colorscheme,
    % as the above one only has 12 triplets
    else
        % use df(:,3) - which has indices corresponding to roots,
        % as our color scheme
        scatter(df(:,1), df(:,2), pointsize, df(:,3), 'filled');
        colormap(hsv);
    end
    
    xlabel("Re");
    ylabel("Im");
   
    if ~(exist('step', 'var'))
        print("../plots/"+name+"Plot.pdf", '-dpdf', '-bestfit');
    else
        title("Step: "+step);
    end
end