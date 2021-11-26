clf;
df = csvread('../data/smallData.csv', 1);
solns = csvread('../data/smallSolns.csv', 1);

color_map = [0 0.4470 0.7410; 0.8500 0.3250 0.0980; 0.4940 0.1840 0.5560; ...
    0.4660 0.6740 0.1880; 0.6350 0.0780 0.1840; ...
    0.3010 0.7450 0.9330; 0.9290 0.6940 0.1250];

% is a vector from 1:3 for 3 solutions
soln_vals = (1:length(solns))';

% map indices to colors
soln_colors = color_map(soln_vals, :);
df_colors = color_map(df(:,3)+1, :);

pointsize = 100;

scatter(df(:,1), df(:,2), pointsize, df_colors, 'filled');
hold on;

scatter(solns(:,1), solns(:,2), pointsize+10, soln_colors, ...
    'filled', 'MarkerEdgeColor', 'k');

xlabel("Re");
ylabel("Im");

print('../plots/smallPlot.pdf', '-dpdf', '-bestfit', '-painters');
% movefile('smallPlot.pdf', '../plots/smallPlot.pdff');