df = csvread('../data/smallData.csv', 1);
solns = csvread('../data/smallSolns.csv', 1);

%color_map = ['r'; 'g'; 'b'; 'c'; 'm'; 'y'; 'k'];
%color_map = [[1 0 0]; [0 1 0]; [0 0 1]; [0 1 1]];
color_map = [1 1 0; 0 1 0; 0 0 1; 0 1 1];
soln_vals = (1:length(solns))';
soln_colors = color_map(soln_vals, :);
df_colors = color_map(df(:,3)+1, :);
pointsize = 20;

% scatter(df(:,1), df(:,2), pointsize, df(:,3), 'filled');
scatter(df(:,1), df(:,2), pointsize, df_colors, 'filled');
% colormap(gca, 'winter');
hold on;

scatter(solns(:,1), solns(:,2), pointsize+10, soln_colors, ...
    'filled', 'MarkerEdgeColor', 'k');

xlabel("Re");
ylabel("Im");