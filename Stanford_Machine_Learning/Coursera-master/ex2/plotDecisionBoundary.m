%% Function to plot classifier’s decision boundary
function plotDecisionBoundary(X, y, theta)

plotData(X(:, 2:3), y);
hold on;

if size(X, 2) <= 3
	plot_x = [min(X(:, 2)) - 2, max(X(:, 2)) + 2];
	plot_y = (theta(1) + theta(2) .* plot_x) * (-1 ./ theta(3));
	plot(plot_x, plot_y);
	legend('Admitted', 'Not admitted', 'Decision Boundary');
	axis([30, 100, 30, 100]);
else 
	% Here is the grid range
	u = linspace(-1, 1.5, 50);
	v = linspace(-1, 1.5, 50);
	z = zeros(length(u), length(v));
	for i = 1:length(u)
		for j = 1:length(v)
			z(i,j) = mapFeature(u(i), v(j)) * theta;
		end
	end
	z = z';
	contour(u, v, z, [0,0], 'LineWidth', 2);
	hold on;
	title('lambda = 100')
	% Labels and Legend
	xlabel('Microchip Test 1')
	ylabel('Microchip Test 2')
	legend('y = 1', 'y = 0', 'Decision boundary')
	hold off
	pause;
	figure;
	surf(u, v, z);
end

end