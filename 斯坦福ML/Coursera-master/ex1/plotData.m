function plotData (x, y)

figure; %open a new figure window

plot(x, y, 'rx', 'MarkerSize', 10);
ylabel('Profit in $10.000s');
xlabel('Population of city in 10,000');

end