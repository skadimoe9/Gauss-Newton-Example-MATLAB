%% Tugas 7 - Nonlinear Regression : Gauss Newton Method
% Fadhli Ammar Taqiyuddin Hakim - 2206817396

clear, clc 

%% 14.8
% Initialize 

tol = 1e-6; 
cnt = 1; 
data_num = 9; % Total of the provided data

syms a b x y

par = [a;b;x];

f(par) = a*x*exp(b*x); 
y_funct = f(a,b,x); 

res_funct([a;b;x;y]) = a*x*exp(b*x) - y; 
Jc =  jacobian(res_funct,[a;b]);

x_data = [0.1 0.2 0.4 0.6 0.9 1.3 1.5 1.7 1.8]; 
y_data = [0.75 1.25 1.45 1.25 0.85 0.55 0.35 0.28 0.18]; 

alpha = 9; % Initial value 
beta = -2; % Initial value

phi = [alpha;beta]; 

% Initialize y_hat and residual vector and calculate phi
y_hat = zeros(1,data_num); % Estimated output from the model
res_vector = zeros(data_num,1); % Residual vector of all data
Jc_res = zeros(data_num,2); % Jacobian matrix of the residual vector of all data

while true 
    fprintf("--- Iter %d ---\n", cnt)
    for i = 1:data_num
        y_hat(i) = vpa(subs(y_funct, [a;b;x], [phi(1);phi(2);x_data(i)]));  
        res_vector(i) = y_hat(i) - y_data(i);
        Jc_res(i,:) = vpa(subs(Jc, [a;b;x;y], [phi(1);phi(2);x_data(i);0]));
    end 
    
    res_vector % Just for display
    Jc_res % Just for display

    phi_old = phi;
    phi = phi - inv((Jc_res.')*Jc_res)*(Jc_res.')*res_vector
    ea = norm(phi-phi_old)/norm(phi)

    if ea < tol
        break 
    end 

    cnt = cnt+1; 
end 

plot_x = 0:0.01:2; 
plot_y = phi(1).*plot_x.*exp(phi(2).*plot_x);

figure
plot(plot_x,plot_y) 
title('Fadhli Ammar T.H. - 14.8 : Gauss Newton')
hold on 
n = 1; 
while(n<data_num+1) 
    plot_titik = scatter(x_data(n), y_data(n),'bo','LineWidth',1,...
                            'MarkerEdgeColor','k',...
                            'MarkerFaceColor','b'); 
    n = n+1; 
end  

alpha = phi(1); 
beta = phi(2); 

fprintf("\nalpha = %f\nbeta = %f", phi(1), phi(2)) 
fprintf("\nTotal iteration = %d", cnt) 

fprintf("\n\n-- 14.8 Finished --\n\n")

%% 14.14 
clear 

% Initialize 

tol = 1e-6; 
cnt = 1; 
data_num = 5; % Total of the provided data

syms k_m cs c k

par = [k_m;cs;c];

f(par) = (k_m*(c^2))/(cs+(c^2)); 
y_funct = f(k_m,cs,c); 

res_funct([k_m;cs;c;k]) = (k_m*(c^2))/(cs+(c^2)) - k; 
Jc =  jacobian(res_funct,[k_m;cs]);

x_data = [0.5 0.8 1.5 2.5 4]; 
y_data = [1.1 2.5 5.3 7.6 8.9]; 

k_max = 2; % Initial value 
c_s = 2; % Initial value

phi = [k_max;c_s]; 

% Initialize y_hat and residual vector and calculate phi
y_hat = zeros(1,data_num); % Estimated output from the model
res_vector = zeros(data_num,1); % Residual vector of all data
Jc_res = zeros(data_num,2); % Jacobian matrix of the residual vector of all data

while true 
    fprintf("--- Iter %d ---\n", cnt)
    for i = 1:data_num
        y_hat(i) = vpa(subs(y_funct, [k_m;cs;c], [phi(1);phi(2);x_data(i)]));  
        res_vector(i) = y_hat(i) - y_data(i);
        Jc_res(i,:) = vpa(subs(Jc, [k_m;cs;c;k], [phi(1);phi(2);x_data(i);0]));
    end 
    
    res_vector % Just for display
    Jc_res % Just for display

    phi_old = phi;
    phi = phi - inv((Jc_res.')*Jc_res)*(Jc_res.')*res_vector
    ea = norm(phi-phi_old)/norm(phi)

    if ea < tol
        break 
    end 

    cnt = cnt+1; 
end 

plot_x = 0:0.1:5; 
plot_y = (phi(1).*(plot_x.^2))./(phi(2)+(plot_x.^2)); 

figure
plot(plot_x,plot_y)
title('Fadhli Ammar T.H. - 14.14 : Gauss Newton')
hold on 
n = 1; 
while(n<data_num+1) 
    plot_titik = scatter(x_data(n), y_data(n),'bo','LineWidth',1,...
                            'MarkerEdgeColor','k',...
                            'MarkerFaceColor','b'); 
    n = n+1; 
end  

k_max = phi(1); 
c_s = phi(2); 

fprintf("\nk_max = %f\ncs = %f", phi(1), phi(2)) 
fprintf("\nTotal iteration = %d", cnt)

fprintf("\n\n-- 14.14 Finished --\n\n")