% UKF_TESTS
%
% This is the MATLAB script used to generate "ground truth" Unscented Kalman
% Filter/Smoother output. To use, download Sarkka's EKF/UKF MATLAB Toolbox
% (v1.3) from http://becs.aalto.fi/en/research/bayes/ekfukf/ , then place this
% file in the "ekfukf/" folder, and execute it. The output should match the
% contents of test_unscented.py.

clear all; close all; clc;

%% setup problem
A = [1, 1; 0, 1];
C = [0.5, -0.3];

x0  = [1; 1];
P0  = [  1, 0.1;
       0.1,   1];
Q   = eye(2) * 2;
R   = 0.5;
Z   = [0, 1, 2, 3];
T   = length(Z);

mu_filt = zeros(2, T);           % mu_filt(:,k) = corrected state estimate for time step k
P_filt  = zeros(2, 2, T);
mu_pred = zeros(size(mu_filt));  % mu_pred(:,k) = predicted state estimate for time step k
P_pred  = zeros(size(P_filt));
K       = zeros(size(P_filt));   % Kalman gain matrices

alpha = 1.0;
beta  = 0.0;
kappa = 3.0 - 2;

%% run Additive Kalman Filter
f   = @(x,param) A*x;
g   = @(x,param) C*x;
for k=1:length(Z)
  % correct
  if k > 1
    [mu_filt(:,k), P_filt(:,:,k)] = ukf_update1(mu_pred(:,k-1), P_pred(:,:,k-1), Z(k), g, R);
  else
    % no correct at t = 1
    mu_filt(:,k)  = x0;
    P_filt(:,:,k) = P0;
  end
  
  % predict
  [mu_pred(:,k), P_pred(:,:,k)] = ukf_predict1(mu_filt(:,k), P_filt(:,:,k), f, Q);
end

%% run Additive Kalman Smoother
[mu_smooth, P_smooth] = urts_smooth1(mu_filt, P_filt, f, Q);

%% print output

fprintf(1, 'Additive Kalmans Filter/Smoother:\n')
mu_filt
P_filt

% ipdb> mu_est
% array([[ 1.        ,  1.        ],
%        [ 2.35637584,  0.9295302 ],
%        [ 4.39153259,  1.1514893 ],
%        [ 6.71906244,  1.52810614]])
% ipdb> sigma_est
% array([[[ 1.        ,  0.1       ],
%         [ 0.1       ,  1.        ]],
% 
%        [[ 2.09738255,  1.51577181],
%         [ 1.51577181,  2.91778523]],
% 
%        [[ 3.62532578,  3.14443734],
%         [ 3.14443734,  4.65898912]],
% 
%        [[ 4.39024659,  3.90194407],
%         [ 3.90194407,  5.40957304]]])

mu_smooth
P_smooth

% ipdb> mu_est
% array([[ 1.21559491,  1.34601118],
%        [ 2.92725011,  1.63582509],
%        [ 4.8744743 ,  1.64678689],
%        [ 6.71906244,  1.52810614]])
% ipdb> sigma_est
% array([[[ 0.77444588, -0.09477913],
%         [-0.09477913,  0.67033352]],
% 
%        [[ 0.99379976,  0.21601451],
%         [ 0.21601451,  1.25274857]],
% 
%        [[ 1.5708688 ,  1.03741786],
%         [ 1.03741786,  2.49806236]],
% 
%        [[ 4.39024659,  3.90194407],
%         [ 3.90194407,  5.40957304]]])

%% run Augmented Kalman Filter
f = @(xw, param) A * xw(1:2) + xw(3:4);
g = @(xw, param) C * xw(1:2) + xw(5);
mu_filt = zeros(size(mu_filt));
P_filt  = zeros(size(P_filt));
mu_pred = zeros(size(mu_pred));
P_pred  = zeros(size(P_pred));
X       = zeros([5, 11, 4]);
w       = {};
for k=1:length(Z)
  % correct
  if k > 1
    [mu_filt(:,k), P_filt(:,:,k)] = ukf_update3(mu_pred(:,k-1), P_pred(:,:,k-1), Z(k), g, R, X(:,:,k-1), w{k-1}, {}, alpha, beta, kappa);
  else
    % no correct at t = 1
    mu_filt(:,k)  = x0;
    P_filt(:,:,k) = P0;
  end
  
  % predict
  [mu_pred(:,k), P_pred(:,:,k), X(:,:,k), w{k}] = ukf_predict3(mu_filt(:,k), P_filt(:,:,k), f, Q, R, {}, alpha, beta, kappa);
end

%% run Augmented Kalman Smoother
[mu_smooth, P_smooth] = urts_smooth2(mu_filt, P_filt, f, Q);

fprintf(1, 'Augmented Kalmans Filter/Smoother:\n')
mu_filt
P_filt

% ipdb> mu_est
% array([[ 1.        ,  1.        ],
%        [ 2.35637584,  0.9295302 ],
%        [ 4.39153259,  1.1514893 ],
%        [ 6.71906244,  1.52810614]])
% ipdb> sigma_est
% array([[[ 1.        ,  0.1       ],
%         [ 0.1       ,  1.        ]],
% 
%        [[ 2.09738255,  1.51577181],
%         [ 1.51577181,  2.91778523]],
% 
%        [[ 3.62532578,  3.14443734],
%         [ 3.14443734,  4.65898912]],
% 
%        [[ 4.39024659,  3.90194407],
%         [ 3.90194407,  5.40957304]]])

mu_smooth
P_smooth

% mu_true = np.zeros((3, 2), dtype=float)
% mu_true[0] = [2.92725011530645, 1.63582509442842]
% mu_true[1] = [4.87447429684622,  1.6467868915685]
% mu_true[2] = [6.71906243764755, 1.52810614201467]
% 
% sigma_true = np.zeros((3, 2, 2), dtype=float)
% sigma_true[0] = [[0.993799756492982, 0.216014513083516],
%                  [0.216014513083516, 1.25274857496387]]
% sigma_true[1] = [[1.57086880378025, 1.03741785934464],
%                  [1.03741785934464, 2.49806235789068]]
% sigma_true[2] = [[4.3902465859811, 3.90194406652627],
%                  [3.90194406652627, 5.40957304471697]]
