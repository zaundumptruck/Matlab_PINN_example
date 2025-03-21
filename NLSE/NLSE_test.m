clc;
clear all;

%% Generate training data
freq = 2000;                     % Frequency (Hz)
c0 = 340;                       % Speed of sound in air (m/s)
k = 2*pi*freq/c0;               % Wavenumber (1/m)

x0BC1 = 0;                      % Left boundary (m)
x0BC2 = 1;                      % Right boundary (m)

u0BC1 = 1;                      % Left boundary value (Pa)
u0BC2 = -1;                     % Right boundary value (Pa)

X0 = [x0BC1 x0BC2];     
U0 = [u0BC1 u0BC2];
