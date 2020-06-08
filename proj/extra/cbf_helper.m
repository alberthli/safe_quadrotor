%% overview
% this is a helper script to symbolically compute the various cbfs and
% corresponding derivatives for the quadrotor system.

%% dynamics and symbols
syms x y z phi theta psi u v w p q r real; % states
syms g m Ix Iy Iz real; % constants/params
syms d_s ang_s v_s real; % safety params
syms x_o y_o z_o real; % obstacle pos
s = [x y z phi theta psi u v w p q r]';

%% intermediates
cphi = cos(phi);
sphi = sin(phi);
cth = cos(theta);
sth = sin(theta);
cpsi = cos(psi);
spsi = sin(psi);
tth = tan(theta);

Rwb = [cth * cpsi, sphi * sth * cpsi - cphi * spsi, cphi * sth * cpsi + sphi * spsi;
       cth * spsi, sphi * sth * spsi + cphi * cpsi, cphi * sth * spsi - sphi * cpsi;
       -sth, sphi * cth, cphi * cth];
Twb = [1, sphi * tth, cphi * tth;
       0, cphi, -sphi;
       0, sphi / cth, cphi / cth];
   
%% dynamics
f = [Rwb * [u;v;w];
     Twb * [p;q;r];
     r * v - q * w + g * sth;
     p * w - r * u - g * sphi * cth;
     q * u - p * v - g * cth * cphi;
     ((Iy - Iz) * q * r) / Ix;
     ((Iz - Ix) * p * r) / Iy;
     ((Ix - Iy) * p * q) / Iz];
gdyn = [zeros(8,4);
     1 / m, 0, 0, 0;
     0, 1 / Ix, 0, 0;
     0, 0, 1 / Iy, 0;
     0, 0, 0, 1 / Iz];
 
%% linear velocity functions
h_v = simplify(v_s^2 - norm([u;v;w])^2);
dh_vds = simplify([diff(h_v,s(1));
                   diff(h_v,s(2));
                   diff(h_v,s(3));
                   diff(h_v,s(4));
                   diff(h_v,s(5));
                   diff(h_v,s(6));
                   diff(h_v,s(7));
                   diff(h_v,s(8));
                   diff(h_v,s(9));
                   diff(h_v,s(10));
                   diff(h_v,s(11));
                   diff(h_v,s(12))]);
lfh_v = simplify(dh_vds' * f);
lgh_v = simplify(dh_vds' * gdyn);

%% roll limit functions
h_phi = ang_s^2 - phi^2;
dh_phids = simplify([diff(h_phi,s(1));
                     diff(h_phi,s(2));
                     diff(h_phi,s(3));
                     diff(h_phi,s(4));
                     diff(h_phi,s(5));
                     diff(h_phi,s(6));
                     diff(h_phi,s(7));
                     diff(h_phi,s(8));
                     diff(h_phi,s(9));
                     diff(h_phi,s(10));
                     diff(h_phi,s(11));
                     diff(h_phi,s(12))]);
lfh_phi = simplify(dh_phids' * f);
dlfh_phids = simplify([diff(lfh_phi,s(1));
                       diff(lfh_phi,s(2));
                       diff(lfh_phi,s(3));
                       diff(lfh_phi,s(4));
                       diff(lfh_phi,s(5));
                       diff(lfh_phi,s(6));
                       diff(lfh_phi,s(7));
                       diff(lfh_phi,s(8));
                       diff(lfh_phi,s(9));
                       diff(lfh_phi,s(10));
                       diff(lfh_phi,s(11));
                       diff(lfh_phi,s(12))]);
lf2h_phi = simplify(dlfh_phids' * f);
lglfh_phi = simplify(dlfh_phids' * gdyn);

%% pitch limit functions
h_th = ang_s^2 - theta^2;
dh_thds = simplify([diff(h_th,s(1));
                    diff(h_th,s(2));
                    diff(h_th,s(3));
                    diff(h_th,s(4));
                    diff(h_th,s(5));
                    diff(h_th,s(6));
                    diff(h_th,s(7));
                    diff(h_th,s(8));
                    diff(h_th,s(9));
                    diff(h_th,s(10));
                    diff(h_th,s(11));
                    diff(h_th,s(12))]);
lfh_th = simplify(dh_thds' * f);
dlfh_thds = simplify([diff(lfh_th,s(1));
                      diff(lfh_th,s(2));
                      diff(lfh_th,s(3));
                      diff(lfh_th,s(4));
                      diff(lfh_th,s(5));
                      diff(lfh_th,s(6));
                      diff(lfh_th,s(7));
                      diff(lfh_th,s(8));
                      diff(lfh_th,s(9));
                      diff(lfh_th,s(10));
                      diff(lfh_th,s(11));
                      diff(lfh_th,s(12))]);
lf2h_th = simplify(dlfh_thds' * f);
lglfh_th = simplify(dlfh_thds' * gdyn);

%% spherical obstacle constraints
h_o = simplify(norm(s(1:3) - [x_o;y_o;z_o])^2 - d_s^2);
dh_ods = simplify([diff(h_o,s(1));
                    diff(h_o,s(2));
                    diff(h_o,s(3));
                    diff(h_o,s(4));
                    diff(h_o,s(5));
                    diff(h_o,s(6));
                    diff(h_o,s(7));
                    diff(h_o,s(8));
                    diff(h_o,s(9));
                    diff(h_o,s(10));
                    diff(h_o,s(11));
                    diff(h_o,s(12))]);
lfh_o = simplify(dh_ods' * f);
dlfh_ods = simplify([diff(lfh_o,s(1));
                     diff(lfh_o,s(2));
                     diff(lfh_o,s(3));
                     diff(lfh_o,s(4));
                     diff(lfh_o,s(5));
                     diff(lfh_o,s(6));
                     diff(lfh_o,s(7));
                     diff(lfh_o,s(8));
                     diff(lfh_o,s(9));
                     diff(lfh_o,s(10));
                     diff(lfh_o,s(11));
                     diff(lfh_o,s(12))]);
lf2h_o = simplify(dlfh_ods' * f);
lglfh_o = simplify(dlfh_ods' * gdyn);

