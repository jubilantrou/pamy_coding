function [k] = Ziegler_Nichols(kc, Tc, mode_name);

switch mode_name
    case "P"
        disp('P-Controller')
        kp = 0.5 * kc;
        ki = 0;
        kd = 0;      
    case "PI"
        disp('PI-Controller')
        kp = 0.45 * kc;
        ki = 0.54 * kc / Tc;
        kd = 0;      
    case "PD"
        disp('PD-Controller')
        kp = 0.8 * kc;
        ki = 0;
        kd = 0.1 * kc * Tc;
    case "PID"
        disp('PID-Controller')
        kp = 0.6 * kc;
        ki = 1.2 * kc  / Tc;
        kd = 0.075 * kc * Tc;
    case "PIDso"
        disp('PID-Controller with some overshoot')
        kp = 0.33 * kc;
        ki = 0.66 * kc  / Tc;
        kd = 0.11 * kc * Tc;
    case "PIDno"
        disp('PID-Controller with no overshoot')
        kp = 0.20 * kc;
        ki = 0.40 * kc  / Tc;
        kd = 0.066 * kc * Tc;
    case "PIR"
        disp('PID-Controller with PIR')
        kp = 0.7 * kc;
        ki = 1.75 * kc  / Tc;
        kd = 0.105 * kc * Tc;
    otherwise
        disp('no such mode')
        kp = 0;
        ki = 0;
        kd = 0;
end

k = [kp, ki, kd];
end

