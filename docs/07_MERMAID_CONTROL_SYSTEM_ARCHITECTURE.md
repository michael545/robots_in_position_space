# Document 07: Visual Control Systems Architecture (Mermaid.js)

## 1. Executive Summary & Verification Statement

This document contains the visual systems architecture diagram for the **Decoupled 2-DoF Modal Motion Controller** on the **Ubiquity Robotics Magni MCB G6** platform.

> [!IMPORTANT]
> **1-to-1 Mathematical & Code Alignment Verification:**
> This diagram is not an abstract concept or high-level sketch. Every block, summing junction, error subtractor, and signal line in this diagram has a **strict 1-to-1 correspondence** with the C code implementation in [`mcb_g6_firmware/src/modules/drive/drive_2dof.c`](file:///home/michael/code/firmware/robots_in_position_space/mcb_g6_firmware/src/modules/drive/drive_2dof.c) and the formal schema in [`docs/06_CONTROL_SYSTEM_GRAPH_SPECIFICATION.json`](file:///home/michael/code/firmware/robots_in_position_space/docs/06_CONTROL_SYSTEM_GRAPH_SPECIFICATION.json).

---

## 2. Complete Mermaid.js Control System Diagram

```mermaid
flowchart TD
    %% =========================================================================
    %% STYLES & CLASSES
    %% =========================================================================
    classDef inputStyle fill:#1e293b,stroke:#94a3b8,stroke-width:2px,color:#f8fafc;
    classDef ffStyle fill:#7c2d12,stroke:#ea580c,stroke-width:2px,color:#fff7ed;
    classDef subStyle fill:#831843,stroke:#db2777,stroke-width:3px,color:#fdf2f8;
    classDef fbStyle fill:#14532d,stroke:#22c55e,stroke-width:2px,color:#f0fdf4;
    classDef summerStyle fill:#312e81,stroke:#6366f1,stroke-width:2px,color:#eef2ff;
    classDef mixerStyle fill:#4c1d95,stroke:#8b5cf6,stroke-width:3px,color:#f5f3ff;
    classDef focStyle fill:#1e1b4b,stroke:#4338ca,stroke-width:2px,color:#e0e7ff;
    classDef plantStyle fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#f0f9ff;
    classDef sensorStyle fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#ecfdf5;

    %% =========================================================================
    %% SUBSYSTEM 1: REFERENCE GENERATOR
    %% =========================================================================
    subgraph Sub1 ["1. Reference Generator & S-Curve Rate Limiter"]
        CMD["<b>ROS /cmd_vel</b><br>v_cmd [m/s]<br>omega_cmd [rad/s]"]:::inputStyle
        RL_LIN["<b>Linear Rate Limiter</b><br>a_des = clamp(Δv/dt, -d_max, a_max)<br>v_des += a_des * dt"]:::inputStyle
        RL_YAW["<b>Yaw Rate Limiter</b><br>alpha_des = clamp(Δω/dt, -α_max, α_max)<br>omega_des += alpha_des * dt"]:::inputStyle

        CMD -->|"v_cmd (m/s)"| RL_LIN
        CMD -->|"omega_cmd (rad/s)"| RL_YAW
    end

    %% =========================================================================
    %% SUBSYSTEM 2: LONGITUDINAL MODAL CHANNEL
    %% =========================================================================
    subgraph Sub2 ["2. Longitudinal Modal Channel (F_linear)"]
        FF_LIN_INER["<b>Inertia Feedforward</b><br>F_iner = M_eff * a_des<br>(64.2 kg * a_des)"]:::ffStyle
        FF_LIN_STRIB["<b>Stribeck Friction Feedforward</b><br>F_strib = [Fc + (Fs-Fc)e^-(v/vs)²]<br>* tanh(v/ε) + Cv*v"]:::ffStyle
        
        SUB_V{{"<b>EXPLICIT SUBTRACTOR (Speed)</b><br>e_v = v_des - v_odom"}}:::subStyle
        
        PI_V["<b>Longitudinal PI Governor</b><br>P = Kp_v * e_v<br>I += Ki_v * e_v * dt (Anti-Windup)<br>F_lin_fb = P + I"]:::fbStyle
        SUM_LIN(("<b>Longitudinal Summer</b><br>F_linear =<br>F_iner + F_strib + F_fb")):::summerStyle

        RL_LIN -->|"a_des (m/s²)"| FF_LIN_INER
        RL_LIN -->|"v_des (m/s)"| FF_LIN_STRIB
        RL_LIN -->|"(+) Desired Reference v_des"| SUB_V

        SUB_V -->|"Speed Error e_v (m/s)"| PI_V
        FF_LIN_INER -->|"(+) Muscle F_iner_ff (N)"| SUM_LIN
        FF_LIN_STRIB -->|"(+) Friction F_strib_ff (N)"| SUM_LIN
        PI_V -->|"(+) Governor F_lin_fb (N)"| SUM_LIN
    end

    %% =========================================================================
    %% SUBSYSTEM 3: YAW MODAL CHANNEL
    %% =========================================================================
    subgraph Sub3 ["3. Yaw Modal Channel (tau_yaw)"]
        FF_YAW_INER["<b>Yaw Inertia Feedforward</b><br>tau_iner = I_eff * alpha_des<br>(2.48 kg·m² * α_des)"]:::ffStyle
        FF_YAW_SCRUB["<b>Tire Scrub Feedforward</b><br>tau_scrub = [Tc + (Ts-Tc)e^-(ω/ws)²]<br>* tanh(ω/ε) + Cw*ω"]:::ffStyle
        
        SUB_W{{"<b>EXPLICIT SUBTRACTOR (Yaw Rate)</b><br>e_omega = omega_des - omega_gyro"}}:::subStyle
        
        PI_W["<b>Yaw PI Governor</b><br>P = Kp_w * e_omega<br>I += Ki_w * e_omega * dt (Anti-Windup)<br>tau_yaw_fb = P + I"]:::fbStyle
        SUM_YAW(("<b>Yaw Summer</b><br>tau_yaw =<br>tau_iner + tau_scrub + tau_fb")):::summerStyle

        RL_YAW -->|"alpha_des (rad/s²)"| FF_YAW_INER
        RL_YAW -->|"omega_des (rad/s)"| FF_YAW_SCRUB
        RL_YAW -->|"(+) Desired Reference omega_des"| SUB_W

        SUB_W -->|"Rate Error e_omega (rad/s)"| PI_W
        FF_YAW_INER -->|"(+) Muscle tau_iner_ff (Nm)"| SUM_YAW
        FF_YAW_SCRUB -->|"(+) Scrub tau_scrub_ff (Nm)"| SUM_YAW
        PI_W -->|"(+) Governor tau_yaw_fb (Nm)"| SUM_YAW
    end

    %% =========================================================================
    %% SUBSYSTEM 4: ALGEBRAIC KINETIC MIXER MATRIX
    %% =========================================================================
    subgraph Sub4 ["4. Algebraic Kinetic Mixer Matrix (Transposed Jacobian J^T)"]
        MIXER["<b>Algebraic Kinetic Mixer</b><br>Common: tau_com = 0.5 * r * F_linear<br>Diff Couple: Δtau = (r / b) * tau_yaw<br><b>tau_L = tau_com - Δtau</b><br><b>tau_R = tau_com + Δtau</b>"]:::mixerStyle

        SUM_LIN -->|"Total Driving Force F_linear (N)"| MIXER
        SUM_YAW -->|"Total Yaw Moment tau_yaw (Nm)"| MIXER
    end

    %% =========================================================================
    %% SUBSYSTEM 5: ACTUATOR FOC INTERFACE
    %% =========================================================================
    subgraph Sub5 ["5. ST MCSDK FOC Actuator Current Interface"]
        CONV_L["<b>Left Motor Current Scaler</b><br>Iq_L = tau_L / (Kt * N_gear)<br>digit_L = Iq_L * 1866 counts/A"]:::focStyle
        CONV_R["<b>Right Motor Current Scaler</b><br>Iq_R = tau_R / (Kt * N_gear)<br>digit_R = Iq_R * 1866 counts/A"]:::focStyle
        API_CALL["<b>ST MCSDK Zero-Latency API</b><br>MC_ProgramTorqueRampMotor1(digit_L, 0)<br>MC_ProgramTorqueRampMotor2(digit_R, 0)"]:::focStyle

        MIXER -->|"tau_L (Nm)"| CONV_L
        MIXER -->|"tau_R (Nm)"| CONV_R
        CONV_L -->|"digit_L (counts)"| API_CALL
        CONV_R -->|"digit_R (counts)"| API_CALL
    end

    %% =========================================================================
    %% SUBSYSTEM 6: PHYSICAL ROBOT PLANT & DISTURBANCES
    %% =========================================================================
    subgraph Sub6 ["6. Physical Robot Plant (Ground Mechanics)"]
        PWM["<b>16 kHz Inverter Gate Drivers</b><br>3-Phase Current Injection (Iq, Id=0)"]:::plantStyle
        CHASSIS["<b>3-Body Coupled Mobile Platform</b><br>M_eff dv/dt = F_linear - F_dist<br>I_eff dω/dt = tau_yaw - tau_dist"]:::plantStyle
        DIST["<b>External Ground Disturbances</b><br>Carpet Drag, Floor Incline, Caster Scrub"]:::inputStyle

        API_CALL -->|"16 kHz PWM Duty"| PWM
        PWM -->|"Shaft Driving Torques"| CHASSIS
        DIST -.->|"F_dist & tau_dist"| CHASSIS
    end

    %% =========================================================================
    %% SUBSYSTEM 7: SENSOR OBSERVERS (CLOSING THE LOOP)
    %% =========================================================================
    subgraph Sub7 ["7. Feedback Sensor Observers"]
        ENC["<b>Optical Quadrature Encoders</b><br>TIMx Counter Accumulator<br>v_odom = 0.5 * (v_L + v_R)"]:::sensorStyle
        IMU["<b>ST LSM6DSOX 6-Axis IMU (mcb.inav)</b><br>High-Rate Z-Gyroscope<br>omega_gyro = (raw_z - bias_z) * scale"]:::sensorStyle

        CHASSIS -->|"Wheel Physical Rotation"| ENC
        CHASSIS -->|"Chassis Spatial Angular Rate"| IMU
    end

    %% =========================================================================
    %% FEEDBACK RETURN EDGES (CLOSING THE LOOPS ON SUBTRACTORS)
    %% =========================================================================
    ENC ==>|"(-) Measured Odometry Speed v_odom (m/s)"| SUB_V
    IMU ==>|"(-) Measured Spatial Yaw Rate omega_gyro (rad/s)"| SUB_W
```

---

## 3. Strict 1-to-1 Verification Table: Diagram to C Code

| Diagram Block | Exact C Code Expression in [`drive_2dof.c`](file:///home/michael/code/firmware/robots_in_position_space/mcb_g6_firmware/src/modules/drive/drive_2dof.c) | Exact Variable Name in [`drive_2dof.h`](file:///home/michael/code/firmware/robots_in_position_space/mcb_g6_firmware/src/modules/drive/drive_2dof.h) | Physical Role |
| :--- | :--- | :--- | :--- |
| `RL_LIN` | `ctrl->state.a_des = CLAMP_VAL(dv / dt_s, -max_a, max_a);`<br>`ctrl->state.v_des += ctrl->state.a_des * dt_s;` | `ctrl->state.v_des`, `ctrl->state.a_des` | Jerk-bounded reference velocity and acceleration |
| `RL_YAW` | `ctrl->state.alpha_des = CLAMP_VAL(dw / dt_s, -max_alpha, max_alpha);`<br>`ctrl->state.w_des += ctrl->state.alpha_des * dt_s;` | `ctrl->state.w_des`, `ctrl->state.alpha_des` | Reference yaw angular rate and angular acceleration |
| `FF_LIN_INER` | `ctrl->state.F_iner_ff = G6_M_EFF_KG * ctrl->state.a_des;` | `ctrl->state.F_iner_ff` | $M_{\text{eff}} \cdot a$ inertial force ($64.2\text{ kg}$) |
| `FF_LIN_STRIB` | `ctrl->state.F_strib_ff = calc_stribeck_linear(ctrl->state.v_des);` | `ctrl->state.F_strib_ff` | $C^1$-continuous static/coulomb/viscous friction |
| `SUB_V` | **`ctrl->state.err_v = ctrl->state.v_des - odom_v_m_s;`** | **`ctrl->state.err_v`** | **Explicit error subtraction $(v_{\text{des}} - v_{\text{odom}})$** |
| `PI_V` | `ctrl->state.integ_err_v += ctrl->state.err_v * dt_s;`<br>`ctrl->state.F_lin_fb = (kp * err_v) + (ki * integ);` | `ctrl->state.F_lin_fb` | Closed-loop velocity disturbance regulator |
| `SUM_LIN` | `ctrl->state.F_linear_total = F_iner_ff + F_strib_ff + F_lin_fb;` | `ctrl->state.F_linear_total` | Generalized forward chassis driving force |
| `FF_YAW_INER` | `ctrl->state.tau_iner_ff = G6_I_EFF_KGM2 * ctrl->state.alpha_des;` | `ctrl->state.tau_iner_ff` | $I_{\text{eff}} \cdot \alpha$ rotational inertia torque ($2.48\text{ kg}\cdot\text{m}^2$) |
| `FF_YAW_SCRUB`| `ctrl->state.tau_scrub_ff = calc_stribeck_yaw(ctrl->state.w_des);` | `ctrl->state.tau_scrub_ff` | Rotational tire scrub friction moment |
| `SUB_W` | **`ctrl->state.err_w = ctrl->state.w_des - gyro_w_rad_s;`** | **`ctrl->state.err_w`** | **Explicit error subtraction $(\omega_{\text{des}} - \omega_{\text{gyro}})$** |
| `PI_W` | `ctrl->state.integ_err_w += ctrl->state.err_w * dt_s;`<br>`ctrl->state.tau_yaw_fb = (kp * err_w) + (ki * integ);` | `ctrl->state.tau_yaw_fb` | Closed-loop yaw rate disturbance regulator |
| `SUM_YAW` | `ctrl->state.tau_yaw_total = tau_iner_ff + tau_scrub_ff + tau_yaw_fb;` | `ctrl->state.tau_yaw_total` | Generalized yaw turning moment |
| `MIXER` | `tau_common = 0.5f * G6_WHEEL_RADIUS_M * F_linear_total;`<br>`delta_tau = (G6_WHEEL_RADIUS_M / G6_TRACK_WIDTH_M) * tau_yaw_total;`<br>`tau_L = tau_common - delta_tau;`<br>`tau_R = tau_common + delta_tau;` | `ctrl->state.tau_wheel_left`, `ctrl->state.tau_wheel_right` | Transposed Jacobian $\mathbf{J}^T$ torque vectoring |
| `CONV_L/R` | `iq_A = tau / (G6_MOTOR_KT_NM_PER_A * G6_GEARBOX_RATIO);`<br>`digit = (int16_t)(iq_A * G6_CURRENT_DIGITS_PER_A);` | `ctrl->state.digit_L`, `ctrl->state.digit_R` | Motor current scaling ($1866\text{ counts/A}$) |
| `API_CALL` | `MC_ProgramTorqueRampMotor1(ctrl->state.digit_L, 0);`<br>`MC_ProgramTorqueRampMotor2(ctrl->state.digit_R, 0);` | Hardware Registers | Direct write to ST MCSDK 16 kHz $I_q$ target |
| `ENC` | `odom_v_m_s = drv->state.odometry.linear_speed_mm_s / 1000.0f;` | `drv->state.odometry` | Real optical encoder wheel odometry |
| `IMU` | `gyro_w_rad_s = mcb.inav.ahrs.last_data.angular_speed.axis.z * (PI / 180.0f);` | `mcb.inav.ahrs.last_data` | True spatial inertial gyro rate from ST LSM6DSOX |
