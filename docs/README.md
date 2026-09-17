# Motion Control in Position Space: Engineering Knowledge Base

Welcome to the definitive engineering documentation suite for the **Ubiquity Robotics Magni MCB G6** motion control and trajectory architecture.

This documentation captures the first-principles mathematics, multi-body physical modeling, and firmware implementation of the transition from legacy velocity-based control (`cmd_vel`) to a deterministic **Position-Space Continuous-to-Discrete (C2D) Trajectory Architecture**.

---

## Documentation Index

### [Document 01: Executive Summary & Legacy Control Audit](01_EXECUTIVE_SUMMARY_AND_LEGACY_AUDIT.md)
* **The Legacy Paradigm:** Deconstructing the Ubiquity Robotics slide: *"cmd_vel gives us speed, and defines a 'dead man' time. Distance is thus implicitly defined."*
* **The Firmware Reality (`drive.c`):** Exposing that `current_speed` is a fictional software simulation, showing that odometry is 100% ignored on curves, and analyzing the stopping distance run-out.
* **The Startup Dead-Time:** Why pure feedback position control ($K_p \cdot e$) produces zero initial torque and causes violent velocity overshoot humps.

### [Document 02: Decoupled 3-Body Dynamics & Continuous Friction Modeling](02_DECOUPLED_3BODY_DYNAMICS_AND_PHYSICS.md)
* **The 3-Body Model:** Rigid body assembly of Chassis ($M, I_z$) + Left Wheel ($m_w, I_w$) + Right Wheel ($m_w, I_w$).
* **Kinetic Energy Derivations:** Mathematical proofs for effective linear mass $M_{eff} = M + 2m_w + 2\frac{I_w}{r^2}$ and effective yaw inertia $I_{eff} = I_z + \frac{m_w b^2}{2} + \frac{I_w b^2}{2r^2}$.
* **Continuous Stribeck Friction Tribology:** Replacing discontinuous $\text{sgn}(v)$ steps with $[C_r + C_s e^{-(v/v_s)^2}]\tanh(v/\epsilon)$ to strictly bound derivative rates and eliminate infinite jerk and motor current buzzing.
* **Tire Scrub Mechanics:** Modeling the lateral scrub moment opposing chassis rotation during differential turns.

### [Document 03: The Algebraic Kinetic Mixer & Torque Vectoring](03_ALGEBRAIC_KINETIC_MIXER_AND_TORQUE_VECTORING.md)
* **Newtonian Derivation:** Inverting the $2 \times 2$ force-moment balance matrix to map Modal Space $[F_{linear}, \tau_{yaw}]^T$ to Actuator Space $[\tau_L, \tau_R]^T$.
* **Virtual Work Derivation:** Proving via D'Alembert's principle that $\boldsymbol{\tau} = \mathbf{J}^T \mathbf{W}$ is an exact, energy-conserving coordinate transformation.
* **True Electronic Torque Vectoring:** Direct Yaw-Moment Control (DYC) actively vectoring differential torque ($\pm \Delta \tau$) to snap into turns with zero steering lag.
* **Three Turning Regimes:** Highway curves ($R \gg b/2$), Pivot turns ($R = b/2$), and Reverse-wheel hairpin turns ($R < b/2$).

### [Document 04: Closed-Loop Modal Feedback & ST MCSDK FOC Integration](04_CLOSED_LOOP_MODAL_FEEDBACK_AND_FOC_INTEGRATION.md)
* **Modal Closed-Loop Trimming:** Wrapping feedback around longitudinal distance ($e_s$) and yaw heading ($e_\theta$) rather than independent wheel PIDs.
* **The 3-Layer Control Stack:** High-level Modal PIDs $\to$ Algebraic Kinetic Mixer $\to$ ST MCSDK 16–32 kHz Hardware FOC Current Loops.
* **Extended Non-Linear PID:** $P_1 E + S_p P_2 E^2$ with $w$-cycle windowed derivative filtering ($\delta_w E / \delta_w t$) for noise-free velocity feedback.
* **Firmware Delivery:** Injecting feedforward torque directly into ST MCSDK's `Iq_ff` register for microsecond electromagnetic response.

### [Document 05: Traction Control & Multi-Sensor Slip Detection](05_TRACTION_CONTROL_AND_SLIP_DETECTION.md)
* **The Wheel Encoder Blindspot:** Why wheel encoders alone cannot detect their own slip.
* **Method 1 (Effective Mass Collapse):** Detecting loss of traction when wheel angular acceleration spikes beyond $\frac{\tau}{M_{eff} r^2}$ due to inertia collapsing from $40\,\text{kg}$ to $1.2\,\text{kg}$.
* **Method 2 (Kinematic vs. Gyroscope Disparity):** Cross-checking wheel differential speed against the onboard IMU $Z$-axis rate gyro.
* **Method 3 (Accelerometer Disparity):** Cross-checking wheel linear acceleration against IMU $X$-axis accelerometer.
* **Active Traction Control (TCS):** Torque clamping, odometry protection, and stall watchdogs.

### [Document 06: Formal Control System Graph Specification (JSON)](06_CONTROL_SYSTEM_GRAPH_SPECIFICATION.json)
* **Unambiguous Topological JSON Graph:** 7 subsystems, 20 functional nodes, and 31 explicit directed edges.
* **Exact Mathematical Port Formulations:** Defines every input/output port, physical engineering units, and transfer equations.
* **Explicit Feedback Error Subtraction:** Mathematically defines the exact nodes where measured wheel odometry ($v_{\text{odom}}$) and IMU gyro ($\omega_{\text{gyro}}$) subtract from desired reference states.

### [Document 07: Visual Control Systems Architecture (Mermaid.js)](07_MERMAID_CONTROL_SYSTEM_ARCHITECTURE.md)
* **High-Resolution Color-Coded Mermaid.js Flowchart:** Visualizes the complete decoupled modal control architecture.
* **1-to-1 Code Verification Proof:** Direct mapping table linking every diagram block to its exact variable and line of C code in `drive_2dof.c`.

### [Document 08: Physical AI & The Embodied Dynamics Contract](08_PHYSICAL_AI_AND_EMBODIED_DYNAMICS_CONTRACT.md)
* **The Classical Robotics Paradox:** Resolving the gap where the SBC attempts to command `effort` without real physical model telemetry.
* **Cerebrum vs. Brainstem Division:** High-rate local impedance damping and FOC current control on the MCU; macro-physics and geometric clothoids on the SBC.
* **The Generalized Impedance Contract:** Formulating $(position, velocity, effort)$ as $\tau = \tau_{ff} + K_p(\theta_{des} - \theta) + B_d(\dot{\theta}_{des} - \dot{\theta})$.
* **Online Adaptive Identification:** Estimating payload mass ($M$) and surface friction ($\mu$) dynamically from real-time current and acceleration.

### [Document 09: Outdoor Terrain Dynamics, Incline Physics & Gravity Compensation](09_OUTDOOR_TERRAIN_DYNAMICS_AND_INCLINE_PHYSICS.md)
* **The Outdoor Robotics Dilemma:** Why dynamic outdoor environments (mud, gravel, slopes) make firmware dynamics mandatory, not redundant.
* **Invariants vs. Variables:** Separating physical invariants ($M, I_{zz}, r, b, K_t$) from terrain variables ($\theta_{\text{pitch}}, \mu, \tau_{\text{scrub}}$).
* **Incline Physics & Gravity Force Balance:** $F_g = M \cdot g \cdot \sin(\theta)$ and zero-latency hill-start anti-rollback via onboard IMU pitch telemetry.
* **Tire Scrub & Friction Adaptivity:** Dynamic trimming of skid-steer yaw scrub via closed-loop onboard IMU rate gyroscope.
* **The Sub-Millisecond Bandwidth Imperative:** Why the MCU's $< 1\text{ ms}$ reaction time is required to prevent slip-trenching and rollback that a $50\text{ ms}$ SBC transport loop cannot catch.

---



## The Master Architecture Flow

```text
========================================================================================================================
                                     COMPLETE 2-DoF MODAL CONTROL TOPOLOGY
========================================================================================================================

 [COMMAND]                  cmd_vel (v_cmd, w_cmd)  OR  200 Hz Trajectory Waypoint
                                   │
                                   ▼
 [REFERENCE ENGINE]         State Evaluator: [ v_tgt, a_tgt, w_tgt, alpha_tgt ]
                                   │
                                   ├──────────────────────────────────────────────────────┐
                                   ▼                                                      ▼
 [MODAL DYNAMICS ENGINE]    CHANNEL A: LONGITUDINAL MODE                           CHANNEL B: LATERAL-YAW MODE
                            F_lin_ff = M_eff*a + Stribeck(v) + Drag(v)             tau_rot_ff = I_eff*alpha + Scrub(w)
                                   │                                                      │
                                   ▼                                                      ▼
 [MODAL CLOSED-LOOP MOP]    + Linear Error Trim: Kp*e_s + Kd*e_v                   + Yaw Error Trim: Kp*e_theta + Kd*e_w
                                   │                                                      │
                                   ▼                                                      ▼
 [TOTAL MODAL WRENCH]       F_linear_total [Newtons]                               tau_rot_total [Newton-meters]
                                   │                                                      │
                                   └──────────────────────┬───────────────────────────────┘
                                                          │
                                                          ▼
 [ALGEBRAIC KINETIC MIXER]                 ┌──────────────────────────────┐
                                           │  tau = J^T * W_total         │
                                           │  tau_L = (F/2 - tau/b) * r   │
                                           │  tau_R = (F/2 + tau/b) * r   │
                                           └──────────────┬───────────────┘
                                                          │
                                                          ▼
 [ACTUATOR TORQUE SPACE]                   tau_Left [N*m]  and  tau_Right [N*m]
                                                          │
                                                          ▼ Convert: Iq = tau / Kt
 [HARDWARE FOC EXECUTION]                  Iq_Left [Amps]  and  Iq_Right [Amps]
                                           (Injected into ST MCSDK TIM1 Interrupt at 16-32 kHz)
========================================================================================================================
```
