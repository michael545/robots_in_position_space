# Document 04: Closed-Loop Modal Feedback & ST MCSDK FOC Integration

## 1. Executive Summary

While open-loop physical feedforward provides 90–95% of the torque required to accelerate and navigate curves, real-world disturbances (carpet friction, floor irregularities, payload weight changes, tire wear) require **closed-loop feedback**.

Rather than placing independent feedback loops on each wheel, the new architecture closes the loop in **Modal Space**:
1. **Longitudinal Distance PID ($e_s$):** Regulates forward position along the trajectory.
2. **Yaw Heading PID ($e_{yaw}$):** Regulates angular orientation relative to the path tangent.

The resulting corrective forces are added to the feedforward wrench and passed through the **Kinetic Mixer**, injecting into the STM32's **Field Oriented Control (FOC) current loop ($I_q$)**.

---

## 2. The 3-Layer Control Stack

```text
===================================================================================================
                                THE COMPLETE 3-LAYER CONTROL STACK
===================================================================================================

  LAYER 3: MODAL ERROR FEEDBACK (Chassis Space: Distance & Heading)
  
    [Distance Error: e_s = s_tgt - s_act] ──► [ Longitudinal PID ] ──► F_lin_fb   [Newtons]
    
    [Heading Error:  e_yaw = θ_tgt - θ_act] ──► [ Yaw Heading PID ]   ──► tau_yaw_fb [N*m]
                                                                              │
                                                                              ▼ Add Feedforward (90%)
                                                                       [ F_total , tau_total ]
                                                                              │
 ─────────────────────────────────────────────────────────────────────────────┼────────────────────
  LAYER 2: THE ALGEBRAIC KINETIC MIXER (Jacobian Transpose)                   ▼
                                                               ┌─────────────────────────────┐
                                                               │ tau = J^T * W_total         │
                                                               └──────────────┬──────────────┘
                                                                              │
                                                                              ▼
                                                               [ tau_Left , tau_Right ]
                                                               (Motor Shaft Torques in N*m)
 ─────────────────────────────────────────────────────────────────────────────┼────────────────────
  LAYER 1: HARDWARE FOC CURRENT LOOPS (Inside ST MCSDK at 16-32 kHz)          │
                                                                              ▼ Convert: Iq = tau / Kt
                                                               [ Iq_ref_Left , Iq_ref_Right ]
                                                                              │
    [Left Current Error:  e_Iq_L]  ──► [ Current PI Loop L ] ──► PWM Voltages to Left Motor
    
    [Right Current Error: e_Iq_R]  ──► [ Current PI Loop R ] ──► PWM Voltages to Right Motor
===================================================================================================
```

---

## 3. Why Modal Feedback is Superior to Wheel Feedback

| Disturbance Event | Traditional Wheel-Space Feedback (Left PID & Right PID) | Modal-Space Feedback (Longitudinal & Yaw PID) |
| :--- | :--- | :--- |
| **Thick Carpet / Slope** (Symmetric drag) | Both wheel PIDs fight independently; minor gain mismatches cause the robot to **veer off course**. | **Longitudinal PID** senses distance lag and injects $+F_{linear}$, pushing both wheels equally. **Robot stays straight.** |
| **Pebble / Bump on Right Wheel** (Asymmetric disturbance) | Right wheel slows; Right PID commands massive current surge, causing **chassis jerk and steering kick**. | **Yaw PID** senses angular deviation and commands $+\Delta \tau$ to right wheel and $-\Delta \tau$ to left wheel. **Restores heading instantly.** |
| **Cruising on Flat Floor** | High PID gains cause high-frequency motor vibration and hunting. | Feedback output is zero ($e \approx 0$); **100% smooth, silent feedforward.** |

---

## 4. The Extended Non-Linear PID Regulator

To prevent high-frequency buzzing while maintaining aggressive disturbance rejection, we implement an **Extended Non-Linear PID Loop**:

$$u_{PID}(E) = \underbrace{P_1 \cdot E}_{\text{Linear Term}} + \underbrace{\text{sgn}(E) \cdot P_2 \cdot E^2}_{\text{Quadratic Term}} + \underbrace{D_1 \cdot \frac{\delta_w E}{\delta_w t}}_{\text{Windowed Derivative}}$$

```text
===================================================================================================
                         LINEAR vs. EXTENDED NON-LINEAR PID
===================================================================================================

  Control Effort u
         ▲
         │                                       _.-' (Quadratic P2*E^2 takes over!)
         │                                  _.-''      Aggressive restoring force on bumps!
         │                             _.-''
         │                       _.-''
         │                 _.-'' ◄── Linear P1*E region
         │           _.-''
         │     _.-''
         │ .-' ◄── Smooth, low gain around zero: ZERO VIBRATION OR CHATTER!
       0 ┼────────────────────────────────────────────────────────► Position Error E
===================================================================================================
```

### The $w$-Cycle Windowed Derivative ($\delta_w E / \delta_w t$)
Direct numerical differentiation of encoder counts ($D_1 \frac{E[k] - E[k-1]}{\Delta t}$) amplifies integer quantization noise. 

A rolling circular buffer of size $w = 4$ to $8$ samples computes:
$$\frac{\delta_w E}{\delta_w t} = \frac{E[k] - E[k - w]}{w \cdot \Delta t}$$
This acts as a finite-impulse-response (FIR) low-pass filter with zero numerical phase lag.

---

## 5. Firmware Integration: STM32 ST MCSDK FOC

### 5.1 Motor Torque Constant Conversion ($K_t$)
Inside the ST Motor Control SDK, electromagnetic motor torque is governed by quadrature axis current ($I_q$):
$$\tau = K_t \cdot I_q \implies I_q = \frac{\tau}{K_t}$$

Where $K_t$ is computed from motor pole pairs and permanent magnet flux linkage $\lambda_m$:
$$K_t = \frac{3}{2} \cdot p \cdot \lambda_m \quad [\text{N}\cdot\text{m} / \text{A}]$$

### 5.2 Direct Injection into `Iq_ff`
In [`drive.c`](file:///home/michael/code/firmware/robots_in_position_space/mcb_g6_firmware/src/modules/drive/drive.c):
```c
// Convert commanded shaft torques into digital current counts
int16_t Iq_target_L = (int16_t)(tau_L * NM_TO_CURRENT_DIGITAL);
int16_t Iq_target_R = (int16_t)(tau_R * NM_TO_CURRENT_DIGITAL);

// Inject into ST MCSDK FOC current loop reference:
Mci[DRIVE_M1].pPosCtrl->Iq_ff = Iq_target_L;
Mci[DRIVE_M2].pPosCtrl->Iq_ff = Iq_target_R;

// Position targets continue to update in Follow Mode:
MC_ProgramPositionCommandMotor1(drv->state.target_rad[DRIVE_M1], 0);
MC_ProgramPositionCommandMotor2(drv->state.target_rad[DRIVE_M2], 0);
```

Because `Iq_ff` injects directly into the PWM timer interrupt (running at 16–32 kHz), **motor flux establishes within microseconds**, completely eliminating control lag.
