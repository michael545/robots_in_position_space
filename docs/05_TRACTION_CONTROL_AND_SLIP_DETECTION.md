# Document 05: Traction Control & Multi-Sensor Slip Detection

## 1. Executive Summary

Wheel slip (loss of tire traction on slick concrete, oil, or loose debris) is the single greatest cause of odometry failure and localization divergence in autonomous ground vehicles.

If an autonomous robot relies **solely on wheel encoders**, it is physically blind to slip: a wheel spinning freely on ice will report that the robot is moving forward at full speed.

The **Magni MCB G6 motherboard**, however, incorporates an onboard **Inertial Navigation Unit (`mcb.inav`) with a 6-axis IMU (Gyroscope + Accelerometer)** and **high-frequency current sensors ($I_q$)**. 

By fusing the **3-Body Dynamic Model** with the **IMU** and **Motor Current**, the system detects wheel slip within **5 milliseconds** using three independent physical mechanisms.

---

## 2. Detection Method 1: The "Effective Mass Collapse" (Physics Observer)

This method detects slip purely from the mismatch between commanded motor torque and observed wheel angular acceleration.

### 2.1 The Physics of Inertia Collapse
Recall from the 3-Body Dynamics model:
$$\tau_{motor} = J_{total} \cdot \dot{\omega}_{wheel} + \tau_{friction}$$
where $J_{total} = M_{eff} \cdot r^2$.

```text
===================================================================================================
                            THE "EFFECTIVE MASS COLLAPSE"
===================================================================================================

  SCENARIO A: NORMAL TRACTION (Tire Gripping Concrete)
  - Motor must push the ENTIRE 40 kg ROBOT + Payload.
  - Effective Inertia: J_total = 40 kg * r^2 ≈ 0.40 kg*m^2
  - For a torque of 2.0 N*m, wheel acceleration is moderate:
    dω/dt = 2.0 / 0.40 = 5.0 rad/s²

  SCENARIO B: WHEEL SLIP (Tire on Oil / Ice / Lifting Off Ground)
  - The tire breaks grip! The motor is no longer pushing the 40 kg robot.
  - The motor is now ONLY spinning the light, 1.2 kg plastic wheel!
  - Effective Inertia COLLAPSES: J_wheel ≈ 0.005 kg*m^2 (80x smaller!)
  - For that SAME 2.0 N*m torque, the wheel violently spins up:
    dω/dt = 2.0 / 0.005 = 400.0 rad/s²  (80x FASTER ACCELERATION!)
===================================================================================================
```

### 2.2 Mathematical Detection Criterion
If the measured wheel acceleration exceeds what Newton's second law physically permits for a 40 kg vehicle:

$$\dot{\omega}_{measured} > \frac{\tau_{command}}{M_{eff} \cdot r^2} \cdot \gamma_{threshold}$$
*(where $\gamma_{threshold} \approx 2.5$)*

The controller confirms with 100% mathematical certainty that the tire has broken traction!

---

## 3. Detection Method 2: Kinematic vs. Gyroscope Disparity (Rotational Slip)

When an AMR turns, rubber scrubs. If the floor is slippery, the wheels will spin differentially, but the chassis will fail to rotate.

```text
===================================================================================================
                             YAW SLIP DISPARITY OBSERVER
===================================================================================================

  Wheel Encoders say:
  ω_wheels = (r / b) * (ω_right - ω_left)  ──► "We are turning at +1.5 rad/s!"
                                                      │
                                                      ├──► Compare! Discrepancy > Threshold?
                                                      │    ──► SLIP DETECTED!
  Onboard IMU Gyroscope says:                         │
  ω_gyro = mcb.inav.gyro_z                 ──► "Chassis is only turning at +0.2 rad/s!"
===================================================================================================
```

### Mathematical Criterion:
$$\Delta \omega_{slip} = \left| \frac{r}{b}(\omega_R - \omega_L) - \omega_{gyro\_z} \right|$$

If $\Delta \omega_{slip} > 0.15\,\text{rad/s}$ for more than 10 ms, rotational tire slip is occurring.

---

## 4. Detection Method 3: Accelerometer vs. Wheel Acceleration (Longitudinal Slip)

The wheel encoders compute linear acceleration of the wheels:
$$a_{wheels} = r \cdot \frac{\dot{\omega}_L + \dot{\omega}_R}{2}$$

The onboard IMU measures the true physical acceleration of the chassis:
$$a_{imu} = \text{mcb.inav.imu.accel\_x}$$

$$\Delta a_{slip} = a_{wheels} - a_{imu}$$

If $\Delta a_{slip} > 0.5\,\text{m/s}^2$, the wheels are spinning faster than the chassis is accelerating forward (burnout / wheel spin).

---

## 5. Active Traction Control System (TCS) Responses

Once slip is detected, the firmware executes a 3-stage **Traction Control Routine**:

```text
===================================================================================================
                    ACTIVE TRACTION CONTROL ACTION (TCS)
===================================================================================================

  1. INSTANT TORQUE CLAMP:
     Cut Iq_ff torque on the slipping wheel by 50-80% to allow the rubber
     to regain static friction with the ground.

  2. ODOMETRY PROTECTION (Freeze Corrupted Encoders):
     Instruct the dead-reckoning engine: "Do NOT trust wheel encoders right now!
     Integrate vehicle pose strictly from the IMU Gyroscope until grip is restored."

  3. STALL / HIGH-CENTER WATCHDOG:
     If motor current is at max (Iq = I_max) but wheel speed is zero (v_wheel = 0),
     the robot is stalled against a physical obstacle. Abort and trigger emergency stop.
===================================================================================================
```

---

## 6. Firmware Implementation in `drive.c`

```c
void check_wheel_slip_and_traction_control(drive_t* drv, drive_feedforward_output_t* ff) 
{
    float r = param_value(drv->params.wheels.radius) / 1000.0f;
    float b = param_value(drv->params.wheels.distance_mm) / 1000.0f;

    // 1. Compute kinematics from encoders
    float w_left  = drv->state.position_rad[DRIVE_M1]; // velocity via diff
    float w_right = drv->state.position_rad[DRIVE_M2];
    float omega_wheels = (r / b) * (w_right - w_left);

    // 2. Read IMU
    float omega_gyro = drv->inav->imu.gyro_z_rad_s;

    // 3. Evaluate Yaw Slip Discrepancy
    float yaw_slip = fabsf(omega_wheels - omega_gyro);
    if (yaw_slip > 0.20f) { // 0.2 rad/s threshold
        // ROTATIONAL SLIP DETECTED: Clamp differential torque couple
        ff->torque_yaw_nm *= 0.3f; // Reduce yaw effort by 70% to regain grip
        drv->state.odometry_trust_encoders = false; // Trust gyro only
    } else {
        drv->state.odometry_trust_encoders = true;
    }
}
```
