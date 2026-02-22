# Magni MCB G6 Firmware: Full Context Snapshot

This document captures the complete technical understanding of the MCB G6 Firmware and the architectural shift to Position Space Control as of February 2026.

## 1. System Architecture
- **Microcontroller:** STM32G474QETX (Cortex-M4 with FPU).
- **Operating System:** Azure RTOS (ThreadX) for real-time task management.
- **Motor Control Stack:** ST Motor Control SDK (MCSDK) v5.4.8. 
  - Implements Field Oriented Control (FOC) for dual brushless DC motors.
  - High-frequency loops (Current/PWM) managed by SDK interrupts.
- **Middleware:** Pico-ROS (Zenoh-pico) for standard ROS 2 communication over Serial.

## 2. Firmware Module Breakdown
- **Drive Logic (drive.c/h):** 
  - Implements the differential drive kinematics and state machine.
  - Modes: STRAIGHT, CURVE, ROTATE, and the new EXTERNAL_TRAJ.
- **Communication (thread_communication.c):**
  - Bridge between ROS 2 topics and internal C structs.
  - Key Topics: /cmd_vel (Legacy), /cmd_traj (New Joint Trajectory), /odometry, /imu.
- **MCSDK Bridge (motorcontrol.c):** 
  - Manages raw handles to PID regulators and FOC structures (MCI_Handle_t).

## 3. Position Space Implementation Details
The firmware has been updated to support deterministic trajectory tracking.

### Data Struct (drive_joint_cmd_t)
Used to store the high-fidelity state for both wheels:
- pos[2]: Target angle in radians.
- vel[2]: Target feedforward velocity in radians/s.
- effort[2]: Feedforward torque/current.
- valid_ms: Watchdog timeout for safety.

### Control Law
When in DRIVE_MODE_EXTERNAL_TRAJ, the firmware executes:
Output = Kp * (Pos_target - Pos_measured) + Kv * (Vel_target)
- The target position is injected directly into the position loop.
- The target velocity is injected as Feedforward into the MCSDK Omega term to eliminate lag.

## 4. Operational Conventions
- **Units:** Linear speeds in mm/s, Rotational in mrad/s, Joint positions in radians.
- **Watchdog:** 100ms timeout for trajectory points; 300ms for cmd_vel.
- **Coordinate Frames:** Reports base_link relative to odom.

## 5. Porting Note
To resume this context in a new Gemini session, show this file to the agent. It contains the Source of Truth for the Magni G6 firmware architecture and the mathematical basis for its motion control.
