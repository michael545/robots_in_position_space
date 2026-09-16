# Document 03: The Algebraic Kinetic Mixer & Torque Vectoring

## 1. Executive Summary

The **Algebraic Kinetic Mixer** is the linear coordinate transformation that maps the generalized chassis wrench from the **Modal Force Frame** ($F_{linear}, \tau_{yaw}$) into the physical **Actuator Torque Frame** ($\tau_L, \tau_R$).

$$\begin{bmatrix} \tau_L \\ \tau_R \end{bmatrix} = \begin{bmatrix} \frac{r}{2} & -\frac{r}{b} \\ \frac{r}{2} & \frac{r}{b} \end{bmatrix} \begin{bmatrix} F_{linear} \\ \tau_{yaw} \end{bmatrix}$$

This transformation operates as an **Electronic Torque Vectoring (Direct Yaw-Moment Control)** system. It continuously calculates the required differential torque couple ($\Delta \tau = \frac{r}{b} \tau_{yaw}$) to steer the robot into turns on millisecond zero, eliminating corner understeer and wheel slip.

---

## 2. Derivation Method 1: Classical Newtonian Statics & Force Equilibrium

Consider the ground contact patches of the two driving wheels separated by track width $b$:

```text
===================================================================================================
                             FREE-BODY DIAGRAM AT WHEEL CONTACTS
===================================================================================================

                        ▲ Forward (X_body)
                        │
         Left Wheel     │     Right Wheel
          Force F_L     │      Force F_R
              ▲         │          ▲
              │         │          │
         ┌────┴─────────┼──────────┴────┐
         │              │               │
         │       ◄──────┼──────►        │
         │         b/2  │  b/2          │
         └──────────────┴───────────────┘
         ◄────────────── b ─────────────► Track width
===================================================================================================
```

### Step 1: Force and Moment Balance
1. **Total Forward Thrust ($F_{linear}$):**
   The net linear force moving the chassis forward is the algebraic sum of the traction forces exerted by both wheels:
   $$F_{linear} = F_L + F_R \quad \text{--- (Eq. 1)}$$

2. **Total Yaw Twisting Moment ($\tau_{yaw}$):**
   Summing moments about the chassis vertical axis through the center of mass:
   $$\tau_{yaw} = F_R \left(\frac{b}{2}\right) - F_L \left(\frac{b}{2}\right) = \frac{b}{2} (F_R - F_L) \quad \text{--- (Eq. 2)}$$

### Step 2: Matrix Formulation
Writing Equations 1 and 2 in matrix form:
$$\begin{bmatrix} F_{linear} \\ \tau_{yaw} \end{bmatrix} = \begin{bmatrix} 1 & 1 \\ -\frac{b}{2} & \frac{b}{2} \end{bmatrix} \begin{bmatrix} F_L \\ F_R \end{bmatrix}$$

### Step 3: Inverting the $2 \times 2$ Linear System
To compute the required tire ground forces $F_L$ and $F_R$, we invert the matrix:
$$\mathbf{A} = \begin{bmatrix} 1 & 1 \\ -\frac{b}{2} & \frac{b}{2} \end{bmatrix}$$

$$\det(\mathbf{A}) = (1)\left(\frac{b}{2}\right) - (1)\left(-\frac{b}{2}\right) = b$$

$$\mathbf{A}^{-1} = \frac{1}{b} \begin{bmatrix} \frac{b}{2} & -1 \\ \frac{b}{2} & 1 \end{bmatrix} = \begin{bmatrix} \frac{1}{2} & -\frac{1}{b} \\ \frac{1}{2} & \frac{1}{b} \end{bmatrix}$$

Therefore, the required tire forces are:
$$F_L = \frac{F_{linear}}{2} - \frac{\tau_{yaw}}{b}$$
$$F_R = \frac{F_{linear}}{2} + \frac{\tau_{yaw}}{b}$$

### Step 4: Converting Tire Forces to Motor Shaft Torques
Since shaft torque is $\tau = \text{Force} \times \text{Radius} = F \cdot r$:
$$\tau_L = F_L \cdot r = \mathbf{\left(\frac{F_{linear}}{2} - \frac{\tau_{yaw}}{b}\right) \cdot r}$$
$$\tau_R = F_R \cdot r = \mathbf{\left(\frac{F_{linear}}{2} + \frac{\tau_{yaw}}{b}\right) \cdot r}$$

In matrix form:
$$\begin{bmatrix} \tau_L \\ \tau_R \end{bmatrix} = \begin{bmatrix} \frac{r}{2} & -\frac{r}{b} \\ \frac{r}{2} & \frac{r}{b} \end{bmatrix} \begin{bmatrix} F_{linear} \\ \tau_{yaw} \end{bmatrix}$$

---

## 3. Derivation Method 2: Principle of Virtual Work & Jacobian Duality

In analytical mechanics, the mapping from generalized forces to actuator torques is uniquely defined by the transpose of the kinematic Jacobian.

### Step 1: Forward Kinematic Jacobian $\mathbf{J}$
Let the chassis velocity vector be $\dot{\mathbf{q}} = \begin{bmatrix} v \\ \omega \end{bmatrix}$ and the wheel rotational speed vector be $\boldsymbol{\Omega} = \begin{bmatrix} \omega_L \\ \omega_R \end{bmatrix}$.

The forward kinematic mapping is:
$$\begin{bmatrix} v \\ \omega \end{bmatrix} = \begin{bmatrix} \frac{r}{2} & \frac{r}{2} \\ -\frac{r}{b} & \frac{r}{b} \end{bmatrix} \begin{bmatrix} \omega_L \\ \omega_R \end{bmatrix} \quad \implies \quad \dot{\mathbf{q}} = \mathbf{J} \boldsymbol{\Omega}$$

### Step 2: Conservation of Virtual Power
By D'Alembert's principle, the mechanical power delivered by the motor shafts must identically equal the mechanical power exerted on the chassis:
$$P = \boldsymbol{\tau}^T \boldsymbol{\Omega} \equiv \mathbf{W}^T \dot{\mathbf{q}}$$
where $\mathbf{W} = \begin{bmatrix} F_{linear} \\ \tau_{yaw} \end{bmatrix}$ is the generalized wrench.

Substitute $\dot{\mathbf{q}} = \mathbf{J} \boldsymbol{\Omega}$:
$$\boldsymbol{\tau}^T \boldsymbol{\Omega} = \mathbf{W}^T (\mathbf{J} \boldsymbol{\Omega}) = (\mathbf{J}^T \mathbf{W})^T \boldsymbol{\Omega}$$

Because this must hold for any arbitrary motion $\boldsymbol{\Omega}$, the actuator torque vector is strictly:
$$\boldsymbol{\tau} = \mathbf{J}^T \mathbf{W}$$

Transposing $\mathbf{J}$:
$$\mathbf{J}^T = \begin{bmatrix} \frac{r}{2} & -\frac{r}{b} \\ \frac{r}{2} & \frac{r}{b} \end{bmatrix}$$

$$\begin{bmatrix} \tau_L \\ \tau_R \end{bmatrix} = \begin{bmatrix} \frac{r}{2} & -\frac{r}{b} \\ \frac{r}{2} & \frac{r}{b} \end{bmatrix} \begin{bmatrix} F_{linear} \\ \tau_{yaw} \end{bmatrix}$$

---

## 4. Physical Meaning of the Matrix Coefficients

```text
===================================================================================================
                               COEFFICIENT BREAKDOWN MATRIX
===================================================================================================

                    Linear Force (F_lin)         Yaw Torque (tau_yaw)
                        Column 1                      Column 2
                    ┌─────────────────────────┬─────────────────────────┐
    Left Wheel (Row 1):│      + r / 2            │        - r / b          │
                    ├─────────────────────────┼─────────────────────────┤
   Right Wheel (Row 2):│      + r / 2            │        + r / b          │
                    └─────────────────────────┴─────────────────────────┘
===================================================================================================
```

### 1. The Linear Thrust Terms ($+\frac{r}{2}$)
* The coefficients in Column 1 are identical and positive.
* Forward thrust $F_{linear}$ is split symmetrically: each wheel carries exactly 50% of the linear translation work.
* Units: $[\text{meters}] \times [\text{Newtons}] = \mathbf{N \cdot m}$ (Mechanical Shaft Torque).

### 2. The Differential Steering Terms ($\pm \frac{r}{b}$)
* The coefficients in Column 2 have opposite signs:
  - **Right Wheel ($+\frac{r}{b}$):** Pushes forward with extra torque for counter-clockwise turning.
  - **Left Wheel ($-\frac{r}{b}$):** Pulls backward (or reduces forward torque) to create the steering couple.
* Units: $\left[\frac{\text{meters}}{\text{meters}}\right] \times [\text{N}\cdot\text{m}] = \mathbf{N \cdot m}$ (Mechanical Shaft Torque).

---

## 5. Torque Vectoring Across the Three Turning Regimes

Depending on the commanded arc radius $R = v / \omega$, the kinetic mixer dynamically shifts between three operating regimes:

```text
===================================================================================================
                               THE THREE TURNING REGIMES
===================================================================================================

 1. WIDE HIGHWAY ARC (R >> b/2):
    Both wheels push forward, but the outside wheel provides majority thrust:
    tau_Left  > 0 (Moderate drive)
    tau_Right > 0 (High drive)

 2. PIVOT ARC (R = b/2, Turning around inside tire contact patch):
    Inside wheel is held at zero velocity while outside wheel swings:
    tau_Left  ≈ 0 (Zero net force; holds ground)
    tau_Right >> 0 (Provides 100% of vehicle forward thrust)

 3. TIGHT HAIRPIN ARC (R < b/2):
    Inside wheel must physically spin in REVERSE:
    tau_Left  < 0 (Active electrical regenerative braking / reverse thrust!)
    tau_Right >> 0 (Massive forward propulsion)
===================================================================================================
```

---

## 6. Energy Conservation Proof (Power Invariance)

We verify that the kinetic mixer does not introduce non-conservative mathematical artifacts:

$$P_{actuators} = \tau_L \omega_L + \tau_R \omega_R$$
$$P_{actuators} = \left( \frac{F_{lin}}{2} - \frac{\tau_{yaw}}{b} \right) r \omega_L + \left( \frac{F_{lin}}{2} + \frac{\tau_{yaw}}{b} \right) r \omega_R$$
$$P_{actuators} = F_{lin} \cdot \underbrace{\left( \frac{r\omega_L + r\omega_R}{2} \right)}_{\equiv v} + \tau_{yaw} \cdot \underbrace{\left( \frac{r\omega_R - r\omega_L}{b} \right)}_{\equiv \omega}$$
$$P_{actuators} = F_{linear} \cdot v + \tau_{yaw} \cdot \omega \equiv P_{body}$$

The mechanical power generated at the motor shafts identically equals the mechanical power exerted on the vehicle body.
