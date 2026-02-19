# **Advanced Stochastic Control Paradigms for Autonomous Ground Vehicles: A Comprehensive Analysis of Model Predictive Path Integral Methodologies**

The technical evolution of Automated Guided Vehicles (AGVs) has transitioned from rigid, line-following industrial tools to highly agile, autonomous mobile robots capable of navigating complex, non-convex, and dynamic environments. At the heart of this transition is the advancement of control architectures that can manage the non-linear dynamics of vehicle motion while simultaneously accounting for the inherent uncertainties of the physical world. Model Predictive Path Integral (MPPI) control has emerged as a state-of-the-art framework that bridges the gap between traditional optimal control and stochastic sampling-based optimization. This report provides an exhaustive exploration of MPPI methodologies, their mathematical foundations, advanced variants, and practical implementation strategies for the next generation of AGVs.

## **The Theoretical Genesis of Path Integral Control**

The conceptual underpinnings of Model Predictive Path Integral control represent a fundamental departure from the deterministic optimization techniques that dominated 20th-century robotics. Traditional Model Predictive Control (MPC) typically solves a constrained optimization problem at each time step to find a single optimal control sequence. However, in environments with high degrees of uncertainty or non-smooth cost landscapes—such as those encountered by AGVs in crowded warehouses or during high-speed off-road maneuvers—deterministic approaches often fail due to their reliance on gradient-based solvers that are prone to becoming trapped in local minima.1

MPPI addresses these limitations by reframing the control problem through the lens of stochastic optimal control and information theory. The methodology was first formalized in the landmark paper presented at the IEEE International Conference on Robotics and Automation (ICRA) in 2016\.3 This foundational work introduced the derivation of MPPI using information-theoretic dualities between free energy and relative entropy, establishing a link between the physics of path integrals and the mathematics of decision-making under uncertainty.3

### **From the HJB Equation to Linearized Stochastic Control**

The mathematical core of MPPI is derived from the Hamilton-Jacobi-Bellman (HJB) equation, which provides the necessary conditions for optimal control in a dynamic system. In non-linear systems, solving the HJB equation is notoriously difficult, often requiring the solution of complex second-order partial differential equations.5 MPPI simplifies this by utilizing a logarithmic transformation—often referred to as the Cole-Hopf transformation in the context of physics—to linearize the stochastic HJB equation.5

This transformation is applicable to a specific class of systems characterized by input-affine dynamics with additive white Gaussian noise (AWGN) and cost functions that are quadratic in the control inputs.5 By transforming the problem, the optimal control can be calculated not through traditional gradient descent, but by evaluating the weighted average of multiple sampled stochastic trajectories.5 This "zero-order" optimization approach allows the controller to ignore the derivatives of the dynamics and cost functions, making it exceptionally robust to discontinuities such as those found in obstacle avoidance logic or contact-heavy manipulation tasks.7

### **The Information-Theoretic Duality and Free Energy**

A profound insight provided by the original researchers is the relationship between the optimal control problem and the concept of free energy from statistical mechanics. The MPPI algorithm effectively seeks a probability distribution over potential trajectories that minimizes a cost function while maintaining a degree of proximity—measured by the Kullback-Leibler (KL) divergence—to a baseline or "uninformed" distribution.3

This duality allows the controller to handle non-convex cost landscapes naturally. Instead of finding a single "best" path, MPPI explores a distribution of paths. When the environment changes rapidly or an unexpected obstacle appears, the sampling mechanism ensures that some trajectories will inevitably discover the new feasible corridors of the state space. The weighted update rule then shifts the entire control distribution toward these successful samples, providing a level of adaptability and agility that deterministic MPC struggles to match.1

## **Mathematical Architecture and Algorithmic Flow**

The execution of an MPPI controller follows a receding horizon logic, where a sequence of future control actions is optimized at every time step, but only the immediate command is executed.6 The process is highly iterative and relies on the generation of thousands of potential future "lives" for the robot, known as rollouts.11

### **The Core Optimization Loop**

The MPPI algorithm can be decomposed into several critical stages that occur within each control cycle, typically running at frequencies between 30 Hz and 100 Hz depending on the hardware and the complexity of the AGV's dynamics.11

| Algorithmic Stage | Description and Mechanism | Key Mathematical Parameters |
| :---- | :---- | :---- |
| **Noise Generation** | Random perturbations are generated from a multivariate normal distribution to explore the control space around the current nominal sequence. | ![][image1] 1 |
| **Trajectory Rollout** | The system’s non-linear dynamics model is used to predict the future states of the robot for each perturbed control sequence. | ![][image2] 6 |
| **Cost Evaluation** | Each predicted trajectory is assigned a scalar cost based on path following, obstacle avoidance, smoothness, and goal progress. | ![][image3] 6 |
| **Exponential Weighting** | A weight is assigned to each sample, where lower-cost trajectories receive exponentially higher influence in the final update. | ![][image4] 15 |
| **Control Update** | The new optimal control sequence is computed as the weighted average of the sampled perturbations. | ![][image5] 13 |

### **The Role of the Temperature Parameter**

The hyperparameter ![][image6], often called the temperature parameter, is fundamental to the behavior of the MPPI controller. It regulates the exploration-exploitation trade-off. A very low temperature causes the controller to become "greedy," essentially selecting only the single best trajectory among the thousands of samples.15 While this can lead to very precise tracking, it risks making the controller brittle if the best sample was an outlier or if the environment changes between the time of sampling and execution. Conversely, a high temperature leads to a more diverse averaging across many samples, which increases robustness and smoothness but may lead to conservative behavior or "blurring" of the control actions near sharp obstacles.15

### **Handling System Dynamics and Constraints**

One of the primary advantages of MPPI for AGVs is its ability to utilize high-fidelity, non-linear dynamics models without the need for linearization or simplification.1 For industrial AGVs operating at high speeds, simple kinematic models (like the bicycle model) often fail to capture critical effects such as tire slip, load transfer, or suspension dynamics.18 MPPI allows engineers to embed these complex physics directly into the rollout stage.

Research has shown that for an AGV to operate at its dynamic limits—such as during emergency obstacle avoidance—a 2-degree-of-freedom (DoF) model accounting for non-linear tire characteristics is necessary.19 Because MPPI is a derivative-free method, these complex tire models do not need to be differentiable, allowing for the use of empirical "Pacejka" magic formula models or even neural-network-based dynamics predictors.7

## **Comparative Analysis: MPPI vs. Traditional Control Strategies**

The decision to deploy MPPI over established methods like Nonlinear MPC (NMPC) or Proportional-Integral-Derivative (PID) control is driven by the specific operational requirements of the AGV. While PID remains the industry standard for simple line-following tasks due to its ease of implementation, it lacks the predictive foresight required for high-speed navigation or obstacle-cluttered environments.18

### **NMPC vs. MPPI Benchmarking**

Recent studies comparing NMPC and MPPI on embedded hardware have revealed significant performance differences. In trajectory tracking experiments, MPPI has demonstrated an 18.6% improvement in overall Root Mean Square Error (RMSE) compared to NMPC.16 This superiority is primarily attributed to MPPI’s ability to handle non-convexity and its lack of reliance on successive linearization.16

| Performance Metric | Traditional NMPC | MPPI |
| :---- | :---- | :---- |
| **Mathematical Basis** | Deterministic Optimization | Stochastic Optimal Control |
| **Solver Type** | Gradient-based (SQP, Interior Point) | Sampling-based (Monte Carlo) |
| **Computational Peak** | CPU-bound, sequential logic | GPU-bound, parallel logic |
| **Global Convergence** | Prone to local minima traps | Higher probability of global discovery |
| **Trajectory Smoothness** | High (naturally structured) | Variable (often requires filtering) |
| **Non-linear Handling** | Requires Jacobians/linearization | Directly uses non-linear ODEs |

Source: 16

NMPC’s primary strength lies in its ability to enforce hard constraints, such as keeping a robot strictly within a lane or ensuring actuator voltages never exceed a certain limit.20 MPPI, by contrast, typically relies on soft constraints (penalty functions), which can lead to slight constraint violations if the penalty weights are not carefully tuned.20 However, MPPI's computational speed on parallel hardware often allows it to compensate for these minor inaccuracies by replanning at much higher frequencies.16

### **The Stochastic Jitter Challenge**

A notable drawback of sampling-based methods is the introduction of high-frequency noise or "jitter" in the control output.20 Because the final control command is a weighted average of random samples, the resulting sequence can lack the inherent smoothness of a gradient-descended trajectory.20 To mitigate this, practical implementations often incorporate temporal smoothing filters, such as Savitzky-Golay filters, or utilize advanced sampling techniques like Halton-MPPI.16 Halton-MPPI replaces pseudo-random Gaussian noise with low-discrepancy Halton sequences, which provide more uniform coverage of the search space and result in significantly smoother control sequences while maintaining the same computational efficiency.21

## **High-Performance Hardware and Implementation Ecosystems**

The computational demand of running thousands of simultaneous rollouts is the primary barrier to MPPI deployment. However, the recent explosion in embedded AI hardware has provided a variety of platforms capable of meeting these needs.

### **GPU Acceleration and CUDA-Based Libraries**

The parallel nature of MPPI makes it a perfect candidate for Graphics Processing Units (GPUs).15 By mapping each rollout to an individual GPU thread, the total computation time becomes nearly independent of the number of samples, up to the limits of the hardware's core count.4

The MPPI-Generic library, written in C++/CUDA, represents a significant advancement in making these tools accessible to the research community.4 It allows researchers to swap out dynamics models and cost functions while maintaining real-time performance on hardware ranging from the NVIDIA Jetson Orin series to desktop-class RTX GPUs.4 In aerial robotics experiments, which share many of the high-speed requirements of agile AGVs, GPU-accelerated MPPI has enabled real-time obstacle avoidance at speeds up to 10 m/s.24

### **FPGA and Domain-Specialized Hardware**

For battery-constrained AGVs, GPUs may be too power-hungry. Field Programmable Gate Arrays (FPGAs) offer a compelling alternative by allowing for hardware-software co-design.15 An FPGA-optimized MPPI design can eliminate the memory-access stalls and synchronization bottlenecks found in GPUs.15 Experimental results indicate that FPGAs can achieve a 3.1x to 7.5x speedup over embedded GPUs while consuming significantly less energy, which is critical for the long-endurance requirements of autonomous warehouse fleets.15

### **The Nav2 MPPI Controller for ROS 2**

For practitioners in the industrial space, the Navigation2 (Nav2) stack provides a robust implementation of MPPI that does not require specialized GPU hardware.11 The Nav2 MPPI controller is optimized for CPU-only execution, achieving 30–50 Hz on standard processors by using efficient vectorization and limited sample counts (typically 1000–2000).11

A key feature of the Nav2 implementation is its plugin-based "Critic" architecture. This allows users to modularly expand the controller's behavior by adding new cost-evaluation functions.11

| Nav2 MPPI Critic | Functional Purpose | Industrial Benefit |
| :---- | :---- | :---- |
| **Obstacle Penalty** | Repels robot from costmap obstacles. | Ensures collision-free navigation. |
| **Path Penalty** | Encourages adherence to a global plan. | Prevents deviations in narrow aisles. |
| **Goal Penalty** | Penalties for distance from terminal goal. | Ensures precise docking at stations. |
| **Velocity Critic** | Discourages excessive speed or reversals. | Increases safety and reduces battery wear. |
| **Inflation Critic** | Weights trajectories based on "soft" obstacles. | Smooths paths around corner edges. |

Source: 11

The Nav2 implementation highlights the importance of "Smooth Inflation" in the environment's costmap. Without smooth gradients, the MPPI sampler cannot "sense" the direction of increasing cost, leading to poor convergence and suboptimal paths.11

## **Advanced Variants: Navigating Risk and Uncertainty**

In high-stakes industrial environments, such as a factory floor with human workers, being "optimal on average" is insufficient. AGVs must be risk-aware, accounting for the worst-case possibilities of their actions.

### **Unscented and Robust MPPI (U-MPPI and RMPPI)**

Standard MPPI is "risk-neutral," meaning it optimizes the expected value of the cost function. Unscented MPPI (U-MPPI) enhances this by using the Unscented Transform to propagate not just the state mean, but also the state covariance through the system dynamics.25 This allows the controller to explicitly "see" how uncertain it is about its future position. By incorporating this uncertainty into the cost function, U-MPPI can proactively slow down or steer wider around obstacles when sensor noise is high or the environment is cluttered.26

Robust MPPI (RMPPI) takes a different approach by augmenting the dynamics representation and incorporating a low-level tracking controller inside the stochastic optimization module.3 This "Tube-based" architecture ensures that the actual robot stays within a bounded "tube" around the planned nominal trajectory, providing formal guarantees on stability even in the presence of bounded external disturbances.3

### **Risk-Awareness via CVaR**

Risk-Aware MPPI (RA-MPPI) introduces the Conditional Value-at-Risk (CVaR) measure into the optimization.13 Instead of averaging across all samples, RA-MPPI focuses on the "tail" of the cost distribution—the 5% or 10% of samples that represent the most dangerous outcomes.13 In simulations of aggressive driving, RA-MPPI has been shown to achieve the same lap times as vanilla MPPI while suffering from significantly fewer collisions, as it avoids maneuvers that have a high probability of failure under uncertain conditions.13

### **Dynamic Risk-Awareness in Human Crowds (DRA-MPPI)**

A unique challenge for AGVs is the "freezing robot problem," where a robot stops moving because it cannot find a path that is guaranteed to be 100% safe among moving humans.9 Dynamic Risk-Aware MPPI (DRA-MPPI) addresses this by efficiently approximating the joint Collision Probability (CP) for several hundred samples in real-time using Monte Carlo methods.29 This allows the robot to accept a small, quantified amount of risk to maintain progress, mirroring the way human drivers navigate busy intersections.29

## **Multi-Agent Systems and Swarm Logistics**

In the context of "Swarm Production" and large-scale warehouse management, the coordination of multiple AGVs becomes a bottleneck. The transition from single-robot to multi-robot MPPI involves managing inter-robot collisions and shared resource allocation.31

### **Decentralized Coordination and CoRL-MPPI**

Centralized controllers that plan for an entire fleet of AGVs simultaneously suffer from exponential computational complexity as the number of robots increases.9 Decentralized MPPI allows each robot to plan its own path while treating other robots as dynamic obstacles or cooperative agents.34

A cutting-edge approach is CoRL-MPPI, which fuses Cooperative Reinforcement Learning with the MPPI framework.34 In this architecture, a deep neural network is trained in simulation to learn cooperative collision avoidance behaviors, such as yielding at intersections or forming platoons.34 This learned policy is then used to bias the MPPI sampling distribution, ensuring that the random samples generated are not just mathematically feasible, but "socially" intelligent.34 Importantly, CoRL-MPPI preserves the theoretical safety guarantees of the original MPPI algorithm while significantly improving navigation efficiency and success rates in dense multi-robot environments.34

### **Traffic Management and Deadlock Prevention**

For industrial traffic management, a three-layer control architecture is often proposed 37:

1. **Topological Layer:** Models the high-level traffic flow between different zones of the warehouse using graph theory.  
2. **Middle Layer:** Computes traffic-sensitive paths for each AGV to prevent congestion in "weakly connected" areas like narrow corridors.  
3. **Coordination Layer:** Uses real-time algorithms like MPPI or Conflict-Based Search (CBS) to prevent local collisions and deadlocks due to motion errors or delays.37

By considering the temporal dimension—predicting where other robots will be in the future—MPPI-based coordination can resolve conflicts with a much longer horizon than reactive methods like Velocity Obstacles or ORCA.35

## **Deep Dive into Recommended Literature**

For researchers and engineers looking to implement or advance these technologies, the following papers provide the essential theoretical and practical building blocks.

### **The Foundation: Williams et al. (2016)**

The original paper, *"Model Predictive Path Integral Control: From Theory to Parallel Control"* (ICRA 2016), is the definitive starting point. It provides the information-theoretic derivation and the first successful application to high-speed off-road driving using the GT-AutoRally vehicle.3 This work established that MPPI could effectively handle learned, non-linear dynamics and non-convex costs in real-time.

### **Robustness and Uncertainty: Gandhi et al. (2021)**

Published in *Robotics and Automation Letters* (RAL), the paper *"Robust Model Predictive Path Integral Control"* (RMPPI) addresses the gap between sampling-based planning and robust control.3 It is critical reading for understanding how to provide stability guarantees in stochastic environments.

### **The Industrial Standard: Macenski et al. (2023)**

While not exclusively about MPPI, the survey paper *"From the desks of ROS maintainers: A survey of modern & capable mobile robotics algorithms in the robot operating system 2"* provides the context for how the Nav2 MPPI controller was designed to be production-ready and hardware-agnostic.11

### **Agile Navigation and Clutter: U-MPPI and log-MPPI**

The papers *"Autonomous Navigation of AGVs in Unknown Cluttered Environments: log-MPPI Control Strategy"* 39 and *"Towards Efficient MPPI Trajectory Generation with Unscented Guidance: U-MPPI Control Strategy"* 26 are essential for those working in unstructured or high-speed settings. They introduce the concept of Normal Log-Normal (NLN) mixture distributions and Unscented Transforms to improve sample efficiency and state-space exploration.

### **Multi-Agent Coordination: CoRL-MPPI (2025)**

The recent work *"CoRL-MPPI: Enhancing MPPI With Learnable Behaviours For Efficient And Provably-Safe Multi-Robot Collision Avoidance"* (IROS 2025/ArXiv) represents the state-of-the-art in combining learning-based behaviors with model-based safety.34 It is highly recommended for those interested in swarm robotics and decentralized fleet management.

## **Industrial Application Case Studies**

The practical utility of MPPI is best demonstrated through its application in diverse industrial scenarios, ranging from high-speed logistics to delicate manipulation.

### **High-Speed Off-Road AGVs**

In aggressive driving scenarios, MPPI has been used to navigate scale-model and full-size vehicles through muddy, unpredictable tracks.3 By using deep convolutional neural networks to predict cost maps directly from video input, these systems can identify "driveable" surfaces and obstacles on-the-fly, allowing for autonomous racing and rapid emergency response in areas without pre-existing maps.40

### **Agile Manipulators and Non-Prehensile Tasks**

Beyond ground locomotion, MPPI is proving effective for complex manipulation tasks like "caging" objects or performing non-prehensile pushing.8 Because these tasks involve frequent, discontinuous contacts (switching between "touching" and "not touching"), traditional gradient-based controllers often fail.8 MPPI’s sampling-based nature allows it to optimize through these contact discontinuities by simply evaluating the "physics in dreams"—simulating the results of different pushes or grasps in a physics engine before executing the best one.8

### **Warehouse Swarm Logistics**

In automated industrial plants, AGVs must coordinate to transport goods between production lines and shipping bays.32 MPPI is used here to manage "Lifelong Multi-Agent Path Finding" (Lifelong MAPF), where goals are constantly reassigned as new orders arrive.37 By replanning at 50 Hz, MPPI-controlled AGVs can adjust their speed and heading to yield to other vehicles or take detours around temporary obstacles, such as human-operated forklifts or spilled cargo, maintaining high throughput even in congested conditions.30

## **Future Outlook: The Convergence of Learning and Control**

The future of AGV control lies in the tighter integration of data-driven models with the robust framework of MPPI. Several emerging trends are likely to define the next decade of development.

### **Data-Driven Dynamics and Identification**

The reliance on hand-tuned physics models is a significant bottleneck. Future systems will likely use "Echo-State Networks" (ESNs) or other reservoir computing models to identify non-linear dynamics in real-time.17 This would allow an AGV to "learn" its own mass and tire-friction coefficients after picking up a heavy payload, immediately updating its internal rollout model to maintain optimal performance.17

### **Anchor-Guided and Multi-Homotopy Exploration**

One of the remaining weaknesses of MPPI is its susceptibility to "local traps"—if the sampling distribution is centered on a path that is blocked, the robot may get stuck even if an alternative route exists elsewhere in the environment.12 "Anchor-guided" ensembles, like AERO-MPPI, address this by running multiple MPPI optimizers in parallel, each guided by a different potential path (or "anchor").12 This ensures that the robot explores multiple "homotopy classes" (different ways to go around an obstacle), significantly increasing robustness in highly cluttered environments like dense forests or crowded shipping ports.12

### **Energy-Efficiency and Sustainable Automation**

As global supply chains focus on sustainability, the role of control algorithms in reducing energy consumption is becoming paramount. MPPI’s flexible cost functions allow for the direct optimization of energy use, penalizing erratic accelerations and choosing paths that maintain a constant, efficient velocity.10 In industrial settings, this can lead to longer battery life for AGV fleets and reduced operational costs.15

## **Synthesis of Operational Insights**

The analysis of Model Predictive Path Integral control reveals a highly versatile and powerful framework that is uniquely suited to the challenges of modern autonomous ground vehicles. By leveraging the massive parallelization capabilities of modern hardware, MPPI provides a level of agility, robustness, and flexibility that deterministic methods struggle to achieve.

Key takeaways for industrial deployment include:

* **Hardware Selection:** For high-performance, agile maneuvering, GPU acceleration (NVIDIA Jetson) is essential to maintain high sample counts and horizons. For standard warehouse logistics, the CPU-optimized Nav2 implementation offers a more accessible path.  
* **Risk Management:** In environments shared with humans, the adoption of risk-aware variants (RA-MPPI, DRA-MPPI) is critical to prevent collisions and avoid the "freezing robot" problem.  
* **Model Fidelity:** The effectiveness of MPPI is directly tied to the accuracy of the rollout dynamics. High-speed AGVs require 2-DoF tire models, while complex manipulators require physics-engine integration.  
* **Cooperative Intelligence:** The future of fleet management lies in decentralized, learning-augmented MPPI architectures like CoRL-MPPI, which enable "socially" aware coordination without the computational burden of centralized planning.

The Model Predictive Path Integral paradigm represents more than just an algorithm; it is a shift toward a more probabilistic, resilient, and intelligent way for robots to interact with the world. As the technology matures and computational power continues to follow Moore's Law, MPPI is positioned to become the foundational control layer for the entire spectrum of autonomous mobile systems.

#### **Works cited**

1. Application of Model Predictive Path Integral Controller in Autonomous Driving: A Simulation Study \- Atlantis Press, accessed on February 19, 2026, [https://www.atlantis-press.com/article/126004082.pdf](https://www.atlantis-press.com/article/126004082.pdf)  
2. Trajectory Distribution Control for Model Predictive Path Integral Control using Covariance Steering \- arXiv, accessed on February 19, 2026, [https://arxiv.org/pdf/2109.12147](https://arxiv.org/pdf/2109.12147)  
3. Model Predictive Path Integral (MPPI) control – Autonomous Control ..., accessed on February 19, 2026, [https://sites.gatech.edu/acds/mppi/](https://sites.gatech.edu/acds/mppi/)  
4. MPPI-Generic: A CUDA Library for Stochastic Trajectory Optimization \- arXiv, accessed on February 19, 2026, [https://arxiv.org/html/2409.07563v2](https://arxiv.org/html/2409.07563v2)  
5. Feature-Based MPPI Control with Applications to Maritime Systems \- MDPI, accessed on February 19, 2026, [https://www.mdpi.com/2075-1702/10/10/900](https://www.mdpi.com/2075-1702/10/10/900)  
6. Model Predictive Path Integral (MPPI) Control in C++ | by Markus Buchholz | Medium, accessed on February 19, 2026, [https://markus-x-buchholz.medium.com/model-predictive-path-integral-mppi-control-in-c-b13ea594ca20](https://markus-x-buchholz.medium.com/model-predictive-path-integral-mppi-control-in-c-b13ea594ca20)  
7. Model Predictive Path Integral Control for Agile Unmanned Aerial Vehicles \- arXiv, accessed on February 19, 2026, [https://arxiv.org/html/2407.09812v1](https://arxiv.org/html/2407.09812v1)  
8. Model Predictive Path Integral Control for Interaction-Rich Local Motion Planning in Dynamic Environments \- TU Delft Research Portal, accessed on February 19, 2026, [https://research.tudelft.nl/en/publications/model-predictive-path-integral-control-for-interaction-rich-local/](https://research.tudelft.nl/en/publications/model-predictive-path-integral-control-for-interaction-rich-local/)  
9. Dynamic Risk-Aware MPPI for Mobile Robots in Crowds via Efficient Monte Carlo Approximations, accessed on February 19, 2026, [https://autonomousrobots.nl/paper\_websites/dra-mppi](https://autonomousrobots.nl/paper_websites/dra-mppi)  
10. Model Predictive Control for Autonomous Vehicle: An In-depth Guide \- Level Up Coding, accessed on February 19, 2026, [https://levelup.gitconnected.com/model-predictive-control-for-autonomous-vehicle-an-in-depth-guide-de984308ba10](https://levelup.gitconnected.com/model-predictive-control-for-autonomous-vehicle-an-in-depth-guide-de984308ba10)  
11. On Use of Nav2 MPPI Controller | ROSCon, accessed on February 19, 2026, [https://roscon.ros.org/2023/talks/On\_Use\_of\_Nav2\_MPPI\_Controller.pdf](https://roscon.ros.org/2023/talks/On_Use_of_Nav2_MPPI_Controller.pdf)  
12. AERO-MPPI: Anchor-Guided Ensemble Trajectory Optimization for Agile Mapless Drone Navigation \- arXiv, accessed on February 19, 2026, [https://arxiv.org/html/2509.17340v1](https://arxiv.org/html/2509.17340v1)  
13. Risk-Aware Model Predictive Path Integral Control Using Conditional Value-at-Risk \- IEEE Xplore, accessed on February 19, 2026, [https://ieeexplore.ieee.org/iel7/10160211/10160212/10161100.pdf](https://ieeexplore.ieee.org/iel7/10160211/10160212/10161100.pdf)  
14. Control of Legged Robots using Model Predictive Optimized Path Integral \- arXiv, accessed on February 19, 2026, [https://arxiv.org/html/2508.11917v1](https://arxiv.org/html/2508.11917v1)  
15. Real-Time, Energy-Efficient, Sampling-Based Optimal Control via FPGA Acceleration \- arXiv, accessed on February 19, 2026, [https://arxiv.org/html/2601.17231v1](https://arxiv.org/html/2601.17231v1)  
16. Comparison of NMPC and GPU-Parallelized MPPI for Real-Time ..., accessed on February 19, 2026, [https://www.mdpi.com/2076-3417/15/16/9114](https://www.mdpi.com/2076-3417/15/16/9114)  
17. Reservoir Predictive Path Integral Control for Unknown Nonlinear Dynamics \- arXiv, accessed on February 19, 2026, [https://arxiv.org/abs/2509.03839](https://arxiv.org/abs/2509.03839)  
18. Simulating Automated Guided Vehicles in Unity: A Case Study on PID Controller Tuning, accessed on February 19, 2026, [https://www.mdpi.com/2571-5577/8/6/170](https://www.mdpi.com/2571-5577/8/6/170)  
19. Obstacle avoidance in real time with Nonlinear Model Predictive Control of autonomous vehicles | Request PDF \- ResearchGate, accessed on February 19, 2026, [https://www.researchgate.net/publication/286650865\_Obstacle\_avoidance\_in\_real\_time\_with\_Nonlinear\_Model\_Predictive\_Control\_of\_autonomous\_vehicles](https://www.researchgate.net/publication/286650865_Obstacle_avoidance_in_real_time_with_Nonlinear_Model_Predictive_Control_of_autonomous_vehicles)  
20. Model Predictive Controllers. This article briefly describes two… | by Fracursious | Medium, accessed on February 19, 2026, [https://medium.com/@cursious92/model-predictive-controllers-291039b9b264](https://medium.com/@cursious92/model-predictive-controllers-291039b9b264)  
21. Stable and Smooth Trajectory Optimization for Autonomous Ground Vehicles via Halton-Sampling-Based MPPI \- MDPI, accessed on February 19, 2026, [https://www.mdpi.com/2504-446X/10/2/96](https://www.mdpi.com/2504-446X/10/2/96)  
22. Benchmarking Model Predictive Control Algorithms in Building Optimization Testing Framework (BOPTEST) \- IBPSA Publications, accessed on February 19, 2026, [https://publications.ibpsa.org/proceedings/bs/2023/papers/bs2023\_1371.pdf](https://publications.ibpsa.org/proceedings/bs/2023/papers/bs2023_1371.pdf)  
23. GPU-Accelerated Motion Planning \- Emergent Mind, accessed on February 19, 2026, [https://www.emergentmind.com/topics/gpu-accelerated-motion-planning](https://www.emergentmind.com/topics/gpu-accelerated-motion-planning)  
24. Geometric Model Predictive Path Integral for Agile UAV Control with Online Collision Avoidance \- arXiv, accessed on February 19, 2026, [https://arxiv.org/html/2510.12924v1](https://arxiv.org/html/2510.12924v1)  
25. Towards Efficient MPPI Trajectory Generation with Unscented Guidance: U-MPPI Control Strategy \- arXiv, accessed on February 19, 2026, [https://arxiv.org/html/2306.12369v3](https://arxiv.org/html/2306.12369v3)  
26. Towards Efficient MPPI Trajectory Generation with Unscented Guidance: U-MPPI Control Strategy \- ResearchGate, accessed on February 19, 2026, [https://www.researchgate.net/publication/387712440\_Towards\_Efficient\_MPPI\_Trajectory\_Generation\_with\_Unscented\_Guidance\_U-MPPI\_Control\_Strategy](https://www.researchgate.net/publication/387712440_Towards_Efficient_MPPI_Trajectory_Generation_with_Unscented_Guidance_U-MPPI_Control_Strategy)  
27. (PDF) Performance Enhanced Risk-Aware MPPI using Gaussian Process for Robust Mobile Robot Control \- ResearchGate, accessed on February 19, 2026, [https://www.researchgate.net/publication/398355403\_Performance\_Enhanced\_Risk-Aware\_MPPI\_using\_Gaussian\_Process\_for\_Robust\_Mobile\_Robot\_Control](https://www.researchgate.net/publication/398355403_Performance_Enhanced_Risk-Aware_MPPI_using_Gaussian_Process_for_Robust_Mobile_Robot_Control)  
28. Risk-Aware Model Predictive Path Integral Control Using ..., accessed on February 19, 2026, [https://ieeexplore.ieee.org/document/10161100](https://ieeexplore.ieee.org/document/10161100)  
29. Dynamic Risk-Aware MPPI for Mobile Robots in Crowds via Efficient Monte Carlo Approximations \- arXiv, accessed on February 19, 2026, [https://arxiv.org/html/2506.21205v1](https://arxiv.org/html/2506.21205v1)  
30. \[Literature Review\] Dynamic Risk-Aware MPPI for Mobile Robots in Crowds via Efficient Monte Carlo Approximations, accessed on February 19, 2026, [https://www.themoonlight.io/en/review/dynamic-risk-aware-mppi-for-mobile-robots-in-crowds-via-efficient-monte-carlo-approximations](https://www.themoonlight.io/en/review/dynamic-risk-aware-mppi-for-mobile-robots-in-crowds-via-efficient-monte-carlo-approximations)  
31. Traffic Management for Swarm Production \- Aalborg University's Research Portal, accessed on February 19, 2026, [https://vbn.aau.dk/en/publications/traffic-management-for-swarm-production/](https://vbn.aau.dk/en/publications/traffic-management-for-swarm-production/)  
32. multi robots control in warehouse \- Advances in Mechanics, accessed on February 19, 2026, [https://www.advancesinmechanics.com/page.php?b=dwn-2024-185](https://www.advancesinmechanics.com/page.php?b=dwn-2024-185)  
33. Decoupled MPPI-Based Multi-Arm Motion Planning \- arXiv, accessed on February 19, 2026, [https://arxiv.org/html/2602.10114v1](https://arxiv.org/html/2602.10114v1)  
34. CoRL-MPPI: Enhancing MPPI With Learnable Behaviours For Efficient And Provably-Safe Multi-Robot Collision Avoidance | Request PDF \- ResearchGate, accessed on February 19, 2026, [https://www.researchgate.net/publication/397556526\_CoRL-MPPI\_Enhancing\_MPPI\_With\_Learnable\_Behaviours\_For\_Efficient\_And\_Provably-Safe\_Multi-Robot\_Collision\_Avoidance](https://www.researchgate.net/publication/397556526_CoRL-MPPI_Enhancing_MPPI_With_Learnable_Behaviours_For_Efficient_And_Provably-Safe_Multi-Robot_Collision_Avoidance)  
35. \[2507.20293\] Decentralized Uncertainty-Aware Multi-Agent Collision Avoidance with Model Predictive Path Integral \- arXiv.org, accessed on February 19, 2026, [https://arxiv.org/abs/2507.20293](https://arxiv.org/abs/2507.20293)  
36. \[2511.09331\] CoRL-MPPI: Enhancing MPPI With Learnable Behaviours For Efficient And Provably-Safe Multi-Robot Collision Avoidance \- arXiv.org, accessed on February 19, 2026, [https://arxiv.org/abs/2511.09331](https://arxiv.org/abs/2511.09331)  
37. AGV Traffic Management in Automated Industrial Plants: An Enhanced Lifelong Multi-Agent Path Finding Approach | Request PDF \- ResearchGate, accessed on February 19, 2026, [https://www.researchgate.net/publication/385184398\_AGV\_Traffic\_Management\_in\_Automated\_Industrial\_Plants\_An\_Enhanced\_Lifelong\_Multi-Agent\_Path\_Finding\_Approach](https://www.researchgate.net/publication/385184398_AGV_Traffic_Management_in_Automated_Industrial_Plants_An_Enhanced_Lifelong_Multi-Agent_Path_Finding_Approach)  
38. (PDF) Decentralized coordination of automated guided vehicles \- ResearchGate, accessed on February 19, 2026, [https://www.researchgate.net/publication/234788053\_Decentralized\_coordination\_of\_automated\_guided\_vehicles](https://www.researchgate.net/publication/234788053_Decentralized_coordination_of_automated_guided_vehicles)  
39. Autonomous Navigation of AGVs in Unknown Cluttered Environments: log-MPPI Control Strategy \- ResearchGate, accessed on February 19, 2026, [https://www.researchgate.net/publication/362144398\_Autonomous\_Navigation\_of\_AGVs\_in\_Unknown\_Cluttered\_Environments\_log-MPPI\_Control\_Strategy](https://www.researchgate.net/publication/362144398_Autonomous_Navigation_of_AGVs_in_Unknown_Cluttered_Environments_log-MPPI_Control_Strategy)  
40. Aggressive Deep Driving: Combining Convolutional Neural Networks and Model Predictive Control \- Proceedings of Machine Learning Research, accessed on February 19, 2026, [http://proceedings.mlr.press/v78/drews17a/drews17a.pdf](http://proceedings.mlr.press/v78/drews17a/drews17a.pdf)  
41. Publications \- Andy Morgan, accessed on February 19, 2026, [https://asmorgan24.github.io/pubs.html](https://asmorgan24.github.io/pubs.html)  
42. Benchmarking Model Predictive Control and Reinforcement Learning Based Control for Legged Robot Locomotion in MuJoCo Simulation \- arXiv, accessed on February 19, 2026, [https://arxiv.org/html/2501.16590v1](https://arxiv.org/html/2501.16590v1)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFcAAAAVCAYAAAAzWHILAAABAElEQVR4Xu3WMUtCURiH8YMIOYo1hBHo5tQQFLS1OUhfoMCxIXARB1uCaImG1gjagyY/g7hF0BS0NDi2CLkpos/l3uCel/CCOoT3/8Bv8H3PdDhcdE4ppZRSSqn17xQ9jDDGtr9Wi5TBE15RRdZfq2Vq4wdlu0joHN/4wBV2/bXXhR2koRwGmBon8UN/dIAO6qjhEl84jp35rYSWHaahigsv89ouEnrEhpnt4RNNbEWzIrpu/qte2w5deLkNu0jo1g6igkt9ceFn5g1DnHknUtQmJrizi4SCFzmv4FUfYccu0tYD+shHvwsu/JaqFRT89brBO55xj33vhFJKKfWPmwF7JiQYm+u6fQAAAABJRU5ErkJggg==>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAJIAAAAVCAYAAACtzrfuAAAEYklEQVR4Xu2aaahVVRTHl2MK5YCEgZFhKQ6ZaJAJmQ+HDEKUVEQculaODR8iBJXwCeaEMwQZBc6YgYVKAwni/JrVog+ivUgsRD8IIYEi+v+7ztZ9F/ecu+8977373uX84A/3rHXPfefttfbaa+97RTIyMjIyWi4vQausMSOjFJ6AjkLtrSMjI5QHoJ+hwdaRkeHTAdoA1UGnoQH5bnkbOmZs1cZz0AJrzCiNtdAf0EroP2hkvlu+h3LGVm2MhhZaY0Y4XLYuQ+ui1z3y3fIkdBvqbuzVxouSJVIqRokmysvWETETumaNVchYaYBEykGnoF+hHdBQ6DNoj8QPcFOwSTTIoTqitwXBgfsOuiB6L19TdllbD/1gbI620GpR/xnRQMyAvoS2Qn3uvbN8XoG+gL4S7WMcj0DnoeGeLQ2pEykHLYbaQB1FB/UXqJtoYJhg1cw2qN4aPfZC31pjxMeiSwJhIP6Htosm0N/QmshXLv1Fk4ichHZ6vjdFY/WUZ0tD6kTaD7WKXvcVfTjuUphUrEgjIp9jFtTO2ELg33gWesE6Ksxv0EFr9GB14ThYeotWK8cc0bEbBA2EdkE9Pf+D0HTvOoT3oeehXqKf/a7n+xz6x7t2FIsPK5mt5MU05u6dJTBf9EY+eCH4ENzVxD3oeGuIGAZ9Cp2DXjO+SsLg3hLdrcXBikQVY7foMhnHFOgbawzkA9G4uI0AJ+UV0TbEp1h8kkhdkXw4GAx2HBMleTCWWoOBPUgpiVRqj3RYbwuGs533MchxbJH4pc3BwLI6fGIdHhtFW4hyOCs6do5nRJ/bjmWx+CSRKpHYLDLbJ4ke/V8VrRyOCaKlkhyALkK/R68LUWsNhlITqbHhARwDwmUqDn63VqhP5OxnkrFH4eaEn5Pz/Fz2WNnZLvD/vg4dFx1vH/rHGZsPD0v52Ys823uR7XHRXSX7tJD4JJEqkbhD4QO9Bc2LXq+IfA+LzsSHomvC090a79qyzBoMzS2R2Cxz8iTBiVToPbWizTX7PvZDHDt+qUtY6dh0OziGTCQmheWS6L1DrCOiNfSv3K/2nUVXDX4e4e7aff9XLD5JpEqkTqKNJHdnTBqebnJ3wEaOScGdm6OL6PrLQzvHNNGG0OmQueZs8WEivW5sleRHaJ81Glyj28/Y2VBzo8Jlh4eZc6ETohX9HdEEcLDi+EuTD8f+hiR/PVEj+pyMC0/huWFhnKip0XsKxacUUiVSKXC2ucGIO+UNqUhvWGOFYEN6E5ptHQWoF11OyoVVnhOLFBq7yZJ+goXEJ4kmSySu0RwQZvwS43OEJFJI4BoTDtiHogd5rDTsdYrB/50HjuXytWig+VOUV42PsKLx7CkNIfFJgktmUq/YYPAnFCzl7HEeMz5HXCI9LToj2Q9wUN3srAR/iSbFcuijfFcs7G3+FD3GKAdOns2iyx83OD7c0iedY4USEp8WA3um5g4P9eqgn6CuxpcEk4jHCzYR0sJ+6lFrbEncAdYQ9udqpAIOAAAAAElFTkSuQmCC>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAL8AAAAVCAYAAAAXbT+OAAAEoElEQVR4Xu2aaahVVRTH/80D2URlTvUIg2ZpNAdSK5qEKMgGCqKSsrKIolLUsEHKqGwusOLRRDNqMw1QKZUNflD8UITiEA0fmuhDhej/39qHs+9677533rnnvnsf7h/84ey17j33nnv3XnutdQ6QSCQSicTWykHUa95YMXtRi6jdvSORaCaadPXYifqU6nD2ZnAK9T61nXckEs1gf+p1b4x4krrBG5vIM9QMb0y0BxOopdRX1OfUK7At+wUMzC37ItSf3IdQf6N/r2s09TO1i3cU5GbqKG9MNM6t1Arq2Mh2JPUD9UVkG0g8Rh3vjYH7qKe9sR9YQ033xoLcQZ3ojYnGOJnaTB3qHeQp6n5vHCB8gvo59gbqfG/sBzqpj7yxIHchTf7KUe6ryb+jd5AHqVO9sY1RqvMBLHX7lnoXXVOfA2HX24oU4grqD28syDykyV85C2CTQYvgMGrbWnelqAjVZ/VFp/3/zp7ZlXqZWgXrrJxNXUNtT71IPZ6/FOfCzrtnZIsZA1tAX8Ki9AmwYlW1z8zodWU4CfbZ+h36Spr8TUDF3y/IJ9tvsIh5RvyiNucN6jtqSBjfjbx+0YLWdR0TxtdRm8KxR3XOc9RuYaxWqH6PUdQ9sPNkvjJot9E5jvCOAqTJ3yTU9TgHVlR9A/uD/oSlCO2OJrW+78WR7R1Y1BdaEPJfGcY3UT+FY8/D1AHheBtqHazrJe5Ffg7VEvKpGaBdYmMY61gdMy2u7tJInVvfZaKzx0xB192vJ/2LxhbkVkt3k1tpzyOwH/Zy52tHFJG1UHcOY928ei934zzYtWTpkyK/drreOBz2vsu8g0yCtR4zllE3RuOXouOY4bBzKjXrKynyV4hy3re9MTAe9idd4B0NUibn763gVi7+dTQeR90ejXWXV6lLluNfQv2Xu+tyNezzO5xd3IY8sqtvr+h7dO7GndFxjNIdnfM47yhAmvwVooi43BsDKhBXI4+m11JvwqLgHNjOoIh7IXU99SEsGraC2bDWZcYtyKO8euqabJfmbpwVbIMiW4YWhlI/sYT6PvJ1wJoDIkuNhBanFlfcKBgRHcdkQWU/7yhAmvwV8gT1OzU1sillUNRUTjwy2FQPTIO9/lXYn7wvLNplUXluUCvQbrIWNumFHmnYGxad9R21cGP0MJsmYBypha5dubq6Rip89d7Pgk/1w/OovQmYoShfbwf1aBEWSbm6I03+ClEbbyhs0qjD8Rasu/EArAOUoWJKKYP65opc4kzU7hqLYT3sVjGYeohaSf1KrYe1J7MOj2cNdZU3kvmwekEtU+1k6npJuiOsBdEdHyNfeL3xKGxHKUOa/C1C0V9b+w5hrFRDE0Io59WNm2Eo17+uErUks9SkJ7QwtGAbRWnhP7A6owgKGEoVy5Amf4uYDNsZMuL7AMqhte2PheXMrUQpXJHHFvRdVfRq52sEPRqi9EgpU28cTP2IvAXbV9R9KvtQXKIBZqF2a19L7RGOdddSz9F0ovwfWxXPwnagIqgdqQf6yqDcX4W/evp/UQup02te0ZVO1BbeiUSlqCAvyj6wRxiKLpZGUP2guirRYrYA9H3x3Az1a0YAAAAASUVORK5CYII=>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAIEAAAAVCAYAAABhV40FAAABc0lEQVR4Xu2YPyjFURTHjzAifwYKZbOzkVdKymRTMslgI0REISmb3SQJGxkMiEkMkk1RRillUxj4ns79cdyJvOXdvp/61Lnn3N9b3n3nd+4TIYQQQgghhBBCCPlJBTyPk44cvIKNcYGkwzQcipOOCfgBB+LCP9GDpZ/7W5fsMZJvSuEBLIoLDt1zB3vjAkmDLjgZ5fbgSJTbhU1RjiTCCmxz63qx1jvocsXw1K1JYuzAFrduhS+wxuV64IJb54u/zgSL9hjJN5twPMQ6F6zBZ9gccnViX1ZVWOtcsAGn4BycgfuwA47CbTgW9pICQW8Gb/BQbPi7hvfwFZ7ABzifbRabH/rhjXx3Cz0ksyHuhMchJgVCtdgv+R0eic0E2v5vg8Ow5Gu3SANcFTs8SpnYgck6hV4n10NMEuYMtoc4By9cTTtKH6x1OZIY5fBJ7L8DRTvCsqs9ir0m9NZBEkUHQH19ZGzB7hBXwkux4VBfG4SQQuATOEVO4dM7BegAAAAASUVORK5CYII=>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAALMAAAAVCAYAAAANUd8AAAADyElEQVR4Xu2aWahNYRTHl9mLmczzEPJgKvOQeYjkxaxrSMoYIlOGjCU8Gh4QMpUhZcgYGQtR5ulBkQgvCA/8/9be93xnuc49+7r33HOP71f/+Nb69rmus/b61lp7i3g8Ho/HUxQYAK2zRo+nIKgOfYV+QrehMwl0HnoV7HXVWnKmMXQJKm0dnsxkEjTUGlPMfNGgvCLJBV4laAh0TvS6jfHu35SBbkFtrMOTefSHVkFToJHQWqhr3I7UUUw06zIwlxhfIngdS4i3UAnjmwFdNjZPhsIMOFA0e92HxsufAZFK6kMfoC9QW+PLjWNQH2O7AWUZmydDaQhtg3ZAO6FTUDV3QyEwTjQ7P4YqGl8iWBv3c9ZNRD+H9bjnP4DB3EpiZUYzqGbcjsJhv2ggHrWOCPCU+WSNARWg69bo0AO6A9WzDk/qGQ4dgU5AHR17DegZ1M2xkc4S7VjnF22nCYnEujwKDLbnotfONb5kYUN40xoDFkKTrdFhnujPHmsdntTSUjSQyVVoj+ObJvolMRunOx2gH9A3qKnxJcNB6LQ1glLQSdHG8W9wD2/6YdbhSS1LRScSjUQDd47jOwS9dtbpzmrR34E3JQMsCixRWK5Y+oqOAV3YPM4yNl7PMiy/GAMdhtpZhyd3wkCoHayZid5Bu7N3pD9lRQOZv8dg48sNZmbKsh7q4qzriH7+RMfGic5FZ50f8Obgz5luHZ7cuSf6xCyEGcF+aXklas28Ui/LEzOhLdaYBFsl5zLjgMRnx/aio8Cqjm0QtMJZ5wd8yDMaKm4dnsQwozGI2OiEhE1NA9FO3x1jpSu9RW/IqCUG4YOUa9YI9kqsqeRptR36CDUPbJzm8GatHKxZN/M0WyBawi2CjkPdodmipYxbynnyGd79b6BlwZrTgSfQ52DN7JTMI+PChI0sH3qEQRWVCdB7axS9wb+L3iRs8u5CL0UbzQui/2/Lw82i9TUz6iOJZW8Ge/iUspfoI3VPAdJTtOFg07dBNJOw/qRGxbalJQwaBhnn3nklbIBbGHsV0czKSclZ0ZqZZcXTQFOhktm7RepCmyV2ypUTDfzwJuOJtyv4u8cTB5svZk1OHaLAmbA97plxGWz/ChNA+M5KD9ETI4T/1hGiM3yPJ45Nok1fFGqJBm4DY2c2ZYb/F8qLlith3c7PXOP4+IITTxJOSTyebLJEpxBRYM3KgHWzZQgb4RdQJ+uIAMszliUh+0TfNiScUPAlLTaBLEc8nt/wjbeHwZ89E4jvMHMawSnEA4mN/xZLzjCQ2di5dbCnCPALPnTKwLykUfgAAAAASUVORK5CYII=>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAZCAYAAADnstS2AAAAvklEQVR4XmNgGJRgMRDfBuJcdAlcAKThPxBLo0tgA+YMEMWF6BK4wAMgPo4uiAt0M0BMV0aXwAYsGSCKq9AlcAFQqFxAF8QGGIF4PwPEdB00ORTABMRzgXgOA0RxPao0KiBaMQsQrwTifij/JhDfQEgjADsQbwTiYwwQTSDQxQAx3QymCAQ4gXg3EF8DYlEkcQsGiOI+mADI59uB+B0D9kgAOeUBA0QdAxcQPwfiQCQFyCCHAWK6B7rEKCAaAADDjSN4tn7/vAAAAABJRU5ErkJggg==>