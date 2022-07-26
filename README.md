<!-- PROJECT SHIELDS -->
<!-- These badges can be used once we make the project public -->
<!-- [![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url] -->

<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/ADVRHumanoids/tree-motor-tests">
    <img src="https://alberobotics.it/images/apple-touch-icon.png" alt="Logo" width="80" height="80">
  </a>

  <h2 align="center">tree-motor-tests</h2>

  <p align="center">
    A set of tests to evaluate new actuators
    <br />
    <a href="https://github.com/ADVRHumanoids/tree-motor-tests"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <!-- <a href="https://github.com/ADVRHumanoids/tree-motor-tests">View Demo</a>
    ·
    <a href="https://github.com/ADVRHumanoids/tree-motor-tests/issues">Request Feature</a>
    ·
    <a href="https://github.com/ADVRHumanoids/tree-motor-tests/issues">Report Bug</a>
    -->
  </p>
</p>

<!--
[![Build Status](https://app.travis-ci.com/ADVRHumanoids/tree-motor-tests.svg?token=zJseufwSAzkrEc1mqg8v&branch=development)](https://app.travis-ci.com/ADVRHumanoids/tree-motor-tests)
[![codecov](https://codecov.io/gh/ADVRHumanoids/tree-motor-tests/branch/development/graph/badge.svg?token=aW77dBlb1w)](https://codecov.io/gh/ADVRHumanoids/tree-motor-tests)
-->

---

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#documentation">Documentation</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <!-- <li><a href="#license">License</a></li> -->
    <li><a href="#contact">Contact</a></li>
    <!-- <li><a href="#acknowledgements">Acknowledgements</a></li> -->
  </ol>
</details>

---

<!-- ABOUT THE PROJECT -->

## About The Project

This repo contains Alberobotics' set of tests to evaluate new electronics, motors, and components.

## Getting Started

To get a local copy up and running follow these simple example steps.

### Prerequisites

EtherCAT libraries:

- [**soem_advr**](https://gitlab.advr.iit.it/xeno-ecat/soem_advr) (branch **xeno-3**) at commit `8697f06`.
- [**ADVRHumanoids/ecat_master_tree**](https://github.com/ADVRHumanoids/ecat_master_tree) (branch **mt_old_master**) which detached from [ecat_master_advr](https://gitlab.advr.iit.it/xeno-ecat/ecat_master_advr) at commit `6e9ac5b`.
- [**ADVRHumanoids/ec_master_app**](https://github.com/ADVRHumanoids/ec_master_app) (branch **mt_stable**) which detached from [ec_master_tests](https://gitlab.advr.iit.it/xeno-ecat/ec_master_tests) at commit `a40d8184`.

If you need to use the old ecat master, use the `mt_old_master` branch.

### Installation

In the instruction below this repo has been cloned with the command `git@github.com:ADVRHumanoids/tree-motor-tests.git motor_tests` (it renames the folder from `tree-motor-tests` to `motor_tests`). For these instructions, we also assume to have both `ec_master_app` and `motor_test` located in `~/ecat_dev/`.

## Usage

The tests have to be run manually, below they are all listed and along with instructions on to run/process them.

### Naming

These tests follow this naming structure `test-${CONTROLLER_TYPE}-${OUTPUT_FLANGE_STATUS}-${REFERENCY_TYPE}`, where:

- `CONTROLLER_TYPE` can be: `current`, `torque`, or `velocity`.
- `OUTPUT_FLANGE_STATUS` can be: `free` (the output flange has nothing attached), `locked` (the output flange is fully locked), or `lever`
- `REFERENCY_TYPE` tries to describe the reference trajectory that will be given to the motor.

e.g.: `test-current-locked-chirp` is a test where a motor with fully locked output receives a chirp current reference.

The only exceptions to this naming convention are `test-pdo` and `test-phase`.

### 0. Test PDO

|                  |                                                                                                                                                                                                                                                                                                                   |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Motor locked:    | Not relevant                                                                                                                                                                                                                                                                                                      |
| Description:     | Passively listen to makes sure we receive data from all elements of the PDO.                                                                                                                                                                                                                                      |
| Code location:   | `~/ecat_dev/ec_master_app/motor-tests/test-pdo`                                                                                                                                                                                                                                                                   |
| Data aquisition: | `~/ecat_dev/ecat_master_tree/build_rt/tools/motor-tests/test-pdo/test-pdo ~/ecat_dev/ec_master_app/tools/motor-tests/config.yaml`                                                                                                                                                                                 |
| Data processing: | None                                                                                                                                                                                                                                                                                                              |
| Notes:           | The executable will print the path of the newly generated yaml file, it is generally convinent to set it as an environment varible `$RESUTS` to more easily refer to this file while rinning the next tests ( eg. set `RESULTS="/logs/AOR02-EOR02-H3236_2020-10-16--17-37-42_test-results.yaml && echo $RESULTS`) |

### 1. Test phase angle

|                  |                                                                                                                                                                                                                                                                                                                      |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Motor locked:    | No.                                                                                                                                                                                                                                                                                                                  |
| Description:     | In JOINT_CURRENT_MODE we command a back and forth motion to the motor and find the commutation offset angle that maximizes speed.                                                                                                                                                                                    |
| Code location:   | `~/ecat_dev/ecat_master_tree/tools/motor-tests/test-phase`                                                                                                                                                                                                                                                           |
| Data aquisition: | `~/ecat_dev/ecat_master_tree/build_rt/tools/motor-tests/test-phase/test-phase $RESUTS`                                                                                                                                                                                                                               |
| Data processing: | `python3 ~/ecat_dev/motor_tests/utils/process_phase.py $RESULTS`                                                                                                                                                                                                                                                     |
| Notes:           | The full list of paramters that can be used is visible in the config file as [`test_phase`](https://github.com/ADVRHumanoids/ecat_master_tree/blob/75d4fce6dfab3bc9b8e0de9105ae42e3fbe9cc3f/tools/motor-tests/config.yaml#L60-L89). Some of them are motor-specific, make sure `$RESULTS` contains the correct ones. |

### 2. test-current-free-\*

#### 2a. test-current-free-spline

|                  |                                                                                                                                                                                                                                                                                                                                                                                        |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Motor locked:    | No.                                                                                                                                                                                                                                                                                                                                                                                    |
| Description:     | In JOINT_CURRENT_MODE, we command positive/negative 3rd order splines of current to reach iq_ref = 6.7A in 50s.                                                                                                                                                                                                                                                                        |
| Code location:   | `~/ecat_dev/ecat_master_tree/tools/motor-tests/test-current-free-spline`                                                                                                                                                                                                                                                                                                               |
| Data aquisition: | `~/ecat_dev/ecat_master_tree/build_rt/tools/motor-tests/test-current-free-spline/test-current-free-spline $RESUTS`                                                                                                                                                                                                                                                                     |
| Data processing: | `python3 ~/ecat_dev/motor_tests/utils/process_current_free.py $RESULTS`                                                                                                                                                                                                                                                                                                                |
| Notes:           | The full list of paramters that can be used is visible in the config file as [`test-current-free`](https://github.com/ADVRHumanoids/ecat_master_tree/blob/75d4fce6dfab3bc9b8e0de9105ae42e3fbe9cc3f/tools/motor-tests/config.yaml#L91-L99). Some of them are motor-specific, make sure `$RESULTS` contains the correct ones. _Make also sure that in the confing file: `type: spline`_. |

#### 2b. test-current-free-smooth

|                  |                                                                                                                                                                                                                                                                                                                                                                                        |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Motor locked:    | No.                                                                                                                                                                                                                                                                                                                                                                                    |
| Description:     | In JOINT_CURRENT_MODE, we command positive/negative smoothed steps of current to reach iq_ref = +/-6.7A in 50s.                                                                                                                                                                                                                                                                        |
| Code location:   | `~/ecat_dev/ecat_master_tree/tools/motor-tests/test-current-free-smooth`                                                                                                                                                                                                                                                                                                               |
| Data aquisition: | `~/ecat_dev/ecat_master_tree/build_rt/tools/motor-tests/test-current-free-smooth/test-current-free-smooth $RESUTS`                                                                                                                                                                                                                                                                     |
| Data processing: | `python3 ~/ecat_dev/motor_tests/utils/process_current_free.py $RESULTS`                                                                                                                                                                                                                                                                                                                |
| Notes:           | The full list of paramters that can be used is visible in the config file as [`test-current-free`](https://github.com/ADVRHumanoids/ecat_master_tree/blob/75d4fce6dfab3bc9b8e0de9105ae42e3fbe9cc3f/tools/motor-tests/config.yaml#L91-L99). Some of them are motor-specific, make sure `$RESULTS` contains the correct ones. _Make also sure that in the confing file: `type: smooth`_. |

#### 2c. test-current-free-ramp

|                  |                                                                                                                                                                                                                                                                                                                                                                                      |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Motor locked:    | No.                                                                                                                                                                                                                                                                                                                                                                                  |
| Description:     | In JOINT_CURRENT_MODE, we command rising/descending ramps of current to reach iq_ref = +/-6.7A in 50s.                                                                                                                                                                                                                                                                               |
| Code location:   | `~/ecat_dev/ecat_master_tree/tools/motor-tests/test-current-free-ramp`                                                                                                                                                                                                                                                                                                               |
| Data aquisition: | `~/ecat_dev/ecat_master_tree/build_rt/tools/motor-tests/test-current-free-ramp/test-current-free-ramp $RESUTS`                                                                                                                                                                                                                                                                       |
| Data processing: | `python3 ~/ecat_dev/motor_tests/utils/process_current_free.py $RESULTS`                                                                                                                                                                                                                                                                                                              |
| Notes:           | The full list of paramters that can be used is visible in the config file as [`test-current-free`](https://github.com/ADVRHumanoids/ecat_master_tree/blob/75d4fce6dfab3bc9b8e0de9105ae42e3fbe9cc3f/tools/motor-tests/config.yaml#L91-L99). Some of them are motor-specific, make sure `$RESULTS` contains the correct ones. _Make also sure that in the confing file: `type: ramp`_. |

### 3. test-velocity-free-\*

#### 3a. test-velocity-free-spline

|                  |                                                                                                                                                                                                                                                                                                                                                                                           |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Motor locked:    | No.                                                                                                                                                                                                                                                                                                                                                                                       |
| Description:     | In JOINT_SPEED_MODE, we command positive/negative 3rd order splines velocity reaference to reach 3.0 rad/s in 50s.                                                                                                                                                                                                                                                                        |
| Code location:   | `~/ecat_dev/ecat_master_tree/tools/motor-tests/test-velocity-free-spline`                                                                                                                                                                                                                                                                                                                 |
| Data aquisition: | `~/ecat_dev/ecat_master_tree/build_rt/tools/motor-tests/test-velocity-free-spline/test-velocity-free-spline $RESUTS`                                                                                                                                                                                                                                                                      |
| Data processing: | `python3 ~/ecat_dev/motor_tests/utils/process_velocity_free.py $RESULTS`                                                                                                                                                                                                                                                                                                                  |
| Notes:           | The full list of paramters that can be used is visible in the config file as [`test_velocity_free`](https://github.com/ADVRHumanoids/ecat_master_tree/blob/75d4fce6dfab3bc9b8e0de9105ae42e3fbe9cc3f/tools/motor-tests/config.yaml#L162-L170). Some of them are motor-specific, make sure `$RESULTS` contains the correct ones. _Make also sure that in the confing file: `type: spline`_. |

#### 3b. test-velocity-free-smooth

|                  |                                                                                                                                                                                                                                                                                                                                                                                           |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Motor locked:    | No.                                                                                                                                                                                                                                                                                                                                                                                       |
| Description:     | In JOINT_SPEED_MODE, we command positive/negative smoothed steps velocity reaference to reach 3.0 rad/s in 50s.                                                                                                                                                                                                                                                                           |
| Code location:   | `~/ecat_dev/ecat_master_tree/tools/motor-tests/test-velocity-free-smooth`                                                                                                                                                                                                                                                                                                                 |
| Data aquisition: | `~/ecat_dev/ecat_master_tree/build_rt/tools/motor-tests/test-velocity-free-smooth/test-velocity-free-smooth $RESUTS`                                                                                                                                                                                                                                                                      |
| Data processing: | `python3 ~/ecat_dev/motor_tests/utils/process_velocity_free.py $RESULTS`                                                                                                                                                                                                                                                                                                                  |
| Notes:           | The full list of paramters that can be used is visible in the config file as [`test_velocity_free`](https://github.com/ADVRHumanoids/ecat_master_tree/blob/75d4fce6dfab3bc9b8e0de9105ae42e3fbe9cc3f/tools/motor-tests/config.yaml#L162-L170). Some of them are motor-specific, make sure `$RESULTS` contains the correct ones. _Make also sure that in the confing file: `type: smooth`_. |

#### 3c. test-velocity-free-ramp

|                  |                                                                                                                                                                                                                                                                                                                                                                                         |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Motor locked:    | No.                                                                                                                                                                                                                                                                                                                                                                                     |
| Description:     | In JOINT_SPEED_MODE, we command rising/descending ramps velocity reaference to reach 3.0 rad/s in 50s.                                                                                                                                                                                                                                                                                  |
| Code location:   | `~/ecat_dev/ecat_master_tree/tools/motor-tests/test-velocity-free-ramp`                                                                                                                                                                                                                                                                                                                 |
| Data aquisition: | `~/ecat_dev/ecat_master_tree/build_rt/tools/motor-tests/test-velocity-free-ramp/test-velocity-free-ramp $RESUTS`                                                                                                                                                                                                                                                                        |
| Data processing: | `python3 ~/ecat_dev/motor_tests/utils/process_velocity_free.py $RESULTS`                                                                                                                                                                                                                                                                                                                |
| Notes:           | The full list of paramters that can be used is visible in the config file as [`test_velocity_free`](https://github.com/ADVRHumanoids/ecat_master_tree/blob/75d4fce6dfab3bc9b8e0de9105ae42e3fbe9cc3f/tools/motor-tests/config.yaml#L162-L170). Some of them are motor-specific, make sure `$RESULTS` contains the correct ones. _Make also sure that in the confing file: `type: ramp`_. |

### 4. test-velocity-free-steps

|                  |                                                                                                                                                                                                                                                                                                                                                                                          |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Motor locked:    | No.                                                                                                                                                                                                                                                                                                                                                                                      |
| Description:     | In JOINT_SPEED_MODE, we command In JOINT_SPEED_MODE we command constant velocities of 0.02, 0.5, 1.5, 3.5 rad/s in overall 40s.                                                                                                                                                                                                                                                          |
| Code location:   | `~/ecat_dev/ecat_master_tree/tools/motor-tests/test-velocity-free-steps`                                                                                                                                                                                                                                                                                                                 |
| Data aquisition: | `~/ecat_dev/ecat_master_tree/build_rt/tools/motor-tests/test-velocity-free-steps/test-velocity-free-steps $RESUTS`                                                                                                                                                                                                                                                                       |
| Data processing: | `python3 ~/ecat_dev/motor_tests/utils/process_velocity_steps.py $RESULTS`                                                                                                                                                                                                                                                                                                                |
| Notes:           | The full list of paramters that can be used is visible in the config file as [`test_velocity_steps`](https://github.com/ADVRHumanoids/ecat_master_tree/blob/75d4fce6dfab3bc9b8e0de9105ae42e3fbe9cc3f/tools/motor-tests/config.yaml#L172-L176). Some of them are motor-specific, make sure `$RESULTS` contains the correct ones. _Make also sure that in the confing file: `type: ramp`_. |

### 5. test-current-locked-ramp

|                  |                                                                                                                                                                                                                                                                                                                                                                                          |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Motor locked:    | Output fully locked.                                                                                                                                                                                                                                                                                                                                                                     |
| Description:     | In JOINT_CURRENT_MODE, we command rising/descending ramps current to reach 15A in 15s.                                                                                                                                                                                                                                                                                                   |
| Code location:   | `~/ecat_dev/ecat_master_tree/tools/motor-tests/test-current-locked-ramp`                                                                                                                                                                                                                                                                                                                 |
| Data aquisition: | `~/ecat_dev/ecat_master_tree/build_rt/tools/motor-tests/test-current-locked-ramp/test-current-locked-ramp $RESUTS`                                                                                                                                                                                                                                                                       |
| Data processing: | `python3 ~/ecat_dev/motor_tests/utils/process_current_locked.py $RESULTS`                                                                                                                                                                                                                                                                                                                |
| Notes:           | The full list of paramters that can be used is visible in the config file as [`test_current_locked`](https://github.com/ADVRHumanoids/ecat_master_tree/blob/75d4fce6dfab3bc9b8e0de9105ae42e3fbe9cc3f/tools/motor-tests/config.yaml#L101-L111). Some of them are motor-specific, make sure `$RESULTS` contains the correct ones. _Make also sure that in the confing file: `type: ramp`_. |

### 6. test-current-locked-step

|                  |                                                                                                                                                                                                                                                                                                                                       |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Motor locked:    | Output fully locked.                                                                                                                                                                                                                                                                                                                  |
| Description:     | In JOINT_CURRENT_MODE we command +/- constant step currents (0.5s rise time for each smooth step) of 1, 4, 7, 10, 15Amps in overall 2s each.                                                                                                                                                                                          |
| Code location:   | `~/ecat_dev/ecat_master_tree/tools/motor-tests/test-current-locked-step`                                                                                                                                                                                                                                                              |
| Data aquisition: | `~/ecat_dev/ecat_master_tree/build_rt/tools/motor-tests/test-current-locked-step/test-current-locked-step $RESUTS`                                                                                                                                                                                                                    |
| Data processing: | `python3 ~/ecat_dev/motor_tests/utils/process_current_locked_steps.py $RESULTS`                                                                                                                                                                                                                                                       |
| Notes:           | The full list of paramters that can be used is visible in the config file as [`test_current_locked_steps`](https://github.com/ADVRHumanoids/ecat_master_tree/blob/75d4fce6dfab3bc9b8e0de9105ae42e3fbe9cc3f/tools/motor-tests/config.yaml#L120-L126). Some of them are motor-specific, make sure `$RESULTS` contains the correct ones. |

### 7. test-current-locked-chirp

|                  |                                                                                                                                                                                                                                                                                                                                       |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Motor locked:    | Output fully locked.                                                                                                                                                                                                                                                                                                                  |
| Description:     | In JOINT_CURRENT_MODE we command a chirp of iq_ref = A \* sin(2pi(f_0+ (f_f- f_0)/duration/2\* t)\* t), where A=1.5, f_0=0.01, f_f=50, duration = 200s.                                                                                                                                                                               |
| Code location:   | `~/ecat_dev/ecat_master_tree/tools/motor-tests/test-current-locked-chirp`                                                                                                                                                                                                                                                             |
| Data aquisition: | `~/ecat_dev/ecat_master_tree/build_rt/tools/motor-tests/test-current-locked-chirp/test-current-locked-chirp $RESUTS`                                                                                                                                                                                                                  |
| Data processing: | `python3 ~/ecat_dev/motor_tests/utils/process_current_locked_chirp.py $RESULTS`                                                                                                                                                                                                                                                       |
| Notes:           | The full list of paramters that can be used is visible in the config file as [`test_current_locked_chirp`](https://github.com/ADVRHumanoids/ecat_master_tree/blob/75d4fce6dfab3bc9b8e0de9105ae42e3fbe9cc3f/tools/motor-tests/config.yaml#L128-L135). Some of them are motor-specific, make sure `$RESULTS` contains the correct ones. |

### 8. test-current-locked-smooth

|                  |                                                                                                                                                                                                                                                                                                                                        |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Motor locked:    | Output fully locked.                                                                                                                                                                                                                                                                                                                   |
| Description:     | In JOINT_CURRENT_MODE we command a few smooth steps up +/- 15A in a short time (20ms max).                                                                                                                                                                                                                                             |
| Code location:   | `~/ecat_dev/ecat_master_tree/tools/motor-tests/test-current-locked-smooth`                                                                                                                                                                                                                                                             |
| Data aquisition: | `~/ecat_dev/ecat_master_tree/build_rt/tools/motor-tests/test-current-locked-smooth/test-current-locked-smooth $RESUTS`                                                                                                                                                                                                                 |
| Data processing: | `python3 ~/ecat_dev/motor_tests/utils/process_current_locked_smooth.py $RESULTS`                                                                                                                                                                                                                                                       |
| Notes:           | The full list of paramters that can be used is visible in the config file as [`test_current_locked_smooth`](https://github.com/ADVRHumanoids/ecat_master_tree/blob/75d4fce6dfab3bc9b8e0de9105ae42e3fbe9cc3f/tools/motor-tests/config.yaml#L128-L135). Some of them are motor-specific, make sure `$RESULTS` contains the correct ones. |

### 9. test-velocity-lever-steps

|                  |                                                                                                                                                                                                                                                                                                                                       |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Motor locked:    | Loaded lever at the end-effector (overall 85N at 28.5cm distance =24.2Nm).                                                                                                                                                                                                                                                            |
| Description:     | In JOINT_SPEED_MODE we command executing two turns forward and then two turns backward at (controlled to be) constant velocities of 0.5 and 1 rad/s.                                                                                                                                                                                  |
| Code location:   | `~/ecat_dev/ecat_master_tree/tools/motor-tests/test-velocity-lever-steps`                                                                                                                                                                                                                                                             |
| Data aquisition: | `~/ecat_dev/ecat_master_tree/build_rt/tools/motor-tests/test-velocity-lever-steps/test-velocity-lever-steps $RESUTS`                                                                                                                                                                                                                  |
| Data processing: | `python3 ~/ecat_dev/motor_tests/utils/process_velocity_lever_steps.py $RESULTS`                                                                                                                                                                                                                                                       |
| Notes:           | The full list of paramters that can be used is visible in the config file as [`test_velocity_lever_steps`](https://github.com/ADVRHumanoids/ecat_master_tree/blob/75d4fce6dfab3bc9b8e0de9105ae42e3fbe9cc3f/tools/motor-tests/config.yaml#L128-L135). Some of them are motor-specific, make sure `$RESULTS` contains the correct ones. |

### 10-11. test-torque-lever-drops

|                  |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Motor locked:    | Loaded lever at the end-effector (long lever with 2Kg weight, 37N overall at 26cm distance).                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| Description:     | In JOINT_INPEDANCE_MODE, using torque gains of Kp=1 and Ks=0.003, friction compensation gain of 0.9, and impedance gain of 200, we command a rotation by 90 degrees (so that the lever becomes horizontal to the ground). The impedance gain is then set to zero to see the zero torque control behavior under the effect of gravity. This is repeated for friction compensation 0.99.                                                                                                                                                     |
| Code location:   | `~/ecat_dev/ecat_master_tree/tools/motor-tests/test-torque-lever-drops`                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| Data aquisition: | `~/ecat_dev/ecat_master_tree/build_rt/tools/motor-tests/test-torque-lever-drops/test-torque-lever-steps $RESUTS`                                                                                                                                                                                                                                                                                                                                                                                                                           |
| Data processing: | `python3 ~/ecat_dev/motor_tests/utils/process_torque_lever_drops.py $RESULTS`                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| Notes:           | The full list of paramters that can be used is visible in the config file as [`test_torque_lever_drops`](https://github.com/ADVRHumanoids/ecat_master_tree/blob/75d4fce6dfab3bc9b8e0de9105ae42e3fbe9cc3f/tools/motor-tests/config.yaml#L147-L151). The gains can be set in the [`inpedance`](https://github.com/ADVRHumanoids/ecat_master_tree/blob/75d4fce6dfab3bc9b8e0de9105ae42e3fbe9cc3f/tools/motor-tests/config.yaml#L43) field of the config file. Some of them are motor-specific, make sure `$RESULTS` contains the correct ones. |

### 12. test-torque-locked-smooth

|                  |                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Motor locked:    | Output fully locked.                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| Description:     | In JOINT_INPEDANCE_MODE, using torque gains of Kp=1 and Ks=0.003, friction compensation gain of 0.9, and impedance gain of 200, we command a set of smooth (rise time of 1s) steps of overall (up and down) 5s, with different amplitudes 1, 5, 10, 20, 50, and 90Nm.                                                                                                                                                                                       |
| Code location:   | `~/ecat_dev/ecat_master_tree/tools/motor-tests/test-torque-locked-smooth`                                                                                                                                                                                                                                                                                                                                                                                   |
| Data aquisition: | `~/ecat_dev/ecat_master_tree/build_rt/tools/motor-tests/test-torque-locked-smooth/test-torque-locked-smooth $RESUTS`                                                                                                                                                                                                                                                                                                                                        |
| Data processing: | `python3 ~/ecat_dev/motor_tests/utils/process_torque_locked_smooth.py $RESULTS`                                                                                                                                                                                                                                                                                                                                                                             |
| Notes:           | The full list of paramters that can be used is visible in the config file as [`test_torque_locked_smooth`](https://github.com/ADVRHumanoids/ecat_master_tree/blob/75d4fce6dfab3bc9b8e0de9105ae42e3fbe9cc3f/tools/motor-tests/config.yaml#L137-L145). The gains can be set in the [`inpedance`](https://github.com/ADVRHumanoids/ecat_master_tree/blob/75d4fce6dfab3bc9b8e0de9105ae42e3fbe9cc3f/tools/motor-tests/config.yaml#L43) field of the config file. |

### 13. test-torque-locked-chirp

|                  |                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Motor locked:    | Output fully locked.                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| Description:     | In JOINT_INPEDANCE_MODE, using torque gains of Kp=1 and Ks=0.003, friction compensation gain of 0.9, and impedance gain of 0, we command a chirp of 150s duration with 15Nm amplitude starting from 0.01Hz to a target frequency of 5Hz. The torque tracking bode is obtained as (output/input = torque/reference).                                                                                                                                        |
| Code location:   | `~/ecat_dev/ecat_master_tree/tools/motor-tests/test-torque-locked-chirp`                                                                                                                                                                                                                                                                                                                                                                                   |
| Data aquisition: | `~/ecat_dev/ecat_master_tree/build_rt/tools/motor-tests/test-torque-locked-chirp/test-torque-locked-chirp $RESUTS`                                                                                                                                                                                                                                                                                                                                         |
| Data processing: | `python3 ~/ecat_dev/motor_tests/utils/process_torque_locked_chirp.py $RESULTS`                                                                                                                                                                                                                                                                                                                                                                             |
| Notes:           | The full list of paramters that can be used is visible in the config file as [`test_torque_locked_chirp`](https://github.com/ADVRHumanoids/ecat_master_tree/blob/75d4fce6dfab3bc9b8e0de9105ae42e3fbe9cc3f/tools/motor-tests/config.yaml#L137-L145). The gains can be set in the [`inpedance`](https://github.com/ADVRHumanoids/ecat_master_tree/blob/75d4fce6dfab3bc9b8e0de9105ae42e3fbe9cc3f/tools/motor-tests/config.yaml#L43) field of the config file. |

## Documentation

Documentation can be found in the [Github Wiki page](git@github.com:ADVRHumanoids/tree-motor-tests.git).

## Roadmap

See the [open issues](https://github.com/ADVRHumanoids/tree-motor-tests/issues) for a list of proposed features (and known issues).

<!-- CONTRIBUTING -->

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<!-- TODO:LICENSE - ->
## License

Distributed under the MIT License. See `LICENSE` for more information. -->

<!-- CONTACT -->

## Contact

Alberobotics team - alberobotics@iit.it

Project Link: [https://github.com/ADVRHumanoids/tree-motor-tests](https://github.com/ADVRHumanoids/tree-motor-tests)

<!-- ACKNOWLEDGEMENTS - ->
## Acknowledgements -->

<!-- MARKDOWN LINKS & IMAGES -->
<!-- These will be used once we make the project public -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links - ->

[contributors-shield]: https://img.shields.io/github/contributors/ADVRHumanoids/tree-motor-tests.svg?style=for-the-badge
[contributors-url]: https://github.com/ADVRHumanoids/tree-motor-tests/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/ADVRHumanoids/tree-motor-tests.svg?style=for-the-badge
[forks-url]: https://github.com/ADVRHumanoids/tree-motor-tests/network/members
[stars-shield]: https://img.shields.io/github/stars/ADVRHumanoids/tree-motor-tests.svg?style=for-the-badge
[stars-url]: https://github.com/ADVRHumanoids/tree-motor-tests/stargazers
[issues-shield]: https://img.shields.io/github/issues/ADVRHumanoids/tree-motor-tests.svg?style=for-the-badge
[issues-url]: https://github.com/ADVRHumanoids/tree-motor-tests/issues
[license-shield]: https://img.shields.io/github/license/ADVRHumanoids/tree-motor-tests.svg?style=for-the-badge
[license-url]: https://github.com/ADVRHumanoids/tree-motor-tests/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png -->
