#!/usr/bin python3
# -*- coding: utf-8 -*-

import os
import sys
import yaml
import numpy as np
# tell matplotlib not to try to load up GTK as it returns errors over ssh
from matplotlib import use as plt_use
plt_use("Agg")
from matplotlib import pyplot as plt

def process(yaml_file, plot_all=False):
    plt.rcParams['savefig.dpi'] = 300

    # read parameters from yaml file
    print('[i] Using yaml_file: ' + yaml_file)
    with open(yaml_file) as f:
        try:
            out_dict = yaml.safe_load(f)
        except Exception:
            raise Exception('error in yaml parsing')

    # find logs
    if 'location' in out_dict['log']:
        head = out_dict['log']['location']
    else:
        head, _ = os.path.split(yaml_file)

    if 'name' in out_dict['log']:
        code_string = out_dict['log']['name']
    else:
        _, tail = os.path.split(yaml_file)
        code_string = tail[:-len("-results.yaml")]

    if 'test_type' in out_dict['test_torque_locked_smooth']:
        test_type = out_dict['test_torque_locked_smooth']['test_type']
    else:
        test_type = "ramp"

    # set path to save graphs
    if len(head)>6 and head[-6:]=='/logs/':
        new_head = f'{head[:-6]}/images/'
    else:
        new_head = f'{head}/images/'

    if os.path.isdir(new_head) is False:
        try:
            os.makedirs(new_head)
        except OSError:
            print("Creation of the directory %s failed" % new_head)
    image_base_path = new_head +f'{code_string}_test-torque-locked-{test_type}'

    log_file = head + f'{code_string}_test-torque-locked-{test_type}.log'
    print('[i] Reading log_file: ' + log_file)

    # log format: '%u64\t%u\t%u\t%u\t%u\t%f\t%f\t%d\t%f\t%f\t%f'
    ns         = [np.uint64(x.split('\t')[0]) for x in open(log_file).readlines()]
    motor_tor  = [    float(x.split('\t')[1]) for x in open(log_file).readlines()]
    torque_ref = [    float(x.split('\t')[2]) for x in open(log_file).readlines()]
    i_fb       = [    float(x.split('\t')[3]) for x in open(log_file).readlines()]

    if not 'test_torque_locked_smooth' in out_dict:
        raise Exception("missing 'test_torque_locked_smooth' in yaml parsing")

    torque_const = float(out_dict['results']['flash_params']['motorTorqueConstant'])
    gear_ratio = float(out_dict['results']['flash_params']['Motor_gear_ratio'])

    print(f'[i] Processing data (loaded {len(ns)} points)')
    print(f"    torque_const: {torque_const}\n    gear_ratio:   {gear_ratio}")

# motor_vel vs time ------------------------------------------------------
    fig, axs = plt.subplots()
    l0 = axs.plot([s/1e9 for s in ns], i_fb, color='#8e8e8e', marker='.', markersize=0.2, zorder=0, label='i_q*torque_const*gear_ratio')

    # make plot pretty
    axs.set_xlabel('Time (s)')
    axs.set_ylabel('Current (A)')
    axs.set_xlim(ns[0]/1e9, ns[-1]/1e9)
    axs.grid(b=True, which='major', axis='x', linestyle=':')
    axs.grid(b=True, which='major', axis='y', linestyle='-')
    axs.grid(b=True, which='minor', axis='y', linestyle=':')
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)

    # Save the graph
    fig_name = image_base_path + '_0.png'
    plt.savefig(fname=fig_name, format='png', bbox_inches='tight')
    print('[i] Saved graphSaved graph as: ' + fig_name)
    if plot_all:
        plt.show()

    # motor_vel vs time ------------------------------------------------------
    fig, axs = plt.subplots()
    l1 = axs.plot([s/1e9 for s in ns], torque_ref, color='#1e1e1e', marker='.', markersize=0.2, linestyle="-", label='Reference')
    # l2 = axs.plot([s/1e9 for s in ns], i_fb, color='#2ca02c', marker='.', markersize=0.2, linestyle=":", label='current ref fb (A)')
    l3 = axs.plot([s/1e9 for s in ns], motor_tor, color='#1f77b4', marker='.', markersize=0.2, label='Sensor readings')
    l0 = axs.plot([s/1e9 for s in ns], [i*torque_const*gear_ratio for i in  i_fb], color='#8e8e8e', marker='.', markersize=0.2, linestyle="", zorder=0, label='i_q*torque_const*gear_ratio')
    lgnd = axs.legend(loc='lower left')
    for handle in lgnd.legendHandles:
        handle._legmarker.set_markersize(6)

    # make plot pretty

    axs.set_xlabel('Time (s)')
    axs.set_ylabel('Torque (Nm)')
    axs.set_xlim(ns[0]/1e9, ns[-1]/1e9)
    plt_max = (max(max(torque_ref),max(motor_tor)) -min(min(torque_ref),min(motor_tor))) * 0.05
    axs.set_ylim(min(min(torque_ref),min(motor_tor))-plt_max, max(max(torque_ref),max(motor_tor))+plt_max)
    axs.grid(b=True, which='major', axis='x', linestyle=':')
    axs.grid(b=True, which='major', axis='y', linestyle='-')
    axs.grid(b=True, which='minor', axis='y', linestyle=':')
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)

    # Save the graph
    fig_name = image_base_path + '_1.png'
    plt.savefig(fname=fig_name, format='png', bbox_inches='tight')
    print('[i] Saved graphSaved graph as: ' + fig_name)
    if plot_all:
        plt.show()

    # motor_tor vs torque_ref ------------------------------------------------------
    fig, axs = plt.subplots()
    #l0 = axs.plot(i_fb,  motor_tor, color='#8e8e8e', marker='.', markersize=0.2, linestyle="", label='current out fb (A)')
    l1 = axs.plot([s/1e9 for s in ns],[u-y for u,y in zip(torque_ref, motor_tor)], color='#e10e0e', marker='.', markersize=1.5, linestyle="", label='torque reference (Nm)')
    # l2 = axs.plot(i_fb,  motor_tor, color='#2ca02c', marker='.', markersize=0.2, linestyle=":", label='current ref fb (A)')
    # lgnd = axs.legend()
    # for handle in lgnd.legendHandles:
    #     handle._legmarker.set_markersize(6)

    # make plot pretty
    # axs.set_xlabel('Current (A)')
    axs.set_xlabel('Time (s)')
    axs.set_ylabel('Torque tracking error (Nm)')
    axs.set_xlim(ns[0]/1e9, ns[-1]/1e9)
    axs.grid(b=True, which='major', axis='x', linestyle=':')
    axs.grid(b=True, which='major', axis='y', linestyle='-')
    axs.grid(b=True, which='minor', axis='y', linestyle=':')
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)

    # Save the graph
    fig_name = image_base_path + '_2.png'
    plt.savefig(fname=fig_name, format='png', bbox_inches='tight')
    print('[i] Saved graphSaved graph as: ' + fig_name)
    if plot_all:
        plt.show()

    return yaml_file

if __name__ == "__main__":
    try:
        from utils import plot_utils
    except ImportError:
        import plot_utils
    plot_utils.print_alberobotics()
    print(plot_utils.bcolors.OKBLUE + "[i] Starting process_torque_free_smooth" + plot_utils.bcolors.ENDC)
    yaml_file = process(yaml_file=sys.argv[1], plot_all=False)

    print(plot_utils.bcolors.OKGREEN + u'[\u2713] Ending program successfully' + plot_utils.bcolors.ENDC)
