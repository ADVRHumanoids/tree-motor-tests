#!/usr/bin python3
# -*- coding: utf-8 -*-

import os
import sys
import yaml
import numpy as np

#import costum
try:
    from utils import plot_utils
    from matplotlib import pyplot as plt
except ImportError:
    import plot_utils
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

    if 'test_type' in out_dict['test_current_locked_smooth']:
        test_type = out_dict['test_current_locked_smooth']['test_type']
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
    image_base_path = new_head +f'{code_string}_test-current-locked-{test_type}'

    log_file = head + f'{code_string}_test-current-locked-{test_type}.log'
    print('[i] Reading log_file: ' + log_file)

    # log format: '%u64\t%u\t%u\t%u\t%u\t%f\t%f\t%d\t%f\t%f\t%f'
    ns        = [np.uint64(x.split('\t')[0]) for x in open(log_file).readlines()]
    motor_tor = [    float(x.split('\t')[1]) for x in open(log_file).readlines()]
    i_q       = [    float(x.split('\t')[2]) for x in open(log_file).readlines()]
    i_fb      = [    float(x.split('\t')[3]) for x in open(log_file).readlines()]

    if 'test_current_locked_smooth' in out_dict:
        number_of_iters = out_dict['test_current_locked_smooth']['number_of_iters']
        rise_time = out_dict['test_current_locked_smooth']['rise_time']/1000
        wait_time = out_dict['test_current_locked_smooth']['wait_time']/1000
        t_steps = [ns[0]/1e9]
        t_steps.append(t_steps[-1]+rise_time)
        if len(out_dict['test_current_locked_smooth']['target_cur']):
            for n in range(1,2*len(out_dict['test_current_locked_smooth']['target_cur'])):
                t_steps.append(t_steps[-1]+rise_time*2+wait_time)
        if number_of_iters > 1:
            tmp_steps=t_steps
            for n in range(1,number_of_iters):
                for tp in tmp_steps:
                    t_steps.append(tp)
    else:
        raise Exception("missing 'test_current_locked_smooth' in yaml parsing")
    print(t_steps)

    print(f'[i] Processing data (loaded {len(ns)} points)')

    # motor_vel vs time ------------------------------------------------------
    fig, axs = plt.subplots(2,1)
    l0 = axs[0].plot([s/1e9 for s in ns], i_fb, color='#8e8e8e', marker='.', markersize=0.2, linestyle="", label='current out fb (A)')
    l1 = axs[0].plot([s/1e9 for s in ns], i_q, color='#1e1e1e', marker='.', markersize=0.2, linestyle="-", label='current reference (A)')
    # l2 = axs.plot([s/1e9 for s in ns], i_fb, color='#2ca02c', marker='.', markersize=0.2, linestyle=":", label='current ref fb (A)')
    l3 = axs[1].plot([s/1e9 for s in ns], motor_tor, color='#1f77b4', marker='.', markersize=0.2, label='motor torque (Nm)')
    lgnd = axs[0].legend(loc='upper left')
    for handle in lgnd.legendHandles:
        handle._legmarker.set_markersize(6)

    for l in t_steps:
        axs[0].axvline(l, linestyle='--', color='r', alpha=0.5, zorder=1)
        axs[1].axvline(l, linestyle='--', color='r', alpha=0.5, zorder=1)

    # make plot pretty
    axs[0].set_ylabel('Current (A)')
    axs[0].set_xlim(ns[0]/1e9, ns[-1]/1e9)
    plt_max = (max(i_q) -min(i_q)) * 0.05
    axs[0].set_ylim(min(i_q)-plt_max,max(i_q)+plt_max)
    axs[0].grid(b=True, which='major', axis='x', linestyle=':')
    axs[0].grid(b=True, which='major', axis='y', linestyle='-')
    axs[0].grid(b=True, which='minor', axis='y', linestyle=':')
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)

    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Motor Toruqe (Nm)')
    axs[1].set_xlim(ns[0]/1e9, ns[-1]/1e9)
    plt_max = (max(motor_tor) -min(motor_tor)) * 0.05
    axs[1].set_ylim(min(motor_tor)-plt_max,max(motor_tor)+plt_max)
    axs[1].grid(b=True, which='major', axis='x', linestyle=':')
    axs[1].grid(b=True, which='major', axis='y', linestyle='-')
    axs[1].grid(b=True, which='minor', axis='y', linestyle=':')
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)

    # Save the graph
    fig_name = image_base_path + '_1.png'
    plt.savefig(fname=fig_name, format='png', bbox_inches='tight')
    print('[i] Saved graphSaved graph as: ' + fig_name)
    if plot_all:
        plt.show()

    # motor_vel vs i_q ------------------------------------------------------
    fig, axs = plt.subplots()
    l0 = axs.plot(i_fb,  motor_tor, color='#8e8e8e', marker='.', markersize=0.2, linestyle="", label='current out fb (A)')
    l1 = axs.plot(i_q, motor_tor, color='#1e1e1e', marker='.', markersize=1.5, linestyle="", label='current reference (A)')
    # l2 = axs.plot(i_fb,  motor_vel, color='#2ca02c', marker='.', markersize=0.2, linestyle=":", label='current ref fb (A)')
    lgnd = axs.legend()
    for handle in lgnd.legendHandles:
        handle._legmarker.set_markersize(6)

    # make plot pretty
    axs.set_xlabel('Current (A)')
    axs.set_ylabel('motor torque (Nm)')
    plt_max = (max(i_q) -min(i_q)) * 0.05
    axs.set_xlim(min(i_q)-plt_max, max(i_q)+plt_max)
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
    plot_utils.print_alberobotics()
    print(plot_utils.bcolors.OKBLUE + "[i] Starting process_current_free_smooth" + plot_utils.bcolors.ENDC)
    yaml_file = process(yaml_file=sys.argv[1], plot_all=False)

    print(plot_utils.bcolors.OKGREEN + u'[\u2713] Ending program successfully' + plot_utils.bcolors.ENDC)
