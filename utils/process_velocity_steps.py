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

    if 'test_type' in out_dict['test_velocity_steps']:
        test_type = out_dict['test_velocity_steps']['test_type']
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
    image_base_path = new_head +f'{code_string}_test-velocity-free-steps'

    log_file = head + f'{code_string}_test-velocity-free-steps.log'
    print('[i] Reading log_file: ' + log_file)

    # log format: '%u64\t%u\t%u\t%u\t%u\t%f\t%f\t%d\t%f\t%f\t%f'
    ns        = [np.uint64(x.split('\t')[0]) for x in open(log_file).readlines()]
    motor_pos = [    float(x.split('\t')[1]) for x in open(log_file).readlines()]
    motor_vel = [ np.int16(x.split('\t')[2]) for x in open(log_file).readlines()]
    ref_vel   = [    float(x.split('\t')[3]) for x in open(log_file).readlines()]
    i_fb      = [    float(x.split('\t')[4]) for x in open(log_file).readlines()]

    if 'test_velocity_steps' in out_dict:
        v_steps = out_dict['test_velocity_steps']['target_vel']
        t_steps = [ns[0]/1e9+t for t in out_dict['test_velocity_steps']['time']]
    else:
        raise Exception("missing 'test_velocity_steps' in yaml parsing")

    print(f'[i] Processing data (loaded {len(ns)} points)')

    # motor_vel vs time ------------------------------------------------------
    fig, axs = plt.subplots()
    v=[float(v)/1e3 for v in motor_vel]
    l0 = axs.plot([s/1e9 for s in ns], i_fb, color='#8e8e8e', marker='.', markersize=0.2, linestyle="", label='current out fb (A)')
    # l2 = axs.plot([s/1e9 for s in ns], i_fb, color='#2ca02c', marker='.', markersize=0.2, linestyle=":", label='current ref fb (A)')
    l3 = axs.plot([s/1e9 for s in ns], v, color='#1f77b4', marker='.', markersize=0.2, label='motor velocity (rad/s)')
    l1 = axs.plot([s/1e9 for s in ns], ref_vel, color='#1e1e1e', marker='.', markersize=0.2, linestyle="-", label='velocity reference (rad/s)')
    axs.legend()
    for l in t_steps:
        axs.axvline(l, linestyle='--', color='r', alpha=0.5, zorder=1)

    # make plot pretty
    axs.set_xlabel('Time (s)')
    axs.set_xlim(ns[0]/1e9, ns[-1]/1e9)
    plt_max = (max(max(ref_vel),max(v)) -min(min(ref_vel),min(v))) * 0.05
    axs.set_ylim(min(min(ref_vel),min(v))-plt_max,max(max(ref_vel),max(v))+plt_max)
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

    # motor_vel vs i_q ------------------------------------------------------
    fig, axs = plt.subplots()
    step_ids=[id for id in range(1,len(ref_vel)) if ref_vel[id]!=ref_vel[id-1]]
    for i in range(1,len(step_ids)):
        axs.plot(i_fb[step_ids[i-1]:step_ids[i]], [mv/1000 for mv in motor_vel[step_ids[i-1]:step_ids[i]]], marker='.', markersize=0.5, linestyle="", label=f'ref: {ref_vel[step_ids[i-1]]:5.2f} rad/s')
    # l0 = axs.plot(i_fb, motor_vel, color='#1f77b4', marker='.', markersize=0.2, linestyle="",  label='current out fb (A)')
    # l1 = axs.plot(i_q,  motor_vel, color='#1e1e1e', marker='.', markersize=0.2, linestyle="-", label='current reference (A)')
    # l2 = axs.plot(i_fb, motor_vel, color='#2ca02c', marker='.', markersize=0.2, linestyle=":", label='current ref fb (A)')
    lgnd = axs.legend()
    for handle in lgnd.legendHandles:
        handle._legmarker.set_markersize(6)

    # make plot pretty
    axs.set_xlabel('current out fb (A)')
    axs.set_ylabel('motor velocity (rad/s)')
    plt_max = (max(i_fb) -min(i_fb)) * 0.05
    axs.set_xlim(min(i_fb)-plt_max, max(i_fb)+plt_max)
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
    import plot_utils
    plot_utils.print_alberobotics()
    print(plot_utils.bcolors.OKBLUE + "[i] Starting process_current_free_smooth" + plot_utils.bcolors.ENDC)
    yaml_file = process(yaml_file=sys.argv[1], plot_all=False)

    print(plot_utils.bcolors.OKGREEN + u'[\u2713] Ending program successfully' + plot_utils.bcolors.ENDC)
