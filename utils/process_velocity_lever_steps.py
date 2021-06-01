#!/usr/bin python3
# -*- coding: utf-8 -*-

import os
import sys
from numpy.lib.function_base import append
import yaml
import statistics
import numpy as np
from scipy import odr

# tell matplotlib not to try to load up GTK as it returns errors over ssh
from matplotlib import use as plt_use
plt_use("Agg")
from matplotlib import pyplot as plt

#import costum
try:
    from utils import plot_utils
except ImportError:
    import plot_utils

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
    image_base_path = new_head +f'{code_string}_test-velocity-lever-steps'

    log_file = head + f'{code_string}_test-velocity-lever-steps.log'
    print('[i] Reading log_file: ' + log_file)

    # log format: '%u64\t%u\t%u\t%u\t%u\t%f\t%f\t%d\t%f\t%f\t%f'
    ns        = [np.uint64(x.split('\t')[0])      for x in open(log_file).readlines()]
    torque    = [    float(x.split('\t')[1])      for x in open(log_file).readlines()]
    motor_pos = [    float(x.split('\t')[2])      for x in open(log_file).readlines()]
    real_pos  = [    float(x.split('\t')[3])      for x in open(log_file).readlines()]
    motor_vel = [    float(x.split('\t')[4])/1000 for x in open(log_file).readlines()]
    ref_vel   = [    float(x.split('\t')[5])      for x in open(log_file).readlines()]
    i_q       = [    float(x.split('\t')[6])      for x in open(log_file).readlines()]

    if 'test_velocity_lever_steps' in out_dict:
        v_steps = out_dict['test_velocity_lever_steps']['target_vel']
    else:
        raise Exception("missing 'test_velocity_lever_steps' in yaml parsing")

    torque_const = float(out_dict['results']['flash_params']['motorTorqueConstant'])
    gear_ratio = float(out_dict['results']['flash_params']['Motor_gear_ratio'])

    print(f'[i] Processing data (loaded {len(ns)} points)')
    print(f"\ttorque_const: {torque_const}\n\tgear_ratio:   {gear_ratio}")


    # motor_vel vs time ------------------------------------------------------
    fig, axs = plt.subplots()
    step_ids=[id for id in range(1,len(ref_vel)) if ref_vel[id]!=ref_vel[id-1]]
    #l0 = axs.plot([s/1e9 for s in ns], i_q, color='#8e8e8e', marker='.', markersize=0.2, linestyle="", label='current out fb (A)')
    #l2 = axs.plot([s/1e9 for s in ns], i_q, color='#2ca02c', marker='.', markersize=0.2, linestyle=":", label='current ref fb (A)')
    l3 = axs.plot([s/1e9 for s in ns], motor_vel, color='#1f77b4', marker='.', markersize=0.2, label='motor velocity (rad/s)')
    l1 = axs.plot([s/1e9 for s in ns], ref_vel, color='#1e1e1e', marker='.', markersize=0.2, linestyle="-", label='velocity reference (rad/s)')
    axs.legend()
    for l in step_ids:
        axs.axvline(ns[l]/1e9, linestyle='--', color='r', alpha=0.5, zorder=1)
    axs.axvline(ns[-1]/1e9, linestyle='--', color='r', alpha=0.5, zorder=1)

    # make plot pretty
    axs.set_xlabel('Time (s)')
    axs.set_xlim(ns[0]/1e9, ns[-1]/1e9)
    plt_max = (max(max(ref_vel),max(motor_vel)) -min(min(ref_vel),min(motor_vel))) * 0.05
    axs.set_ylim(min(min(ref_vel),min(motor_vel))-plt_max,max(max(ref_vel),max(motor_vel))+plt_max)
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

    # motor_vel vs motor_pos ------------------------------------------------------
    fig, axs = plt.subplots()
    l0 = axs.plot(real_pos, i_q, color='#8e8e8e', marker='.', markersize=0.2, linestyle="", label='current out fb (A)')
    l1 = axs.plot(real_pos, ref_vel, color='#1e1e1e', marker='.', markersize=0.2, linestyle="", label='velocity reference (rad/s)')
    #l2 = axs.plot([s/1e9 for s in ns], i_q, color='#2ca02c', marker='.', markersize=0.2, linestyle=":", label='current ref fb (A)')
    #l3 = axs.plot(real_pos, motor_vel, color='#1f77b4', marker='.', markersize=0.2, linestyle="", label='motor velocity (rad/s)')
    for i in range(1,len(step_ids)):
        if ref_vel[step_ids[i-1]]!=ref_vel[0]:
            axs.plot(real_pos[step_ids[i-1]:step_ids[i]], motor_vel[step_ids[i-1]:step_ids[i]], marker='.', markersize=0.5, linestyle="", label=f'motor velocity for ref: {ref_vel[step_ids[i-1]]:5.2f} rad/s')
    lgnd = axs.legend()
    for handle in lgnd.legendHandles:
        handle._legmarker.set_markersize(6)

    # make plot pretty
    axs.set_xlabel('Position (rad)')
    axs.set_xlim(min(real_pos), max(real_pos))
    axs.set_ylim(-2.5, 2.5)
    axs.grid(b=True, which='major', axis='x', linestyle=':')
    axs.grid(b=True, which='major', axis='y', linestyle='-')
    axs.grid(b=True, which='minor', axis='y', linestyle=':')
    axs.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    axs.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 6))
    axs.xaxis.set_major_formatter(
        plt.FuncFormatter(
            plot_utils.multiple_formatter(denominator=4,
                                          number=np.pi,
                                          latex='\pi')))
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)

    # Save the graph
    fig_name = image_base_path + '_2.png'
    plt.savefig(fname=fig_name, format='png', bbox_inches='tight')
    print('[i] Saved graphSaved graph as: ' + fig_name)
    if plot_all:
        plt.show()

    # motor_tor vs curr ------------------------------------------------------
    i_start=[]
    i_end=[]
    v_ref=[]
    for i in range(1,len(step_ids)):
        if ref_vel[step_ids[i-1]]!=ref_vel[0]:
            i_start.append(step_ids[i-1])
            i_end.append(step_ids[i])
            v_ref.append(ref_vel[step_ids[i-1]])

    print(f"int(len(i_start)/2): {int(len(i_start)/2)}, int(len(i_start)%2: {int(len(i_start)%2)}")
    fig, axs = plt.subplots(int(len(i_start)/2), 2)

    def linear_func(p, x):
            m, c = p
            return m*x + c

    for i in range(0,len(i_start)):
        id=int(i/2)
        odr_linear = odr.Model(linear_func)
        tor_x = [iq*torque_const*gear_ratio for iq in i_q[i_start[i]:i_end[i]]]
        tor_y = torque[i_start[i]:i_end[i]]
        sx=statistics.stdev(tor_x)
        sy=statistics.stdev(tor_y)
        odr_data = odr.RealData(np.array(tor_x), np.array(tor_y), sx=sx, sy=sy)
        odr_obj = odr.ODR(odr_data, odr_linear, beta0=[1.0, 0.0])
        odr_out = odr_obj.run()
        tor_odr = [linear_func(odr_out.beta,x) for x in tor_x]
        nmsre = np.sqrt(np.mean(np.square([tor_y[v] - tor_odr[v] for v in range(0,len(tor_x))])))/(max(tor_y)-min(tor_y))

        print(f"\tref: {v_ref[i]:5.2f} rad/s-> fit slope: {odr_out.beta[0]:6.3f} with NMSRE:{nmsre:6.3f}")
        axs[id,i%2].plot( tor_x,
                          tor_y,
                          marker='.',
                          markersize=0.5,
                          linestyle="",
                          color='#8e8e8e',
                          label=f'ref: {v_ref[i]:5.2f} rad/s')
        axs[id,i%2].plot( [min(torque), max(torque)],
                          [linear_func(odr_out.beta,min(torque)), linear_func(odr_out.beta,max(torque))],
                          marker='',
                          linestyle="--",
                          color='#ff7f0e',
                          label=f'model')
        # l0 = axs.plot(i_fb, motor_vel, color='#1f77b4', marker='.', markersize=0.2, linestyle="",  label='current out fb (A)')
        # l1 = axs.plot(i_q,  motor_vel, color='#1e1e1e', marker='.', markersize=0.2, linestyle="-", label='current reference (A)')
        # l2 = axs.plot(i_fb, motor_vel, color='#2ca02c', marker='.', markersize=0.2, linestyle=":", label='current ref fb (A)')
        lgnd = axs[id,i%2].legend(loc='upper left')
        for handle in lgnd.legendHandles:
            handle._legmarker.set_markersize(6)

        # make plot pretty
        if id == int(len(i_start)/2)-1:
            axs[id,i%2].set_xlabel('current*Kt*N (Nm/rad)')
        if i%2 == 0:
            axs[id,i%2].set_ylabel('motor torque (Nm)')
        # plt_max = (max(i_q) -min(i_q)) * torque_const * 0.05
        # axs[id,i%2].set_xlim(min(i_q)*torque_const-plt_max, max(i_q)*torque_const+plt_max)
        plt_max = (max(torque) -min(torque)) * 0.05
        axs[id,i%2].set_xlim(min(torque)-plt_max, max(torque)+plt_max)
        axs[id,i%2].set_ylim(min(torque)-plt_max, max(torque)+plt_max)
        axs[id,i%2].grid(b=True, which='major', axis='x', linestyle=':')
        axs[id,i%2].grid(b=True, which='major', axis='y', linestyle='-')
        axs[id,i%2].grid(b=True, which='minor', axis='y', linestyle=':')

        axs[id,i%2].spines['top'].set_visible(False)
        axs[id,i%2].spines['right'].set_visible(False)

    # Save the graph
    fig_name = image_base_path + '_3.png'
    plt.savefig(fname=fig_name, format='png', bbox_inches='tight')
    print('[i] Saved graphSaved graph as: ' + fig_name)
    if plot_all:
        plt.show()

    return yaml_file

if __name__ == "__main__":
    import plot_utils
    plot_utils.print_alberobotics()
    print(plot_utils.bcolors.OKBLUE + "[i] Starting process_velocity_lever_steps" + plot_utils.bcolors.ENDC)
    yaml_file = process(yaml_file=sys.argv[1], plot_all=False)

    print(plot_utils.bcolors.OKGREEN + u'[\u2713] Ending program successfully' + plot_utils.bcolors.ENDC)
