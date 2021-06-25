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
    image_base_path = new_head +f'{code_string}_test-torque-lever-drops'

    log_file = head + f'{code_string}_test-torque-lever-drops.log'
    print('[i] Reading log_file: ' + log_file)

    # log format: '%u64\t%u\t%u\t%u\t%u\t%f\t%f\t%d\t%f\t%f\t%f'
    ns        = [np.uint64(x.split('\t')[0])      for x in open(log_file).readlines()]
    ref_pos   = [    float(x.split('\t')[1])      for x in open(log_file).readlines()]
    motor_pos = [    float(x.split('\t')[2])      for x in open(log_file).readlines()]
    motor_vel = [    float(x.split('\t')[3])/1000 for x in open(log_file).readlines()]
    torque    = [    float(x.split('\t')[4])      for x in open(log_file).readlines()]
    i_q       = [    float(x.split('\t')[5])      for x in open(log_file).readlines()]

    if 'test_torque_lever_drops' in out_dict:
        t_drop =  int(out_dict['test_torque_lever_drops']['pos_motion_time'])*1000
    else:
        raise Exception("missing 'test_torque_lever_drops' in yaml parsing")

    torque_const = float(out_dict['results']['flash_params']['motorTorqueConstant'])
    gear_ratio = float(out_dict['results']['flash_params']['Motor_gear_ratio'])

    print(f'[i] Processing data (loaded {len(ns)} points)')
    print(f"    torque_const: {torque_const}\n    gear_ratio:   {gear_ratio}")

    # use only data about drop
    ts_drop = ns[t_drop]/1e9
    # filter current and torque
    from scipy import signal
    b,a = signal.butter(2, 0.001, btype='lowpass')
    # padlen is needed: see https://dsp.stackexchange.com/questions/11466/#47945
    i_q_filt = signal.filtfilt(b,a, i_q[t_drop:], padlen=3*(max(len(b),len(a))-1))
    torque_filt = signal.filtfilt(b,a, torque[t_drop:], padlen=3*(max(len(b),len(a))-1))

    # motor_vel vs time ------------------------------------------------------
    fig, axs = plt.subplots(2)
    l0 = axs[0].plot([s/1e9 for s in ns], motor_pos, color='#1f77b4', marker='.', markersize=0.2, label='motor position (rad)')
    l1 = axs[0].plot([s/1e9 for s in ns], ref_pos, color='#1e1e1e', marker='.', markersize=0.2, linestyle="-", label='position reference (rad)')
    l2 = axs[1].plot([s/1e9 for s in ns], motor_vel, color='#1f77b4', marker='.', markersize=0.2, linestyle=":", label='motor velocity (rad/s)')
    axs[0].legend()
    axs[0].axvline(ts_drop, linestyle='--', color='r', alpha=0.5, zorder=1)
    axs[1].axvline(ts_drop, linestyle='--', color='r', alpha=0.5, zorder=1)

    # make plot pretty
    axs[0].set_ylabel('motor position (rad)')
    axs[0].set_xlim(ns[0]/1e9, ns[-1]/1e9)
    plt_max = (max(max(ref_pos),max(motor_pos)) -min(min(ref_pos),min(motor_pos))) * 0.05
    axs[0].set_ylim(min(min(ref_pos),min(motor_pos))-plt_max,max(max(ref_pos),max(motor_pos))+plt_max)
    axs[0].grid(b=True, which='major', axis='x', linestyle=':')
    axs[0].grid(b=True, which='major', axis='y', linestyle='-')
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    axs[0].yaxis.set_major_locator(plt.MultipleLocator(np.pi / 4))
    axs[0].yaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
    axs[0].yaxis.set_major_formatter(
            plt.FuncFormatter(
                plot_utils.multiple_formatter(denominator=4,
                                            number=np.pi,
                                            latex='\pi')))

    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('motor vel (rad/s)')
    axs[1].set_xlim(ns[0]/1e9, ns[-1]/1e9)
    # plt_max = max(motor_vel) -min(motor_vel) * 0.05
    # axs[1].set_ylim(min(motor_vel)-plt_max, max(motor_vel)+plt_max)
    axs[1].grid(b=True, which='major', axis='x', linestyle=':')
    axs[1].grid(b=True, which='major', axis='y', linestyle='-')
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)

    # Save the graph
    fig_name = image_base_path + '_1.png'
    plt.savefig(fname=fig_name, format='png', bbox_inches='tight')
    print('[i] Saved graphSaved graph as: ' + fig_name)
    if plot_all:
        plt.show()

    # motor_vel vs time ------------------------------------------------------
    fig, axs = plt.subplots(2)
    l0 = axs[0].plot([s/1e9 for s in ns[t_drop:]], i_q[t_drop:], color='#8e8e8e', marker='.', markersize=0.2, linestyle="", label='Raw data')
    l1 = axs[0].plot([s/1e9 for s in ns[t_drop:]], i_q_filt, color='#1e1e1e', marker='.', markersize=0.2, linestyle="", label='Filtered')
    lgnd = axs[0].legend(loc='upper right')
    for handle in lgnd.legendHandles:
        handle._legmarker.set_markersize(6)

    l2 = axs[1].plot([s/1e9 for s in ns[t_drop:]], torque[t_drop:], color='#8e8e8e', marker='.', markersize=0.2, linestyle="", label='Raw data')
    l3 = axs[1].plot([s/1e9 for s in ns[t_drop:]], torque_filt, color='#1e1e1e', marker='.', markersize=0.2, linestyle="", label='Filtered')
    lgnd = axs[1].legend(loc='upper right')
    for handle in lgnd.legendHandles:
        handle._legmarker.set_markersize(6)

    # make plot pretty
    axs[0].set_ylabel('Current (A)')
    axs[0].set_xlim(ns[t_drop]/1e9, ns[-1]/1e9)
    plt_max = (max(i_q) -min(i_q)) * 0.05
    axs[0].set_ylim(min(i_q)-plt_max,max(i_q)+plt_max)
    axs[0].grid(b=True, which='major', axis='x', linestyle=':')
    axs[0].grid(b=True, which='major', axis='y', linestyle='-')
    axs[0].grid(b=True, which='minor', axis='y', linestyle=':')
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)

    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Torque (Nm)')
    axs[1].set_xlim(ns[t_drop]/1e9, ns[-1]/1e9)
    plt_max = (max(torque) -min(torque)) * 0.05
    axs[1].set_ylim(min(torque)-plt_max,max(torque)+plt_max)
    axs[1].grid(b=True, which='major', axis='x', linestyle=':')
    axs[1].grid(b=True, which='major', axis='y', linestyle='-')
    axs[1].grid(b=True, which='minor', axis='y', linestyle=':')
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)


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

    print(f"[i] Fitting sensor reading vs i_q*torque_const*gear_ratio")
    fig, axs = plt.subplots()

    def linear_func(p, x):
            m, c = p
            return m*x + c

    odr_linear = odr.Model(linear_func)
    tor_x =  i_q[t_drop:]
    tor_y = torque[t_drop:]
    # tor_x = i_q_filt
    # tor_y = torque_filt
    sx=statistics.stdev(tor_x)
    sy=statistics.stdev(tor_y)
    odr_data = odr.RealData(np.array(tor_x), np.array(tor_y), sx=sx, sy=sy)
    odr_obj = odr.ODR(odr_data, odr_linear, beta0=[1.0, 0.0])
    odr_out = odr_obj.run()
    tor_odr = [linear_func(odr_out.beta,x) for x in tor_x]
    nmsre = np.sqrt(np.mean(np.square([tor_y[v] - tor_odr[v] for v in range(0,len(tor_x))])))/(max(tor_y)-min(tor_y))

    print(f"    fit slope: {odr_out.beta[0]:6.3f} with NMSRE:{nmsre:6.3f}")
    axs.plot(   tor_x,
                tor_y,
                marker='.',
                markersize=0.5,
                linestyle="",
                color='#8e8e8e',
                label=f'datapoint')
    axs.plot(   tor_x,
                tor_odr,
                marker='',
                linestyle="--",
                color='#ff7f0e',
                label=f'model',
                zorder=1)
    # l0 = axs.plot(i_fb, motor_vel, color='#1f77b4', marker='.', markersize=0.2, linestyle="",  label='current out fb (A)')
    # l1 = axs.plot(i_q,  motor_vel, color='#1e1e1e', marker='.', markersize=0.2, linestyle="-", label='current reference (A)')
    # l2 = axs.plot(i_fb, motor_vel, color='#2ca02c', marker='.', markersize=0.2, linestyle=":", label='current ref fb (A)')
    lgnd = axs.legend(loc='upper left')
    for handle in lgnd.legendHandles:
        handle._legmarker.set_markersize(6)

    # make plot pretty
    axs.set_xlabel('Current (A)')
    axs.set_ylabel('Torque sensor reading (Nm)')
    # plt_max = (max(i_q) -min(i_q)) * torque_const * 0.05
    # axs.set_xlim(min(i_q)*torque_const-plt_max, max(i_q)*torque_const+plt_max)
    plt_max = (max(tor_x) -min(tor_x)) * 0.05
    axs.set_xlim(min(tor_x)-plt_max, max(tor_x)+plt_max)
    plt_max = (max(tor_y) -min(tor_y)) * 0.05
    axs.set_ylim(min(tor_y)-plt_max, max(tor_y)+plt_max)
    axs.grid(b=True, which='major', axis='x', linestyle=':')
    axs.grid(b=True, which='major', axis='y', linestyle='-')
    axs.grid(b=True, which='minor', axis='y', linestyle=':')

    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)

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
