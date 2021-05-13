#!/usr/bin/python3

import os
import sys
import numpy as np
#from matplotlib import pyplot as plt



def process(log_file, plot_all=False):
    print('[i] Using log_file: ' + log_file)

    ## Load matplotlib
    if not plot_all:
        # tell matplotlib not to try to load up GTK as it returns errors over ssh
        from matplotlib import use as plt_use
        plt_use("Agg")
    from matplotlib import pyplot as plt
    plt.rcParams['savefig.dpi'] = 300

    ## define image path
    _, tail =os.path.split(log_file)
    new_head = f'{log_file[:-4]}/'
    if os.path.isdir(new_head) is False:
        try:
            os.makedirs(new_head)
        except OSError:
            print("Creation of the directory %s failed" % new_head)
    image_base_path= new_head +f'{tail[:-4]}'

    ## Load log
    # log format: '%u64\t%f\t%f\t%f\t%d\t%d\t%f\t%u\t%u\t%u\t%u\t%f'
    ns         = [ np.uint64(x.split('\t')[ 0]) for x in open(log_file).readlines()]
    pos_ref    = [     float(x.split('\t')[ 1]) for x in open(log_file).readlines()]
    link_pos   = [     float(x.split('\t')[ 2]) for x in open(log_file).readlines()]
    motor_pos  = [     float(x.split('\t')[ 3]) for x in open(log_file).readlines()]
    # link_vel   = [  np.int16(x.split('\t')[ 4]) for x in open(log_file).readlines()]
    # motor_vel  = [  np.int16(x.split('\t')[ 5]) for x in open(log_file).readlines()]
    link_vel   = [     float(x.split('\t')[ 4]) for x in open(log_file).readlines()]
    motor_vel  = [     float(x.split('\t')[ 5]) for x in open(log_file).readlines()]
    torque     = [     float(x.split('\t')[ 6]) for x in open(log_file).readlines()]
    temp1      = [ np.uint16(x.split('\t')[ 7]) for x in open(log_file).readlines()]
    temp2      = [ np.uint16(x.split('\t')[ 8]) for x in open(log_file).readlines()]
    fault      = [ np.uint16(x.split('\t')[ 9]) for x in open(log_file).readlines()]
    tx_rtt     = [ np.uint16(x.split('\t')[10]) for x in open(log_file).readlines()]
    op_idx_ack = [ np.uint16(x.split('\t')[11]) for x in open(log_file).readlines()]
    tx_aux     = [     float(x.split('\t')[12]) for x in open(log_file).readlines()]


    # position vs time ------------------------------------------------------
    fig2, axs = plt.subplots()
    l0 = axs.plot(ns, pos_ref, color='#2ca02c', marker='.', markersize=0.2, linestyle="-", label='Reference')
    l1 = axs.plot(ns, link_pos, color='#1f77b4', marker='.', markersize=0.2, linestyle="--", label='link_pos')
    l2 = axs.plot(ns, motor_pos, color='#ff7f0e', marker='.', markersize=0.2, linestyle=":", label='motor_pos')
    axs.legend()

    # make plot pretty
    axs.set_xlabel('Time (ns)')
    axs.set_xlim(ns[0], ns[-1])
    axs.set_ylabel('Psoition (rad)')
    axs.grid(b=True, which='major', axis='x', linestyle=':')
    axs.grid(b=True, which='major', axis='y', linestyle='-')
    axs.grid(b=True, which='minor', axis='y', linestyle=':')
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)

    # Save the graph
    fig_name = image_base_path + '_pos.png'
    plt.savefig(fname=fig_name, format='png', bbox_inches='tight')
    print('[i] Saved graph: ' + fig_name)
    if plot_all:
        plt.show()

    # motor_vel vs time ------------------------------------------------------
    fig, axs = plt.subplots()
    l1 = axs.plot(ns,  link_vel, color='#1f77b4', marker='.', markersize=0.2, label='link_vel')
    l2 = axs.plot(ns,  motor_vel, color='#ff7f0e', marker='.', markersize=0.2, linestyle="", label='motor_vel')
    axs.legend()

    # make plot pretty
    axs.set_xlabel('Time (ns)')
    axs.set_xlim(ns[0], ns[-1])
    axs.set_ylabel('Velocity (mrad/s)')
    axs.grid(b=True, which='major', axis='x', linestyle=':')
    axs.grid(b=True, which='major', axis='y', linestyle='-')
    axs.grid(b=True, which='minor', axis='y', linestyle=':')
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)

    # Save the graph
    fig_name = image_base_path + '_vel.png'
    plt.savefig(fname=fig_name, format='png', bbox_inches='tight')
    print('[i] Saved graph: ' + fig_name)
    if plot_all:
        plt.show()

    # torque vs time ------------------------------------------------------
    fig, axs = plt.subplots()
    l1 = axs.plot(ns,  torque, color='#1f77b4', marker='.', markersize=0.2, linestyle="", label='torque')

    # make plot pretty
    axs.set_xlabel('Time (ns)')
    axs.set_xlim(ns[0], ns[-1])
    axs.set_ylabel('Torque (Nm)')
    axs.grid(b=True, which='major', axis='x', linestyle=':')
    axs.grid(b=True, which='major', axis='y', linestyle='-')
    axs.grid(b=True, which='minor', axis='y', linestyle=':')
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)

    # Save the graph
    fig_name = image_base_path + '_torque.png'
    plt.savefig(fname=fig_name, format='png', bbox_inches='tight')
    print('[i] Saved graph: ' + fig_name)
    if plot_all:
        plt.show()

    # temp vs time ------------------------------------------------------
    fig, axs = plt.subplots()
    l1 = axs.plot(ns,  temp1, color='#1f77b4', marker='.', markersize=0.2, linestyle="", label='temp1')
    l2 = axs.plot(ns,  temp2, color='#ff7f0e', marker='.', markersize=0.2, linestyle="", label='temp2')
    axs.legend()

    # make plot pretty
    axs.set_xlabel('Time (ns)')
    axs.set_xlim(ns[0], ns[-1])
    axs.set_ylabel('Temp (°C)')
    axs.grid(b=True, which='major', axis='x', linestyle=':')
    axs.grid(b=True, which='major', axis='y', linestyle='-')
    axs.grid(b=True, which='minor', axis='y', linestyle=':')
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)

    # Save the graph
    fig_name = image_base_path + '_temp.png'
    plt.savefig(fname=fig_name, format='png', bbox_inches='tight')
    print('[i] Saved graph: ' + fig_name)
    if plot_all:
        plt.show()

    # fault vs time ------------------------------------------------------
    fig, axs = plt.subplots()
    l1 = axs.plot(ns,  fault, color='#1f77b4', marker='.', markersize=0.2, linestyle="", label='fault')

    # make plot pretty
    axs.set_xlabel('Time (ns)')
    axs.set_xlim(ns[0], ns[-1])
    axs.set_ylabel('Temp (°C)')
    axs.grid(b=True, which='major', axis='x', linestyle=':')
    axs.grid(b=True, which='major', axis='y', linestyle='-')
    axs.grid(b=True, which='minor', axis='y', linestyle=':')
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)

    # Save the graph
    fig_name = image_base_path + '_fault.png'
    plt.savefig(fname=fig_name, format='png', bbox_inches='tight')
    print('[i] Saved graph: ' + fig_name)
    if plot_all:
        plt.show()

    # tx_rtt vs time ------------------------------------------------------
    fig, axs = plt.subplots()
    l1 = axs.plot(ns,  tx_rtt, color='#1f77b4', marker='.', markersize=0.2, linestyle="", label='tx_rtt')

    # make plot pretty
    axs.set_xlabel('Time (ns)')
    axs.set_xlim(ns[0], ns[-1])
    axs.set_ylabel('tx_rtt (ns)')
    axs.grid(b=True, which='major', axis='x', linestyle=':')
    axs.grid(b=True, which='major', axis='y', linestyle='-')
    axs.grid(b=True, which='minor', axis='y', linestyle=':')
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)

    # Save the graph
    fig_name = image_base_path + '_rtt.png'
    plt.savefig(fname=fig_name, format='png', bbox_inches='tight')
    print('[i] Saved graph: ' + fig_name)
    if plot_all:
        plt.show()


    # tx_rtt vs time ------------------------------------------------------
    fig, axs = plt.subplots()
    l1 = axs.plot(ns[100:200],  tx_rtt[100:200], color='#1f77b4', marker='.', markersize=0.5, linestyle="-", label='tx_rtt')

    # make plot pretty
    axs.set_xlabel('Time (ns)')
    axs.set_xlim(ns[100], ns[200])
    axs.set_ylabel('tx_rtt (ns)')
    axs.grid(b=True, which='major', axis='x', linestyle=':')
    axs.grid(b=True, which='major', axis='y', linestyle='-')
    axs.grid(b=True, which='minor', axis='y', linestyle=':')
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)

    # Save the graph
    fig_name = image_base_path + '_rtt-zoomed.png'
    plt.savefig(fname=fig_name, format='png', bbox_inches='tight')
    print('[i] Saved graph: ' + fig_name)
    if plot_all:
        plt.show()

    # op_idx_ack vs time ------------------------------------------------------
    fig, axs = plt.subplots()
    l1 = axs.plot(ns,  op_idx_ack, color='#1f77b4', marker='.', markersize=0.2, linestyle="", label='op_idx_ack')

    # make plot pretty
    axs.set_xlabel('Time (ns)')
    axs.set_xlim(ns[0], ns[-1])
    axs.set_ylabel('op_idx_ack')
    axs.grid(b=True, which='major', axis='x', linestyle=':')
    axs.grid(b=True, which='major', axis='y', linestyle='-')
    axs.grid(b=True, which='minor', axis='y', linestyle=':')
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)

    # Save the graph
    fig_name = image_base_path + '_op_idx_ack.png'
    plt.savefig(fname=fig_name, format='png', bbox_inches='tight')
    print('[i] Saved graph: ' + fig_name)
    if plot_all:
        plt.show()

    # tx_aux vs time ------------------------------------------------------
    fig, axs = plt.subplots()
    l1 = axs.plot(ns,  tx_aux, color='#1f77b4', marker='.', markersize=0.2, linestyle="", label='tx_aux')

    # make plot pretty
    axs.set_xlabel('Time (ns)')
    axs.set_xlim(ns[0], ns[-1])
    axs.set_ylabel('tx_aux')
    axs.grid(b=True, which='major', axis='x', linestyle=':')
    axs.grid(b=True, which='major', axis='y', linestyle='-')
    axs.grid(b=True, which='minor', axis='y', linestyle=':')
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)

    # Save the graph
    fig_name = image_base_path + '_aux.png'
    plt.savefig(fname=fig_name, format='png', bbox_inches='tight')
    print('[i] Saved graph: ' + fig_name)
    if plot_all:
        plt.show()


if __name__ == "__main__":
    process(log_file=sys.argv[1], plot_all=False)
