#!/usr/bin python3
# -*- coding: utf-8 -*-

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
    ns         = [ np.uint64(x.split('\t')[0]) for x in open(log_file).readlines()]
    load     = [     float(x.split('\t')[1]) for x in open(log_file).readlines()]
    status     = [     float(x.split('\t')[2]) for x in open(log_file).readlines()]
    tx_rtt     = [ np.uint16(x.split('\t')[3]) for x in open(log_file).readlines()]

    # load vs time ------------------------------------------------------
    fig, axs = plt.subplots()
    l1 = axs.plot(ns,  load, color='#1f77b4', marker='.', markersize=0.2, linestyle="", label='load')

    # make plot pretty
    axs.set_xlabel('Time (ns)')
    axs.set_xlim(ns[0], ns[-1])
    axs.set_ylabel('load (Nm)')
    axs.grid(b=True, which='major', axis='x', linestyle=':')
    axs.grid(b=True, which='major', axis='y', linestyle='-')
    axs.grid(b=True, which='minor', axis='y', linestyle=':')
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)

    # Save the graph
    fig_name = image_base_path + '_load.png'
    plt.savefig(fname=fig_name, format='png', bbox_inches='tight')
    print('[i] Saved graph: ' + fig_name)
    if plot_all:
        plt.show()

    # fault vs time ------------------------------------------------------
    fig, axs = plt.subplots()
    l1 = axs.plot(ns,  status, color='#1f77b4', marker='.', markersize=0.2, linestyle="", label='fault')

    # make plot pretty
    axs.set_xlabel('Time (ns)')
    axs.set_xlim(ns[0], ns[-1])
    axs.set_ylabel('status')
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

if __name__ == "__main__":
    process(log_file=sys.argv[1], plot_all=False)
