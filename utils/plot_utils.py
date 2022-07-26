#!/usr/bin python3
# -*- coding: utf-8 -*-

import numpy as np

# tell matplotlib not to try to load up GTK as it returns errors over ssh
from matplotlib import use as plt_use
plt_use("Agg")
from matplotlib import pyplot as plt

def multiple_formatter(denominator=2, number=np.pi, latex='\pi'):
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def _multiple_formatter(x, pos):
        den = denominator
        num = np.int(np.rint(den*x/number))
        com = gcd(num, den)
        (num, den) = (int(num/com), int(den/com))
        if den == 1:
            if num == 0:
                return r'$0$'
            if num == 1:
                return r'$%s$' % latex
            elif num == -1:
                return r'$-%s$' % latex
            else:
                return r'$%s%s$' % (num, latex)
        else:
            if num == 1:
                return r'$\frac{%s}{%s}$' % (latex, den)
            elif num == -1:
                return r'$\frac{-%s}{%s}$' % (latex, den)
            else:
                return r'$\frac{%s%s}{%s}$' % (num, latex, den)
    return _multiple_formatter

class Multiple:
    def __init__(self, denominator=2, number=np.pi, latex='\pi'):
        self.denominator = denominator
        self.number = number
        self.latex = latex

def locator(self):
    return plt.MultipleLocator(self.number / self.denominator)

def formatter(self):
    return plt.FuncFormatter(multiple_formatter(self.denominator, self.number, self.latex))


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_alberobotics():
    print('\x1b[38;5;234m'+'················································································')
    print('\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'······················'+'\x1b[38;5;242m'+'·'+'\x1b[38;5;235m'+'G'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;238m'+'p'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'···················································')
    print('\x1b[38;5;234m'+'·························'+'\x1b[38;5;240m'+'D'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;244m'+'Q'+'\x1b[38;5;242m'+'·'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'·················································')
    print('\x1b[38;5;234m'+'·························'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;95m'+'('+'\x1b[38;5;52m'+'G'+'\x1b[38;5;1m'+'G'+'\x1b[38;5;88m'+'G'+'\x1b[38;5;160m'+'G'+'\x1b[38;5;160m'+'G'+'\x1b[38;5;160m'+'G'+'\x1b[38;5;196m'+'G'+'\x1b[38;5;196m'+'G'+'\x1b[38;5;196m'+'GGGGG'+'\x1b[38;5;160m'+'G'+'\x1b[38;5;160m'+'G'+'\x1b[38;5;160m'+'G'+'\x1b[38;5;160m'+'b'+'\x1b[38;5;167m'+'p'+'\x1b[38;5;131m'+'p'+'\x1b[38;5;131m'+'p'+'\x1b[38;5;95m'+'p'+'\x1b[38;5;95m'+'p'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'·····························')
    print('\x1b[38;5;234m'+'···················'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;131m'+'·'+'\x1b[38;5;131m'+'p'+'\x1b[38;5;160m'+'G'+'\x1b[38;5;196m'+'G'+'\x1b[38;5;196m'+'b'+'\x1b[38;5;196m'+'bb'+'\x1b[38;5;196m'+'G'+'\x1b[38;5;124m'+'Q'+'\x1b[38;5;88m'+'Q'+'\x1b[38;5;1m'+'p'+'\x1b[38;5;131m'+'O'+'\x1b[38;5;131m'+'·'+'\x1b[38;5;95m'+'·'+'\x1b[38;5;95m'+'·'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'······'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;95m'+'·'+'\x1b[38;5;236m'+'d'+'\x1b[38;5;52m'+'Q'+'\x1b[38;5;52m'+'Q'+'\x1b[38;5;88m'+'p'+'\x1b[38;5;160m'+'G'+'\x1b[38;5;160m'+'G'+'\x1b[38;5;160m'+'G'+'\x1b[38;5;160m'+'b'+'\x1b[38;5;167m'+'p'+'\x1b[38;5;131m'+'p'+'\x1b[38;5;95m'+'·'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'·······················')
    print('\x1b[38;5;234m'+'···············'+'\x1b[38;5;95m'+'·'+'\x1b[38;5;131m'+'p'+'\x1b[38;5;160m'+'G'+'\x1b[38;5;196m'+'G'+'\x1b[38;5;196m'+'b'+'\x1b[38;5;196m'+'b'+'\x1b[38;5;196m'+'G'+'\x1b[38;5;160m'+'C'+'\x1b[38;5;131m'+'D'+'\x1b[38;5;131m'+'·'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;240m'+'d'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;237m'+'b'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'··········'+'\x1b[38;5;233m'+'·'+'\x1b[38;5;244m'+'Q'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;243m'+'QQ'+'\x1b[38;5;239m'+'b'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'···'+'\x1b[38;5;95m'+'·'+'\x1b[38;5;131m'+'·'+'\x1b[38;5;131m'+'D'+'\x1b[38;5;160m'+'G'+'\x1b[38;5;160m'+'Gb'+'\x1b[38;5;131m'+'p'+'\x1b[38;5;95m'+'·'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'··················')
    print('\x1b[38;5;234m'+'············'+'\x1b[38;5;131m'+'·'+'\x1b[38;5;160m'+'G'+'\x1b[38;5;196m'+'G'+'\x1b[38;5;196m'+'bb'+'\x1b[38;5;160m'+'Q'+'\x1b[38;5;88m'+'S'+'\x1b[38;5;52m'+'Q'+'\x1b[38;5;233m'+'·'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'······'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;234m'+'G'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'·········'+'\x1b[38;5;233m'+'·'+'\x1b[38;5;244m'+'Q'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;243m'+'QG'+'\x1b[38;5;240m'+'O'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'·········'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;131m'+'·'+'\x1b[38;5;131m'+'D'+'\x1b[38;5;160m'+'G'+'\x1b[38;5;160m'+'G'+'\x1b[38;5;167m'+'p'+'\x1b[38;5;131m'+'p'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'·············')
    print('\x1b[38;5;234m'+'··········'+'\x1b[38;5;131m'+'('+'\x1b[38;5;160m'+'G'+'\x1b[38;5;196m'+'b'+'\x1b[38;5;196m'+'G'+'\x1b[38;5;166m'+'C'+'\x1b[38;5;131m'+'('+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;243m'+'QQQ'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;239m'+'p'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'······'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;234m'+'G'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'··'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'······'+'\x1b[38;5;236m'+'S'+'\x1b[38;5;233m'+'G'+'\x1b[38;5;243m'+'QG'+'\x1b[38;5;242m'+'·'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'···············'+'\x1b[38;5;95m'+'·'+'\x1b[38;5;131m'+')'+'\x1b[38;5;160m'+'G'+'\x1b[38;5;160m'+'G'+'\x1b[38;5;167m'+'p'+'\x1b[38;5;131m'+'p'+'\x1b[38;5;239m'+'d'+'\x1b[38;5;235m'+'Q'+'\x1b[38;5;244m'+'Q'+'\x1b[38;5;235m'+'Q'+'\x1b[38;5;233m'+'·'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'·····')
    print('\x1b[38;5;234m'+'········'+'\x1b[38;5;131m'+'('+'\x1b[38;5;160m'+'G'+'\x1b[38;5;196m'+'G'+'\x1b[38;5;167m'+'C'+'\x1b[38;5;131m'+'·'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'··'+'\x1b[38;5;237m'+'S'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;243m'+'QQQQ'+'\x1b[38;5;244m'+'Q'+'\x1b[38;5;239m'+'p'+'\x1b[38;5;233m'+'·'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;137m'+'·'+'\x1b[38;5;137m'+'p'+'\x1b[38;5;137m'+'p'+'\x1b[38;5;136m'+'D'+'\x1b[38;5;172m'+'s'+'\x1b[38;5;178m'+'s'+'\x1b[38;5;214m'+'p'+'\x1b[38;5;214m'+'ppppppppppps'+'\x1b[38;5;178m'+'b'+'\x1b[38;5;178m'+'p'+'\x1b[38;5;178m'+'p'+'\x1b[38;5;137m'+'p'+'\x1b[38;5;137m'+'p'+'\x1b[38;5;137m'+'·'+'\x1b[38;5;101m'+'·'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'·············'+'\x1b[38;5;240m'+'s'+'\x1b[38;5;235m'+'Q'+'\x1b[38;5;1m'+'Q'+'\x1b[38;5;160m'+'Q'+'\x1b[38;5;160m'+'p'+'\x1b[38;5;1m'+'S'+'\x1b[38;5;238m'+'Q'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'······')
    print('\x1b[38;5;234m'+'······'+'\x1b[38;5;131m'+'·'+'\x1b[38;5;160m'+'G'+'\x1b[38;5;160m'+'C'+'\x1b[38;5;131m'+'·'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'······'+'\x1b[38;5;59m'+'('+'\x1b[38;5;235m'+'G'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;233m'+'G'+'\x1b[38;5;58m'+'G'+'\x1b[38;5;94m'+'D'+'\x1b[38;5;172m'+'s'+'\x1b[38;5;214m'+'s'+'\x1b[38;5;214m'+'p'+'\x1b[38;5;172m'+'Q'+'\x1b[38;5;136m'+'p'+'\x1b[38;5;94m'+'p'+'\x1b[38;5;94m'+'p'+'\x1b[38;5;58m'+'Q'+'\x1b[38;5;236m'+'Q'+'\x1b[38;5;101m'+'·'+'\x1b[38;5;101m'+'·'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'··'+'\x1b[38;5;234m'+'····'+'\x1b[38;5;237m'+'d'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;244m'+'Q'+'\x1b[38;5;235m'+'Q'+'\x1b[38;5;95m'+'b'+'\x1b[38;5;137m'+'·'+'\x1b[38;5;137m'+'·'+'\x1b[38;5;179m'+'·'+'\x1b[38;5;178m'+'G'+'\x1b[38;5;178m'+'G'+'\x1b[38;5;214m'+'b'+'\x1b[38;5;214m'+'pp'+'\x1b[38;5;178m'+'p'+'\x1b[38;5;173m'+'p'+'\x1b[38;5;137m'+'p'+'\x1b[38;5;101m'+'·'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'····'+'\x1b[38;5;242m'+'('+'\x1b[38;5;237m'+'d'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;233m'+'G'+'\x1b[38;5;235m'+'G'+'\x1b[38;5;240m'+'D'+'\x1b[38;5;233m'+'·'+'\x1b[38;5;95m'+'·'+'\x1b[38;5;131m'+'·'+'\x1b[38;5;131m'+'D'+'\x1b[38;5;131m'+'b'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'···')
    print('\x1b[38;5;234m'+'····'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;166m'+'p'+'\x1b[38;5;167m'+'C'+'\x1b[38;5;233m'+'·'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'········'+'\x1b[38;5;137m'+'·'+'\x1b[38;5;137m'+'p'+'\x1b[38;5;178m'+'b'+'\x1b[38;5;214m'+'p'+'\x1b[38;5;178m'+'G'+'\x1b[38;5;136m'+'d'+'\x1b[38;5;58m'+'S'+'\x1b[38;5;244m'+'Q'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;243m'+'QQQQQQ'+'\x1b[38;5;237m'+'b'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'·······'+'\x1b[38;5;234m'+'S'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;243m'+'QQ'+'\x1b[38;5;240m'+'b'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'·····'+'\x1b[38;5;101m'+'·'+'\x1b[38;5;137m'+'·'+'\x1b[38;5;137m'+'·'+'\x1b[38;5;178m'+'G'+'\x1b[38;5;214m'+'G'+'\x1b[38;5;214m'+'pp'+'\x1b[38;5;178m'+'p'+'\x1b[38;5;179m'+'p'+'\x1b[38;5;137m'+'p'+'\x1b[38;5;236m'+'G'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;235m'+'G'+'\x1b[38;5;240m'+'D'+'\x1b[38;5;233m'+'·'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'···········')
    print('\x1b[38;5;234m'+'··'+'\x1b[38;5;240m'+'s'+'\x1b[38;5;1m'+'G'+'\x1b[38;5;1m'+'S'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;235m'+'Q'+'\x1b[38;5;239m'+'p'+'\x1b[38;5;233m'+'·'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'···'+'\x1b[38;5;101m'+'·'+'\x1b[38;5;137m'+'p'+'\x1b[38;5;178m'+'p'+'\x1b[38;5;178m'+'b'+'\x1b[38;5;179m'+'·'+'\x1b[38;5;137m'+'·'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'···'+'\x1b[38;5;242m'+'·'+'\x1b[38;5;239m'+'D'+'\x1b[38;5;236m'+'G'+'\x1b[38;5;233m'+'G'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;243m'+'QQQQ'+'\x1b[38;5;235m'+'Q'+'\x1b[38;5;242m'+'p'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'····'+'\x1b[38;5;233m'+'·'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;243m'+'QQQ'+'\x1b[38;5;242m'+'·'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'··········'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;101m'+'('+'\x1b[38;5;94m'+'p'+'\x1b[38;5;172m'+'p'+'\x1b[38;5;214m'+'p'+'\x1b[38;5;214m'+'p'+'\x1b[38;5;172m'+'s'+'\x1b[38;5;137m'+'p'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'··············')
    print('\x1b[38;5;234m'+'··'+'\x1b[38;5;95m'+')'+'\x1b[38;5;237m'+'S'+'\x1b[38;5;235m'+'G'+'\x1b[38;5;233m'+'G'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;243m'+'QQQ'+'\x1b[38;5;238m'+'b'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;137m'+'·'+'\x1b[38;5;178m'+'p'+'\x1b[38;5;214m'+'p'+'\x1b[38;5;179m'+'·'+'\x1b[38;5;101m'+'·'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'··········'+'\x1b[38;5;240m'+'D'+'\x1b[38;5;233m'+'G'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;243m'+'QQQQ'+'\x1b[38;5;239m'+'p'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'···'+'\x1b[38;5;242m'+'('+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;243m'+'QQ'+'\x1b[38;5;233m'+'G'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'··········'+'\x1b[38;5;233m'+'·'+'\x1b[38;5;236m'+'S'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;243m'+'QQ'+'\x1b[38;5;235m'+'G'+'\x1b[38;5;137m'+'·'+'\x1b[38;5;179m'+'·'+'\x1b[38;5;214m'+'G'+'\x1b[38;5;214m'+'p'+'\x1b[38;5;179m'+'p'+'\x1b[38;5;101m'+'·'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'···········')
    print('\x1b[38;5;234m'+'······'+'\x1b[38;5;233m'+'·'+'\x1b[38;5;242m'+')'+'\x1b[38;5;236m'+'S'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;236m'+'G'+'\x1b[38;5;172m'+'b'+'\x1b[38;5;178m'+'b'+'\x1b[38;5;137m'+'·'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'··············'+'\x1b[38;5;233m'+'·'+'\x1b[38;5;238m'+'S'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;243m'+'QQQQQ'+'\x1b[38;5;235m'+'Q'+'\x1b[38;5;237m'+'Q'+'\x1b[38;5;236m'+'Q'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;243m'+'QQ'+'\x1b[38;5;238m'+'C'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'·········'+'\x1b[38;5;242m'+'('+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;243m'+'QQQ'+'\x1b[38;5;234m'+'G'+'\x1b[38;5;233m'+'·'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'··'+'\x1b[38;5;101m'+'·'+'\x1b[38;5;137m'+'·'+'\x1b[38;5;178m'+'G'+'\x1b[38;5;178m'+'p'+'\x1b[38;5;137m'+'·'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'·········')
    print('\x1b[38;5;234m'+'········'+'\x1b[38;5;101m'+'·'+'\x1b[38;5;136m'+'P'+'\x1b[38;5;178m'+'Q'+'\x1b[38;5;58m'+'S'+'\x1b[38;5;236m'+'b'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'···········'+'\x1b[38;5;101m'+'·'+'\x1b[38;5;143m'+'·'+'\x1b[38;5;143m'+'p'+'\x1b[38;5;143m'+'p'+'\x1b[38;5;184m'+'p'+'\x1b[38;5;184m'+'p'+'\x1b[38;5;184m'+'b'+'\x1b[38;5;184m'+'s'+'\x1b[38;5;184m'+'pQQQQppss'+'\x1b[38;5;178m'+'s'+'\x1b[38;5;142m'+'D'+'\x1b[38;5;178m'+'p'+'\x1b[38;5;143m'+'p'+'\x1b[38;5;143m'+'p'+'\x1b[38;5;137m'+'·'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'·····'+'\x1b[38;5;236m'+'S'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;243m'+'QQQ'+'\x1b[38;5;242m'+'b'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'······'+'\x1b[38;5;101m'+'·'+'\x1b[38;5;137m'+'·'+'\x1b[38;5;178m'+'G'+'\x1b[38;5;137m'+'·'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'·······')
    print('\x1b[38;5;234m'+'·······'+'\x1b[38;5;137m'+'('+'\x1b[38;5;178m'+'s·'+'\x1b[38;5;235m'+'S'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;237m'+'b'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'······'+'\x1b[38;5;143m'+'·'+'\x1b[38;5;179m'+'p'+'\x1b[38;5;184m'+'p'+'\x1b[38;5;220m'+'p'+'\x1b[38;5;220m'+'b'+'\x1b[38;5;184m'+'C'+'\x1b[38;5;179m'+'·'+'\x1b[38;5;143m'+'·'+'\x1b[38;5;143m'+'·'+'\x1b[38;5;101m'+'·'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;233m'+'·'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;243m'+'QQQQQQ'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;234m'+'G'+'\x1b[38;5;101m'+'·'+'\x1b[38;5;143m'+'·'+'\x1b[38;5;143m'+'·'+'\x1b[38;5;178m'+'·'+'\x1b[38;5;184m'+'G'+'\x1b[38;5;220m'+'G'+'\x1b[38;5;220m'+'p'+'\x1b[38;5;184m'+'p'+'\x1b[38;5;178m'+'p'+'\x1b[38;5;143m'+'p'+'\x1b[38;5;101m'+'·'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'S'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;243m'+'QQ'+'\x1b[38;5;234m'+'G'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'··········'+'\x1b[38;5;137m'+'·'+'\x1b[38;5;137m'+'·'+'\x1b[38;5;101m'+'·'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'·····')
    print('\x1b[38;5;234m'+'·····'+'\x1b[38;5;101m'+'·'+'\x1b[38;5;179m'+'C'+'\x1b[38;5;178m'+'C'+'\x1b[38;5;101m'+'·'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;236m'+'S'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;235m'+'Q'+'\x1b[38;5;241m'+'p'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;143m'+'·'+'\x1b[38;5;179m'+'p'+'\x1b[38;5;184m'+'p'+'\x1b[38;5;220m'+'p'+'\x1b[38;5;184m'+'C'+'\x1b[38;5;143m'+'·'+'\x1b[38;5;137m'+'·'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'········'+'\x1b[38;5;240m'+'('+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;243m'+'QQQQQQQ'+'\x1b[38;5;239m'+'C'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'······'+'\x1b[38;5;137m'+'·'+'\x1b[38;5;143m'+'·'+'\x1b[38;5;178m'+'·'+'\x1b[38;5;184m'+'G'+'\x1b[38;5;220m'+'p'+'\x1b[38;5;184m'+'s'+'\x1b[38;5;100m'+'D'+'\x1b[38;5;58m'+'G'+'\x1b[38;5;233m'+'G'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'········'+'\x1b[38;5;233m'+'·'+'\x1b[38;5;233m'+'··'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'·······')
    print('\x1b[38;5;234m'+'·····'+'\x1b[38;5;137m'+'·······'+'\x1b[38;5;240m'+'('+'\x1b[38;5;234m'+'G'+'\x1b[38;5;233m'+'G'+'\x1b[38;5;100m'+'C'+'\x1b[38;5;184m'+'s'+'\x1b[38;5;184m'+'Q'+'\x1b[38;5;136m'+'p'+'\x1b[38;5;101m'+'p'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'·············'+'\x1b[38;5;244m'+'Q'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;243m'+'QQQQQQ'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'··········'+'\x1b[38;5;233m'+'·'+'\x1b[38;5;58m'+'Q'+'\x1b[38;5;100m'+'p'+'\x1b[38;5;184m'+'p'+'\x1b[38;5;220m'+'p'+'\x1b[38;5;178m'+'D'+'\x1b[38;5;100m'+'b'+'\x1b[38;5;101m'+'q'+'\x1b[38;5;241m'+'p'+'\x1b[38;5;59m'+'p'+'\x1b[38;5;239m'+'q'+'\x1b[38;5;238m'+'p'+'\x1b[38;5;236m'+'Q'+'\x1b[38;5;244m'+'Q'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;243m'+'QQQ'+'\x1b[38;5;244m'+'Q'+'\x1b[38;5;241m'+'p'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'·····')
    print('\x1b[38;5;234m'+'·············'+'\x1b[38;5;143m'+'('+'\x1b[38;5;220m'+'p'+'\x1b[38;5;184m'+'C'+'\x1b[38;5;58m'+'G'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;243m'+'QQQ'+'\x1b[38;5;235m'+'Q'+'\x1b[38;5;239m'+'p'+'\x1b[38;5;242m'+'·'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'·········'+'\x1b[38;5;238m'+'S'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;243m'+'QQQQQQ'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'········'+'\x1b[38;5;242m'+'·'+'\x1b[38;5;239m'+'q'+'\x1b[38;5;235m'+'Q'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;243m'+'QQQ'+'\x1b[38;5;244m'+'Q'+'\x1b[38;5;58m'+'b'+'\x1b[38;5;142m'+'Q'+'\x1b[38;5;184m'+'p'+'\x1b[38;5;184m'+'s'+'\x1b[38;5;3m'+'S'+'\x1b[38;5;234m'+'G'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;233m'+'G'+'\x1b[38;5;234m'+'G'+'\x1b[38;5;235m'+'G'+'\x1b[38;5;236m'+'G'+'\x1b[38;5;237m'+'C'+'\x1b[38;5;239m'+'D'+'\x1b[38;5;241m'+')'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'······')
    print('\x1b[38;5;234m'+'·········'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;242m'+')'+'\x1b[38;5;101m'+'p'+'\x1b[38;5;184m'+'p'+'\x1b[38;5;142m'+'p'+'\x1b[38;5;238m'+'p'+'\x1b[38;5;238m'+'p'+'\x1b[38;5;239m'+'p'+'\x1b[38;5;59m'+'p'+'\x1b[38;5;59m'+')'+'\x1b[38;5;239m'+'D'+'\x1b[38;5;236m'+'G'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;243m'+'QQ'+'\x1b[38;5;244m'+'Q'+'\x1b[38;5;237m'+'b'+'\x1b[38;5;242m'+'·'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'······'+'\x1b[38;5;242m'+'·'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;243m'+'QQQQQQ'+'\x1b[38;5;242m'+'p'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'····'+'\x1b[38;5;240m'+'s'+'\x1b[38;5;236m'+'S'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;243m'+'QQQQQQQQQQ'+'\x1b[38;5;234m'+'G'+'\x1b[38;5;100m'+'b'+'\x1b[38;5;184m'+'('+'\x1b[38;5;184m'+'b'+'\x1b[38;5;143m'+'p'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'············')
    print('\x1b[38;5;234m'+'······'+'\x1b[38;5;59m'+'s'+'\x1b[38;5;237m'+'S'+'\x1b[38;5;244m'+'Q'+'\x1b[38;5;233m'+'G'+'\x1b[38;5;100m'+'b'+'\x1b[38;5;142m'+'p'+'\x1b[38;5;244m'+'Q'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;243m'+'QQQQQ'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;236m'+'Q'+'\x1b[38;5;244m'+'Q'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;243m'+'QQQQ'+'\x1b[38;5;244m'+'Q'+'\x1b[38;5;240m'+'p'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;233m'+'·'+'\x1b[38;5;65m'+'·'+'\x1b[38;5;65m'+'·'+'\x1b[38;5;65m'+'p'+'\x1b[38;5;237m'+'S'+'\x1b[38;5;22m'+'G'+'\x1b[38;5;22m'+'GGGGG'+'\x1b[38;5;235m'+'G'+'\x1b[38;5;65m'+'p'+'\x1b[38;5;233m'+'·'+'\x1b[38;5;233m'+'·'+'\x1b[38;5;240m'+'q'+'\x1b[38;5;244m'+'Q'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;243m'+'QQQQQQQQ'+'\x1b[38;5;233m'+'G'+'\x1b[38;5;235m'+'G'+'\x1b[38;5;238m'+'C'+'\x1b[38;5;240m'+'D'+'\x1b[38;5;233m'+'·'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'···'+'\x1b[38;5;143m'+'·'+'\x1b[38;5;179m'+'·'+'\x1b[38;5;143m'+'p'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'··········')
    print('\x1b[38;5;234m'+'····'+'\x1b[38;5;233m'+'·'+'\x1b[38;5;244m'+'Q'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;233m'+'G'+'\x1b[38;5;142m'+'Q'+'\x1b[38;5;58m'+'S'+'\x1b[38;5;235m'+'G'+'\x1b[38;5;238m'+'C'+'\x1b[38;5;240m'+'D'+'\x1b[38;5;59m'+'DD'+'\x1b[38;5;239m'+'D'+'\x1b[38;5;238m'+'S'+'\x1b[38;5;236m'+'G'+'\x1b[38;5;235m'+'G'+'\x1b[38;5;234m'+'G'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;243m'+'QQG'+'\x1b[38;5;22m'+'G'+'\x1b[38;5;22m'+'G'+'\x1b[38;5;22m'+'QQS'+'\x1b[38;5;239m'+'Q'+'\x1b[38;5;65m'+'·'+'\x1b[38;5;65m'+'·'+'\x1b[38;5;233m'+'·'+'\x1b[38;5;242m'+'('+'\x1b[38;5;244m'+'Q'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;243m'+'QQQQ'+'\x1b[38;5;244m'+'Q'+'\x1b[38;5;244m'+'Q'+'\x1b[38;5;244m'+'Q'+'\x1b[38;5;22m'+'Q'+'\x1b[38;5;22m'+'Q'+'\x1b[38;5;22m'+'Q'+'\x1b[38;5;2m'+'Q'+'\x1b[38;5;2m'+'QbG'+'\x1b[38;5;22m'+'G'+'\x1b[38;5;22m'+'G'+'\x1b[38;5;238m'+'Q'+'\x1b[38;5;240m'+'D'+'\x1b[38;5;242m'+'·'+'\x1b[38;5;233m'+'·'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'······················')
    print('\x1b[38;5;234m'+'····'+'\x1b[38;5;242m'+'·'+'\x1b[38;5;237m'+'S'+'\x1b[38;5;236m'+'G'+'\x1b[38;5;237m'+'G'+'\x1b[38;5;142m'+'C'+'\x1b[38;5;242m'+')'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'········'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;65m'+'s'+'\x1b[38;5;29m'+'p'+'\x1b[38;5;29m'+'G'+'\x1b[38;5;65m'+'D'+'\x1b[38;5;237m'+'G'+'\x1b[38;5;234m'+'G'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;243m'+'QQQ'+'\x1b[38;5;244m'+'Q'+'\x1b[38;5;236m'+'Q'+'\x1b[38;5;235m'+'Q'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;243m'+'QQQQQQQQQQQQG'+'\x1b[38;5;236m'+'C'+'\x1b[38;5;240m'+'D'+'\x1b[38;5;233m'+'·'+'\x1b[38;5;65m'+'·'+'\x1b[38;5;65m'+')'+'\x1b[38;5;65m'+'S'+'\x1b[38;5;35m'+'G'+'\x1b[38;5;35m'+'G'+'\x1b[38;5;65m'+'p'+'\x1b[38;5;233m'+'·'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'·····················')
    print('\x1b[38;5;234m'+'··················'+'\x1b[38;5;65m'+'·'+'\x1b[38;5;35m'+'S'+'\x1b[38;5;35m'+'G'+'\x1b[38;5;65m'+'D'+'\x1b[38;5;233m'+'·'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'···'+'\x1b[38;5;233m'+'·'+'\x1b[38;5;239m'+'D'+'\x1b[38;5;235m'+'G'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;243m'+'QQQQQQQQQQQQQQ'+'\x1b[38;5;235m'+'G'+'\x1b[38;5;240m'+'D'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'·······'+'\x1b[38;5;65m'+'·'+'\x1b[38;5;65m'+'D'+'\x1b[38;5;65m'+'G'+'\x1b[38;5;65m'+'p'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'···················')
    print('\x1b[38;5;234m'+'················'+'\x1b[38;5;65m'+'·'+'\x1b[38;5;65m'+'G'+'\x1b[38;5;65m'+')'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'·········'+'\x1b[38;5;233m'+'·'+'\x1b[38;5;240m'+'D'+'\x1b[38;5;235m'+'G'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;243m'+'QQQQQQQQQ'+'\x1b[38;5;234m'+'G'+'\x1b[38;5;240m'+'D'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'·············'+'\x1b[38;5;233m'+'·'+'\x1b[38;5;65m'+')'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'··················')
    print('\x1b[38;5;234m'+'················'+'\x1b[38;5;233m'+'·················'+'\x1b[38;5;236m'+'S'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;233m'+'G'+'\x1b[38;5;235m'+'G'+'\x1b[38;5;236m'+'G'+'\x1b[38;5;23m'+'G'+'\x1b[38;5;23m'+'D'+'\x1b[38;5;23m'+'D'+'\x1b[38;5;29m'+'Q'+'\x1b[38;5;29m'+'p'+'\x1b[38;5;72m'+'b'+'\x1b[38;5;72m'+'p'+'\x1b[38;5;72m'+'pp'+'\x1b[38;5;72m'+'p'+'\x1b[38;5;66m'+'p'+'\x1b[38;5;66m'+'p'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'····························')
    print('\x1b[38;5;234m'+'······························'+'\x1b[38;5;66m'+'·'+'\x1b[38;5;66m'+'p'+'\x1b[38;5;72m'+'p'+'\x1b[38;5;65m'+'Q'+'\x1b[38;5;23m'+'p'+'\x1b[38;5;23m'+'S'+'\x1b[38;5;236m'+'Q'+'\x1b[38;5;244m'+'Q'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;243m'+'QQ'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;59m'+'b'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'··'+'\x1b[38;5;66m'+'·'+'\x1b[38;5;66m'+'·'+'\x1b[38;5;66m'+'·'+'\x1b[38;5;66m'+'·'+'\x1b[38;5;72m'+'·'+'\x1b[38;5;72m'+'G'+'\x1b[38;5;72m'+'b'+'\x1b[38;5;66m'+'·'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'·························')
    print('\x1b[38;5;234m'+'··························'+'\x1b[38;5;66m'+'·'+'\x1b[38;5;66m'+'·'+'\x1b[38;5;66m'+'··'+'\x1b[38;5;66m'+'·'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;235m'+'S'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;243m'+'QQQQQQQ'+'\x1b[38;5;233m'+'·'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'·········'+'\x1b[38;5;234m'+'···························')
    print('\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'························'+'\x1b[38;5;234m'+'·······'+'\x1b[38;5;237m'+'S'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;243m'+'QQQQQQQQ'+'\x1b[38;5;59m'+'C'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'····································')
    print('\x1b[38;5;234m'+'·······························'+'\x1b[38;5;233m'+'·'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;243m'+'QQQQQQQQQ'+'\x1b[38;5;236m'+'b'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'····································')
    print('\x1b[38;5;234m'+'·······························'+'\x1b[38;5;242m'+'('+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;243m'+'QQQQQQQQQQ'+'\x1b[38;5;242m'+'p'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'···································')
    print('\x1b[38;5;234m'+'······························'+'\x1b[38;5;233m'+'·'+'\x1b[38;5;234m'+'S'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;243m'+'QQQQQQQQQQ'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;242m'+'p'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'··································')
    print('\x1b[38;5;234m'+'·····························'+'\x1b[38;5;233m'+'·'+'\x1b[38;5;234m'+'S'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;243m'+'QQQQQQQQQQQQ'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'··································')
    print('\x1b[38;5;234m'+'····························'+'\x1b[38;5;242m'+'('+'\x1b[38;5;244m'+'Q'+'\x1b[38;5;243m'+'Q'+'\x1b[38;5;243m'+'QQ'+'\x1b[38;5;234m'+'G'+'\x1b[38;5;235m'+'G'+'\x1b[38;5;23m'+'G'+'\x1b[38;5;24m'+'G'+'\x1b[38;5;24m'+'D'+'\x1b[38;5;25m'+'P'+'\x1b[38;5;31m'+'b'+'\x1b[38;5;31m'+'bb'+'\x1b[38;5;24m'+'D'+'\x1b[38;5;24m'+'D'+'\x1b[38;5;24m'+'G'+'\x1b[38;5;67m'+'p'+'\x1b[38;5;67m'+'p'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'·······························')
    print('\x1b[38;5;234m'+'···························'+'\x1b[38;5;242m'+'('+'\x1b[38;5;233m'+'G'+'\x1b[38;5;23m'+'G'+'\x1b[38;5;24m'+'G'+'\x1b[38;5;31m'+'b'+'\x1b[38;5;32m'+'b'+'\x1b[38;5;32m'+'G'+'\x1b[38;5;31m'+'C'+'\x1b[38;5;67m'+'·'+'\x1b[38;5;67m'+'·'+'\x1b[38;5;66m'+'·'+'\x1b[38;5;66m'+'·'+'\x1b[38;5;66m'+'·'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'····'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'··································')
    print('\x1b[38;5;234m'+'··························'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;67m'+'G'+'\x1b[38;5;67m'+')'+'\x1b[38;5;66m'+'·'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'·'+'\x1b[38;5;234m'+'················································')
    print('\x1b[38;5;234m'+'·······························'+'\x1b[38;5;65m'+'ALBEROBOTICS'+'\x1b[38;5;234m'+'·····································')
    print('\x1b[38;5;234m'+'················································································'+'\x1b[0m')

def print_alberobotics_bw():
    print("""
················································································
························GQQp····················································
·························DQQQ,··················································
·························.,GGGGGGGGGGGGGGGGbppppp·······························
···················.,pGGbbbGQQpO············.dQQpGGGbp,.························
···············.pGGbbGC'····(QQb···········.QQQQb·····'DGGbp.···················
············.GGbbQSQ·········QQG··········.QQQGO············'GGp,···············
··········,GbGC'QQQQQp·······QQG····.....·SGQG··················'GGp,dQQQ,······
········,GGC····SQQQQQQp·..ppDssppppppppppppsbpppp..··············sQQQpSQ·······
······.GC········'GQGGDsspQpppQQ·········dQQQb''''Gbpppp,.·····.dQQGG'··'Db·····
····.p'··········.pbpGdSQQQQQQQQb········SQQQb········''GpppppGQQG'·············
··sGSQQQp·····.ppb'······'GGQQQQQQ,······QQQQ·············.ppppsp···············
··'SGGQQQQb·.pp'············DGQQQQQp····(QQQG···········.SQQQG''Gpp.············
·······'SQGbb'················SQQQQQQQQQQQQQC··········(QQQQG·····'Gp.··········
········.PQSb············..ppppb'pQQQQpps'sDpp,.·······SQQQQ'········'G,········
·······(s'SQQb·······.pppb'''····QQQQQQQQQG··''GGppp,.·SQQQG············'·······
·····.CC···SQQQ,·.ppp''··········(QQQQQQQQC········''Gp'DGGQ·········...········
············'GGCsQp,··············QQQQQQQQ············.QpppDbq,pqpQQQQQQQp······
·············(p'GQQQQQp,··········SQQQQQQQ·········.qQQQQQQbQp'SGQGGGGC''·······
··········,,pppppp,DGQQQQb,········QQQQQQQ,·····sSQQQQQQQQQQQQGb'b,·············
······sSQGbpQQQQQQQQQQQQQQQQp··..,,SGGGGGGG,..qQQQQQQQQQQGGCD······',···········
····.QQQGQSGC''''SGGGQQQGGGQQSQ···.QQQQQQQQQQQQQQbGGGQD·························
····'SGG''···········,pG'GGQQQQQQQQQQQQQQQQQQQQGCD··''GGp.······················
··················.SG'······'GQQQQQQQQQQQQQQQGD··········'G,····················
················.G'············'GQQQQQQQQQQG'···············'···················
·································SQGGGGDDQpbppppp,.·····························
······························.,pQpSQQQQQQb····''''Gb,··························
··························.·''···SQQQQQQQQ······································
································SQQQQQQQQQC·····································
································QQQQQQQQQQb·····································
·······························(QQQQQQQQQQQ,····································
······························.SQQQQQQQQQQQQ,···································
·····························.SQQQQQQQQQQQQQQ···································
····························.QQQQGGGGDPbbbDDGp,.································
···························(GGGbbGC''···········································
··························.G'···················································
·······························ALBEROBOTICS·····································
················································································
    """)

if __name__ == "__main__":
    print_alberobotics()
    print_alberobotics_bw()
