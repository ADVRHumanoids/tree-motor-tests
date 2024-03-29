#!/usr/bin python3
# -*- coding: utf-8 -*-
"""Function to interact with the user"""

def single_yes_or_no_question(question, default_no=True):
    """Simple yes/no function."""
    choices = ' [y/N]: ' if default_no else ' [Y/n]: '
    default_answer = 'n' if default_no else 'y'
    reply = str(input(question + choices)).lower().strip() or default_answer
    if reply[0] == 'y':
        return True
    if reply[0] == 'n':
        return False
    else:
        return False if default_no else True
