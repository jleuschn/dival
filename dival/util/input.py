# -*- coding: utf-8 -*-


def input_yes_no(default='y'):
    """
    Demand user input y[es] or n[o].

    The user is asked repeatedly, until the input is valid.

    Parameters
    ----------
    default : {``'y'``, ``'n'``}, optional
        The output if the user enters empty input.

    Returns
    -------
    inp : {``'y'``, ``'n'``}
        The users input (or `default`).
    """
    def _input():
        inp = input()
        inp = inp.lower()
        if inp in ['y', 'yes']:
            inp = 'y'
        elif inp in ['n', 'no']:
            inp = 'n'
        elif inp == '':
            inp = default
        else:
            print('please input y[es] or n[o]')
            return None
        return inp

    inp = _input()
    while inp not in ['y', 'n']:
        inp = _input()

    return inp == 'y'
