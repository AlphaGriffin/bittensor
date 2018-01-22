#!/usr/bin/env python
"""
A recursive json printer. And other fun stuff.
"""

import os
import sys
import shlex
import struct
import platform
import subprocess
import json
# import options

__author__ = "Eric Petersen @Ruckusist"
__copyright__ = "Copyright 2017, The Alpha Griffin Project"
__credits__ = ["Eric Petersen", "Shawn Wilson", "@alphagriffin"]
__license__ = "***"
__version__ = "0.1.0"
__maintainer__ = "Eric Petersen"
__email__ = "ruckusist@alphagriffin.com"
__status__ = "Alpha"


class Printer(object):
    """A class that formats a cleaner CLI look."""

    def __init__(self, options=None):
        """Probably dont need options but you never know."""
        self.options = options
        if not self.options:
            self.options.border = '|~|'
            self.options.filler = ' '
            self.options.spacer = ' |#|'
            self.options.blocker = '    '
        self.working_res = False
        try:
            self.res_x, self.res_y = self._get_terminal_size_windows() if not None else self.getTerminalSize()
            self.working_res = True
        except Exception as exp:
            # cant know the screen size, might be visual studio or Jupyter
            # use backup printer that doesnt need screen size
            # not working in powershell either.
            print("Failing to get screen size!")
            pass

    def __call__(self, data=None):
        """Call this Class without any functions."""
        if self.working_res:
            self.printer(data)
        else:
            self.no_size_printer(data)
        return True

    def __str__(self):
        """Tie up those loose ends."""
        return str("AG_Printer Ver: {}".format(__version__))

    def no_size_printer(self, data=None):
        """Basic printer with no known size."""
        border = self.options.border
        filler = self.options.filler
        spacer = self.options.spacer
        blocker = self.options.blocker
        # catch a blank line
        if data is None:
            data = spacer * int((70/len(self.options.spacer)))
        msg = "{border}{0:{filler}^{size}}{border}".format(
            data[:76], size=70, border=border, filler=filler
            )
        if self.options.verbose:
            print(msg)
        # return msg

    def printer(self, data=None):
        """Pretty recursive json printer, or line printer."""
        border = self.options.border
        filler = self.options.filler
        spacer = self.options.spacer
        blocker = self.options.blocker
        scr_size = self.res_x-(len(border)*2)
        if data == None:
            print("{border}{0:{filler}^{size}}{border}".format(
            '#'*scr_size, size=scr_size, border=border, filler=filler
            ))
        if type(data) == str:
            print("{border}{0:{filler}^{size}}{border}".format(
            data[:self.res_x-8], size=scr_size, border=border, filler=filler
            ))
        if type(data) == dict:
            for i in data:
                print("{border}{0:{filler}<{size}}{border}".format(
                i[:self.res_x-4], size=scr_size, border=border, filler=filler
                ))
                try:
                    for x in data[i]:
                        if type(data[i][x]) == dict:
                            try:
                                print("{border}{block}{0:{filler}<{size}}{border}".format(
                                x[:self.res_x-4], size=scr_size-len(spacer), border=border, filler=filler, block=blocker
                                ))
                                for y in data[i][x]:
                                    msg = "{}: {}".format(y, data[i][x][y])
                                    print("{border}{space}{0:{filler}<{size}}{border}".format(
                                    msg[:self.res_x-4], size=scr_size-len(spacer)*2, border=border, filler=filler, space=spacer*2
                                    ))
                            except:
                                pass
                        if type(data[i][x]) == list:
                            try:
                                print("{border}{block}{0:{filler}<{size}}{border}".format(
                                x[:self.res_x-4], size=scr_size-len(spacer), border=border, filler=filler, block=blocker
                                ))
                                for index, y in enumerate(data[i][x]):
                                    msg = "{}: {}".format(y, data[i][x][index])
                                    print("{border}{space}{0:{filler}<{size}}{border}".format(
                                    msg[:self.res_x-4], size=scr_size-len(spacer)*2, border=border, filler=filler, space=spacer*2
                                    ))
                            except:
                                pass
                        else:
                            msg = "{}: {}".format(x, data[i][x])
                            print("{border}{space}{0:{filler}<{size}}{border}".format(
                            msg[:self.res_x-4], size=scr_size-4, border=border, filler=filler, space=spacer
                            ))
                except:
                    pass

    def main(self):
        """Sanity Check."""
        print("ResX: {}, ResY: {}".format(self.res_x, self.res_y))
        return True

    def _get_terminal_size_windows(self):
        """Source: https://gist.github.com/jtriley/1108174"""
        try:
            from ctypes import windll, create_string_buffer
            # stdin handle is -10
            # stdout handle is -11
            # stderr handle is -12
            h = windll.kernel32.GetStdHandle(-12)
            csbi = create_string_buffer(22)
            res = windll.kernel32.GetConsoleScreenBufferInfo(h, csbi)
            if res:
                (bufx, bufy, curx, cury, wattr,
                 left, top, right, bottom,
                 maxx, maxy) = struct.unpack("hhhhHhhhhhh", csbi.raw)
                sizex = right - left + 1
                sizey = bottom - top + 1
                return sizex -1, sizey
        except:
            # src: http://stackoverflow.com/questions/263890/how-do-i-find-the-width-height-of-a-terminal-window
            try:
                cols = int(subprocess.check_call(shlex.split('tput cols')))
                rows = int(subprocess.check_call(shlex.split('tput lines')))
                return (cols, rows)
            except:
                pass
            pass

    def getTerminalSize(self):
        """Source: https://stackoverflow.com/questions/566746/how-to-get-linux-console-window-width-in-python"""
        env = os.environ
        def ioctl_GWINSZ(fd):
            try:
                import fcntl, termios, struct, os
                cr = struct.unpack('hh', fcntl.ioctl(fd, termios.TIOCGWINSZ,
            '1234'))
            except:
                return
            return cr
        cr = ioctl_GWINSZ(0) or ioctl_GWINSZ(1) or ioctl_GWINSZ(2)
        if not cr:
            try:
                fd = os.open(os.ctermid(), os.O_RDONLY)
                cr = ioctl_GWINSZ(fd)
                os.close(fd)
            except:
                pass
        if not cr:
            cr = (env.get('LINES', 25), env.get('COLUMNS', 80))

            ### Use get(key[, default]) instead of a try/catch
            #try:
            #    cr = (env['LINES'], env['COLUMNS'])
            #except:
            #    cr = (25, 80)
        return int(cr[1]), int(cr[0])


def main():
    """Launcher for the app."""
    config = options.Options()
    app = Printer(config)
    if app.main():
        app()
        app("This is just a test")
        app.printer("this test string is clearly way way too long to fit into this field, surely, too long.")
        app(str(app))
        app.printer("AlphaGriffin.com | 2017")
        app()
        dick = {'test': {'thing1': '1', 'Thing2': '2', 'thing3': {'thing4' : '4'}}}
        app(dick)
        app()
        sys.exit('Alphagriffin.com | 2017')
    return True

if __name__ == '__main__':
    try:
        import options
        # os.system("mode con cols=80 lines=75")
        os.system("clear")
        main()
    except Exception as e:
        print("and thats okay too.")
        sys.exit(e)
