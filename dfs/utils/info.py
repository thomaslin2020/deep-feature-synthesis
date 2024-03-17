import locale
import os
import platform
import struct
import sys

deps = [
    "polars",
    "tqdm",
    "pip",
    "setuptools",
]


def print_sys_info():
    print("\nSYSTEM INFO")
    print("-----------")
    sys_info = get_sys_info()
    for k, stat in sys_info:
        print("{k}: {stat}".format(k=k, stat=stat))


# Modified from here
# https://github.com/pandas-dev/pandas/blob/d9a037ec4ad0aab0f5bf2ad18a30554c38299e57/pandas/util/_print_versions.py#L11
def get_sys_info():
    """Returns system information as a dict"""

    info = []

    try:
        sys_name, node_name, release, version, machine, processor = platform.uname()
        info.extend(
            [
                ("python", ".".join(map(str, sys.version_info))),
                ("python-bits", struct.calcsize("P") * 8),
                ("OS", "{sysname}".format(sysname=sys_name)),
                ("OS-release", "{release}".format(release=release)),
                ("machine", "{machine}".format(machine=machine)),
                ("processor", "{processor}".format(processor=processor)),
                ("byteorder", "{byteorder}".format(byteorder=sys.byteorder)),
                ("LC_ALL", "{lc}".format(lc=os.environ.get("LC_ALL", "None"))),
                ("LANG", "{lang}".format(lang=os.environ.get("LANG", "None"))),
                ("LOCALE", ".".join(map(str, locale.getlocale()))),
            ],
        )
    except (KeyError, ValueError):
        pass

    return info
