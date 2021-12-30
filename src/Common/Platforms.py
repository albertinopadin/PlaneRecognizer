import platform


MAC_OS = 'Darwin'
LINUX = 'Linux'
WINDOWS = 'Windows'


def in_mac_os():
    return platform.system() == MAC_OS


def in_linux():
    return platform.system() == LINUX


def in_windows():
    return platform.system() == WINDOWS
