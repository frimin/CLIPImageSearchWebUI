import os

class CommandLineArgs():
    __data = None
    def __init__(self, cli_args) -> None:
        self.__data = cli_args

def init_clip_args(args):
    if not hasattr(CommandLineArgs, "__data") or CommandLineArgs.__data is None:
        CommandLineArgs.__data = CommandLineArgs(args)

def get_cli_args():
    return CommandLineArgs.__data