from argparse import ArgumentParser


def configure_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-c",
                            "--config",
                            help="Path to YAML configuration file",
                            default="../configs/config.yaml",
                            type=str)
    return arg_parser
