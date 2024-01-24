from configparser import ConfigParser

def get_parser(config_file):
    config = ConfigParser()
    config.read(config_file, encoding='UTF-8')
    return config