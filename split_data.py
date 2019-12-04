from data.DataSplitter import Splitter
import configparser


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("split_data.properties")
    splitter_info = dict(config.items("split"))
    splitter = Splitter(splitter_info)
    splitter.split()
