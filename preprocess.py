from reckit import Preprocessor
from reckit import Configurator

config = Configurator()

config.add_config("Preprocess.ini", section="Preprocess")
config.parse_cmd()

data = Preprocessor()
data.load_data(config.filename, sep=config.separator, columns=config.file_column)
if config.drop_duplicates is True:
    data.drop_duplicates(keep=config.keep)

data.filter_data(user_min=config.user_min, item_min=config.item_min)
if config.remap_id is True:
    data.remap_data_id()

if config.splitter == "leave_out":
    data.split_data_by_leave_out(valid=config.valid, test=config.test,
                                 by_time=config.by_time)
elif config.splitter == "ratio":
    data.split_data_by_ratio(train=config.train, valid=config.valid,
                             test=config.test, by_time=config.by_time)

data.save_data()
