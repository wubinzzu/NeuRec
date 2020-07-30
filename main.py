from reckit import Configurator
from importlib.util import find_spec
from importlib import import_module
from reckit import typeassert
import os


def _set_random_seed(seed=2020):
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)

    try:
        import tensorflow as tf
        tf.set_random_seed(seed)
        print("set tensorflow seed")
    except:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        print("set pytorch seed")
    except:
        pass


@typeassert(recommender=str, platform=str)
def find_recommender(recommender, platform="pytorch"):
    model_dirs = set(os.listdir("model"))
    model_dirs.remove("base")

    module = None
    if platform == "pytorch":
        platforms = ["pytorch", "tensorflow"]
    elif platform == "tensorflow":
        platforms = ["tensorflow", "pytorch"]
    else:
        raise ValueError(f"unrecognized platform: '{platform}'.")

    for platform in platforms:
        if module is not None:
            break
        for tdir in model_dirs:
            spec_path = ".".join(["model", tdir, platform, recommender])
            if find_spec(spec_path):
                module = import_module(spec_path)
                break

    if module is None:
        raise ImportError(f"Recommender: {recommender} not found")

    if hasattr(module, recommender):
        Recommender = getattr(module, recommender)
    else:
        raise ImportError(f"Import {recommender} failed from {module.__file__}!")
    return Recommender


def main():
    config = Configurator()
    config.add_config("NeuRec.ini", section="NeuRec")
    config.parse_cmd()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config["gpu_id"])

    Recommender = find_recommender(config.recommender, platform=config.platform)

    model_cfg = os.path.join("conf", config.recommender+".ini")
    config.add_config(model_cfg, section="hyperparameters", used_as_summary=True)

    recommender = Recommender(config)
    recommender.train_model()


if __name__ == "__main__":
    _set_random_seed()
    main()
