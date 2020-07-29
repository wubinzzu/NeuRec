from reckit import Configurator
from importlib.util import find_spec
from importlib import import_module
from reckit import typeassert
import os


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
        raise ValueError("unrecognized platform: '%s'." % platform)

    for platform in platforms:
        if module is not None:
            break
        for tdir in model_dirs:
            spec_path = ".".join(["model", tdir, platform, recommender])
            if find_spec(spec_path):
                module = import_module(spec_path)
                break

    if module is None:
        raise ImportError("Recommender: {} not found".format(recommender))

    if hasattr(module, recommender):
        Recommender = getattr(module, recommender)
    else:
        raise ImportError("Import '%s' failed from '%s'!" % (recommender, module.__file__))
    return Recommender


def main():
    config = Configurator()
    config.add_config("NeuRec.ini", section="NeuRec")
    config.parse_cmd()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config["gpu_id"])

    Recommender = find_recommender(config.recommender, platform=config.platform)

    rs_path = Recommender.__module__
    if "pytorch" in rs_path:
        platform = "pytorch"
    elif "tensorflow" in rs_path:
        platform = "tensorflow"
    else:
        raise ImportError("unrecognized platform")

    model_cfg = os.path.join("conf", platform, config.recommender+".ini")
    config.add_config(model_cfg, section="hyperparameters", used_as_summary=True)

    recommender = Recommender(config)
    recommender.train_model()


if __name__ == "__main__":
    main()
