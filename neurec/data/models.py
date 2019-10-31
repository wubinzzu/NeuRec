from neurec.model.general_recommender.APR import APR
from neurec.model.general_recommender.CDAE import CDAE
from neurec.model.general_recommender.CFGAN import CFGAN
from neurec.model.general_recommender.ConvNCF import ConvNCF
from neurec.model.general_recommender.DAE import DAE
from neurec.model.general_recommender.DeepICF import DeepICF
from neurec.model.general_recommender.DMF import DMF
from neurec.model.general_recommender.FISM import FISM
from neurec.model.general_recommender.IRGAN import IRGAN
from neurec.model.general_recommender.JCA import JCA
from neurec.model.general_recommender.MF import MF
from neurec.model.general_recommender.MLP import MLP
from neurec.model.general_recommender.MultiDAE import MultiDAE
from neurec.model.general_recommender.MultiVAE import MultiVAE
from neurec.model.general_recommender.NAIS import NAIS
from neurec.model.general_recommender.NeuMF import NeuMF
from neurec.model.general_recommender.NGCF import NGCF
from neurec.model.general_recommender.SpectralCF import SpectralCF
from neurec.model.general_recommender.WRMF import WRMF
from neurec.model.sequential_recommender.Caser import Caser
from neurec.model.sequential_recommender.Fossil import Fossil
from neurec.model.sequential_recommender.FPMC import FPMC
from neurec.model.sequential_recommender.FPMCplus import FPMCplus
from neurec.model.sequential_recommender.GRU4Rec import GRU4Rec
from neurec.model.sequential_recommender.GRU4RecPlus import GRU4RecPlus
from neurec.model.sequential_recommender.HRM import HRM
from neurec.model.sequential_recommender.NPE import NPE
from neurec.model.sequential_recommender.SASRec import SASRec
from neurec.model.sequential_recommender.SRGNN import SRGNN
from neurec.model.sequential_recommender.TransRec import TransRec
from neurec.model.social_recommender.DiffNet import DiffNet
from neurec.model.social_recommender.SBPR import SBPR

models = {
    "apr": APR,
    "cdae": CDAE,
    "cfgan": CFGAN,
    "convncf": ConvNCF,
    "dae": DAE,
    "deepicf":DeepICF,
    "dmf": DMF,
    "fism": FISM,
    "irgan": IRGAN,
    "jca": JCA,
    "mf": MF,
    "mlp": MLP,
    "multidae": MultiDAE,
    "multivae": MultiVAE,
    "nais": NAIS,
    "neumf": NeuMF,
    "ngcf": NGCF,
    "spectralcf": SpectralCF,
    "wrmf": WRMF,
    "caser": Caser,
    "fossil": Fossil,
    "fpmc": FPMC,
    "fpmcplus": FPMCplus,
    "gru4rec": GRU4Rec,
    "gru4recplus": GRU4RecPlus,
    "hrm": HRM,
    "npe": NPE,
    "sasrec": SASRec,
    "srgnn": SRGNN,
    "transrec": TransRec,
    "diffnet": DiffNet,
    "sbpr": SBPR
}
