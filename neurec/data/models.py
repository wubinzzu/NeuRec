from neurec.model.item_ranking.MF import MF
from neurec.model.item_ranking.FISM import FISM
from neurec.model.item_ranking.NAIS import NAIS
from neurec.model.item_ranking.MLP import MLP
from neurec.model.item_ranking.NeuMF import NeuMF
from neurec.model.item_ranking.APR import APR
from neurec.model.item_ranking.DMF import DMF
from neurec.model.item_ranking.ConvNCF import ConvNCF
from neurec.model.item_ranking.CDAE import CDAE
from neurec.model.item_ranking.DAE import DAE
from neurec.model.seq_ranking.NPE import NPE
from neurec.model.item_ranking.IRGAN import IRGAN
from neurec.model.item_ranking.MultiDAE import MultiDAE
from neurec.model.item_ranking.MultiVAE import MultiVAE
from neurec.model.item_ranking.JCA import JCA
from neurec.model.item_ranking.CFGAN import CFGAN
from neurec.model.item_ranking.SBPR import SBPR
from neurec.model.item_ranking.WRMF import WRMF
from neurec.model.item_ranking.SpectralCF import SpectralCF
from neurec.model.seq_ranking.FPMC import FPMC
from neurec.model.seq_ranking.FPMCplus import FPMCplus
from neurec.model.seq_ranking.HRM import HRM
from neurec.model.seq_ranking.TransRec import TransRec

models = {
    "mf": MF,
    "fpmc": FPMC,
    "fpmcplus": FPMCplus,
    "fism": FISM,
    "apr": APR,
    "nais": NAIS,
    "mlp": MLP,
    "hrm": HRM,
    "dmf": DMF,
    "neumf": NeuMF,
    "convncf": ConvNCF,
    "transrec": TransRec,
    "cdae": CDAE,
    "dae": DAE,
    "npe": NPE,
    "multidae": MultiDAE,
    "multivae": MultiVAE,
    "irgan": IRGAN,
    "cfgan": CFGAN,
    "jca": JCA,
    "sbpr": SBPR,
    "spectralcf": SpectralCF,
    "wrmf": WRMF
}
