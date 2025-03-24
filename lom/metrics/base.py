from torch import Tensor, nn
from os.path import join as pjoin
# from .mr import MRMetrics
# from .t2m import TM2TMetrics
# from .t2m_exp import TM2TMetrics_Exp
# from .mm import MMMetrics
# from .m2t import M2TMetrics
# from .m2m import PredMetrics
# from .a2t import A2TMetrics
# from .a2m import AM2AMetrics
# from .a2m_emage import AM2AMetrics_Emage
# from .a2m_exp import AM2AMetrics_Exp
# from .m2e import M2EMetrics
from .rotation_metric import RotationMetrics
from .h3d_metric_bk import H3DMetrics
from .co_speech import CoSpeechMetrics

class BaseMetrics(nn.Module):
    def __init__(self, cfg, datamodule, debug, **kwargs) -> None:
        super().__init__()

        for metric in cfg.METRIC.TYPE:
            setattr(self, metric, globals()[metric](
                    cfg=cfg,
                    dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP,
                ))
