import logging
from . import predictor_config
from .model_selector import ModelPairError

logger = logging.getLogger(__name__)


class PredictorProvider:
    """Singleton-style provider for OH and NPS predictors.

    V2: an optional *model family* may be set via
    :meth:`set_model_family` or :meth:`load_pair`.  When a family is
    active, ``get_oh_predictor`` / ``get_nps_predictor`` lazily load
    ``{family}_OH.pkl`` / ``{family}_NPS.pkl``.  When no family is set
    the legacy filenames are used (backward compatibility).
    """

    _oh = None
    _nps = None
    _model_family = None

    # ---------------------------------------------------------------
    # Model-family management
    # ---------------------------------------------------------------

    @classmethod
    def get_model_family(cls):
        """Return the currently selected model family name, or ``None``."""
        return cls._model_family

    @classmethod
    def set_model_family(cls, family):
        """Set the active model family and clear cached predictors.

        The next call to ``get_oh_predictor`` / ``get_nps_predictor``
        will lazily load the new family's model files.
        """
        cls._model_family = family
        cls._oh = None
        cls._nps = None

    @classmethod
    def load_pair(cls, family):
        """Load and validate a complete OH+NPS pair by family name.

        Returns ``(oh_predictor, nps_predictor)``.
        Raises :class:`ModelPairError` if the pair is incomplete or
        invalid.
        """
        oh, nps = predictor_config.load_model_pair(family)
        cls._oh = oh
        cls._nps = nps
        cls._model_family = family
        return oh, nps

    # ---------------------------------------------------------------
    # Predictor access
    # ---------------------------------------------------------------

    @classmethod
    def get_oh_predictor(cls):
        if cls._oh is None:
            cls._oh = predictor_config.create_oh_predictor(cls._model_family)
        return cls._oh

    @classmethod
    def get_nps_predictor(cls):
        if cls._nps is None:
            cls._nps = predictor_config.create_nps_predictor(cls._model_family)
        return cls._nps

    @classmethod
    def set_oh_predictor(cls, predictor):
        cls._oh = predictor

    @classmethod
    def set_nps_predictor(cls, predictor):
        cls._nps = predictor

    @classmethod
    def reset(cls):
        cls._oh = None
        cls._nps = None
        cls._model_family = None


# Module-level convenience aliases (preserve V1 surface)
get_oh_predictor = PredictorProvider.get_oh_predictor
get_nps_predictor = PredictorProvider.get_nps_predictor
set_oh_predictor = PredictorProvider.set_oh_predictor
set_nps_predictor = PredictorProvider.set_nps_predictor
set_model_family = PredictorProvider.set_model_family
get_model_family = PredictorProvider.get_model_family
load_pair = PredictorProvider.load_pair
reset = PredictorProvider.reset
