import logging
from typing import Optional

from kimmdy.plugins import Parameterizer
from kimmdy.topology.topology import Topology

import grappa
from grappa.grappa import Grappa

# Conditional import based on grappa version to allow for backward compatibility
if grappa.__version__ <= "1.4.1":
    from grappa.utils.kimmdy_utils import (
        KimmdyGrappaParameterizer as GrappaParameterizer,
    )
else:
    from grappa.utils.gromacs_utils import GrappaParameterizer


logger = logging.getLogger("kimmdy.grappa_interface")


class GrappaInterface(Parameterizer):
    """
    Wrapper of the GrappaParameterizer used in grappa. Initialised with a tag instead of a model.
    """

    def __init__(
        self, *args, grappa_tag: str = "latest", charge_model: str = "amber99", **kwargs
    ):
        super().__init__(*args, **kwargs)
        logger.info(f"Instantiating Grappa with tag '{grappa_tag}'.")
        grappa_instance = Grappa.from_tag(grappa_tag)
        self.kimmdy_grappa_parameterizer = GrappaParameterizer(
            grappa_instance=grappa_instance,
            charge_model=charge_model,
        )

    def parameterize_topology(
        self, current_topology: Topology, focus_nrs: Optional[set[str]] = None
    ) -> Topology:
        return self.kimmdy_grappa_parameterizer.parameterize_topology(
            current_topology=current_topology, focus_nrs=focus_nrs
        )
