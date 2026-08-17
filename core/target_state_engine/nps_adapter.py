"""
AxiPulseAI NPS Output Adapter

Converts model survey distribution outputs
into NPS score.
"""

import numpy as np


def convert_output(raw):

    raw=np.asarray(raw)


    if raw.ndim == 1:
        raw=raw.reshape(1,-1)


    if raw.shape[1] == 11:

        detractors = raw[:,:7].sum(axis=1)

        promoters = (
            raw[:,9]
            +
            raw[:,10]
        )

        total = raw.sum(axis=1)


        return (
            (
                promoters
                -
                detractors
            )
            /
            total
            *
            100
        )


    return raw.flatten()
