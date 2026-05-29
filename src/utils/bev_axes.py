from __future__ import annotations

from typing import Dict


def get_bev_axis_config() -> Dict[str, object]:
    """Return the canonical BEV axis convention for ego-frame plots.

    Ego-frame trajectory data in this repo uses:
    - x: forward positive
    - y: left positive

    Matplotlib horizontal axes increase to the right, so we invert the x-axis
    after plotting y on the horizontal axis to keep "left positive" visually
    consistent with the camera views.
    """

    return {
        "xlabel": "Y (left positive)",
        "ylabel": "X (up positive)",
        "invert_x": True,
    }


def configure_bev_axes(ax) -> None:
    """Apply the canonical BEV axis convention to a matplotlib axis."""

    config = get_bev_axis_config()
    ax.set_xlabel(config["xlabel"])
    ax.set_ylabel(config["ylabel"])
    if config["invert_x"]:
        ax.invert_xaxis()
