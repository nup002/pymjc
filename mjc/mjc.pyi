import numpy as np
import numpy.typing as npt
from typing import Union, Tuple, List

def get_overlapping_region(s1: npt.NDArray[Union[np.integer, np.floating]],
                           s2: npt.NDArray[Union[np.integer, np.floating]]) -> Tuple[int, ...]: ...


def check_input(s1: Union[List, npt.NDArray[Union[np.integer, np.floating]]],
                s2: Union[List, npt.NDArray[Union[np.integer, np.floating]]],
                override_checks: bool) -> Tuple[npt.NDArray[Union[np.integer, np.floating]],
                                                npt.NDArray[Union[np.integer, np.floating]]]: ...

def minimum_mjc(s1: Union[List, npt.NDArray[Union[np.integer, np.floating]]],
                s2: Union[List, npt.NDArray[Union[np.integer, np.floating]]],
                dxy_limit: float = np.inf,
                beta: float = 1.,
                show_plot: bool = False,
                std_s1: float = None, std_s2: float = None,
                tavg_s1: float = None, tavg_s2: float = None,
                override_checks=False) -> Tuple[float, bool]: ...

def mjc(s1: Union[List, npt.NDArray[Union[np.integer, np.floating]]],
        s2: Union[List, npt.NDArray[Union[np.integer, np.floating]]],
        dxy_limit: float = np.inf,
        beta: float = 1.,
        show_plot: bool = False,
        std_s1: float = None, std_s2: float = None,
        tavg_s1: float = None, tavg_s2: float = None,
        return_args=False,
        override_checks=False) -> Union[Tuple[float, bool],
                                        Tuple[float, bool, float, float, float, float,
                                              npt.NDArray[Union[np.integer, np.floating]],
                                              npt.NDArray[Union[np.integer, np.floating]]]]: ...