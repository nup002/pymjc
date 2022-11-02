import numpy as np
import numpy.typing as npt
from typing import Union, Tuple, List


def analyse_timeseries(std_s1: Union[float, None],
                       std_s2: Union[float, None],
                       tavg_s1: Union[float, None],
                       tavg_s2: Union[float, None],
                       s1: npt.NDArray[Union[np.integer, np.floating]],
                       s2: npt.NDArray[Union[np.integer, np.floating]]) -> Tuple[float, ...]: ...

def get_overlapping_region(s1: npt.NDArray[Union[np.integer, np.floating]],
                           s2: npt.NDArray[Union[np.integer, np.floating]]
                           ) -> Tuple[npt.NDArray[Union[np.integer, np.floating]],
                                      npt.NDArray[Union[np.integer, np.floating]]]: ...


def check_input(s1: Union[List, npt.NDArray[Union[np.integer, np.floating]]],
                s2: Union[List, npt.NDArray[Union[np.integer, np.floating]]],
                override_checks: bool) -> Tuple[npt.NDArray[Union[np.integer, np.floating]],
                                                npt.NDArray[Union[np.integer, np.floating]]]: ...

def dmjc(s1: Union[List, npt.NDArray[Union[np.integer, np.floating]]],
                s2: Union[List, npt.NDArray[Union[np.integer, np.floating]]],
                dxy_limit: float = np.inf,
                beta: float = 1.,
                show_plot: bool = False,
                std_s1: float = None, std_s2: float = None,
                tavg_s1: float = None, tavg_s2: float = None,
                override_checks: bool = False) -> Tuple[float, bool]: ...

def mjc(s1: Union[List, npt.NDArray[Union[np.integer, np.floating]]],
        s2: Union[List, npt.NDArray[Union[np.integer, np.floating]]],
        dxy_limit: float = np.inf,
        beta: float = 1.,
        show_plot: bool = False,
        std_s1: float = None, std_s2: float = None,
        tavg_s1: float = None, tavg_s2: float = None,
        return_args: bool = False,
        override_checks: bool = False) -> Union[Tuple[float, bool],
                                                Tuple[float, bool, float, float, float, float,
                                                npt.NDArray[Union[np.integer, np.floating]],
                                                npt.NDArray[Union[np.integer, np.floating]]]]: ...

def jump_cost(x: npt.NDArray[Union[np.integer, np.floating]],
              y: npt.NDArray[Union[np.integer, np.floating]],
              dxy_limit: float,
              beta: float,
              std: float,
              tavg_x: float,
              tavg_y: float) -> Union[float, npt.NDArray[np.integer]]: ...

def cmin(x: npt.NDArray[Union[np.integer, np.floating]],
         idx_x: int,
         y:npt.NDArray[Union[np.integer, np.floating]],
         idx_y: int,
         n: int,
         phi: float,
         t_avg_x: float,
         t_avg_y: float) -> Tuple[float, int, int, int, int]: ...
