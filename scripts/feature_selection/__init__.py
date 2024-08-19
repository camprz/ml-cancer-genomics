from .feature_selection import (select_vars, 
                                choose_variable_to_drop, 
                                corr_comparison, 
                                mutual_information)

__all__ = ["select_vars", 
           "choose_variable_to_drop", 
           "corr_comparison", 
           "mutual_information"]