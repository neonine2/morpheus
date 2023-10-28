"""
This module defines the default metadata and data dictionaries.
Note that the "name" field is automatically populated upon initialization of the corresponding
Explainer class.
"""

# CFProto
DEFAULT_META_CFP = {"name": None,
                    "type": ["blackbox", "tensorflow", "keras"],
                    "explanations": ["local"],
                    "params": {},
                    "version": None}  # type: dict
"""
Default counterfactual prototype metadata.
"""

DEFAULT_DATA_CFP = {"cf": None,
                    "all": [],
                    "orig_class": None,
                    "orig_proba": None,
                    "id_proto": None
                    }  # type: dict
"""
Default counterfactual prototype metadata.
"""

