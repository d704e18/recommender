import json
import os
import sys

import recommender

settings = {}

this_module = sys.modules[__name__]
this_module.settings = {}


def set_settings(config):
    with open(os.path.join(recommender.PROJECT_ROOT, 'config.json'), 'r') as f:
        this_module.settings = json.load(f)[config]
