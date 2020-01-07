# -*- coding: utf-8 -*-
from pathlib import Path
import re
conversion_dict = {'title': 'str', 'min_step': 'float', 'foc': 'float',
                   'Cw': 'float', 'Cm': 'float', 'SAMPLE_START': 'int',
                   'SAMPLE_END': 'int', 'imgL': 'int', 'imgR': 'int',
                   'ymin': 'int', 'ymax': 'int'}


def load_conf(FOL):
    FOL = Path(FOL)
    path = FOL/'conf.txt'
    f = open(path, 'r')
    conf = f.read().split('\n')
    f.close()
    conf_dict = {}
    for c in conf:
        key =  re.search('(.*):', c).group(1)
        val = re.search('{(.*)}', c).group(1)
        if conversion_dict[key] == 'str':
            val = str(val)
        elif conversion_dict[key] == 'int':
            val = int(val)
        elif conversion_dict[key] == 'float':
            val = float(val)
        conf_dict[key] = val
    return conf_dict

#if __name__ == '__main__':
#    DATA_FOLDER: str = "FLAT50CM-PURE-60um"
#    directory_path: str = "C:/Users/indra/Documents/GitHub"
#    ARR_FOL = join(directory_path, DATA_FOLDER)
#    conf = load_conf(ARR_FOL)