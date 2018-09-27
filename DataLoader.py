# load OASIS data
from glob import glob
import os, re
import logging

def loadOASISData():
    possible_oasis_paths = [
    "/data/johann/datasets_prepared/OASIS1_*.nii.gz",
    "/data/Johann/johann_docker/datasets_prepared/OASIS1_*.nii.gz",
    "/mnt/hdd1/datasets_prepared/OASIS1_*.nii.gz",
    ]

    oasis_path = None
    for op in possible_oasis_paths:
        if len(list(glob(op))) > 0:
            logging.info("Found OASIS1 data at: {}".format(os.path.dirname(op)))
            # oasis path found!
            oasis_path = op
            break

    if oasis_path is None:
        raise Exception("No OASIS path found!")

    subjects = list(glob(oasis_path))
    def subjectProperties(subj_root):
        props = {}
        pattern = re.compile(r'OASIS1_(?P<id>[0-9]+).nii.gz$')
        props['id'] = pattern.findall(subj_root)[0]
        props['img'] = subj_root
        return props

    props = sorted([ subjectProperties(sb) for sb in subjects ],key=lambda obj:obj['id'])
    return props
