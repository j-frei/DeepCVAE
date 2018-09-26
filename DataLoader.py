# load OASIS data
from glob import glob
import os, re
import socket

if socket.gethostname() == "JohannFrei-PC":
    oasis_path = "/mnt/hdd1/oasis1/out/OAS1_*_MR1"
elif socket.gethostname() == "mango2gpu":
    oasis_path = "/data/johann/OASIS1/OAS1_*_MR1"
elif socket.gethostname() == "09eae6835cdc":
    oasis_path = "/data/johann/OASIS1/OAS1_*_MR1"

else:
    oasis_path = "/data/Johann/johann_docker/OASIS1/OAS1_*_MR1"

def loadOASISData():
    subjects = list(glob(oasis_path))


    def subjectProperties(subj_root):
        props = {}
        pattern = re.compile(r'OAS1_(?P<id>[0-9]+)_MR1$')

        props['id'] = pattern.findall(subj_root)[0]
        txt_path = os.path.join(subj_root, "OAS1_{}_MR1.txt".format(props['id']))

        with open(txt_path, "r", encoding="utf8") as txtf:
            txt = txtf.read()

        props['age'] = int(re.compile(r'\nAGE:[\s]+(?P<age>[0-9]+)\n').findall(txt)[0])
        props['gender'] = re.compile(r'\nM/F:[\s]+(?P<gender>Female|Male)\n').findall(txt)[0]
        try:
            props['cdr'] = float(re.compile(r'\nCDR:[\s]+(?P<cdr>[0-9\.]+)\n').findall(txt)[0])
        except:
            props['cdr'] = 0.
        x = os.path.join(subj_root, "PROCESSED", "MPRAGE", "T88_111", "OAS1_*_MR1_mpr_n*_anon_111_t88_masked_gfc.hdr")

        props['img'] = glob(x)[0]
        return props

    props = sorted([ subjectProperties(sb) for sb in subjects ],key=lambda obj:obj['id'])
    return props
