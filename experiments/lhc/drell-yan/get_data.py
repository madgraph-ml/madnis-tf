""" Get interference datasets """

import wget

URLS = [
    "https://www.dropbox.com/s/t41yi2m9jt7ecm3/events_G1_novegas.lhe.gz?dl=1",
    "https://www.dropbox.com/s/8myo0llpbhfj6hr/events_G1.lhe.gz?dl=1",
    "https://www.dropbox.com/s/uz59z0xsm2lu462/events_G2_no_vegas.lhe.gz?dl=1",
    "https://www.dropbox.com/s/0j0mksuw5y2xt28/events_G2.lhe.gz?dl=1",
]

NAMES = ["events_G1_novegas.lhe.gz", "events_G1.lhe.gz", "events_G2_no_vegas.lhe.gz", "events_G2.lhe.gz"]

for url, name in zip(URLS, NAMES):
    wget.download(url, f"{name}")