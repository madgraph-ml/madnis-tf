""" Get interference datasets """

import wget
import numpy as np
import pandas as pd
import re
from dataclasses import dataclass, field
from xml.etree import ElementTree
from skhep.math import LorentzVector
import gzip
import os
from os.path import exists


@dataclass
class Particle:
    pdgid: int
    px: float
    py: float
    pz: float
    energy: float
    mass: float
    spin: float
    status: int
    vtau: float
    parent: int

    def p4(self):
        return LorentzVector(self.px, self.py, self.pz, self.energy)


@dataclass
class Event:
    particles: list = field(default_factory=list)
    weights: list = field(default_factory=list)
    scale: float = -1

    def add_particle(self, particle):
        self.particles.append(particle)


class LHEReader:
    def __init__(self, file_path, weight_mode="list", weight_regex=".*"):
        """
        Constructor.

        :param file_path: Path to input LHE file
        :type file_path: string
        :param weight_mode: Format to return weights as. Can be dict or list. If dict, weight IDs are used as keys.
        :type weight_mode: string
        :param weight_regex: Regular expression to select weights to be read. Defaults to reading all.
        :type weight_regex: string
        """
        self.file_path = file_path
        self.iterator = ElementTree.iterparse(self.file_path, events=("start", "end"))
        self.current = None
        self.current_weights = None

        assert weight_mode in ["list", "dict"]
        self.weight_mode = weight_mode
        self.weight_regex = re.compile(weight_regex)

    def unpack_from_iterator(self):
        # Read the lines for this event
        lines = self.current[1].text.strip().split("\n")

        # Create a new event
        event = Event()
        event.scale = float(lines[0].strip().split()[3])
        event.weights = float(lines[0].strip().split()[2])

        # Read header
        event_header = lines[0].strip()
        num_part = int(event_header.split()[0].strip())

        # Iterate over particle lines and push back
        for ipart in range(1, num_part + 1):
            part_data = lines[ipart].strip().split()
            p = Particle(
                pdgid=int(part_data[0]),
                status=int(part_data[1]),
                parent=int(part_data[2]) - 1,
                px=float(part_data[6]),
                py=float(part_data[7]),
                pz=float(part_data[8]),
                energy=float(part_data[9]),
                mass=float(part_data[10]),
                vtau=float(part_data[11]),
                spin=int(float(part_data[12])),
            )
            event.add_particle(p)

        return event

    def __iter__(self):
        return self

    def __next__(self):
        # Clear XML iterator
        if self.current:
            self.current[1].clear()

        # Find beginning of new event in XML
        element = next(self.iterator)
        while element[1].tag != "event":
            element = next(self.iterator)

        # Loop over tags in this event
        element = next(self.iterator)

        if self.weight_mode == "list":
            self.current_weights = []
        elif self.weight_mode == "dict":
            self.current_weights = {}

        while not (element[0] == "end" and element[1].tag == "event"):
            if element[0] == "end" and element[1].tag == "wgt":

                # If available, use "id" identifier as
                # 1. filter which events to read
                # 2. key for output dict
                weight_id = element[1].attrib.get("id", "")

                if self.weight_regex.match(weight_id):
                    value = float(element[1].text)
                    if self.weight_mode == "list":
                        self.current_weights.append(value)
                    elif self.weight_mode == "dict":
                        self.current_weights[weight_id] = value
            element = next(self.iterator)

        # Find end up this event in XML
        # use it to construct particles, etc
        while not (element[0] == "end" and element[1].tag == "event"):
            element = next(self.iterator)
        self.current = element

        return self.unpack_from_iterator()


def lhe_to_array(reader: LHEReader):
    """LHE to numpy array"""
    events = []
    weights = []
    for _, event in enumerate(reader):
        particles_out = filter(lambda x: x.status == 1, event.particles)
        momenta = []
        for particle in particles_out:
            mom = np.array([particle.energy, particle.px, particle.py, particle.pz])
            momenta.append(mom)
        momenta = np.hstack(momenta)
        events.append(momenta)
        weights.append(event.weights)
    events = np.stack(events)
    weights = np.stack(weights)[..., None]
    wgt_events = np.concatenate((events, weights), axis=1)
    return wgt_events


def array_to_hdf5(array: np.ndarray, header: list, name="."):
    """Numpy array to HDF5"""
    filename = name + ".h5"
    store = pd.HDFStore(filename)

    dataframe = pd.DataFrame(array, columns=header)
    store.append("data", dataframe)

    store.close()


if __name__ == "__main__":

    URLS = [
        "https://www.dropbox.com/s/923h4e199jbl3yt/events_G1_novegas.lhe.gz?dl=1",
        "https://www.dropbox.com/s/tv2fe1k4bqoc05l/events_G1.lhe.gz?dl=1",
        "https://www.dropbox.com/s/fic3l3jn4xpehfr/events_G2_no_vegas.lhe.gz?dl=1",
        "https://www.dropbox.com/s/4a713m3pexn3kj4/events_G2.lhe.gz?dl=1",
    ]

    NAMES = ["events_G1_novegas", "events_G1", "events_G2_no_vegas", "events_G2"]

    for url, name in zip(URLS, NAMES):
        if exists(f"{name}.h5"):
            print(f"Nothing to do. Dataset `{name}.h5`` already exists")
            continue
        print("Download....")
        wget.download(url, f"{name}.lhe.gz")
        print(f"\nOpen zipped file: {name}.lhe.gz")
        with gzip.open(f"{name}.lhe.gz", "rt") as f:
            print("Read Les Houches file...")
            red = LHEReader(f)
            data = lhe_to_array(red)
            headers = [
                "Eep",
                "pxep",
                "pyep",
                "pzep",
                "Eem",
                "pxem",
                "pyem",
                "pzem",
                "wgt",
            ]
            print("Save as hdf5 file...")
            array_to_hdf5(data, headers, name=name)
        print(f"Delete {name}.lhe.gz....")
        os.remove(f"{name}.lhe.gz")
