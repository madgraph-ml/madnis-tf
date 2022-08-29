import numpy as np


class LHEWriter:
    """
    This class creates a writer to output the events
    in the Les Houches Event file format.
    """

    def __init__(self, file):

        self.file = open(file, "w")

    def write_lhe(self, events, masses, pdg_codes):
        """
        This function takes as inputs:
        - events: 2D array containing events in rows. Each row is an array
                  of 4 momenta, one after the other.
        - masses: list with masses for the particles, in the same order
                  the momenta are given.
        - pdg_codes: list with pdg codes for the particles, in the same
                     order the momenta are given.

        Output:
        A lhe file containing all the events.
        """

        self.num_particles = events.shape[1] // 4

        if len(pdg_codes) != self.num_particles:
            raise Exception(
                "You have passed %i pdg codes for %i particles."
                % (len(pdg_codes), self.num_particles)
            )

        self.file.write('<LesHouchesEvents version="3.0">\n')

        self.write_intro()

        for event in events:
            event = event.reshape(self.num_particles, 4)

            self.write_event(event, masses, pdg_codes)

        self.file.write("</LesHouchesEvents>\n")
        self.file.close()

    def write_intro(self):

        self.file.write("<header>\n")

        header = """
#################################################################################
#                       THIS FILE HAS BEEN PRODUCED BY MADNIS                   #
#                                                                               #
#                                                                               #
#                                  Authors: xyz                                 #
#                                 reference, paper                              #
#                                                                               #
#                                                                               #
#################################################################################
#################################################################################
# The original Les Houches Event (LHE) format is defined in hep-ph/0609017      #
#################################################################################

"""

        self.file.write(header)
        self.file.write("</header>\n")

        # here there are supposed to be general information on the process
        self.file.write("<init>\n")
        self.file.write("</init>\n")

    def write_event(self, event, masses, pdg_codes):

        self.file.write("<event>\n")

        self.file.write(
            "%i   0  -1.0000000E+00 -1.0000000E+00 -1.0000000E+00 -1.0000000E+00\n"
            % (self.num_particles)
        )

        for momentum, mass, pdg in zip(event, masses, pdg_codes):

            self.write_particle(momentum, mass, pdg)

        self.file.write("</event>\n")

    def write_particle(self, momentum, mass, pdg):

        E, px, py, pz = momentum

        self.file.write(
            "      %2i    1    0    0    0    0  %13.6e  %13.6e  %13.6e  %13.6e  %13.6e  0.00000  0.00000\n"
            % (pdg, px, py, pz, E, mass)
        )
