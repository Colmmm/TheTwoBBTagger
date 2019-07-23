import numpy as np
from skhep.math.vectors import Vector3D, LorentzVector
import pandas as pd
from fnmatch import filter
from numpy import array
from math import sqrt, atan2, cos, sin, acos, degrees, log, pi, sinh

def perp(v):
    """defining a function to find the perpendicular vector to our flight vector"""
    if v[0] == 0 and v[1] == 0:
        if v[2] == 0:
            raise ValueError('zero vector')
        else:
            return np.cross(v, [0, 1, 0])
    return np.cross(v, [0, 0, 1])

#simple function for calculating the magnitude of a vector
mag = lambda x: np.sqrt(x.dot(x))


def eta(Vector3D):
        """Return the pseudorapidity. for some reason, the libary I used, only
        allowed calculation of eta for lorentz vectors and not Vector3D as well, so I made my own"""
        if abs(Vector3D.costheta()) < 1.:
            return -0.5 * log( (1. - Vector3D.costheta())/(1. + Vector3D.costheta()) )
        else:
            return 10E10 if Vector3D.z > 0 else -10E10



def my_SetPtEtaPhiM(pt, eta, phi, m):
    """ Create a Lorentz 4-momentum vector defined from the transverse momentum, the pseudorapidity,
    the angle phi and the mass."""
    px, py, pz = pt * cos(phi), pt * sin(phi), pt * sinh(eta)

    self = LorentzVector()
    self.x = px;
    self.y = py;
    self.z = pz
    if m > 0.:
        self.t = sqrt(px ** 2 + py ** 2 + pz ** 2 + m ** 2)
    else:
        self.t = sqrt(px ** 2 + py ** 2 + pz ** 2 - m ** 2)

    return self

def boostvector(self):
    """Return the spatial component divided by the time component."""
    return Vector3D(self.x / self.t, self.y / self.t, self.z / self.t)


def setpxpypzm(px, py, pz, m):
    """Creates a Lorentz four momentum vector using the 3momentum and mass as inputs. I have to define this myself
    rather than use function in the skhep library as that one doesnt output the four vector but updates the
    parameters which messes up the .apply method"""
    self = LorentzVector()
    self.x = px;
    self.y = py;
    self.z = pz
    if m > 0.:
        self.t = sqrt(px ** 2 + py ** 2 + pz ** 2 + m ** 2)
    else:
        self.t = sqrt(px ** 2 + py ** 2 + pz ** 2 - m ** 2)

    return self


def track_four_momentum(x, three_momentum_labels, PID_labels):
    """This function calculates the lorentz four momentum of a given track. It uses the PID information (which are
    just probabilities and then interpreted by PID_classifier)  which determines the particle identity, we then use the
     known rest masses of whatever particle was identified along with the measured 3momentum to initialise the four momentum"""
    # defining a lookup table for mass hypothesis in the track_four_momentum function below
    mass_PID_lookup_table = pd.Series(data=[938.27, 493.677, 139.57, 0.511, 105.658], index=PID_labels)
    PID_probs = x.loc[PID_labels]
    PID_pred = [idx for idx in PID_labels if x[idx] == PID_probs.max()] #this will hopefully be changed with ML later
    if PID_probs.loc[filter(PID_probs.index, '*mu')[0]] >= 0.16: #as relative probability for muons are low
        PID_pred = filter(PID_probs.index, '*mu')
    p4_vector = setpxpypzm(x[three_momentum_labels[0]], x[three_momentum_labels[1]], x[three_momentum_labels[2]], mass_PID_lookup_table.loc[PID_pred][0])

    return p4_vector

def TOF_COM_calculation(df, B_M_nominal = 5279.5 ):
    """Main function to calculate COM values, input is df, and output is same
    df but with added columns for the COM values"""

    # set up flight related vectors
    df['pv_vector'] = df.apply(lambda x: array([x.TwoBody_OWNPV_X, x.TwoBody_OWNPV_Y, x.TwoBody_OWNPV_Z]), axis=1)
    df['sv_vector'] = df.apply(lambda x: array([x.TwoBody_ENDVERTEX_X, x.TwoBody_ENDVERTEX_Y, x.TwoBody_ENDVERTEX_Z]),
                      axis=1)
    df['flight'] = df.apply(lambda x: x.sv_vector - x.pv_vector, axis=1)
    df['tan_theta'] = df.apply(lambda x: mag(perp(x.flight) / x.flight[-1]), axis=1)

    # this is setting the 4momentum of the B meson from kinematic info from Two Body vertex
    df['p4B'] = df.apply(lambda x: LorentzVector(x.TwoBody_PX, x.TwoBody_PY, x.TwoBody_PZ, x.TwoBody_PE), axis=1)

    #Define the right parameters for the track_four_momentum function
    #the names of the columns of df containing the 3momentum of track1, track2 and the extra tracks
    track1_3p_labels = ['Track1_PX', 'Track1_PY', 'Track1_PZ']
    track2_3p_labels = ['Track2_PX', 'Track2_PY', 'Track2_PZ']
    extra_3p_labels = ['TwoBody_Extra_Px', 'TwoBody_Extra_Py', 'TwoBody_Extra_Pz']
    #the names of the columns containing the PID probabilities outputted by the NN, order makes a difference here (maybe)!!!!
    track1_PID_labels = ['Track1_ProbNNp', 'Track1_ProbNNk', 'Track1_ProbNNpi', 'Track1_ProbNNe', 'Track1_ProbNNmu']
    track2_PID_labels = ['Track2_ProbNNp', 'Track2_ProbNNk', 'Track2_ProbNNpi', 'Track2_ProbNNe', 'Track2_ProbNNmu']
    extra_PID_labels = ['TwoBody_Extra_NNp','TwoBody_Extra_NNk', 'TwoBody_Extra_NNpi', 'TwoBody_Extra_NNe', 'TwoBody_Extra_NNmu']
    #the models of the PID_classifiers
    PID_classifier = ['track1_PID_classifier.pkl', 'track2_PID_classifier.pkl', 'extra_PID_classifier.pkl' ]
    #Now calculate the four momentum of track1, track2 and the extra tracks
    #NB: In the C++ code, the variable track1_p4 is called p4Track1,etc
    df['track1_p4'] = df.apply(track_four_momentum, axis=1, args=(track1_3p_labels, track1_PID_labels))
    df['track2_p4'] = df.apply(track_four_momentum, axis=1, args=(track2_3p_labels, track2_PID_labels))
    df['extra_track_p4'] = df.apply(track_four_momentum, axis=1, args=(extra_3p_labels, extra_PID_labels))

    # PT estimate based on reconstructed mass and flight vector
    df['pt_est'] = df.apply(lambda x: (B_M_nominal / x.TwoBody_M) * x.tan_theta * x.TwoBody_PZ, axis=1)
    # calculating the eta and phi of the flight vector
    df['flight_eta'] = df.apply(lambda x: eta(Vector3D(x.flight[0], x.flight[1], x.flight[2]).unit()), axis=1)
    df['flight_phi'] = df.apply(lambda x: Vector3D(x.flight[0], x.flight[1], x.flight[2]).unit().phi(), axis=1)
    # estimated B candidate for this estimated momentum, measured flight direction and expected true B mass
    df['p4B_est'] = df.apply(lambda x: my_SetPtEtaPhiM(x.pt_est, x.flight_eta, x.flight_phi, B_M_nominal), axis=1)

    #estimating the boost needed to get to the B's rest frame
    df['boost_est'] = df.apply(lambda x: boostvector(x.p4B_est), axis=1)
    #boosting the tracks to the B's rest frame
    df['track1_p4_boosted'] = df.apply(lambda x: x.track1_p4.boost(x.boost_est), axis=1)
    df['track2_p4_boosted'] = df.apply(lambda x: x.track2_p4.boost(x.boost_est), axis=1)
    df['extra_track_p4_boosted'] = df.apply(lambda x: x.extra_track_p4.boost(x.boost_est), axis=1)

    #calculating the missing mass^2 - this can go negative with resolution
    df['mm2'] = df.apply(lambda x: (x.p4B_est - x.p4B).mass2, axis=1)
    #calculate the energy of all tracks in the B's rest frame, ie the COM energy
    df['Etrack1'] = df.apply(lambda x: x.track1_p4_boosted.t, axis=1)
    df['Etrack2'] = df.apply(lambda x: x.track2_p4_boosted.t, axis=1)
    df['Etrack_extra'] = df.apply(lambda x: x.extra_track_p4_boosted.t, axis=1)

    return df



