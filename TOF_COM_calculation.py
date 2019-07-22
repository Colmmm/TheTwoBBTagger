import numpy as np
from skhep.math.vectors import Vector3D, LorentzVector

def perp(v):
    """defining a function to find the perpendicular vector to our flight vector"""
    if v[0] == 0 and v[1] == 0:
        if v[2] == 0:
            raise ValueError('zero vector')
        else:
            return np.cross(v, [0, 1, 0])
    return np.cross(v, [0, 0, 1])


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

def track_four_momentum(x, three_momentum_labels, PID_labels, PID_classifier):
    """This function calculates the lorentz four momentum of a given track. It uses the PID information which is passed
    onto the PID_classifier which """

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


