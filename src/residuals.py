import numpy as np
import pickle
import matplotlib.pyplot as plt
from collections import OrderedDict


class residuals:
    radius = 35.44  # mm

    def __init__(self, mesh, start, end, key):
        self.mesh = mesh
        self.start = start
        self.end = end
        self.key = key

    @classmethod
    def from_pickle(cls, mesh_path, start_path, end_path, key):
        mesh_whole = pickle.load(open(mesh_path, "rb"))
        xM = mesh_whole["xMesh"]
        yM = mesh_whole["yMesh"]
        zM = mesh_whole["zMesh"]
        xG = mesh_whole["xGrid"]
        yG = mesh_whole["yGrid"]
        zG = mesh_whole["zGrid"]
        mesh = {
            "xMesh": xM,
            "yMesh": yM,
            "zMesh": zM,
            "xGrid": xG,
            "yGrid": yG,
            "zGrid": zG,
        }

        start = pickle.load(open(start_path, "rb"))[key]
        end = pickle.load(open(end_path, "rb"))[key]

        return cls(mesh, start, end, key)

    def calc_coordinates(self, radius=radius):
        "Calculate mesh coordinates from Lagranian frame"
        xMesh = self.mesh["xMesh"] + self.start[0] / 2 / radius
        yMesh = self.mesh["yMesh"] + self.start[1] / 2 / radius
        zMesh = self.mesh["zMesh"] + self.start[2] / 2 / radius
        return xMesh, yMesh, zMesh

    def calc_cylindrical_coordinates(self, radius=radius):
        xMesh, yMesh, zMesh = self.calc_coordinates()
        (rMesh, thetaMesh) = cart2pol(xMesh * 2 * radius, zMesh * 2 * radius)
        vMesh = yMesh * 2 * radius

        return rMesh, thetaMesh, vMesh

    def calc_residuals(self):
        u, v, w = self.calc_disp_whole()
        rMesh, thetaMesh, vMesh = self.calc_cylindrical_coordinates()

        # Calculate radius and tangential displacement fields
        rDisp = cal_drdt(u, w, thetaMesh)[0]
        phiDisp = cal_drdt(u, w, thetaMesh)[1]
        yDisp = v

        # averaged displacement field along the specimen height
        rDisp_avg = np.average(rDisp, axis=1)
        phiDisp_avg = np.average(phiDisp, axis=1)
        yDisp_avg = np.average(yDisp, axis=1)

        # residual displacement field
        rDisp_rsdl = (rDisp.T - rDisp_avg.T).T
        phiDisp_rsdl = (phiDisp.T - phiDisp_avg.T).T
        v_rsdl = (yDisp.T - yDisp_avg.T).T

        # Back to cartesian coordinates
        u_rsdl = cal_dudw(rDisp_rsdl, phiDisp_rsdl, thetaMesh)[0]
        w_rsdl = cal_dudw(rDisp_rsdl, phiDisp_rsdl, thetaMesh)[1]

        return u_rsdl, v_rsdl, w_rsdl

    def calc_disp_whole(self):
        xDisp = self.end[0] - self.start[0]
        yDisp = self.end[1] - self.start[1]
        zDisp = self.end[2] - self.start[2]
        return xDisp, yDisp, zDisp

    def save_residuals(self, filePath):
        residualDisp_all = OrderedDict()
        u_rsdl, v_rsdl, w_rsdl = self.calc_residuals()
        residualDisp_all.update({self.key: [u_rsdl, v_rsdl, w_rsdl]})

        with open(filePath, "wb") as fp:  # Pickling
            pickle.dump(residualDisp_all, fp)

    def calc_gradient(self, r=radius):
        u, v, w = self.calc_disp_whole()
        rMesh, thetaMesh, vMesh = self.calc_cylindrical_coordinates()

        # Calculate radius and tangential displacement fields
        rDisp = cal_drdt(u, w, thetaMesh)[0]
        phiDisp = cal_drdt(u, w, thetaMesh)[1]
        yDisp = v

        # Gradient tensor of vector field: Nine components
        dUr_r = rDisp
        dUr_phi = np.gradient(rDisp, axis=1)
        dUr_z = np.gradient(rDisp, axis=0)
        dUPhi_r = phiDisp
        dUPhi_phi = np.gradient(phiDisp, axis=1)
        dUPhi_z = np.gradient(phiDisp, axis=0)
        yDisp_avg = np.average(yDisp, axis=1)
        dUz_r = (yDisp.T - yDisp_avg.T).T
        dUz_phi = np.gradient(yDisp, axis=1)
        dUz_z = np.gradient(yDisp, axis=0)

        dr = rMesh
        dPhi = np.gradient(thetaMesh, axis=1)
        dz = np.gradient(vMesh, axis=0)

        return [
            dUr_r / dr,
            dUr_phi / dPhi / r - phiDisp / r,
            dUr_z / dz,
            dUPhi_r / dr,
            dUPhi_phi / dPhi / r + rDisp / r,
            dUPhi_z / dz,
            dUz_r / dr,
            dUz_phi / dPhi / r,
            dUz_z / dz,
        ]

    def divergence(self):
        gradient = self.calc_gradient()
        div = gradient[0] + gradient[4] + gradient[8]
        return div

    def vorticity(self):
        """curl around r axis"""
        gradient = self.calc_gradient()
        curl_r = gradient[7] - gradient[5]
        return curl_r


def cal_drdt(u, w, thetaMesh):
    """
    Calculate radius and tangential displacement, counter clockwise positive
    """
    dTheta = np.arctan2(w, u)
    dt = -np.sqrt(u**2 + w**2) * np.sin(thetaMesh - dTheta)
    dr = np.sqrt(u**2 + w**2) * np.cos(thetaMesh - dTheta)

    return [dr, dt]


def cal_dudw(dr, dt, thetaMesh):
    """
    Calculate horizontal and out-of-plane displacement
    """
    dTheta = np.arctan2(dt, dr)
    w = np.sqrt(dr**2 + dt**2) * np.sin(thetaMesh + dTheta)
    u = np.sqrt(dr**2 + dt**2) * np.cos(thetaMesh + dTheta)

    return [u, w]


def cart2pol(x, y):
    """
    Transform cartesian to polor coordinates
    """
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    return (rho, phi)


def pol2cart(rho, phi):
    """
    Transform polor to cartesian coordinates
    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)

    return (x, y)


def main():
    mesh_path = "./data/mesh_hiRes.txt"
    start_path = "./data/strain_5.0_extrap_hiRes.txt"
    end_path = "./data/strain_7.0_extrap_hiRes.txt"
    key = "Z40C_092903b-192"
    residual_path = "./data/" + key + "_residuals_5_7.txt"
    residual_5_7 = residuals.from_pickle(
        mesh_path=mesh_path,
        start_path=start_path,
        end_path=end_path,
        key=key,
    )
    residual_5_7.save_residuals(filePath=residual_path)


if __name__ == "__main__":
    main()
