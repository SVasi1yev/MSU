import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from astropy.cosmology import FlatLambdaCDM, z_at_value
from astropy.coordinates import SkyCoord
from astropy import units as u
from sklearn.neighbors import KDTree
from scipy.spatial import ConvexHull
import copy
from pathlib import Path
import json
import time

# TODO
# Настройка угла соединения филаментов в рассчете метрик
# Переменные указывающие было ли выполненно некоторое действие
# Перевод из декартовых в сферические
# Куда сохранять случайные кластера?
# plt.hline() в visual_metrics
# доделать генерацию данных для сегметации


def dist(x1, y1, z1, x2, y2, z2):
    """
    Расстояние между двумя точками
    """
    return (
        (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2)
    ) ** 0.5


def intersec_line_sphere(x1, y1, z1, x2, y2, z2, x3, y3, z3, r):
    """
    Проверяет пересекает ли отрезок (x1, y1, z1), (x2, y2, z2)
    шар с центром в точке (x3, y3, z3) и радиусом r
    """
    a = (x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2
    b = 2 * ((x2 - x1) * (x1 - x3) + (y2 - y1) * (y1 - y3) + (z2 - z1) * (z1 - z3))
    c = (
        x3 ** 2
        + y3 ** 2
        + z3 ** 2
        + x1 ** 2
        + y1 ** 2
        + z1 ** 2
        - 2 * (x3 * x1 + y3 * y1 + z3 * z1)
        - r ** 2
    )
    d = b ** 2 - 4 * a * c
    if a == 0:
        if dist(x1, y1, z1, x3, y3, z3) <= r:
            return True
        else:
            return False
    if d < 0:
        return False
    if d == 0:
        u_ = -b / (2 * a)
        if 0 <= u_ <= 1:
            return True
        else:
            return False
    if d > 0:
        u1 = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
        u2 = (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
        if (0 <= u1 <= 1) or (0 <= u2 <= 1):
            return True
        if (u1 <= 0) and (u2 >= 1) or (u2 <= 0) and (u1 >= 1):
            return True
        return False
    raise Exception("intersec_line_sphere_ERROR")


def dot(x1, y1, z1, x2, y2, z2):
    """
    Скалярное произведение
    """
    return x1 * x2 + y1 * y2 + z1 * z2


def dist_point_line(x1, y1, z1, x2, y2, z2, x3, y3, z3):
    """
    Расстояние от точки до прямой
    """
    p = np.array([x1, y1, z1])
    q = np.array([x2, y2, z2])
    r = np.array([x3, y3, z3])

    def t(p, q, r):
        x = p - q
        return np.dot(r - q, x) / np.dot(x, x)

    return np.linalg.norm(t(p, q, r) * (p - q) + q - r)


def dist_point_section(x1, y1, z1, x2, y2, z2, x3, y3, z3):
    """
    Расстояние от точки до отрезка
    """
    v = [x2 - x1, y2 - y1, z2 - z1]
    w0 = [x3 - x1, y3 - y1, z3 - z1]
    w1 = [x3 - x2, y3 - y2, z3 - z2]
    if np.dot(w0, v) <= 0:
        return dist(x3, y3, z3, x1, y1, z1)
    elif np.dot(w1, v) >= 0:
        return dist(x3, y3, z3, x2, y2, z2)
    else:
        return dist_point_line(x1, y1, z1, x2, y2, z2, x3, y3, z3)


class Disperse3D:
    RANDOM_CLUSTERS_NUM = 5
    random_clusters = None

    @classmethod
    def metrics_visual(cls, metric, metrics, mode="overall", sigma=None, smooth=None):
        """
        Визуализация метрик построенной карты филаментов
        """

        font = {"size": 16}
        plt.rc("font", **font)
        coefs = metrics["rads"]
        fig = plt.figure(figsize=(18, 10))
        rads = metrics["rads"]
        if mode == "overall":
            i = 0
            for sigma in metrics["sigmas"]:
                for smooth in metrics["smooths"]:
                    if i % 3 == 0:
                        linestyle = "-"
                        linewidth = 2
                    if i % 3 == 1:
                        linestyle = "--"
                        linewidth = 2
                    if i % 3 == 2:
                        linestyle = ":"
                        linewidth = 4
                    m = metrics[str(sigma)][str(smooth)][metric]
                    plt.plot(
                        rads,
                        m,
                        linestyle=linestyle,
                        linewidth=linewidth,
                        label=f"{metric}_SIGMA={sigma}_SMOOTH={smooth}",
                    )

            plt.hlines(1, min(rads), max(rads), color="r")
            plt.grid()
            plt.xticks(coefs)

            plt.xlabel(metrics["mode"])
            plt.ylabel(metric)
            plt.legend()
            plt.title(metric)
        else:
            plt.plot(
                rads, metrics[sigma][smooth]["true_" + metric], label="true_" + metric
            )
            plt.plot(
                rads, metrics[sigma][smooth]["false_" + metric], label="false_" + metric
            )
            plt.plot(
                rads, metrics[sigma][smooth]["diff_" + metric], label="diff_" + metric
            )

            plt.grid()
            plt.xticks(coefs)
            # plt.yticks(np.arange(0.0, 1.1, 0.1)) # TODO
            plt.xlabel(metrics["mode"])
            plt.ylabel(metric)
            plt.legend()
            plt.title(f"true/false/diff {metric}: SIGMA={sigma}, SMOOTH={smooth}")

    @classmethod
    def read(cls, dir_path):
        """
        Чтение сохраненной карты филаментов
        """
        if not dir_path.endswith(os.path.sep):
            dir_path = dir_path + os.path.sep
        galaxies = pd.read_csv(dir_path + "galaxies.csv")
        if os.path.exists(dir_path + "clusters.csv"):
            clusters = pd.read_csv(dir_path + "clusters.csv")
        else:
            clusters = None

        with open(dir_path + "meta.json") as f:
            meta = json.load(f)

        with open(dir_path + "dsp_res.json") as f:
            dsp_res = json.load(f)

        DPS = cls(
            galaxies,
            meta["disperse_path"],
            meta["cosmo_H0"],
            meta["cosmo_Om"],
            meta["cosmo_Ol"],
            meta["cosmo_Ok"],
            clusters,
            meta["sph2cart_f"],
            meta["cart2sph_f"],
        )

        DPS.CX_int = meta["CX_int"]
        DPS.CY_int = meta["CX_int"]
        DPS.CZ_int = meta["CX_int"]
        DPS.cart_coords = meta["cart_coords"]
        DPS.disperse_sigma = meta["disperse_sigma"]
        DPS.disperse_smooth = meta["disperse_smooth"]
        DPS.disperse_board = meta["disperse_board"]
        DPS.disprese_asmb_angle = meta["disperse_asmb_angle"]

        DPS.cps = dsp_res["cps"]
        DPS.fils = dsp_res["fils"]
        DPS.maxs = dsp_res["maxs"]

        return DPS

    def __init__(
        self,
        galaxies,
        disperse_path,
        cosmo_H0,
        cosmo_Om,
        cosmo_Ol,
        cosmo_Ok,
        clusters=None,
        sph2cart_f="dist",
        cart2sph_f="dist",
    ):
        """
        galaxies и cluster должны иметь поля RA, DEC, Z
        galaxies - DataFrame со сферическими координатами галактик
        disperse_path - путь до исполняемых файлов пакета disperse
        cosmo_H0, cosmo_Om, cosmo_Ol, cosmo_Ok - космологические параметры
        sph2cart_f, cart2sph_f - функции перевода координат
        clusters - DataFrame со сферическими координатами скоплений для рассчета метрик
        """

        self.disperse_path = disperse_path
        if self.disperse_path[-1] != "/":
            self.disperse_path += "/"
        self.galaxies = galaxies.copy()
        if clusters is not None:
            self.clusters = clusters.copy()
        else:
            self.clusters = None
        self.ra_int = (self.galaxies["RA"].min(), self.galaxies["RA"].max())
        self.dec_int = (self.galaxies["DEC"].min(), self.galaxies["DEC"].max())
        self.z_int = (self.galaxies["Z"].min(), self.galaxies["Z"].max())
        self.CX_int = None
        self.CY_int = None
        self.CZ_int = None
        self.cosmo_H0 = cosmo_H0 / 100
        self.cosmo_Om = cosmo_Om
        self.cosmo_Ol = cosmo_Ol
        self.cosmo_Ok = cosmo_Ok
        self.cosmo = FlatLambdaCDM(H0=cosmo_H0, Om0=cosmo_Om)
        self.COORDS_IN = f"{id(self)}_coords_ascii.txt"
        self.DISPERSE_IN = f"{id(self)}_galaxies_ascii.txt"

        self.sph2cart_f = sph2cart_f
        if sph2cart_f == "min":
            self.sph2cart = self.sph2cart_DTFE_MIN
        elif sph2cart_f == "dist":
            self.sph2cart = self.sph2cart_DIST
        elif sph2cart_f == "astropy":
            self.sph2cart = self.sph2cart_ASTROPY
        else:
            print("WRONG shp2cart_f value")
            self.sph2cart = self.sph2cart_DIST

        self.cart2sph_f = cart2sph_f
        if cart2sph_f == "dist":
            self.cart2sph = self.cart2sph_DIST
        elif cart2sph_f == "astropy":
            self.cart2sph = self.cart2sph_ASTROPY
        else:
            print("WRONG cart2sph_f value")
            self.cart2sph = self.cart2sph_DIST

        self.cart_coords = False

        self.disperse_sigma = None
        self.disperse_smooth = None
        self.disperse_board = None
        self.disprese_asmb_angle = None

        self.cps = None
        self.fils = None
        self.maxs = None

    def save(self, dir_path):
        """
        Сохранение построенной карты филаментов
        """
        if not dir_path.endswith(os.path.sep):
            dir_path = dir_path + os.path.sep
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        self.galaxies.to_csv(dir_path + "galaxies.csv", index=False)
        if self.clusters is not None:
            self.clusters.to_csv(dir_path + "clusters.csv", index=False)

        dsp_res = {
            "cps": self.cps,
            "fils": self.fils,
            "maxs": self.maxs,
        }
        with open(dir_path + "dsp_res.json", "w") as f:
            json.dump(dsp_res, f)

        meta = {
            "disperse_path": self.disperse_path,
            "CX_int": self.CX_int,
            "CY_int": self.CY_int,
            "CZ_int": self.CZ_int,
            "cosmo_H0": self.cosmo_H0 * 100,
            "cosmo_Ol": self.cosmo_Ol,
            "cosmo_Om": self.cosmo_Om,
            "cosmo_Ok": self.cosmo_Ok,
            "sph2cart_f": self.sph2cart_f,
            "cart2sph_f": self.cart2sph_f,
            "cart_coords": self.cart_coords,
            "disperse_sigma": self.disperse_sigma,
            "disperse_smooth": self.disperse_smooth,
            "disperse_board": self.disperse_board,
            "disperse_asmb_angle": self.disprese_asmb_angle,
        }
        with open(dir_path + "meta.json", "w") as f:
            json.dump(meta, f)

    def sph2cart_DTFE_MIN(self, ra, dec, z):
        """
        Преобразование сферических координат в декартовы вызовом
        команды delaunay_3D с параметром -minimal
        """

        uniq_a = []
        uniq_d = {}
        for i in range(len(ra)):
            t = (ra[i], dec[i], z[i])
            if t not in uniq_d:
                uniq_d[t] = None
                uniq_a.append(t)
        with open(f"{self.COORDS_IN}", "w") as f:
            f.write("# ra dec z\n")
            for i in range(len(uniq_a)):
                f.write(f"{uniq_a[i][0]}\t{uniq_a[i][1]}\t{uniq_a[i][2]}\n")
        os.system(
            (
                f"{self.disperse_path}delaunay_3D {self.COORDS_IN} "
                f"-btype void  -minimal "
                f"-cosmo {self.cosmo_Om} {self.cosmo_Ol} {self.cosmo_Ok} {self.cosmo_H0} {-1.0}"
            )
        )
        os.system(f"{self.disperse_path}netconv {self.COORDS_IN}.NDnet -to NDnet_ascii")

        CX, CY, CZ = [], [], []
        with open(f"{self.COORDS_IN}.NDnet.a.NDnet", "r") as f:
            for i in range(4):
                f.readline()
            n = int(f.readline())
            if n != len(uniq_a):
                print("ERROR!")
                return
            for i in range(n):
                cx, cy, cz = tuple(map(float, f.readline().split()))
                uniq_d[uniq_a[i]] = (cx, cy, cz)

        for i in range(len(ra)):
            t = (ra[i], dec[i], z[i])
            t = uniq_d[t]
            CX.append(t[0])
            CY.append(t[1])
            CZ.append(t[2])

        os.system(f"rm {self.COORDS_IN}*")

        return CX, CY, CZ

    def sph2cart_DIST(self, ra, dec, z):
        """
        Преобразование сферичесих координат в декартовы
        с использованием соответствующих функций из пакета disperse
        """

        with open(self.COORDS_IN, "w") as f:
            for i in range(len(z)):
                f.write(f"{z[i]}\n")
        os.system(
            (
                f"{self.disperse_path}my_dist "
                f"{self.cosmo_Om} {self.cosmo_Ol} 0.0 {self.cosmo_H0} "
                f"{self.COORDS_IN} out_{self.COORDS_IN} s"
            )
        )
        dist = []
        with open(f"out_{self.COORDS_IN}", "r") as f:
            for line in f:
                dist.append(float(line))
        os.system(f"rm {self.COORDS_IN} out_{self.COORDS_IN}")

        CX, CY, CZ = [], [], []
        for i in range(len(ra)):
            x = dist[i] * np.cos(ra[i] * np.pi / 180) * np.cos(dec[i] * np.pi / 180)
            y = dist[i] * np.sin(ra[i] * np.pi / 180) * np.cos(dec[i] * np.pi / 180)
            z = dist[i] * np.sin(dec[i] * np.pi / 180)
            CX.append(x)
            CY.append(y)
            CZ.append(z)

        return CX, CY, CZ

    def cart2sph_DIST(self, CX, CY, CZ):
        """
        Преобразование декартовых координат в сферические
        с использованием соответствующих функций из пакета disperse
        """

        ra, dec, dist = [], [], []
        for i in range(len(CX)):
            if CX[i] == 0 and CY[i] > 0:
                ra.append(90)
            elif CX[i] == 0 and CY[i] < 0:
                ra.append(270)
            elif CX[i] == 0 and CY[i] == 0:
                ra.append(0)
            else:
                ra.append(int(CX[i] < 0) * 180 + np.arctan(CY[i] / CX[i]) * 180 / np.pi)
            dec.append(
                90 - np.arctan((CX[i] ** 2 + CY[i] ** 2) ** 0.5 / CZ[i]) * 180 / np.pi
            )
            dist.append((CX[i] ** 2 + CY[i] ** 2 + CZ[i] ** 2) ** 0.5)

        with open(self.COORDS_IN, "w") as f:
            for i in range(len(dist)):
                f.write(f"{dist[i]}\n")
        os.system(
            (
                f"{self.disperse_path}my_dist "
                f"{self.cosmo_Om} {self.cosmo_Ol} 0.0 {self.cosmo_H0} "
                f"{self.COORDS_IN} out_{self.COORDS_IN} c"
            )
        )
        z = []
        with open(f"out_{self.COORDS_IN}", "r") as f:
            for line in f:
                z.append(float(line))
        os.system(f"rm {self.COORDS_IN} out_{self.COORDS_IN}")

        return ra, dec, z

    def sph2cart_ASTROPY(self, ra, dec, z):
        """
        Преобразование сферичесих координат в декартовы
        с использованием пакета astropy
        """

        CX, CY, CZ = [], [], []
        for i in range(len(ra)):
            c = SkyCoord(
                ra=ra[i],
                dec=dec[i],
                distance=self.cosmo.comoving_distance(z[i]),
                frame="fk5",
            )
            c.representation_type = "cartesian"
            CX.append(c.x.value)
            CY.append(c.y.value)
            CZ.append(c.z.value)

        return CX, CY, CZ

    def cart2sph_ASTROPY(self, CX, CY, CZ):
        """
        Преобразование декартовых кооin_cart_coordsрдинат в сферические
        с использованием пакета astropy
        """

        ra, dec, z = [], [], []
        for i in range(len(CX)):
            c = SkyCoord(
                x=CX[i],
                y=CY[i],
                z=CZ[i],
                frame="fk5",
                unit="Mpc",
                representation_type="cartesian",
            )
            c.representation_type = "spherical"
            ra.append(c.ra.value)
            dec.append(c.dec.value)
            z.append(z_at_value(self.cosmo.comoving_distance, c.distance))

        return ra, dec, z

    def count_cart_coords(self):
        """
        Вычисление декартовых координат по сферическим
        в данных о галактиках и кластерах
        """

        CX, CY, CZ = self.sph2cart(
            self.galaxies["RA"], self.galaxies["DEC"], self.galaxies["Z"]
        )

        self.galaxies = self.galaxies.assign(CX=CX)
        self.galaxies = self.galaxies.assign(CY=CY)
        self.galaxies = self.galaxies.assign(CZ=CZ)

        self.CX_int = (self.galaxies["CX"].min(), self.galaxies["CX"].max())
        self.CY_int = (self.galaxies["CY"].min(), self.galaxies["CY"].max())
        self.CZ_int = (self.galaxies["CZ"].min(), self.galaxies["CZ"].max())

        if self.clusters is not None:
            CX, CY, CZ = self.sph2cart(
                self.clusters["RA"], self.clusters["DEC"], self.clusters["Z"]
            )
            self.clusters = self.clusters.assign(CX=CX)
            self.clusters = self.clusters.assign(CY=CY)
            self.clusters = self.clusters.assign(CZ=CZ)

        self.cart_coords = True

    def count_sph_coords(self):
        """
        Вычисление сферических координат по декартовым
        в данных о галактиках и кластерах
        """

        ra, dec, z = self.cart2sph(
            self.galaxies["CX"], self.galaxies["CY"], self.galaxies["CZ"]
        )
        self.galaxies = self.galaxies.assign(RA=ra)
        self.galaxies = self.galaxies.assign(DEC=dec)
        self.galaxies = self.galaxies.assign(Z=z)

        if self.clusters is not None:
            ra, dec, z = self.cart2sph(
                self.clusters["CX"], self.clusters["CY"], self.clusters["CZ"]
            )
            self.clusters = self.clusters.assign(RA=ra)
            self.clusters = self.clusters.assign(DEC=dec)
            self.clusters = self.clusters.assign(Z=z)

    def apply_disperse(
        self,
        disperse_sigma,
        disperse_smooth,
        disperse_board="smooth",
        disprese_asmb_angle=30,
        in_cart_coords=True,
    ):
        """
        Применение алгоритма disperse к self.galaxies. Построение критических точек и филаментов.
        disperse_sigma - погор персистентности
        disperse_smooth - число сглаживаний dtfe
        disperse_board - способо дополнения данных по границам
        disprese_asmb_angle - минимальный угол между филаментами для соединения
        in_cart_coords - передавать ли декартовы координаты в disperse
        """

        self.fils = None

        self.disperse_sigma = (
            disperse_sigma
            if disperse_sigma != int(disperse_sigma)
            else int(disperse_sigma)
        )
        self.disperse_smooth = disperse_smooth
        self.disperse_board = disperse_board
        self.disprese_asmb_angle = disprese_asmb_angle

        if in_cart_coords:
            with open(self.DISPERSE_IN, "w") as f:
                f.write("# px py pz\n")
                for i in range(self.galaxies.shape[0]):
                    t = self.galaxies.iloc[i]
                    f.write(f"{t.CX}\t{t.CY}\t{t.CZ}\n")
        else:
            with open(self.DISPERSE_IN, "w") as f:
                f.write("# ra dec z\n")
                for i in range(self.galaxies.shape[0]):
                    t = self.galaxies.iloc[i]
                    f.write(f"{t.RA}\t{t.DEC}\t{t.Z}\n")

        print(">>> delaunay_3D starts")
        os.system(
            (
                f"{self.disperse_path}delaunay_3D {self.DISPERSE_IN} "
                f"-btype {self.disperse_board} -smooth {self.disperse_smooth} "
                f"-cosmo {self.cosmo_Om} {self.cosmo_Ol} {self.cosmo_Ok} {self.cosmo_H0} {-1.0}"
            )
        )

        print(">>> mse starts")
        os.system(
            (
                f"{self.disperse_path}mse {self.DISPERSE_IN}.NDnet "
                f"-upSkl -forceLoops -nsig {self.disperse_sigma}"
            )
        )

        print(">>> skelconv starts")
        os.system(
            (
                f"{self.disperse_path}skelconv {self.DISPERSE_IN}.NDnet_s{self.disperse_sigma}.up.NDskl "
                f"-breakdown -to NDskl_ascii -toRaDecZ "
                # f'-assemble 0 {self.disprese_asmb_angle} '
                f"-cosmo {self.cosmo_Om} {self.cosmo_Ol} {self.cosmo_Ok} {self.cosmo_H0} {-1.0}"
            )
        )
        # {f"-assemble 0 {self.disprese_asmb_angle}"}

        print(">>> read_skl_ascii_RaDecZ starts")
        self.read_skl_ascii_RaDecZ(
            f"{self.DISPERSE_IN}.NDnet_s{self.disperse_sigma}.up.NDskl.BRK.RaDecZ.a.NDskl"
        )

        os.system(f"rm {self.DISPERSE_IN}* test_smooth.dat")

    def read_skl_ascii_RaDecZ(self, file_name):
        """
        Чтение результатов применений пакета disperse
        """
        self.cps = []
        self.fils = []
        with open(file_name) as f:
            s = ""
            while s != "[CRITICAL POINTS]":
                s = f.readline().strip()
            cp_num = int(f.readline().strip())
            ras = []
            decs = []
            zs = []
            types = []
            values = []
            for i in range(cp_num):
                type_, ra, dec, z, value, _, _ = tuple(map(float, f.readline().split()))
                type_ = int(type_)
                ras.append(ra)
                decs.append(dec)
                zs.append(z)
                types.append(type_)
                values.append(value)
                for j in range(int(f.readline())):
                    f.readline()
            cx, cy, cz = self.sph2cart(ras, decs, zs)
            for i in range(cp_num):
                self.cps.append(
                    {
                        "RA": ras[i],
                        "DEC": decs[i],
                        "Z": zs[i],
                        "CX": cx[i],
                        "CY": cy[i],
                        "CZ": cz[i],
                        "type": types[i],
                        "value": values[i],
                    }
                )

            while s != "[FILAMENTS]":
                s = f.readline().strip()
            fil_num = int(f.readline())
            ras = []
            decs = []
            zs = []
            for i in range(fil_num):
                fil = {}
                cp1, cp2, sp_num = tuple(map(int, f.readline().split()))
                fil["CP1_id"] = cp1
                fil["CP2_id"] = cp2
                fil["sp_num"] = sp_num
                fil["sample_points"] = []
                for j in range(sp_num):
                    ra, dec, z = tuple(map(float, f.readline().split()))
                    ras.append(ra)
                    decs.append(dec)
                    zs.append(z)
                self.fils.append(fil)
            cx, cy, cz = self.sph2cart(ras, decs, zs)
            k = 0
            for i in range(fil_num):
                for j in range(self.fils[i]["sp_num"]):
                    self.fils[i]["sample_points"].append(
                        {
                            "RA": ras[k + j],
                            "DEC": decs[k + j],
                            "Z": zs[k + j],
                            "CX": cx[k + j],
                            "CY": cy[k + j],
                            "CZ": cz[k + j],
                        }
                    )
                k += self.fils[i]["sp_num"]

        self.maxs = []
        for cp in self.cps:
            if cp["type"] == 3:
                self.maxs.append(cp.copy())
        self.maxs = sorted(self.maxs, key=lambda x: -x["value"])

    def count_conn(self, cl_conn_rads, clusters=None):
        """
        Для всех филаментов рассчитывается числа пересекаемых скоплений,
        для каждого скоплений число пересекших его филаментов
        cl_conn_rads: радиуса скоплений
        clusters: DataFrame со скоплениями, если None, то используется self.clusters
        :return: cl_conn, fils_conn
        """

        if clusters is None:
            clusters = self.clusters

        MIN_SEG_LEN = 1  # Mpc

        points = []
        next_point = []
        fil_num = []
        count = 0
        for i, fil in enumerate(self.fils):
            sp = fil["sample_points"]
            for j in range(len(sp) - 1):
                points.append([sp[j]["CX"], sp[j]["CY"], sp[j]["CZ"]])
                fil_num.append(i)
                count += 1
                next_point.append(count)
                d = dist(
                    sp[j]["CX"],
                    sp[j]["CY"],
                    sp[j]["CZ"],
                    sp[j + 1]["CX"],
                    sp[j + 1]["CY"],
                    sp[j + 1]["CZ"],
                )
                if d > MIN_SEG_LEN:
                    n = int(d // MIN_SEG_LEN + 1)
                    d_x = sp[j + 1]["CX"] - sp[j]["CX"]
                    d_y = sp[j + 1]["CY"] - sp[j]["CY"]
                    d_z = sp[j + 1]["CZ"] - sp[j]["CZ"]
                    for k in range(1, n):
                        points.append(
                            [
                                sp[j]["CX"] + k * d_x / n,
                                sp[j]["CY"] + k * d_y / n,
                                sp[j]["CZ"] + k * d_z / n,
                            ]
                        )
                        fil_num.append(i)
                        count += 1
                        next_point.append(count)
            points.append([sp[-1]["CX"], sp[-1]["CY"], sp[-1]["CZ"]])
            fil_num.append(i)
            count += 1
            next_point.append(None)

        kd_tree = KDTree(points, leaf_size=10)

        cl_conn = [0] * clusters.shape[0]
        fils_conn = [0] * len(self.fils)
        cl_min_dist = [0] * clusters.shape[0]

        CX = clusters["CX"]
        CY = clusters["CY"]
        CZ = clusters["CZ"]

        for i in range(clusters.shape[0]):
            x3 = CX[i]
            y3 = CY[i]
            z3 = CZ[i]

            # r_fil = cl_conn_rads[i]
            # proc_fils = set()

            # close_points_idx = kd_tree.query_radius([[x3, y3, z3]], r=r_fil + MIN_SEG_LEN + 1)
            # for p_idx in close_points_idx[0]:
            #     if fil_num[p_idx] in proc_fils:
            #         continue
            #     if next_point[p_idx] is None:
            #         continue
            #     x1, y1, z1 = tuple(points[p_idx])
            #     x2, y2, z2 = tuple(points[next_point[p_idx]])
            #     if intersec_line_sphere(
            #         x1, y1, z1,
            #         x2, y2, z2,
            #         x3, y3, z3,
            #         r_fil
            #     ):
            #         cl_conn[i] += 1
            #         fils_conn[fil_num[p_idx]] += 1
            #         proc_fils.add(fil_num[p_idx])

            close_points_idx = kd_tree.query([[x3, y3, z3]], k=1, return_distance=False)
            min_dist = 1e10
            for p_idx in close_points_idx[0]:
                if next_point[p_idx] is None:
                    continue
                x1, y1, z1 = tuple(points[p_idx])
                x2, y2, z2 = tuple(points[next_point[p_idx]])
                d = dist_point_section(x1, y1, z1, x2, y2, z2, x3, y3, z3)
                if d < min_dist:
                    min_dist = d
            cl_min_dist[i] = min_dist

        return cl_conn, fils_conn, cl_min_dist

    def gen_random_clusters(self, clusters=None):
        """
        Генерация ложных кластеров
        """
        if clusters is None:
            clusters = self.clusters
        if (
            Disperse3D.random_clusters is None
            or Disperse3D.random_clusters[0].shape[0] != clusters.shape[0]
        ):
            print(">>> Generate random clusters")
            RA_int = (self.clusters["RA"].min(), self.clusters["RA"].max())
            DEC_int = (self.clusters["DEC"].min(), self.clusters["DEC"].max())
            Z_int = (
                self.clusters["Z"].min(),
                np.quantile(self.clusters["Z"], [0.97])[0],
            )
            CX_int = (self.clusters["CX"].min(), self.clusters["CX"].max())
            CY_int = (self.clusters["CY"].min(), self.clusters["CY"].max())
            CZ_int = (self.clusters["CZ"].min(), self.clusters["CZ"].max())

            # points = np.array(self.galaxies[['CX', 'CY', 'CZ']])
            # hull = ConvexHull(points)
            # A, b = hull.equations[:, :-1], hull.equations[:, -1:]
            # EPS = -5
            #
            # def contained(x):
            #     return np.all(np.asarray(x) @ A.T + b.T < EPS, axis=1)

            np.random.seed(0)

            Disperse3D.random_clusters = []
            for i in range(Disperse3D.RANDOM_CLUSTERS_NUM):
                # CX, CY, CZ = [], [], []
                RA, DEC, Z = [], [], []
                # for j in tqdm(range(clusters.shape[0])):
                # fl = False
                # while not fl:
                #     cx = np.random.uniform(CX_int[0], CX_int[1], 1)[0]
                #     cy = np.random.uniform(CY_int[0], CY_int[1], 1)[0]
                #     cz = np.random.uniform(CZ_int[0], CZ_int[1], 1)[0]
                #     ra, dec, z = self.cart2sph([cx], [cy], [cz])
                #     # print(ra[0], dec[0], z[0])
                #     fl = RA_int[0] <= ra[0] <= RA_int[1] \
                #         and DEC_int[0] <= dec[0] <= DEC_int[1] \
                #         and Z_int[0] <= z[0] <= Z_int[1]
                #     # fl = contained([[cx, cy, cz]])
                # CX.append(cx)
                # CY.append(cy)
                # CZ.append(cz)
                RA = np.random.uniform(RA_int[0], RA_int[1], clusters.shape[0])
                DEC = np.random.uniform(DEC_int[0], DEC_int[1], clusters.shape[0])
                Z = np.random.uniform(Z_int[0], Z_int[1], clusters.shape[0])
                df = pd.DataFrame()
                # df = df.assign(CX=CX)
                # df = df.assign(CY=CY)
                # df = df.assign(CZ=CZ)
                # RA, DEC, Z = self.cart2sph(CX, CY, CZ)
                df = df.assign(RA=RA)
                df = df.assign(DEC=DEC)
                df = df.assign(Z=Z)
                CX, CY, CZ = self.sph2cart(RA, DEC, Z)
                df = df.assign(CX=CX)
                df = df.assign(CY=CY)
                df = df.assign(CZ=CZ)
                df = df.assign(R=self.clusters["R"])
                Disperse3D.random_clusters.append(df)

    def count_metrics(self, mode, rads, clusters=None):
        """
        Вычисление метрик

        :param mode: (coefs, mpcs) радиусы скоплений в R200 или Mpc
        :param rads: набор радиусов для рассчета метрик
        :param clusters: DataFrame со скоплениями, если None, то используется self.clusters
        :return: None. Метрики сохраняются в self.metrics
        """

        if clusters is None:
            clusters = self.clusters
        if self.fils is None:
            print("DisPerSe wasn't computed")
            return
        self.gen_random_clusters(clusters)

        metrics = {}
        metrics["sigma"] = self.disperse_sigma
        metrics["smooth"] = self.disperse_smooth
        metrics["angle"] = self.disprese_asmb_angle
        metrics["mode"] = mode
        metrics["rads"] = rads

        cl_num = clusters.shape[0]
        fils_num = len(self.fils)

        true_cl_inter = []
        true_fils_inter = []
        true_cl_conns = []
        for rad in tqdm(rads):
            if mode == "coefs":
                cl_conn, fils_conn, _ = self.count_conn(
                    self.clusters["R"] * rad, clusters
                )
            else:
                cl_conn, fils_conn, _ = self.count_conn(
                    [rad] * self.clusters.shape[0], clusters
                )
            true_cl_inter.append(sum(list(map(lambda x: int(x > 0), cl_conn))))
            true_fils_inter.append(sum(list(map(lambda x: int(x > 0), fils_conn))))
            true_cl_conns.append(sum(cl_conn))
        true_cl_inter = np.array(true_cl_inter)
        true_fils_inter = np.array(true_fils_inter)
        true_cl_conns = np.array(true_cl_conns)

        false_cl_inter = []
        false_fils_inter = []
        false_cl_conns = []
        for i in tqdm(range(Disperse3D.RANDOM_CLUSTERS_NUM)):
            false_cl_inter.append([])
            false_fils_inter.append([])
            false_cl_conns.append([])
            for rad in rads:
                if mode == "coefs":
                    cl_conn, fils_conn, _ = self.count_conn(
                        Disperse3D.random_clusters[i]["R"] * rad,
                        Disperse3D.random_clusters[i],
                    )
                else:
                    cl_conn, fils_conn, _ = self.count_conn(
                        [rad] * Disperse3D.random_clusters[i].shape[0],
                        Disperse3D.random_clusters[i],
                    )
                false_cl_inter[i].append(sum(list(map(lambda x: int(x > 0), cl_conn))))
                false_fils_inter[i].append(
                    sum(list(map(lambda x: int(x > 0), fils_conn)))
                )
                false_cl_conns[i].append(sum(cl_conn))

        false_cl_inter = np.array(false_cl_inter).mean(0)
        false_fils_inter = np.array(false_fils_inter).mean(0)
        false_cl_conns = np.array(false_cl_conns).mean(0)

        diff_cl_inter = true_cl_inter - false_cl_inter
        diff_fils_inter = true_fils_inter - false_fils_inter
        diff_cl_conns = true_cl_conns - false_cl_conns

        true_recall = true_cl_inter / cl_num
        false_recall = false_cl_inter / cl_num
        diff_recall = diff_cl_inter / cl_num

        true_precision = true_fils_inter / fils_num
        false_precision = false_fils_inter / fils_num
        diff_precision = diff_fils_inter / fils_num

        true_f1 = 2 * true_recall * true_precision / (true_recall + true_precision)
        false_f1 = 2 * false_recall * false_precision / (false_recall + false_precision)
        diff_f1 = 2 * diff_recall * diff_precision / (diff_recall + diff_precision)

        metrics["cl_num"] = cl_num
        metrics["fils_num"] = fils_num

        metrics["true_cl_inter"] = [int(e) for e in true_cl_inter]
        metrics["false_cl_inter"] = [float(e) for e in false_cl_inter]
        metrics["diff_cl_inter"] = [float(e) for e in diff_cl_inter]

        metrics["true_cl_conns"] = [int(e) for e in true_cl_conns]
        metrics["false_cl_conns"] = [int(e) for e in false_cl_conns]
        metrics["diff_cl_conns"] = [int(e) for e in diff_cl_conns]

        metrics["true_fils_inter"] = [int(e) for e in true_fils_inter]
        metrics["false_fils_inter"] = [int(e) for e in false_fils_inter]
        metrics["diff_fils_inter"] = [int(e) for e in diff_fils_inter]

        metrics["true_recall"] = [float(e) for e in true_recall]
        metrics["false_recall"] = [float(e) for e in false_recall]
        metrics["diff_recall"] = [float(e) for e in diff_recall]

        metrics["true_precision"] = [float(e) for e in true_precision]
        metrics["false_precision"] = [float(e) for e in false_precision]
        metrics["diff_precision"] = [float(e) for e in diff_precision]

        metrics["true_f1"] = [float(e) for e in true_f1]
        metrics["false_f1"] = [float(e) for e in false_f1]
        metrics["diff_f1"] = [float(e) for e in diff_f1]

        return metrics

    def count_metrics_several_params(self, sigmas, smooths, mode, rads, clusters=None):
        """

        :param sigmas: набор порогов персистентности
        :param smooths: набор значений количества сглаживаний
        :param mode: (coefs, mpcs) радиусы скоплений в R200 или Mpc
        :param rads: набор радиусов для рассчета метрик
        :param clusters: DataFrame со скоплениями, если None, то используется self.clusters
        :return:
        """
        if clusters is None:
            clusters = self.clusters
        metrics = {}
        metrics["sigmas"] = sigmas
        metrics["smooths"] = smooths
        metrics["mode"] = mode
        metrics["rads"] = rads
        for sigma in sigmas:
            metrics[sigma] = {}
            for smooth in smooths:
                self.apply_disperse(sigma, smooth)
                metrics[sigma][smooth] = self.count_metrics(mode, rads, clusters)

        return metrics

    # TODO
    def get_seg_mask(self, voxel_size, fil_rad):
        mask_sizes = (
            int((self.CX_int[1] - self.CX_int[0]) // voxel_size) + 1,
            int((self.CY_int[1] - self.CY_int[0]) // voxel_size) + 1,
            int((self.CZ_int[1] - self.CZ_int[0]) // voxel_size) + 1,
        )
        mask = np.zeros(mask_sizes)

        MIN_SEG_LEN = 1  # Mpc

        points = []
        next_point = []
        fil_num = []
        count = 0
        for i, fil in enumerate(self.fils):
            sp = fil["sample_points"]
            for j in range(len(sp) - 1):
                points.append([sp[j]["CX"], sp[j]["CY"], sp[j]["CZ"]])
                fil_num.append(i)
                count += 1
                next_point.append(count)
                d = dist(
                    sp[j]["CX"],
                    sp[j]["CY"],
                    sp[j]["CZ"],
                    sp[j + 1]["CX"],
                    sp[j + 1]["CY"],
                    sp[j + 1]["CZ"],
                )
                if d > MIN_SEG_LEN:
                    n = int(d // MIN_SEG_LEN + 1)
                    d_x = sp[j + 1]["CX"] - sp[j]["CX"]
                    d_y = sp[j + 1]["CY"] - sp[j]["CY"]
                    d_z = sp[j + 1]["CZ"] - sp[j]["CZ"]
                    for k in range(1, n):
                        points.append(
                            [
                                sp[j]["CX"] + k * d_x / n,
                                sp[j]["CY"] + k * d_y / n,
                                sp[j]["CZ"] + k * d_z / n,
                            ]
                        )
                        fil_num.append(i)
                        count += 1
                        next_point.append(count)
            points.append([sp[-1]["CX"], sp[-1]["CY"], sp[-1]["CZ"]])
            fil_num.append(i)
            count += 1
            next_point.append(None)

        kd_tree = KDTree(points, leaf_size=2)

        for i in range(mask_sizes[0]):
            for j in range(mask_sizes[1]):
                for k in range(mask_sizes[2]):
                    x3, y3, z3 = (
                        self.CX_int[0] + (i + 0.5) * voxel_size,
                        self.CY_int[0] + (j + 0.5) * voxel_size,
                        self.CZ_int[0] + (k + 0.5) * voxel_size,
                    )
                    close_points_idx = kd_tree.query_radius(
                        [[x3, y3, z3]], r=fil_rad + MIN_SEG_LEN + 1
                    )
                    for p_idx in close_points_idx[0]:
                        if next_point[p_idx] is None:
                            continue
                        x1, y1, z1 = tuple(points[p_idx])
                        x2, y2, z2 = tuple(points[next_point[p_idx]])
                        if intersec_line_sphere(
                            x1, y1, z1, x2, y2, z2, x3, y3, z3, fil_rad
                        ):
                            mask[i, j, k] = 1

        input_ = np.zeros(mask_sizes)

        CX = self.galaxies["CX"]
        CY = self.galaxies["CY"]
        CZ = self.galaxies["CZ"]

        for i in tqdm(range(len(CX))):
            input_[
                int((CX[i] - self.CX_int[0]) // voxel_size),
                int((CY[i] - self.CY_int[0]) // voxel_size),
                int((CZ[i] - self.CZ_int[0]) // voxel_size),
            ] += 1

        return input_, mask

    def plot_2d(
        self,
        plot_galaxies=True,
        plot_clusters=True,
        plot_cps=True,
        plot_only_max=True,
        plot_fils=True,
        cl_fils=None,
        cl_maxs=None,
        title=None,
        clusters=None,
    ):
        if clusters is None:
            clusters = self.clusters

        font = {"size": 16}
        plt.rc("font", **font)
        fig = plt.figure(figsize=(18, 12))
        ax = fig.add_subplot(111)
        print(ax)

        if plot_galaxies:
            ax.scatter(self.galaxies["RA"], self.galaxies["DEC"], c="grey", s=8)

        if plot_clusters:
            ax.scatter(clusters["RA"], clusters["DEC"], c="purple", s=150)
            if cl_maxs is not None:
                t = clusters[cl_maxs]
                ax.scatter(
                    t["RA"],
                    t["DEC"],
                    marker="s",
                    facecolors="none",
                    edgecolors="orange",
                    linewidths=5,
                    s=500,
                )
            if cl_fils is not None:
                t = clusters[cl_fils]
                ax.scatter(
                    t["RA"],
                    t["DEC"],
                    facecolors="none",
                    edgecolors="cyan",
                    linewidths=4,
                    s=300,
                )

        if plot_cps:
            d = {4: "xkcd:brown", 3: "red", 2: "green", 1: "orange", 0: "blue"}
            x, y, c = [], [], []
            for cp in self.cps:
                if cp["type"] != 3 and plot_only_max:
                    continue
                x.append(cp["RA"])
                y.append(cp["DEC"])
                c.append(d[cp["type"]])
            ax.scatter(x, y, c=c, s=100)

        if plot_fils:
            for fil in self.fils:
                points = fil["sample_points"]
                x = []
                y = []
                for i in range(len(points)):
                    x.append(points[i]["RA"])
                    y.append(points[i]["DEC"])
                ax.plot(x, y, "b", linewidth=2, color="b")

        ax.invert_xaxis()
        ax.set_xlabel("RA")
        ax.set_ylabel("DEC")
        if title is None:
            title = (
                f"DisPerSe_2D_smooth:{self.disperse_smooth}_s:{self.disperse_sigma}_"
                f"board:{self.disperse_board}_asmb:{self.disprese_asmb_angle}"
            )
        ax.set_title(title)

        return fig, title

    def plot_3d(
        self,
        plot_galaxies=False,
        plot_clusters=True,
        plot_cps=False,
        plot_only_max=True,
        plot_fils=True,
        cl_fils=None,
        cl_maxs=None,
        title=None,
        clusters=None,
    ):
        if clusters is None:
            clusters = self.clusters

        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlim(self.ra_int)
        ax.set_zlim(self.dec_int)
        ax.set_ylim(self.z_int)

        if plot_galaxies:
            ax.scatter(
                self.galaxies["RA"],
                self.galaxies["Z"],
                self.galaxies["DEC"],
                c="grey",
                s=2,
                alpha=0.3,
            )

        if plot_clusters:
            ax.scatter(
                clusters["RA"],
                clusters["Z"],
                clusters["DEC"],
                color="purple",
                s=40,
                alpha=1,
            )
            if cl_maxs is not None:
                t = clusters[cl_maxs]
                ax.scatter(
                    t["RA"],
                    t["Z"],
                    t["DEC"],
                    marker="s",
                    facecolors="none",
                    edgecolors="orange",
                    linewidths=5,
                    s=50,
                )
            if cl_fils is not None:
                t = clusters[cl_fils]
                ax.scatter(
                    t["RA"],
                    t["Z"],
                    t["DEC"],
                    facecolors="none",
                    edgecolors="cyan",
                    linewidths=2,
                    s=30,
                )

        if plot_cps:
            d = {4: "xkcd:brown", 3: "red", 2: "green", 1: "orange", 0: "blue"}
            x = []
            y = []
            z = []
            c = []
            for cp in self.cps:
                if cp["type"] != 3 and plot_only_max:
                    continue
                x.append(cp["RA"])
                y.append(cp["DEC"])
                z.append(cp["Z"])
                c.append(d[cp["type"]])
            ax.scatter(x, z, y, c=c, s=10)

        if plot_fils:
            for fil in tqdm(self.fils):
                points = fil["sample_points"]
                x = []
                y = []
                z = []
                for i in range(len(points)):
                    x.append(points[i]["RA"])
                    y.append(points[i]["DEC"])
                    z.append(points[i]["Z"])
                ax.plot(x, z, y, "b", linewidth=1, color="b")

        ax.invert_xaxis()
        ax.set_xlabel("RA")
        ax.set_ylabel("Z")
        ax.set_zlabel("DEC")
        if title is None:
            title = (
                f"DisPerSe_3D_smooth:{self.disperse_smooth}_s:{self.disperse_sigma}_"
                f"board:{self.disperse_board}_asmb:{self.disprese_asmb_angle}"
            )
        ax.set_title(title)

        return fig, title
