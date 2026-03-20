from itertools import groupby
import os
import re
import math 
import numpy as np
import pandas as pd
import random
import pickle
import datetime
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from dataclasses import dataclass, field
from scipy.special import ellipe
os.chdir(os.path.dirname(os.path.abspath(__file__)))

@dataclass
class BestLifeSnapshot:
    valid: bool = False
    y: float = 0.0
    z: float = 0.0
    r1: float = None
    rt1: float = None

    sigma_f_List: np.ndarray = field(default_factory=lambda: np.array([]))
    orList: np.ndarray = field(default_factory=lambda: np.array([]))
    rnList: np.ndarray = field(default_factory=lambda: np.array([]))
    Nd: np.ndarray = field(default_factory=lambda: np.array([]))

    PnList: list = field(default_factory=list) 
    RnAS: list = field(default_factory=list)

    BasicParams: object = None
    detailed_results: dict = field(default_factory=dict)

class BasicParameters:
    def __init__(self):
        self.W = 1e9   
        self.T = 20.5
        self.fixed_aspect_ratio = None

        self.y_lim = 5 
        self.z_lim = 5 
        self.grain_size_lim = 0.01         

        self.active_element_area = self.y_lim * self.z_lim 

        self.sigma_nom_list = np.arange(260, 261, 5) 
        self.R_ratio = -1 
        self.crystal = "BCC"
        self.FirstGrain = 'Ferrite'
        self.SecondGrain = 'Pearlite'

        self.steel = "N50R"
        self.iteration_num = np.arange(0,10,1) 

        self.closure_type = 2 

        self.k_tr = 2 * 14  
        self.c_paris = 18 
        self.n_paris = 1.8  
        self.Δδ_th = 0.000063 

        self.thickness = self.T
        self.width = self.W

        self.eval_num_stage1 = None 
        self.eval_num_stage2 = None 
        self.eval_num_total = None 
        self.eval_lenghth_lim = None 
        self.eval_points_full= None 
        self.eval_points_stage1 = None 
        self.eval_points_stage2 = None 
        self.calculate_evaluation_points()

    def SlipPlane(self, ang, DeltaSigma):
        ang = np.atleast_2d(ang)
        if DeltaSigma.ndim == 2:
            DeltaSigma = DeltaSigma[np.newaxis, ...]
        N = ang.shape[0]

        if self.crystal == "BCC":
            nv = np.array([
                [[1., 1., 0.], [1., -1., 1.], [-1., 1., 1.]],
                [[1., -1., 0.], [1., 1., 1.], [-1., -1., 1.]],
                [[1., 0., 1.], [1., 1., -1.], [-1., 1., 1.]],
                [[-1., 0., 1.], [-1., 1., -1.], [1., 1., 1.]],
                [[0., 1., 1.], [1., 1., -1.], [1., -1., 1.]],
                [[0., 1., -1.], [1., 1., 1.], [1., -1., -1.]]
            ])
        else:
            raise ValueError("Only BCC supported currently")

        nv = nv / np.linalg.norm(nv, axis=2, keepdims=True)

        phi, theta, psi = ang[:, 0], ang[:, 1], ang[:, 2]
        C_phi, S_phi = np.cos(phi), np.sin(phi)
        C_theta, S_theta = np.cos(theta), np.sin(theta)
        C_psi, S_psi = np.cos(psi), np.sin(psi)

        g = np.empty((N, 3, 3))
        g[:, 0, 0] = C_psi*C_theta*C_phi - S_psi*S_phi
        g[:, 0, 1] = S_psi*C_theta*C_phi + C_psi*S_phi
        g[:, 0, 2] = -S_theta*C_phi
        g[:, 1, 0] = -C_psi*C_theta*S_phi - S_psi*C_phi
        g[:, 1, 1] = -S_psi*C_theta*S_phi + C_psi*C_phi
        g[:, 1, 2] = S_theta*S_phi
        g[:, 2, 0] = C_psi*S_theta
        g[:, 2, 1] = S_psi*S_theta
        g[:, 2, 2] = C_theta

        nv2 = np.einsum('nij,pdj->npdi', g, nv)

        n_vec = nv2[:, :, 0, :]       
        b_vecs = nv2[:, :, 1:, :]     

        tau_matrix = np.abs(np.einsum('npi,nij,npdj->npd', n_vec, DeltaSigma, b_vecs)) 

        tau_flat = tau_matrix.reshape(N, -1) 
        max_linear_idx = np.argmax(tau_flat, axis=1) 

        max_shear = tau_flat[np.arange(N), max_linear_idx]

        p_idx, d_idx_sub = np.divmod(max_linear_idx, 2) 

        idx_n = np.arange(N)
        theta_n = nv2[idx_n, p_idx, 0, :]           
        theta_s = nv2[idx_n, p_idx, d_idx_sub+1, :] 

        if N == 1:
            return max_shear[0], theta_n[0], theta_s[0]

        return max_shear, theta_n, theta_s

    def makeEulerAngles(self, N):
        phi = np.random.uniform(0, 2 * np.pi, N)
        psi = np.random.uniform(0, 2 * np.pi, N)
        theta = np.arccos(np.random.uniform(-1, 1, N))
        return np.column_stack((phi, theta, psi))

    def calculate_evaluation_points(self):
            self.eval_num_stage1 = 4
            self.eval_lenghth_lim = 0.98 * self.thickness
            def eval_points_stage1(r1):
                return [r1 * (1 - ((self.eval_num_stage1 - i + 1) / self.eval_num_stage1) ** 3) for i in range(1, self.eval_num_stage1 + 1)]
            self.eval_points_stage1 = eval_points_stage1

            points = []

            points.extend(np.arange(0.01, 0.31, 0.01))

            points.extend(np.arange(0.35, 1.01, 0.05))

            points.extend(np.arange(1.1, 1.51, 0.1))

            if self.T > 1.5:
                points.extend(np.arange(2.0, self.T, 0.5))

            self.eval_points_stage2 = [round(p, 2) for p in points if p < self.T]

def get_fatigue_eval_points(r1, rt1, rnList, BasicParams, DataImport):
    Ai1 = BasicParams.eval_points_stage1(r1)
    Ai1[-1] = r1 - DataImport.dave / 1000

    rni0 = np.array([2, 3] + [1 + int(sum(np.heaviside(ai - rnList,0))) for ai in BasicParams.eval_points_stage2])-1
    rni1 = np.unique(rni0)

    rni = rni1[1:] if rni1[0] == 0 else rni1 

    Ai2 = np.array([[rnList[rn - 1] + DataImport.dave / 1000, 0.5 * (rnList[rn] + rnList[rn - 1]), rnList[rn] - DataImport.dave / 1000] for rn in rni]).flatten().tolist()
    ai = sorted(Ai1 + Ai2)
    nai = len(ai) 

    surfai = ai.copy() 
    surfai[:BasicParams.eval_num_stage1] = [2* rt1] * BasicParams.eval_num_stage1

    return ai, len(ai)

class AnalyticalKCalculator:
    @staticmethod
    def calculate_K(a, c,  St):
        if a <= 0 or c <= 0: return 0.0

        a_c = min(a / c, 1.0)

        phi_rad =np.pi/2
        sin_phi = np.sin(phi_rad)
        cos_phi = np.cos(phi_rad)

        Q = 1 + 1.464 * (a_c)**1.65

        lambda_s = (1.13 - 0.09 * a_c) * (1 + 0.1 * (1 - sin_phi)**2)

        f_phi = (sin_phi**2 + (a_c)**2 * cos_phi**2)**0.25

        K = lambda_s * St * np.sqrt(np.pi * a / Q) * f_phi

        return K

class MaterialDataImporter:
    def __init__(self, BasicParams):
        self.BasicParams = BasicParams

        self.E = 206000.0 
        self.ν = 0.3      
        self.AA = self.E / (4 * np.pi * (1 - self.ν ** 2))

        self.monotonic_tensile_properties()
        self.friction_strength()

        self.FirstGrainDF = pd.read_csv(f"{self.BasicParams.steel}_{BasicParams.FirstGrain} grain size.csv", header=None).to_numpy()
        self.FirstGrainAspectDF = pd.read_csv(f"{self.BasicParams.steel}_{BasicParams.FirstGrain} grain aspect ratio.csv", header=None).to_numpy()
        angle_file = os.path.join(f"{self.BasicParams.steel}_{self.BasicParams.FirstGrain} grain angle.csv")
        if os.path.exists(angle_file):
            self.FirstGrainAngleDF = pd.read_csv(angle_file, header=None).to_numpy()
        else:
            self.FirstGrainAngleDF = None 

        if BasicParams.SecondGrain == "Pearlite":
            self.Pearlite_thickness_files = [file for file in os.listdir() if file.startswith(f"{self.BasicParams.steel}_{BasicParams.SecondGrain} thickness") and file.endswith(".csv")]
            if self.Pearlite_thickness_files:
                self.SecondGrainDF = pd.read_csv(self.Pearlite_thickness_files[0], header=None).to_numpy()
                self.SecondGrainCDF = self.makeCDF(self.SecondGrainDF)
                self.SecondGrainCDFr = self.makeCDFr(self.SecondGrainDF)
                self.Pearlite_fraction()

        self.FirstGrainCDF = self.makeCDF(self.FirstGrainDF)
        self.FirstGrainCDFr = self.makeCDFr(self.FirstGrainDF)
        self.FirstGrainAspectCDFr = self.makeCDFrA(self.FirstGrainAspectDF)
        self.FirstGrainAngleCDF = self.makeCDF(self.FirstGrainAngleDF) if self.FirstGrainAngleDF is not None else None

        self.dave = sum(item[0] ** 3 * item[1] for item in self.FirstGrainDF) / sum(item[0] ** 2 * item[1] for item in self.FirstGrainDF)
        self.dmax = self.FirstGrainCDF(1)

        self.ngAe = (4 * self.BasicParams.active_element_area) / (np.pi * self.dave**2)

        print(f"Microstructure loaded from CSVs: dave={self.dave:.4f}, dmax={self.dmax:.4f}")

        self.calculate_VolumeFraction()

        self.ModelSize()

        self.CreateGrainData()

    def ModelSize(self):
        n_samples = 50000

        r_vals_F = np.random.rand(int(n_samples * (1 - self.PRateN)))
        part1 = (self.FirstGrainCDF(r_vals_F))**2

        if self.PRateN > 0:
            r_vals_P1 = np.random.rand(int(n_samples * self.PRateN))
            r_vals_P2 = np.random.rand(int(n_samples * self.PRateN))
            part2 = self.FirstGrainCDF(r_vals_P1) * self.SecondGrainCDF(r_vals_P2)
            A0_samples = np.concatenate((part1, part2))
        else:
            A0_samples = part1

        mean_A0 = np.mean(A0_samples)

        fw = 3.0 

        self.Ng = round(fw * ((2 * self.BasicParams.T**2) / mean_A0))

        r_vals_d0 = np.random.rand(n_samples)
        d0_samples = self.FirstGrainCDF(r_vals_d0) * self.FirstGrainAspectCDFr(r_vals_d0)
        mean_d0 = np.mean(d0_samples)

        self.Mg = round((self.BasicParams.T * fw) / mean_d0)

        print(f"Dynamic Model Size: Ng={self.Ng}, Mg={self.Mg} (fw={fw})")

    def calculate_VolumeFraction(self):
        n_samples = 100000
        r_vals = np.random.rand(n_samples)
        VP = np.mean(self.SecondGrainCDFr(r_vals) * self.FirstGrainCDFr(r_vals))
        VF = np.mean(self.FirstGrainCDFr(np.random.rand(n_samples)) ** 2)
        self.PRateN = (self.PRate / VP) / (self.PRate / VP + (1 - self.PRate) / VF)

    def monotonic_tensile_properties(self):
        filename = f"{self.BasicParams.steel}_Monotonic tensile test.csv"
        tensile_test_data = pd.read_csv(filename)
        self.σ_YS = tensile_test_data.iloc[0, 0]
        self.σ_UT = tensile_test_data.iloc[0, 1]
        self.Reduction_in_Area = tensile_test_data.iloc[0, 2]
        self.σ_0 = 0.5 * (self.σ_YS + self.σ_UT)
        print(f"Loaded {filename}: σ_YS={self.σ_YS}, σ_UT={self.σ_UT}")

    def friction_strength(self):
        filename = f"{self.BasicParams.steel}_Friction strength.csv"
        df = pd.read_csv(filename, header=None).apply(pd.to_numeric, errors='coerce').dropna()
        self.σ_fF = float(df.iloc[0, 0])
        self.σ_fP = float(df.iloc[0, 1])
        print(f"Loaded {filename}: σ_fF={self.σ_fF}, σ_fP={self.σ_fP}")

    def makeCDF(self, DF):
        CDF0 = np.cumsum(DF[:, 1])
        CDF = CDF0 * 0.99999 + np.arange(len(DF)) / len(DF) * 0.00001
        return interp1d(CDF, DF[:, 0], kind='linear', fill_value='extrapolate')
    def makeCDFr(self, DF):
        CDF0 = np.cumsum(DF[:, 1])
        CDF = CDF0 * 0.99999 + np.arange(len(DF)) / len(DF) * 0.00001
        return interp1d(CDF, np.sqrt(np.pi) / 2. * DF[:, 0], kind='linear', fill_value='extrapolate')
    def makeCDFrA(self, DF):
        CDF0 = np.cumsum(DF[:, 1])
        CDF = CDF0 * 0.99999 + np.arange(len(DF)) / len(DF) * 0.00001
        RA = np.sqrt(DF[:, 0])
        return interp1d(CDF, RA, kind='linear', fill_value='extrapolate')
    def Pearlite_fraction(self):
        match = re.search(r"(\d+\.\d+)", str(self.Pearlite_thickness_files[0]))
        self.PRate = 1.5 * float(match.group(1)) if match else 0.0

    def CreateGrainData(self):
        pkl_name = f"{self.BasicParams.steel}_gData_Analytical.pkl"
        if os.path.exists(pkl_name):
            try:
                with open(pkl_name, "rb") as file: self.gData = pickle.load(file)
                print(f"Loaded {pkl_name}")
                return
            except: pass

        print("Generating gData...")
        nData = 1000000
        R = np.random.rand(5, nData)
        dtrList = [[row[0], self.FirstGrainCDFr(row[1]), self.SecondGrainCDFr(row[2]), self.FirstGrainAspectCDFr(row[3]), np.pi / 180 * self.FirstGrainAngleCDF(row[4])] for row in R.T]

        self.gData = [self.makegList(*x) for x in dtrList]
        with open(pkl_name, "wb") as file: pickle.dump(self.gData, file)
        print("gData generated.")

    def makegList(self, r, d, t, ra, ang):
        if r > self.PRateN:
            return [
                self.σ_fF,
                d * np.sqrt(np.sqrt((np.cos(ang)**2 + ra**4 * np.sin(ang)**2) / (np.sin(ang)**2 + ra**4 * np.cos(ang)**2))),
                d * np.sqrt(np.sqrt((np.sin(ang)**2 + ra**4 * np.cos(ang)**2) / (np.cos(ang)**2 + ra**4 * np.sin(ang)**2)))
            ]
        else:
            return [
                self.σ_fP,
                math.sqrt(t * d * ra * math.sqrt((t**2 * math.cos(ang)**2 + d**2 * ra**2 * math.sin(ang)**2) /(t**2 * math.sin(ang)**2 + d**2 * ra**2 * math.cos(ang)**2))),
                math.sqrt(t * d * ra * math.sqrt((t**2 * math.sin(ang)**2 + d**2 * ra**2 * math.cos(ang)**2) /(t**2 * math.cos(ang)**2 + d**2 * ra**2 * math.sin(ang)**2)))
            ]

def makeFList(BasicParams, DataImport):
    Aej = BasicParams.active_element_area 
    d = 0 
    fww = 25
    n_max= round(fww * DataImport.ngAe) 
    FList0 = [[] for _ in range(n_max)] 

    i = 0

    while d != 1 and i < n_max: 
        PorF = random.random()
        if PorF > DataImport.PRateN:
            i += 1
            fd = DataImport.FirstGrainCDF(random.random()) 

            if Aej < (np.pi * fd**2) / 4: 
                fd = np.sqrt((4 * Aej) / np.pi)
                d = 1
            FList0.append(fd)
            Aej -= (np.pi * fd**2) / 4
        else:
            if not hasattr(DataImport, 'SecondGrainCDF') or DataImport.SecondGrainCDF is None:
                continue 

            r = random.random()
            pw = DataImport.SecondGrainCDF(r) 
            pl = DataImport.FirstGrainCDF(r)

            if Aej < np.pi / 4 * pw * pl:
                pw = (4 * Aej) / (np.pi * pl)
                d = 1
            Aej -= np.pi / 4 * pw * pl

    FList = [item for sublist in FList0 for item in (sublist if isinstance(sublist, list) else [sublist])]
    sorted_FList = sorted(FList, reverse=True) 
    return sorted_FList

def get_crack_aspect_ratio_Wu(a_curr, r1, rt1, BasicParams):
    if BasicParams.fixed_aspect_ratio is not None:
        return BasicParams.fixed_aspect_ratio

    if a_curr <= r1:
         return max(1e-9, a_curr) / rt1

    r0 = r1 / rt1               

    term2 = (r1 / a_curr)**3 * (1 - r0**(-3))
    bracket_term = 1 - term2
    rw_val = bracket_term**(-1/3)

    return rw_val

def analytical_sec_ell(asp):
    asp = max(1e-6, asp)

    if asp <= 1.0:
        m = 1.0 - asp**2
        return ellipe(m)
    else:
        m = 1.0 - (1.0/asp)**2
        return asp * ellipe(m)

def generateGrains(r1, rt1, or1, BasicParams, DataImport, ff00_analytical):
    def L2(asp0, b_val): 
        if b_val * asp0 < BasicParams.W: 
             return b_val * analytical_sec_ell(asp0)
        else:
             return BasicParams.W

    g_data_arr = np.asarray(DataImport.gData) 
    Ng = DataImport.Ng  

    rand_indices = np.random.randint(0, len(g_data_arr), size=Ng)
    selected_gdata = g_data_arr[rand_indices]

    first_grain = np.array([[float(DataImport.σ_fF), float(r1), float(rt1)]])
    g_List = np.vstack((first_grain, selected_gdata)) 

    σ_f_List = g_List[:, 0] 
    dnList = g_List[:, 1]
    tnList = g_List[:, 2]

    orList_full = np.zeros((Ng + 1, 3), dtype=np.float32)
    orList_full[0] = or1
    orList_full[1:] = BasicParams.makeEulerAngles(Ng)

    rnList = np.zeros(DataImport.Mg, dtype=np.float32)
    Nd = np.zeros(DataImport.Mg, dtype=np.int32)

    PnList = [None] * DataImport.Mg
    RnAS = [None] * DataImport.Mg

    rnList[0] = r1
    Nd[0] = 1

    n, n0, n1 = 0, 0, 0
    RnAA_prev = np.array([1.0]) 
    while rnList[n] < 1.05 * BasicParams.T:
        rr = rnList[n] 
        m0 = n1 + 1
        if m0 >= DataImport.Ng: break
        m02 = m0 + (n1 - n0)
        if m02 >= DataImport.Ng: m02 = DataImport.Ng
        gr, nnn = 0, 1
        if rr < BasicParams.T * 0.9:
            rr0 = rr + np.dot(dnList[m0 : m02+1], tnList[m0 : m02+1]) / (2 * np.sum(tnList[m0 : m02+1]))
            current_asp = ff00_analytical(rr0)
            if L2(current_asp, rr0) < np.sum(tnList[m0:m02+1]):
                while True:
                    if (m02 - nnn) >= m0:
                        rr_temp = rr + np.dot(dnList[m0:m02 - nnn + 1], tnList[m0:m02 - nnn + 1]) / (2 * np.sum(tnList[m0:m02 - nnn + 1]))
                        if L2(ff00_analytical(rr_temp), rr_temp) > np.sum(tnList[m0:m02 - nnn + 1]): break
                    else: break
                    nnn += 1
                gr = 1
            else:
                while True:
                    if (m02 + nnn) <= DataImport.Ng:
                        rr_temp = rr + np.dot(dnList[m0:m02 + nnn + 1], tnList[m0:m02 + nnn + 1]) / (2 * np.sum(tnList[m0:m02 + nnn + 1]))
                        if L2(ff00_analytical(rr_temp), rr_temp) < np.sum(tnList[m0:m02 + nnn + 1]): break
                    else: break
                    nnn += 1
                gr = 2
        else:
            rr0 = rr + np.dot(dnList[m0:m02+1], tnList[m0:m02+1]) / (2 * np.sum(tnList[m0:m02+1]))
            asp_90 = ff00_analytical(BasicParams.T * 0.9)
            if L2(asp_90, rr0) < np.sum(tnList[m0:m02+1]):
                while True:
                    if (m02 - nnn) >= m0:
                        rr_temp = rr + np.dot(dnList[m0:m02 - nnn + 1], tnList[m0:m02 - nnn + 1]) / (2 * np.sum(tnList[m0:m02 - nnn + 1]))
                        if L2(asp_90, rr_temp) > np.sum(tnList[m0:m02 - nnn + 1]): break
                    else: break
                    nnn += 1
                gr = 1
            else:
                while True:
                    if (m02 + nnn) <= DataImport.Ng:
                        rr_temp = rr + np.dot(dnList[m0:m02 + nnn + 1], tnList[m0:m02 + nnn + 1]) / (2 * np.sum(tnList[m0:m02 + nnn + 1]))
                        if L2(asp_90, rr_temp) < np.sum(tnList[m0:m02 + nnn + 1]): break
                    else: break
                    nnn += 1
                gr = 2

        n0 = n1 + 1
        n1 = m02 - nnn + 1 if gr == 1 else m02 + nnn - 1
        if n1 >= DataImport.Ng:
             n1 = DataImport.Ng - 1
             if n0 > n1: break

        Nd[n + 1] = n1 - n0 + 1
        tn_sum = np.sum(tnList[n0:n1+1])

        RnA_layer = tnList[n0:n1+1] / tn_sum if tn_sum > 0 else np.ones(Nd[n+1])/Nd[n+1]
        RnAA_layer = np.cumsum(RnA_layer)

        rnList[n + 1] = rnList[n] + np.dot(RnA_layer, dnList[n0:n1+1])

        num_pairs = Nd[n] + Nd[n+1] - 1
        current_Pn = np.zeros((num_pairs, 2), dtype=np.int32) 

        if num_pairs > 0:
            current_Pn[num_pairs - 1] = [Nd[n]-1, Nd[n + 1]-1] 
            current_Pn[0] = [0, 0]

            if n == 0:
                 RnAA_prev = np.array([1.0])
            else:
                 pass 

            k0, k1 = 1, 1
            for j in range(1, num_pairs - 1):
                idx0 = min(k0 - 1, len(RnAA_prev)-1)
                idx1 = min(k1 - 1, len(RnAA_layer)-1)

                if RnAA_prev[idx0] < RnAA_layer[idx1]: k0 += 1
                else: k1 += 1
                current_Pn[j] = [k0-1, k1-1]

        PnList[n] = current_Pn 

        RnAAS_layer = np.sort(np.concatenate((RnAA_prev, RnAA_layer[:-1])))
        RnAS_layer = np.diff(np.concatenate(([0.0], RnAAS_layer)))
        RnAS[n] = RnAS_layer.astype(np.float32) 

        RnAA_prev = RnAA_layer
        n += 1

        if n >= DataImport.Mg - 1:
             print("Warning: Reached max layers (Mg) in generateGrains.")
             break

    return (σ_f_List, 
            rnList[:n+1], 
            Nd[:n+1], 
            PnList[:n], 
            RnAS[:n],   
            orList_full)

def calc_driving_force(i, aii, r1, rt1, BasicParams, DataImport, sigma_nom_val):
    S_nom = sigma_nom_val 

    n_prime = 0.2014
    K_prime = 1206.0  

    E = DataImport.E  

    S_eq = S_nom + E * ((S_nom / K_prime) ** (1.0 / n_prime))

    if aii <= r1: 
        c_curr = rt1
        ar_curr = aii / c_curr
    else: 
        ar_curr = get_crack_aspect_ratio_Wu(aii, r1, rt1, BasicParams)
        c_curr = aii / ar_curr

    K = AnalyticalKCalculator.calculate_K(a=aii, c=c_curr, St=S_eq)

    if aii > 1e-9:
        S_app = K / np.sqrt(np.pi * aii)
    else:
        S_app = S_nom

    S_max = 2 * S_app / (1.0 - BasicParams.R_ratio)
    S_min = BasicParams.R_ratio * S_max
    sop_ratio_long = calculate_closure_Newman(BasicParams.R_ratio, S_max, DataImport.σ_0)

    S_op_long_crack = sop_ratio_long * S_max

    S_op_transient = S_min + (S_op_long_crack - S_min) * (1 - np.exp(-BasicParams.k_tr * aii))

    S_op_transient = min(S_op_transient, S_op_long_crack)

    if S_max - S_min > 0:
        U = (S_max - S_op_transient) / (S_max - S_min)
    else:
        U = 1.0 

    U = max(min(U, 1.0), 0.0) 

    DeltaS_eff = U * (S_max - S_min)
    DeltaSigma_eff = np.zeros((3, 3))
    DeltaSigma_eff[1, 1] = DeltaS_eff
    DeltaSigma_eff[2, 2] = 0.3 * DeltaS_eff     

    sigma_op = S_op_transient

    return ar_curr, K, U, DeltaSigma_eff, sigma_op

def get_arrest_eval_points(r1, BasicParams):
    ai_stage1 = list(BasicParams.eval_points_stage1(r1))
    ai_stage1[-1] = r1 - 1e-6 

    d_start = r1

    d_end = BasicParams.T * 0.98 

    log_d_start = np.log10(d_start)
    log_d_end = np.log10(d_end)

    log_step = 0.01

    log_points_overshoot = np.arange(log_d_start, log_d_end + log_step, log_step)

    log_points = log_points_overshoot[log_points_overshoot <= log_d_end]

    ai_stage2 = 10**log_points

    ai = sorted(list(set(ai_stage1 + ai_stage2.tolist())))
    nai = len(ai)

    print(f"  [R-Curve] Generated {len(ai_stage1)} (Stage I) + {len(ai_stage2)} (Stage II) points (log_step={log_step}).")

    return ai, nai

def optimized_grouping(Pn, RnAS_slice, τ0, t0):
    RnAS_arr = np.array(RnAS_slice)
    τ0_arr = np.array(τ0)
    t0_arr = np.array(t0)

    if τ0_arr.ndim == 0: 
        τ0_arr = τ0_arr.reshape(1)

    if t0_arr.ndim == 2: 
        t0_arr = t0_arr.reshape(1, 3, 3)

    groups = []

    for key, group in groupby(enumerate(Pn), key=lambda x: x[1][1]):
        indices = [idx for idx, _ in group]
        groups.append(indices)

    t_gs, τ_gs = [], []
    for indices in groups:
        idxs = np.array(indices)
        weights = RnAS_arr[idxs]
        rnas = np.sum(weights)

        if rnas <= 1e-30: 
            τ_gs.append(0.0)
            t_gs.append(np.zeros((3,3)))
        else:
            τ_val = np.sum(τ0_arr[idxs] * weights) / rnas
            τ_gs.append(τ_val)

            t_val = np.sum(t0_arr[idxs] * weights.reshape(-1, 1, 1), axis=0) / rnas
            t_gs.append(t_val)

    return t_gs, τ_gs

def evalCTSD(i, aii, Δσ, σ_f_List, orList, rnList, Nd, PnList, RnAS, BasicParams, DataImport):
    def safe_arccos(x):
        return np.arccos(np.clip(x, -1, 1))

    def CC1(a, σ_fr, rnlist):
        def Eq_1(c, a, σ_fr):
            epsilon = 1e-10 
            return np.pi / 2 - σ_fr[0] * safe_arccos(a / (c + epsilon))

        upper_bound = rnlist + 100
        max_upper_bound = 1e8  
        while upper_bound <= max_upper_bound:
            try:
                c_root = brentq(Eq_1, a + 1e-10, upper_bound, args=(a, σ_fr))
                return c_root
            except ValueError:
                upper_bound *= 10
        return c_root

    def CCn(a, n, σ_fr, rnList):
        def Eq_n(c, a, n, σ_fr, rnList):
            epsilon = 1e-10 
            return np.pi * 0.5 - σ_fr[0] * safe_arccos(a / (c + epsilon)) - sum((σ_fr[i+1] - σ_fr[i]) * safe_arccos(rnList[i] / (c + epsilon)) for i in range(n+1))

        upper_bound = rnList[n] + 10000
        try:
            c_root = brentq(Eq_n, a + 1e-10, upper_bound, args=(a, n, σ_fr, rnList))
            return c_root
        except ValueError:
            c_root = 10000
        return c_root

    def gg(x, c, ad):
        if x == ad:
            a = (1 - 1e-3) * ad
        else:
            a = ad

        ca = np.sqrt(c**2 - a**2)
        cx = np.sqrt(c**2 - x**2)
        return a * np.log(abs((ca + cx) / (ca - cx))) - x * np.log(abs((x * ca + a * cx) / (x * ca - a * cx)))

    def CTSD(a, c, n, σ_fr, Δτ_j, rnList_L, DataImport):
        if a == 0:
            term1 = 0
        else:
            term1 = 2 * a * σ_fr[0] * np.log(c / a)

        term2 =  sum((σ_fr[i + 1] - σ_fr[i]) * gg(a, c, rnList_L[i]) for i in range(n+1))
        result = Δτ_j / (np.pi ** 2 * DataImport.AA) * (term1 + term2)
        return result.item()

    def optimized_grouping(Pn, RnAS_slice, τ0, t0):
        groups = []
        for key, group in groupby(enumerate(Pn), key=lambda x: x[1][1]):
            indices = [idx for idx, _ in group]
            groups.append(indices)

        RnAS_arr = np.array(RnAS_slice)      
        τ0_arr = np.array(τ0)                
        t0_arr = np.array(t0)                

        t_gs = []
        τ_gs = []
        for indices in groups:
            indices = np.array(indices)
            rnas = np.sum(RnAS_arr[indices])
            if rnas == 0:
                τ_avg = np.zeros_like(τ0_arr[indices][0])
                t_avg = np.zeros_like(t0_arr[indices][0])
            else:
                weights = RnAS_arr[indices].reshape(-1, 1, 1)
                τ_avg = np.sum(τ0_arr[indices] * weights, axis=0) / rnas
                t_avg = np.sum(t0_arr[indices] * weights, axis=0) / rnas
            τ_gs.append(τ_avg)
            t_gs.append(t_avg)
        return t_gs, τ_gs

    unstable = 0
    goto=0

    rnList_temp = rnList.tolist()
    jj = rnList_temp.index(next(filter(lambda x: x > aii, rnList_temp)))+1
    N0 = sum(Nd[0 : jj])

    if jj == 1:
        τ1, θn1, θs1 = BasicParams.SlipPlane(orList[0], Δσ)

        t_List = [ [] for _ in range(DataImport.Mg) ]
        τ_List = [ [] for _ in range(DataImport.Mg) ]
        σ_fr = [0] * DataImport.Mg
        t_List[0] = [τ1 * np.outer(θn1, θs1)]
        τ_List[0] = [τ1]
        σ_fr[0] = DataImport.σ_fF / τ1
    else:
        τ_t2_temp = [BasicParams.SlipPlane(orList[N0 + item[1]], Δσ) for item in PnList[jj-1]]
        τ_t2 = [list(t) for t in zip(*τ_t2_temp)]

        σ_f0 = [σ_f_List[N0 + item[1]] for item in PnList[jj-1]]

        t_List = [ [] for _ in range(DataImport.Mg) ]
        τ_List = [ [] for _ in range(DataImport.Mg) ]
        σ_fr = [0] * DataImport.Mg

        t_List[0] = [TauT2_0 * np.outer(TauT2_1, TauT2_2) for TauT2_0, TauT2_1, TauT2_2 in zip(*τ_t2)] 
        τ_List[0] = list(τ_t2[0])

        σ_fr[0] = 1 / (RnAS[jj-1].dot([τ_t2[0][i] / σ_f0[i] for i in range(len(σ_f0))]))
        τ1 = RnAS[jj-1].dot(τ_List[0])

    if σ_fr[0] < 1:
        cc = 1.05 * BasicParams.thickness
        Δδ = 0

        for m in range(len(rnList)):

            if cc < rnList[jj + m - 1]:
                Δδ = CTSD(aii, cc, m - 1, σ_fr, τ1, rnList[jj - 1:jj + m],DataImport)
                break
            elif jj + m + 1 > len(Nd):
                unstable = 1
                goto = 1
                break

            N0 = sum(Nd[0:jj + m])

            PnList_1_2 = np.array(PnList[jj + m - 1])

            PnList1 = PnList_1_2[:, 0].tolist()
            PnList2 = PnList_1_2[:, 1].tolist()

            or_array = np.array([orList[N0 + PnList2[i]] for i in range(len(PnList2))])
            t_array = np.array([t_List[m][PnList1[i]] for i in range(len(PnList1))])
            τ_t0 = BasicParams.SlipPlane(or_array, t_array)                            

            τ0 = τ_t0[0]  

            N_grains = or_array.shape[0]

            if N_grains == 1:
                τ0 = [τ_t0[0]]  
                t0 = [τ_t0[0] * np.outer(τ_t0[1], τ_t0[2])]
            else:
                τ0 = τ_t0[0]  
                t0 = [t[0] * np.outer(t[1], t[2]) for t in zip(*τ_t0)]

            Pn_current = PnList[jj + m - 1]
            t_gs, τ_gs = optimized_grouping(Pn_current, RnAS[jj + m - 1], τ0, t0)

            σ_f0 = [σ_f_List[N0 + p[1]] for p in PnList[jj + m - 1]]
            τ0_arr = np.array(τ0)       
            σ_f0_arr = np.array(σ_f0)   
            temp_array = τ0_arr / σ_f0_arr  
            σ_fr[m + 1] = 1 / (RnAS[jj + m - 1].dot(temp_array))

            τ_List[m + 1] = τ_gs.copy()
            t_List[m + 1] = t_gs.copy()

            if σ_fr[m] < 1:
                cc = 1.05 * BasicParams.thickness
            else:
                cc = CCn(aii, m, σ_fr, rnList[jj - 1:jj + m])

    else:
        if i == 0:
            cc=100
            Δδ = 0
            goto=1
        else:
            cc = CC1(aii, σ_fr, rnList[0]) 
            Δδ = 0
            for m in range(len(rnList)):
                if cc < rnList[jj + m - 1]:
                    Δδ = CTSD(aii, cc, m - 1, σ_fr, τ1, rnList[jj - 1:jj + m], DataImport)
                    break
                elif jj + m + 1 > len(Nd):
                    unstable = 1
                    goto = 1
                    break

                N0 = sum(Nd[0:jj + m])

                PnList_1_2 = np.array(PnList[jj + m - 1])

                PnList1 = PnList_1_2[:, 0].tolist()
                PnList2 = PnList_1_2[:, 1].tolist()

                or_array = np.array([orList[N0 + PnList2[i]] for i in range(len(PnList2))])
                t_array = np.array([t_List[m][PnList1[i]] for i in range(len(PnList1))])
                τ_t0 = BasicParams.SlipPlane(or_array, t_array)                            

                N_grains = or_array.shape[0]
                if N_grains == 1:
                    τ0 = [τ_t0[0]]  
                    t0 = [τ_t0[0] * np.outer(τ_t0[1], τ_t0[2])]
                else:
                    τ0 = τ_t0[0]  
                    t0 = [t[0] * np.outer(t[1], t[2]) for t in zip(*τ_t0)]

                Pn_current = PnList[jj + m - 1]
                t_gs, τ_gs = optimized_grouping(Pn_current, RnAS[jj + m - 1], τ0, t0)

                σ_f0 = [σ_f_List[N0 + p[1]] for p in PnList[jj + m - 1]]
                τ0_arr = np.array(τ0)       
                σ_f0_arr = np.array(σ_f0)   
                temp_array = τ0_arr / σ_f0_arr  
                σ_fr[m + 1] = 1 / (RnAS[jj + m - 1].dot(temp_array))

                τ_List[m + 1] = τ_gs.copy()
                t_List[m + 1] = t_gs.copy()

                if σ_fr[m] < 1:
                    cc = 1.05 * BasicParams.thickness
                else:
                    cc = CCn(aii, m, σ_fr, rnList[jj - 1:jj + m])

    return cc, Δδ, unstable, goto

def evalCycle(i, cc, rnList, ai, nTemp, S_Δδ0, Scyc0Copy, dNdaListCopy, LifeMin, BasicParams):
    def NNf(x):
        if x == 0:
            return 10**18
        elif 0 < x < 1:
            return 1 + (-0.7 * math.log(x.item()))**1.5
        else:
            return 1

    S_cyc0 = Scyc0Copy.copy()
    dNdaList = dNdaListCopy.copy()
    Label = 0

    if i < BasicParams.eval_num_stage1:
        if i == 0:
            S_cyc0[i] = 0
            dNda = 0
        else:
            if S_Δδ0[i - 1] == 0 or S_Δδ0[i] == 0:
                return S_cyc0, dNdaList

            dNda0 = 1 / (BasicParams.c_paris * S_Δδ0[i - 1]**BasicParams.n_paris)
            dNda1 = 1 / (BasicParams.c_paris * S_Δδ0[i]**BasicParams.n_paris)
            dNda = 0.5 * (dNda0 + dNda1)

            S_cyc0[i] = dNda * (ai[i] - ai[i - 1]) + S_cyc0[i - 1]
        NSuspend = NNf(S_cyc0[i] / nTemp)

        if S_cyc0[i] > NSuspend * LifeMin(ai[i]):
            Label = 1
            return S_cyc0, dNdaList , Label             

        if S_cyc0[i] > 10 * LifeMin(ai[i]):
            Label = 1
            return S_cyc0, dNdaList, Label

        dNdaList[0] = [0, dNda]

    elif (i - BasicParams.eval_num_stage1 + 1) % 3 == 0:
        ai1, ai2 = ai[i - 1] - ai[i - 2], ai[i] - ai[i - 2]
        dNda0 = 1 / (BasicParams.c_paris * (S_Δδ0[i - 2]**BasicParams.n_paris - BasicParams.Δδ_th**BasicParams.n_paris))
        dNda1 = 1 / (BasicParams.c_paris * (S_Δδ0[i - 1]**BasicParams.n_paris - BasicParams.Δδ_th**BasicParams.n_paris)) - dNda0
        dNda2 = 1 / (BasicParams.c_paris * (S_Δδ0[i]**BasicParams.n_paris - BasicParams.Δδ_th**BasicParams.n_paris)) - dNda0

        if ai1 * ai2 * (ai1 - ai2) == 0:
            return S_cyc0, dNdaList

        αα = (ai2 * dNda1 - ai1 * dNda2) / (ai1 * ai2 * (ai1 - ai2))
        ββ = (ai2**2 * dNda1 - ai1**2 * dNda2) / (ai1 * ai2 * (ai2 - ai1))
        γγ = dNda0

        dNda = (1 / ai2) * ((1/3) * αα * ai2**3 + (1/2) * ββ * ai2**2 + γγ * ai2)
        grNum = int((i -  BasicParams.eval_num_stage1 + 1) / 3)
        lineNum = int(sum(np.heaviside(ai[i] - rnListi, 0) for rnListi in rnList))
        dNdaList[grNum] = [lineNum, dNda]
        lineNumPrev, dNdaPrev = dNdaList[grNum - 1]
        rnPrev = rnList[lineNumPrev]

        S_cyc0[i - 2] = S_cyc0[i - 3] + dNdaPrev * (rnPrev - ai[i - 3]) + ((dNdaPrev + dNda) / 2) * (rnList[lineNum - 1] - rnPrev) + dNda * (ai[i - 2] - rnList[lineNum - 1])
        S_cyc0[i - 1] = S_cyc0[i - 2] + dNda * (ai[i - 1] - ai[i - 2])
        S_cyc0[i] = S_cyc0[i - 1] + dNda * (ai[i] - ai[i - 1])

        if min(S_cyc0[i - 2:i + 1]) > nTemp or min(S_cyc0[i - 2:i + 1]) < 0:
            Label = 1

            return S_cyc0, dNdaList, Label

        for ii in range(i - 2, i + 1):
            if S_cyc0[ii] > nTemp:
                Label = 1

                return S_cyc0, dNdaList, Label
            NSuspend = NNf(S_cyc0[ii] / nTemp)

            if S_cyc0[ii] > NSuspend * LifeMin(ai[ii]):
                Label = 1

                return S_cyc0, dNdaList, Label

            if S_cyc0[ii] > 10 * LifeMin(ai[ii]):
                Label = 1

                return S_cyc0, dNdaList, Label

    return S_cyc0, dNdaList, Label

def calculate_closure_Newman(R, S_max, sigma_0, alpha=3):
    sig_ratio = S_max / sigma_0 if S_max<sigma_0 else 1

    # Coefficients for Newman's equation
    A0 = (0.825 - 0.34 * alpha + 0.05 * alpha**2) * (np.cos(np.pi * sig_ratio / 2.0))**(1.0 / alpha)
    A1 = (0.415 - 0.071 * alpha) * sig_ratio
    A3 = 2.0 * A0 + A1 - 1.0
    A2 = 1.0 - A0 - A1 - A3

    if R >= 0:
        sop_ratio = A0 + A1 * R + A2 * R**2 + A3 * R**3
    elif -1 <= R < 0:
        sop_ratio = A0 + A1 * R
    else:
        sop_ratio = A0 - 2.0 * A1 

    real_sop_ratio = max(sop_ratio, R) 

    return real_sop_ratio

def calculate_effective_range(S_max, S_min, sop_ratio):
    R = S_min / S_max

    if sop_ratio >= R:
        U = (1.0 - sop_ratio) / (1.0 - R)
    else:
        U = 1.0 

    return max(min(U, 1.0), 0.0) 

def CrackLifeCalc(r1, rt1, or1, BasicParams, DataImport, sigma_nom, nTemp, LifeMin):
    ff00_wu_proxy = lambda a: get_crack_aspect_ratio_Wu(a, r1, rt1, BasicParams)
    σ_f_List, rnList, Nd, PnList, RnAS, orList = generateGrains(r1, rt1, or1, BasicParams, DataImport, ff00_analytical=ff00_wu_proxy)

    ai, nai = get_fatigue_eval_points(r1, rt1, rnList, BasicParams, DataImport)

    S_Δδ0 = np.zeros(nai)
    S_K0 = np.zeros(nai)
    S_U0 = np.zeros(nai)
    S_S_op0 = np.zeros(nai)
    S_c0 = np.zeros(nai)
    S_asp0 = np.zeros(nai)
    S_cyc0 = np.full(nai, 1e18) 

    dNdaList_init = [[0,0] for _ in range(nai // 3 + 2)] 

    unstable = 0
    for i in range(nai):
        aii = ai[i]

        ar, K, U, DeltaSigma_eff, sigma_op = calc_driving_force(
            i, aii, r1, rt1, BasicParams, DataImport, sigma_nom
        )

        cc, DeltaDelta, unstable_ctsd, goto = evalCTSD(i, aii, DeltaSigma_eff, 
                                                  σ_f_List, orList, rnList, Nd, PnList, RnAS, 
                                                  BasicParams, DataImport)

        if goto == 1 or unstable_ctsd == 1: 
            unstable = 1
            break

        S_c0[i] = cc
        S_Δδ0[i] = DeltaDelta
        S_K0[i] = K
        S_U0[i] = U
        S_S_op0[i] = sigma_op
        S_asp0[i] = ar

        if i >= BasicParams.eval_num_stage1 and DeltaDelta < BasicParams.Δδ_th:
             unstable = 1 
             break

        S_cyc0, dNdaList_init, Label = evalCycle(
            i, cc, rnList, ai, nTemp, 
            S_Δδ0, S_cyc0, dNdaList_init, LifeMin, BasicParams
        )

        if Label == 1: 
            unstable = 1
            break

    if unstable == 1:
        S_cyc0[i:] = 1e18 

    results_dict = {
        "Crack depth (a)": ai,
        "Cycles (N)": S_cyc0,

        "K_max": S_K0,
        "CTSD (Δδ)": S_Δδ0,
        "Closure (U)": S_U0,
        "Opening Stress": S_S_op0,
        "Slip band (c)": S_c0,
        "Aspect Ratio (a/c)": S_asp0
    }

    micro_data = (σ_f_List, orList, Nd, PnList, RnAS, rnList, rt1, r1)
    return S_cyc0, results_dict, micro_data

def main_monte_carlo():
    print("Starting Analytical Monte Carlo Simulation...")

    BasicParams = BasicParameters()
    DataImport = MaterialDataImporter(BasicParams)

    for sigma_nom in BasicParams.sigma_nom_list:
        print(f"\n=== Processing Sigma_nom = {sigma_nom} MPa ===")

        for iteration_val in BasicParams.iteration_num:
            print(f"\n>>> Starting Run: Iteration {iteration_val} for {sigma_nom} MPa <<<")

            best_snapshot = BestLifeSnapshot(BasicParams=BasicParams)
            min_life = 1e18
            update_count = 0

            nTemp = 1e18
            LifeMin = lambda x: 1e18 

            FList = makeFList(BasicParams, DataImport)
            total_generated_grains = len(FList)
            num_to_test =  int(total_generated_grains * BasicParams.grain_size_lim)
            grains_to_test = FList[:num_to_test]
            total_grains = len(grains_to_test)

            print(f"  Calculating {total_grains} filtered grains...")

            update_step_grain = max(1, total_grains // 100)
            if total_grains > 30: update_step_grain = 10

            for grain_idx, r0 in enumerate(grains_to_test):
                if (grain_idx + 1) % update_step_grain == 0:
                    progress_percent = ((grain_idx + 1) / total_grains) * 100
                    print(f"  ...Grain: {grain_idx + 1}/{total_grains} ({progress_percent:.1f}%) | Min Life: {min_life:.2e} | Updates: {update_count}", end='\r')                

                aspg = DataImport.FirstGrainAspectCDFr(random.random())
                r1 = aspg * r0 * 0.5
                rt1 = r0 * 0.5 / aspg
                or1 = BasicParams.makeEulerAngles(1)[0] 

                try:
                    S_cyc, results_dict, micro_data_packed = CrackLifeCalc(
                        r1, rt1, or1, BasicParams, DataImport, sigma_nom, nTemp, LifeMin 
                    )
                    current_life = S_cyc[-1]

                    if current_life < min_life:
                        min_life = current_life
                        update_count += 1
                        best_snapshot.valid = True

                        (best_snapshot.sigma_f_List, best_snapshot.orList, best_snapshot.Nd, 
                         best_snapshot.PnList, best_snapshot.RnAS, best_snapshot.rnList, 
                         best_snapshot.rt1, best_snapshot.r1) = micro_data_packed
                        best_snapshot.detailed_results = results_dict

                        nTemp = min_life
                        a_champ = results_dict["Crack depth (a)"]
                        N_champ = results_dict["Cycles (N)"]
                        LifeMin = interp1d(a_champ, N_champ, kind='linear', fill_value=1e18, bounds_error=False)

                except Exception as e:
                    pass 

            print() 

            if best_snapshot.valid:
                print(f"--- Finished Iteration {iteration_val}. Best life: {min_life:.2e} cycles. ---")

                final_data = best_snapshot.detailed_results.copy()
                max_len = len(final_data["Crack depth (a)"])
                rn_list = best_snapshot.rnList if isinstance(best_snapshot.rnList, list) else best_snapshot.rnList.tolist()
                rn_padded = rn_list + [np.nan] * (max_len - len(rn_list))
                final_data["Grain Boundaries"] = rn_padded[:max_len]
                df_life = pd.DataFrame(final_data)

                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

                filename_life = f"LifeTo{BasicParams.T}mm_{BasicParams.steel}_{sigma_nom}MPa_R{BasicParams.R_ratio}_Iter{iteration_val}_{timestamp}.csv"

                df_life.to_csv(filename_life, index=False)
                print(f"  Saved: {filename_life}")

            else:
                print(f"\n--- Warning: No valid cracks grew in Iteration {iteration_val}. ---")

if __name__ == "__main__":
    main_monte_carlo()