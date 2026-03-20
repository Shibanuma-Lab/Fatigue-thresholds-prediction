from itertools import groupby
import os
import re
import math # The math library is used to perform mathematical operations
import numpy as np
import pandas as pd
import random
import pickle
import datetime
from scipy import interpolate
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from dataclasses import dataclass, field
import warnings
from scipy.special import ellipe
warnings.filterwarnings("ignore", category=RuntimeWarning)
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# ==============================================================================
# 1. Data Structures & Basic Parameters
# ==============================================================================

@dataclass
class BestLifeSnapshot:
    valid: bool = False
    y: float = 0.0
    z: float = 0.0
    r1: float = None
    rt1: float = None
    # --- MODIFIED: 使用 numpy 数组以节省大量内存 ---
    # 对于大裂纹，这些列表可能包含数百万个元素
    sigma_f_List: np.ndarray = field(default_factory=lambda: np.array([]))
    orList: np.ndarray = field(default_factory=lambda: np.array([]))
    rnList: np.ndarray = field(default_factory=lambda: np.array([]))
    Nd: np.ndarray = field(default_factory=lambda: np.array([]))
    # PnList 和 RnAS 仍然是列表，但它们的 *元素* 将变成紧凑的 numpy 数组
    PnList: list = field(default_factory=list) 
    RnAS: list = field(default_factory=list)
    
    BasicParams: object = None
    detailed_results: dict = field(default_factory=dict)

class BasicParameters:
    def __init__(self):
        # --- 1. Specimen Geometry (Direct Definition) ---
        self.W = 1e9   
        self.T = 20.5
        self.type = "CS" 
        self.fixed_aspect_ratio = None
        # None = 使用 Wu 氏公式演化
        # 1.0  = 强制半圆
        # --- 2. 高风险区域 (Active Zone) 定义 ---
        # (来自您 L107...v1.py 的设置)
        self.y_lim = 5 # [mm]
        self.z_lim = 5 # [mm]
        self.grain_size_lim = 0.1         # Grain size filter ---

        # --- 3. 区域单元 (Area Element) 定义 ---
        # (我们将整个高风险区视为一个“单元”)
        self.active_element_area = self.y_lim * self.z_lim # (1.5 * 2.5 = 3.75 mm^2)



        # --- 2. Loading Conditions ---
        # 名义应力列表 (MPa)
        self.sigma_nom_list = np.arange(260, 261, 5) # 例如: [100, 105]
        self.R_ratio = -1 # Stress Ratio (min/max)
        self.loading_type = "Tension" # "Tension" or "Bending"

        # --- 3. Material & Microstructure Constants ---
        self.crystal = "BCC"
        self.FirstGrain = 'Ferrite'
        self.SecondGrain = 'Pearlite'
        current_dir = os.path.basename(os.path.dirname(os.path.abspath(__file__)))

        # 按 "_" 分割，取第一段
        self.steel = current_dir.split("_")[0]   
        self.iteration_num = np.arange(0,10,1) #iteration times

        # Parameters related to fissure aperture ratio
        self.closure_type = 2 # The closure type of the specimen

        # --- 4. Physics Constants ---
        self.k_tr = 2 * 14  # For opening stress calculation
        self.c_paris = 18 # (Placeholder, adjust as needed)
        self.n_paris = 1.8  # (Placeholder)
        self.Δδ_th = 0.000063 # Threshold CTSD [mm]

        # (Derived parameters for compatibility if needed later)
        self.thickness = self.T
        self.width = self.W
        #Evaluation points
        self.eval_num_stage1 = None #The number of evaluation points in Stage I (within first grains)
        self.eval_num_stage2 = None #The number of evaluation points in Stage II (outside second grains)
        self.eval_num_total = None #The total number of evaluation points
        self.eval_lenghth_lim = None #Evaluation limination of crack lenghth
        self.eval_points_full= None #Position of evaluation points(stage1+stage2)
        self.eval_points_stage1 = None #Position of evaluation points in Stage I (within first grains)
        self.eval_points_stage2 = None #Position of evaluation points in Stage II (outside the first grains)
        self.calculate_evaluation_points()

    def SlipPlane(self, ang, DeltaSigma):
        """
        [FIXED] Calculates maximum shear stress (tau) AND its vectors (normal n, slip s).
        Essential for the full evalCTSD logic.
        """
        ang = np.atleast_2d(ang)
        if DeltaSigma.ndim == 2:
            DeltaSigma = DeltaSigma[np.newaxis, ...]
        N = ang.shape[0]

        if self.crystal == "BCC":
            # BCC Slip Systems {110}<111> (6 planes * 2 directions = 12 systems)
            # Dim 1 (6): Planes
            # Dim 2 (3): [Normal vector, Slip dir 1, Slip dir 2]
            nv = np.array([
                [[1., 1., 0.], [1., -1., 1.], [-1., 1., 1.]],
                [[1., -1., 0.], [1., 1., 1.], [-1., -1., 1.]],
                [[1., 0., 1.], [1., 1., -1.], [-1., 1., 1.]],
                [[-1., 0., 1.], [-1., 1., -1.], [1., 1., 1.]],
                [[0., 1., 1.], [1., 1., -1.], [1., -1., 1.]],
                [[0., 1., -1.], [1., 1., 1.], [1., -1., -1.]]
            ])
        else:
            # (Can add FCC here if needed later)
            raise ValueError("Only BCC supported currently")

        # Normalize vectors
        nv = nv / np.linalg.norm(nv, axis=2, keepdims=True)

        # Euler angles to Rotation Matrix (ZXZ)
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

        # Rotate all slip systems for all grains: (N, 6, 3, 3)
        # Dim 2 is still [Normal, Dir1, Dir2]
        nv2 = np.einsum('nij,pdj->npdi', g, nv)

        # Calculate Resolved Shear Stress for all 12 systems
        # tau = | n * sigma * b |
        # n is nv2[:, :, 0, :], b is nv2[:, :, 1:, :]
        n_vec = nv2[:, :, 0, :]       # (N, 6, 3)
        b_vecs = nv2[:, :, 1:, :]     # (N, 6, 2, 3)
        
        # Einsum to compute shear for all systems at once
        # npi: normal(p) component i
        # nij: sigma component ij
        # npdj: slip_dir(p, d) component j
        # -> result npd: (Sample, Plane, Dir_index)
        tau_matrix = np.abs(np.einsum('npi,nij,npdj->npd', n_vec, DeltaSigma, b_vecs)) # (N, 6, 2)

        # Find best system for each grain
        tau_flat = tau_matrix.reshape(N, -1) # (N, 12)
        max_linear_idx = np.argmax(tau_flat, axis=1) # (N,)

        # Extract max tau value
        max_shear = tau_flat[np.arange(N), max_linear_idx]

        # Unravel index to find which plane (p) and which direction (d)
        # 12 systems = 6 planes * 2 directions/plane
        p_idx, d_idx_sub = np.divmod(max_linear_idx, 2) 
        
        # Extract the corresponding vectors
        # d_idx_sub is 0 or 1. In nv2, directions are at indices 1 and 2.
        # So we need nv2 index to be d_idx_sub + 1.
        
        idx_n = np.arange(N)
        theta_n = nv2[idx_n, p_idx, 0, :]           # Best normal vector (N, 3)
        theta_s = nv2[idx_n, p_idx, d_idx_sub+1, :] # Best slip vector (N, 3)

        # If input was a single grain (1D array), return 1D arrays (scalars/vectors)
        # to match original code's likely behavior for simple calls
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



            # --- 自动生成 eval_points_stage2 开始 ---
            points = []

            # 0.01 ~ 0.30 (step 0.01)
            points.extend(np.arange(0.01, 0.31, 0.01))

            # 0.35 ~ 1.00 (step 0.05)
            points.extend(np.arange(0.35, 1.01, 0.05))

            # 1.1 ~ 1.5 (step 0.1)
            points.extend(np.arange(1.1, 1.51, 0.1))

            # 1.5 之后
            if self.T > 1.5:
                points.extend(np.arange(2.0, self.T, 0.5))

            # 最终清理：
            # 1. 过滤掉任何可能大于等于 self.T 的点 (以防 T 很小，例如 T=0.05 时不应有 0.06)
            # 2. 使用 round(x, 2) 解决浮点数精度问题 (例如避免出现 0.30000000004)
            self.eval_points_stage2 = [round(p, 2) for p in points if p < self.T]


                    
def get_fatigue_eval_points(r1, rt1, rnList, BasicParams, DataImport):
    """
    [RENAMED] Originally 'get_EvalPoints'.
    Used for FAST fatigue life integration during Monte Carlo loops.
    Points are sparse and focused on grain boundaries.
    """

    Ai1 = BasicParams.eval_points_stage1(r1)
    Ai1[-1] = r1 - DataImport.dave / 1000#

    rni0 = np.array([2, 3] + [1 + int(sum(np.heaviside(ai - rnList,0))) for ai in BasicParams.eval_points_stage2])-1
    rni1 = np.unique(rni0)
    #rni1 = np.arange(1, len(rnList) )
    rni = rni1[1:] if rni1[0] == 0 else rni1 

    Ai2 = np.array([[rnList[rn - 1] + DataImport.dave / 1000, 0.5 * (rnList[rn] + rnList[rn - 1]), rnList[rn] - DataImport.dave / 1000] for rn in rni]).flatten().tolist()
    ai = sorted(Ai1 + Ai2)
    nai = len(ai) 

    surfai = ai.copy() 
    surfai[:BasicParams.eval_num_stage1] = [2* rt1] * BasicParams.eval_num_stage1

    return ai, len(ai)

# ==============================================================================
# 2. Analytical K Calculator (Infinite Plate - Figure 2.19)
# ==============================================================================

class AnalyticalKCalculator:
    """
    Implements the simplified stress-intensity factor equation for 
    semi-elliptical surface cracks in an INFINITE plate.
    Reference: Figure 2.19 from user provided text.
    """
    @staticmethod
    def calculate_K(a, c,  St):
        """
        Calculates K_I using the Infinite Plate assumption.
        Note: 't' (thickness) and 'W' (width) arguments are kept for interface 
        compatibility but are IGNORED in this infinite model.
        """
        if a <= 0 or c <= 0: return 0.0
        
        # 限制 a/c <= 1.0，因为图片中的公式注明 "valid ... as a <= c"
        # 如果 a > c，通常需要交换坐标轴定义，这里简单钳位以保持公式有效性
        a_c = min(a / c, 1.0)
        
        phi_rad =np.pi/2
        sin_phi = np.sin(phi_rad)
        cos_phi = np.cos(phi_rad)

        # --- 1. 形状因子 Q ---
        Q = 1 + 1.464 * (a_c)**1.65

        # --- 2. 表面修正因子 lambda_s ---
        # lambda_s = [1.13 - 0.09(a/c)] * [1 + 0.1(1 - sin(phi))^2]
        lambda_s = (1.13 - 0.09 * a_c) * (1 + 0.1 * (1 - sin_phi)**2)

        # --- 3. 角度函数 f(phi) ---
        # f(phi) = [sin^2(phi) + (a/c)^2 * cos^2(phi)]^(1/4)
        f_phi = (sin_phi**2 + (a_c)**2 * cos_phi**2)**0.25

        # --- 4. 计算 K_I ---
        # K_I = lambda_s * sigma * sqrt(pi*a/Q) * f(phi)
        # (这里我们暂只考虑拉伸应力 St，因为无限板通常只讨论远场拉伸)
        K = lambda_s * St * np.sqrt(np.pi * a / Q) * f_phi

        return K
    
    # ==============================================================================
# 3. Material Data Importer (Restored CSV reading for properties)
# ==============================================================================

class MaterialDataImporter:
    def __init__(self, BasicParams):
        self.BasicParams = BasicParams

        # --- 1. Material Elastic Properties ---
        # (Originally from .inp file. Hardcoded now as .inp is removed. 
        #  You can also read these from a CSV if needed.)
        self.E = 206000.0 # Young's Modulus [MPa]
        self.ν = 0.3      # Poisson's ratio
        self.AA = self.E / (4 * np.pi * (1 - self.ν ** 2))


        # --- 2. Read Material Properties from CSV ---
        # Restored original methods to read from CSV files
        self.monotonic_tensile_properties()
        self.friction_strength()

        # --- 3. Read Grain Statistics from CSV ---

        self.FirstGrainDF = pd.read_csv(f"{self.BasicParams.steel}_{BasicParams.FirstGrain} grain size.csv", header=None).to_numpy()
        self.FirstGrainAspectDF = pd.read_csv(f"{self.BasicParams.steel}_{BasicParams.FirstGrain} grain aspect ratio.csv", header=None).to_numpy()
        angle_file = os.path.join(f"{self.BasicParams.steel}_{self.BasicParams.FirstGrain} grain angle.csv")
        if os.path.exists(angle_file):
            self.FirstGrainAngleDF = pd.read_csv(angle_file, header=None).to_numpy()
        else:
            self.FirstGrainAngleDF = None # Handle missing angle file gracefully
        
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

        # --- 新增：恢复 ngAe (来自 L107...v1.py) ---
        self.ngAe = (4 * self.BasicParams.active_element_area) / (np.pi * self.dave**2)
        # --- 修改结束 ---
        
        print(f"Microstructure loaded from CSVs: dave={self.dave:.4f}, dmax={self.dmax:.4f}")
# --- 4. Calculate Volume Fractions (PRateN) ---
        # 必须在 ModelSize 之前计算
        self.calculate_VolumeFraction()

        # --- 5. Dynamic Model Size Calculation (Restored) ---
        # 根据试件尺寸和微观结构动态计算 Ng 和 Mg
        self.ModelSize()

        # --- 6. Generate/Load Microstructure Database (gData) ---
        self.CreateGrainData()

    def ModelSize(self):
        """
        Dynamically calculates the required number of grains (Ng) and layers (Mg)
        based on specimen thickness and microstructure statistics.
        Restored from original user code.
        """
        # 1. Estimate average grain area (A0) using Monte Carlo
        n_samples = 50000
        # Part 1: Ferrite area contributions
        r_vals_F = np.random.rand(int(n_samples * (1 - self.PRateN)))
        part1 = (self.FirstGrainCDF(r_vals_F))**2
        
        # Part 2: Pearlite/Second phase area contributions
        if self.PRateN > 0:
            r_vals_P1 = np.random.rand(int(n_samples * self.PRateN))
            r_vals_P2 = np.random.rand(int(n_samples * self.PRateN))
            part2 = self.FirstGrainCDF(r_vals_P1) * self.SecondGrainCDF(r_vals_P2)
            A0_samples = np.concatenate((part1, part2))
        else:
            A0_samples = part1
        
        mean_A0 = np.mean(A0_samples)

        # 2. Determine safety factor fw (width factor)
        # (Assuming standard specimen if 'type' is not listed, use 3 as default)
        fw = 3.0 
        # if getattr(self.BasicParams, 'type', '') == "Smooth": fw = 6.0

        # 3. Calculate Ng (Total grains in fracture process zone)
        # Ng = fw * (Total Area / Average Grain Area)
        # Area approx = 2 * Thickness^2 (heuristic from original code)
        self.Ng = round(fw * ((2 * self.BasicParams.T**2) / mean_A0))

        # 4. Calculate Mg (Total layers along depth)
        # d0 = average depth increment per layer
        r_vals_d0 = np.random.rand(n_samples)
        d0_samples = self.FirstGrainCDF(r_vals_d0) * self.FirstGrainAspectCDFr(r_vals_d0)
        mean_d0 = np.mean(d0_samples)
        
        self.Mg = round((self.BasicParams.T * fw) / mean_d0)

        print(f"Dynamic Model Size: Ng={self.Ng}, Mg={self.Mg} (fw={fw})")


    def calculate_VolumeFraction(self):
        # Renamed from VolumeFraction to avoid confusion and made it a method
        if self.BasicParams.steel in ["Bainite", "Martensite", 'H'] or self.PRate == 0:
            self.PRateN = 0.0
        else:

            n_samples = 100000
            r_vals = np.random.rand(n_samples)
            VP = np.mean(self.SecondGrainCDFr(r_vals) * self.FirstGrainCDFr(r_vals))
            VF = np.mean(self.FirstGrainCDFr(np.random.rand(n_samples)) ** 2)
            self.PRateN = (self.PRate / VP) / (self.PRate / VP + (1 - self.PRate) / VF)

        # print(f"PRateN calculated: {self.PRateN}")

    # ... (monotonic_tensile_properties, friction_strength, makeCDF... 保持不变) ...
    # ... (请确保包含这些之前已有的辅助方法) ...
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
        if self.BasicParams.steel == "H":
            self.σ_fF = float(df.iloc[0, 0])
            self.σ_fP = self.σ_fF
        else:
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

        if self.BasicParams.steel in ["Bainite", "Martensite", "H"]:
             dtrList = [[row[0], self.FirstGrainCDFr(row[2]), 0, self.FirstGrainAspectCDFr(row[1]), 0] for row in R.T]
        else: 
            if self.BasicParams.type == "CS":
                dtrList = [[row[0], self.FirstGrainCDFr(row[1]), self.SecondGrainCDFr(row[2]), self.FirstGrainAspectCDFr(row[3]), np.pi / 180 * self.FirstGrainAngleCDF(row[4])] for row in R.T]
            else:
                dtrList = [[row[0], self.FirstGrainCDFr(row[1]), self.SecondGrainCDFr(row[1]), self.FirstGrainAspectCDFr(row[3]), np.pi / 180 * self.FirstGrainAngleCDF(row[4])-np.pi/2] for row in R.T]

        # Use a helper to avoid lambda pickling issues if any
        self.gData = [self.makegList(*x) for x in dtrList]
        with open(pkl_name, "wb") as file: pickle.dump(self.gData, file)
        print("gData generated.")

    def makegList(self, r, d, t, ra, ang):
        if r > self.PRateN:
            if self.BasicParams.type == "CS":
                return [
                    self.σ_fF,
                    d * np.sqrt(np.sqrt((np.cos(ang)**2 + ra**4 * np.sin(ang)**2) / (np.sin(ang)**2 + ra**4 * np.cos(ang)**2))),
                    d * np.sqrt(np.sqrt((np.sin(ang)**2 + ra**4 * np.cos(ang)**2) / (np.cos(ang)**2 + ra**4 * np.sin(ang)**2)))
                ]
            else:
                return [
                    self.σ_fF,
                    d * np.sqrt(np.sqrt((np.sin(ang)**2 + ra**4 * np.cos(ang)**2) / (np.cos(ang)**2 + ra**4 * np.sin(ang)**2))),
                    d * np.sqrt(np.sqrt((np.cos(ang)**2 + ra**4 * np.sin(ang)**2) / (np.sin(ang)**2 + ra**4 * np.cos(ang)**2)))
                ]
        else:
            if self.BasicParams.type == "CS":
                return [
                    self.σ_fP,
                    math.sqrt(t * d * ra * math.sqrt((t**2 * math.cos(ang)**2 + d**2 * ra**2 * math.sin(ang)**2) /(t**2 * math.sin(ang)**2 + d**2 * ra**2 * math.cos(ang)**2))),
                    math.sqrt(t * d * ra * math.sqrt((t**2 * math.sin(ang)**2 + d**2 * ra**2 * math.cos(ang)**2) /(t**2 * math.cos(ang)**2 + d**2 * ra**2 * math.sin(ang)**2)))
                ]
            else:
                return [
                    self.σ_fP,
                    math.sqrt(t * d * ra * math.sqrt((t**2 * math.sin(ang)**2 + d**2 * ra**2 * math.cos(ang)**2) /(t**2 * math.cos(ang)**2 + d**2 * ra**2 * math.sin(ang)**2))),
                    math.sqrt(t * d * ra * math.sqrt((t**2 * math.cos(ang)**2 + d**2 * ra**2 * math.sin(ang)**2) /(t**2 * math.sin(ang)**2 + d**2 * ra**2 * math.cos(ang)**2))),

                ]
# ==============================================================================
# 4. Global Helper Functions (Pure Analytical)
# ==============================================================================
def makeFList(BasicParams, DataImport):
    """
    [RESTORED] Original 'space-filling' algorithm from ElementLife.
    Simulates finding the largest grains within a single Area Element.
    """
    Aej = BasicParams.active_element_area # area of area element Ae
    d = 0 # flag for loop termination
    fww = 25
    n_max= round(fww * DataImport.ngAe) #
    FList0 = [[] for _ in range(n_max)] 

    i = 0
    # (此循环用于模拟用随机晶粒“填充”一个区域单元)
    while d != 1 and i < n_max: # (增加 i < n_max 保护)
        PorF = random.random()
        if PorF > DataImport.PRateN:
            i += 1
            fd = DataImport.FirstGrainCDF(random.random()) #

            if Aej < (np.pi * fd**2) / 4: #
                fd = np.sqrt((4 * Aej) / np.pi)
                d = 1
            FList0.append(fd)
            Aej -= (np.pi * fd**2) / 4
        else:
            # (处理第二相，如果存在)
            if not hasattr(DataImport, 'SecondGrainCDF') or DataImport.SecondGrainCDF is None:
                continue # 如果没有第二相，跳过
                
            r = random.random()
            pw = DataImport.SecondGrainCDF(r) #
            pl = DataImport.FirstGrainCDF(r)

            if Aej < np.pi / 4 * pw * pl:
                pw = (4 * Aej) / (np.pi * pl)
                d = 1
            Aej -= np.pi / 4 * pw * pl

    FList = [item for sublist in FList0 for item in (sublist if isinstance(sublist, list) else [sublist])]
    sorted_FList = sorted(FList, reverse=True) #
    return sorted_FList

def get_crack_aspect_ratio_Wu(a_curr, r1, rt1, BasicParams):
    """
    MODIFIED: Now checks BasicParams.fixed_aspect_ratio first.
    If fixed_aspect_ratio is set (e.g., 1.0), it overrides Wu's formula.
    
    Parameters:
    -----------
    a_curr : float
        Current crack depth (a).
    r1 : float
        Crack depth at the end of Stage I (a0).
    rt1 : float
        Crack half-length during Stage I (c0).
    BasicParams : object
        Contains global settings, including fixed_aspect_ratio.
        
    Returns:
    --------
    ar_curr : float
        Current aspect ratio (a/c).
    """
    # --- 新增：检查是否使用固定纵横比 ---
    if BasicParams.fixed_aspect_ratio is not None:
        return BasicParams.fixed_aspect_ratio
    # --- 修改结束 ---

    # --- Stage I (保持不变) ---
    # 表面长度固定，只有深度增加
    if a_curr <= r1:
         # 保护措施：防止 a_curr 为 0 导致纵横比为 0 (如果下游有除法可能会有问题)
         # 但物理上 a/c 确实是从 0 开始增长的。
         return max(1e-9, a_curr) / rt1

    # --- Stage II (Wu's Formula) ---
    r0 = r1 / rt1               # Stage II 起始时的纵横比
    # 计算公式中的各项

    term2 = (r1 / a_curr)**3 * (1 - r0**(-3))
    bracket_term = 1 - term2
    rw_val = bracket_term**(-1/3)
    
    return rw_val

def analytical_sec_ell(asp):
    """
    Analytical replacement for the old DataImport.sec_ell interpolation.
    Calculates the normalized arc length of a quarter ellipse.
    Returns L/c where L is quarter perimeter and c is surface half-length.
    """
    # Ensure asp is not negative or zero to avoid errors
    asp = max(1e-6, asp)
    
    if asp <= 1.0:
        # c is major axis (or equal). Standard formula L = c * E(1 - (a/c)^2)
        # Normalized by c: just E(1 - asp^2)
        m = 1.0 - asp**2
        return ellipe(m)
    else:
        # a is major axis. L = a * E(1 - (c/a)^2)
        # Normalized by c: (a/c) * E(1 - (1/asp)^2)
        m = 1.0 - (1.0/asp)**2
        return asp * ellipe(m)
    
def generateGrains(r1, rt1, or1, BasicParams, DataImport, ff00_analytical):
    """
    MODIFIED: MEMORY OPTIMIZED VERSION.
    Converts massive microstructure lists to compact numpy arrays to prevent RAM overload
    for long cracks.
    """
    def L2(asp0, b_val): 
        if b_val * asp0 < BasicParams.W: 
             return b_val * analytical_sec_ell(asp0)
        else:
             return BasicParams.W

    g_data_arr = np.asarray(DataImport.gData) # Ensure this is numpy read-only if possible
    Ng = DataImport.Ng  

    rand_indices = np.random.randint(0, len(g_data_arr), size=Ng)
    selected_gdata = g_data_arr[rand_indices]
    
    # 1. Compact standard lists
    first_grain = np.array([[float(DataImport.σ_fF), float(r1), float(rt1)]])
    g_List = np.vstack((first_grain, selected_gdata)) # (Ng+1, 3) array

    σ_f_List = g_List[:, 0] # View, no extra memory
    dnList = g_List[:, 1]
    tnList = g_List[:, 2]
    
    # 2. Compact orientations
    # Pre-allocate full orientation array
    orList_full = np.zeros((Ng + 1, 3), dtype=np.float32)
    orList_full[0] = or1
    orList_full[1:] = BasicParams.makeEulerAngles(Ng)
    
    # Pre-allocate layer arrays (using smaller dtypes where possible)
    rnList = np.zeros(DataImport.Mg, dtype=np.float32)
    Nd = np.zeros(DataImport.Mg, dtype=np.int32)
    
    # PnList and RnAS must remain as lists of arrays because each layer has different size (ragged)
    PnList = [None] * DataImport.Mg
    RnAS = [None] * DataImport.Mg

    rnList[0] = r1
    Nd[0] = 1
    # Layer 0 has no Pn/RnAS predecessor

    n, n0, n1 = 0, 0, 0
    RnAA_prev = np.array([1.0]) # <--- 添加这行初始化
    while rnList[n] < 1.05 * BasicParams.T:
        # ... (中间的几何相交判断逻辑保持完全不变) ...
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
        
        # --- 内存优化点：使用 numpy 计算和存储层级数据 ---
        # RnAA (cumulative sum of area fractions)
        RnA_layer = tnList[n0:n1+1] / tn_sum if tn_sum > 0 else np.ones(Nd[n+1])/Nd[n+1]
        RnAA_layer = np.cumsum(RnA_layer)
        
        rnList[n + 1] = rnList[n] + np.dot(RnA_layer, dnList[n0:n1+1])

        # PnList generation (kept logic, but optimized storage)
        # PnList[n] needs to store Nd[n] + Nd[n+1] - 1 pairs.
        num_pairs = Nd[n] + Nd[n+1] - 1
        current_Pn = np.zeros((num_pairs, 2), dtype=np.int32) # Compact numpy array!
        
        if num_pairs > 0:
            current_Pn[num_pairs - 1] = [Nd[n]-1, Nd[n + 1]-1] # 0-based index adjustment
            current_Pn[0] = [0, 0]
            
            # Need previous layer's RnAA. Re-calculate or store?
            # Better re-calculate on the fly to save memory, or keep just last one.
            # Actually we need it for the loop. Let's keep track of just the last two RnAA if possible,
            # but RnAS needs full history.
            
            # Re-generating previous RnAA for pairing logic (a bit slow but saves tons of memory)
            # Actually, let's just store RnAA temporarily for this loop.
            if n == 0:
                 RnAA_prev = np.array([1.0])
            else:
                 # Recover previous RnAA from its grains.
                 # This is tricky. Let's just store RnAA_prev from last iteration.
                 pass # Handled by loop structure below
            
            k0, k1 = 1, 1
            for j in range(1, num_pairs - 1):
                # Safe indexing
                idx0 = min(k0 - 1, len(RnAA_prev)-1)
                idx1 = min(k1 - 1, len(RnAA_layer)-1)
                
                if RnAA_prev[idx0] < RnAA_layer[idx1]: k0 += 1
                else: k1 += 1
                current_Pn[j] = [k0-1, k1-1]
        
        # --- 关键优化：直接存为 numpy 数组 ---
        PnList[n] = current_Pn 
        
        # RnAS generation
        RnAAS_layer = np.sort(np.concatenate((RnAA_prev, RnAA_layer[:-1])))
        RnAS_layer = np.diff(np.concatenate(([0.0], RnAAS_layer)))
        RnAS[n] = RnAS_layer.astype(np.float32) # Compact float32

        # Prepare for next iteration
        RnAA_prev = RnAA_layer
        n += 1

        # Safety break if out of pre-allocated memory
        if n >= DataImport.Mg - 1:
             print("Warning: Reached max layers (Mg) in generateGrains.")
             break

    # Trim and return
    return (σ_f_List, 
            rnList[:n+1], 
            Nd[:n+1], 
            PnList[:n], # List of numpy arrays
            RnAS[:n],   # List of numpy arrays
            orList_full)

def calc_driving_force(i, aii, r1, rt1, BasicParams, DataImport, sigma_nom_val):
    """
    MODIFIED: correctly handles sigma_nom_val as Stress Amplitude.
    """
    # --- 修正步骤 0: 将应力幅 (Amplitude) 转换为最大应力 (Max Stress) ---
    # 公式: S_max = 2 * Amp / (1 - R)
    # 注意: 防止 R=1 导致除以零（虽然疲劳通常 R<1）

    S_nom = sigma_nom_val 
    #S_nom = 2 * sigma_nom_val / (1.0 - BasicParams.R_ratio)

    # === Li: 等效最大应力 S_eq_max（用于算K的驱动力） ===
    n_prime = 0.2014
    K_prime = 1206.0  # MPa

    E = DataImport.E  # 你代码里是 MPa（206000） :contentReference[oaicite:4]{index=4}

    # 你之前讨论的“m' 应为倒数”——这里用 (1/n') 形式：
    # σ_eq = σ_app + E * (σ_app / K')^{1/n'}
    S_eq = S_nom + E * ((S_nom / K_prime) ** (1.0 / n_prime))

    # 1. Determine current crack shape (a/c)
    if aii <= r1: 
        c_curr = rt1
        ar_curr = aii / c_curr
    else: 
        ar_curr = get_crack_aspect_ratio_Wu(aii, r1, rt1, BasicParams)
        c_curr = aii / ar_curr
        
    # 2. Calculate Max K (at deepest point phi=pi/2)
    # --- 修正点: St 输入改为 S_nom_max ---
    K = AnalyticalKCalculator.calculate_K(a=aii, c=c_curr, St=S_eq)

    # --- 关键修改：计算等效远程应力 ---
    if aii > 1e-9:
        S_app = K / np.sqrt(np.pi * aii)
    else:
        # --- 修正点: 对于极小裂纹，等效最大应力应为名义最大应力 ---
        S_app = S_nom

    # 3. Calculate Closure (U)
    # 闭合效应通常基于名义应力比 R 和最大应力水平
    # 使用等效最大应力可能更准确地反映裂尖附近的塑性状态
    S_max = 2 * S_app / (1.0 - BasicParams.R_ratio)
    S_min = BasicParams.R_ratio * S_max
    sop_ratio_long = calculate_closure_Newman(BasicParams.R_ratio, S_max, DataImport.σ_0)

    # 4. Construct Effective Stress Tensor for Microstructure
    # 使用等效应力来构建张量
    S_op_long_crack = sop_ratio_long * S_max
    
    # (来自您原始 FEM 代码 L1653 的逻辑)
    # 张开应力 S_op(a) 从 S_min 增长到 S_op_long_crack
    S_op_transient = S_min + (S_op_long_crack - S_min) * (1 - np.exp(-BasicParams.k_tr * aii))
    
    # 确保 S_op 不会超过长裂纹的稳态值
    S_op_transient = min(S_op_transient, S_op_long_crack)
    
    # --- 3c. 计算当前的有效 U ---
    # U = (S_max - S_op_transient) / (S_max - S_min)
    if S_max - S_min > 0:
        U = (S_max - S_op_transient) / (S_max - S_min)
    else:
        U = 1.0 # 如果 R=1, DeltaS=0
        
    U = max(min(U, 1.0), 0.0) # 钳位在 [0, 1]

    # 4. 构造有效应力张量
    DeltaS_eff = U * (S_max - S_min)
    DeltaSigma_eff = np.zeros((3, 3))
    DeltaSigma_eff[1, 1] = DeltaS_eff
    DeltaSigma_eff[2, 2] = 0.3 * DeltaS_eff     # Δσ_zz (plane strain)
    # 5. 张开应力值 (用于报告)
    sigma_op = S_op_transient

    return ar_curr, K, U, DeltaSigma_eff, sigma_op


def get_arrest_eval_points(r1, BasicParams):
    """
    [Updated] Generates evaluation points for final detailed analysis.
    - Uses FIXED function name as requested.
    - Stage I: Standard sparse points.
    - Stage II: High-density log-spaced grid (log_step=0.02) with DYNAMIC thickness.
    """
    # --- Stage I Points ---
    ai_stage1 = list(BasicParams.eval_points_stage1(r1))
    ai_stage1[-1] = r1 - 1e-6 # Ensure strictly less than r1


    # --- Stage II Points (High Density Log Grid) ---
    d_start = r1
    # MODIFIED: Use dynamic thickness from BasicParams instead of hardcoded 4.3
    d_end = BasicParams.T * 0.98 
    
    log_d_start = np.log10(d_start)
    log_d_end = np.log10(d_end)
    
    # MODIFIED: Requested high density step
    log_step = 0.01
    
    # Generate grid (including potential overshoot)
    log_points_overshoot = np.arange(log_d_start, log_d_end + log_step, log_step)
    # Cut off points exceeding d_end
    log_points = log_points_overshoot[log_points_overshoot <= log_d_end]
    
    ai_stage2 = 10**log_points
    
    # --- Combine & Sort ---
    # Use set to avoid duplicates at r1 boundary, then sort
    ai = sorted(list(set(ai_stage1 + ai_stage2.tolist())))
    nai = len(ai)
    
    print(f"  [R-Curve] Generated {len(ai_stage1)} (Stage I) + {len(ai_stage2)} (Stage II) points (log_step={log_step}).")
    
    return ai, nai

# ==============================================================================
# 4. Global Helper Functions (Continued - Core Physics)
# ==============================================================================
def optimized_grouping(Pn, RnAS_slice, τ0, t0):
    """
    [FINAL FIXED] Calculates weighted average of stress (tau) and tensor (t)
    for grains in the next layer based on area fractions.
    Robustly handles both scalar (N=1) and array (N>1) inputs.
    """
    # 1. Standardize inputs to numpy arrays with guaranteed shapes
    RnAS_arr = np.array(RnAS_slice)
    τ0_arr = np.array(τ0)
    t0_arr = np.array(t0)

    # Safety: Ensure τ0 is at least 1D (shape: [N_pairs])
    if τ0_arr.ndim == 0: 
        τ0_arr = τ0_arr.reshape(1)
    # Safety: Ensure t0 is at least 3D (shape: [N_pairs, 3, 3])
    if t0_arr.ndim == 2: 
        t0_arr = t0_arr.reshape(1, 3, 3)
        
    # 2. Group indices by target grain (assuming Pn is sorted by layer/target)
    groups = []
    # enumerate(Pn) yields (index, element). 
    # If Pn is numpy array, 'element' is a row [g_prev, g_curr].
    # We group by g_curr (x[1][1]).
    for key, group in groupby(enumerate(Pn), key=lambda x: x[1][1]):
        indices = [idx for idx, _ in group]
        groups.append(indices)
        
    # 3. Calculate weighted averages for each group
    t_gs, τ_gs = [], []
    for indices in groups:
        idxs = np.array(indices)
        weights = RnAS_arr[idxs]
        rnas = np.sum(weights)
        
        if rnas <= 1e-30: # Avoid division by zero
            τ_gs.append(0.0)
            t_gs.append(np.zeros((3,3)))
        else:
            # Weighted average for scalar tau
            # (weights automatically broadcasts with 1D τ0_arr[idxs])
            τ_val = np.sum(τ0_arr[idxs] * weights) / rnas
            τ_gs.append(τ_val)
            
            # Weighted average for tensor t
            # (weights must be reshaped to (N, 1, 1) to broadcast against (N, 3, 3) t0_arr)
            t_val = np.sum(t0_arr[idxs] * weights.reshape(-1, 1, 1), axis=0) / rnas
            t_gs.append(t_val)
            
    return t_gs, τ_gs

def evalCTSD(i, aii, Δσ, σ_f_List, orList, rnList, Nd, PnList, RnAS, BasicParams, DataImport):
    """
    MODIFIED: Moved to global scope.
    Calculation of CTSD for each evaluation point
    """
    #aii : crack depth ;
    #Δσ : stress range tensor ;
    #σ_f_List : friction strength list ;
    #orList : orientation list ;
    #rnList : grain boundary depth list ;
    #Nd : number of grains in each layer ;
    #PnList : adjacent grains of each grain in each layer ;
    #RnAS : Percentage of distance between each of the two grain boundaries by combining the two layers
    #----------------------------------------------sub_function__START------------------------------------------------
    def safe_arccos(x):
        return np.arccos(np.clip(x, -1, 1))
    
    
    def CC1(a, σ_fr, rnlist):
        def Eq_1(c, a, σ_fr):
            epsilon = 1e-10 # Avoid dividing by zero
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
            epsilon = 1e-10 # Avoid dividing by zero
            return np.pi * 0.5 - σ_fr[0] * safe_arccos(a / (c + epsilon)) - sum((σ_fr[i+1] - σ_fr[i]) * safe_arccos(rnList[i] / (c + epsilon)) for i in range(n+1))
        
        upper_bound = rnList[n] + 10000
        try:
            c_root = brentq(Eq_n, a + 1e-10, upper_bound, args=(a, n, σ_fr, rnList))
            return c_root
        except ValueError:
            c_root = 10000
        return c_root
    

    def gg(x, c, ad):# 
        if x == ad:
            a = (1 - 1e-3) * ad
        else:
            a = ad

        ca = np.sqrt(c**2 - a**2)
        cx = np.sqrt(c**2 - x**2)
        return a * np.log(abs((ca + cx) / (ca - cx))) - x * np.log(abs((x * ca + a * cx) / (x * ca - a * cx)))

    def CTSD(a, c, n, σ_fr, Δτ_j, rnList_L, DataImport):#calculation of CTSD

        if a == 0:
            term1 = 0
        else:
            term1 = 2 * a * σ_fr[0] * np.log(c / a)
    
        term2 =  sum((σ_fr[i + 1] - σ_fr[i]) * gg(a, c, rnList_L[i]) for i in range(n+1))
        result = Δτ_j / (np.pi ** 2 * DataImport.AA) * (term1 + term2)
        return result.item()
    #----------------------------------------------sub_function__END------------------------------------------------


    def optimized_grouping(Pn, RnAS_slice, τ0, t0):

        # 对 Pn 进行分组：直接对 enumerate(Pn) 按 key 分组，key 为 item[1]
        groups = []
        for key, group in groupby(enumerate(Pn), key=lambda x: x[1][1]):
            indices = [idx for idx, _ in group]
            groups.append(indices)
        
        # 转换权重和数据为 numpy 数组
        RnAS_arr = np.array(RnAS_slice)      # 形状 (n,)
        τ0_arr = np.array(τ0)                # 形状 (n, 3, 3)
        t0_arr = np.array(t0)                # 形状 (n, 3, 3)
        
        t_gs = []
        τ_gs = []
        for indices in groups:
            indices = np.array(indices)
            rnas = np.sum(RnAS_arr[indices])
            if rnas == 0:
                τ_avg = np.zeros_like(τ0_arr[indices][0])
                t_avg = np.zeros_like(t0_arr[indices][0])
            else:
                # 将权重重塑为 (n,1,1) 以便与 (n,3,3) 进行逐元素相乘
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

        σ_fr[0] = 1 / (RnAS[jj-1].dot([τ_t2[0][i] / σ_f0[i] for i in range(len(σ_f0))]))#
        τ1 = RnAS[jj-1].dot(τ_List[0])


    if σ_fr[0] < 1:
        cc = 1.05 * BasicParams.thickness
        Δδ = 0
        
        for m in range(len(rnList)):

            
            # 判断是否满足提前终止条件
            if cc < rnList[jj + m - 1]:
                Δδ = CTSD(aii, cc, m - 1, σ_fr, τ1, rnList[jj - 1:jj + m],DataImport)
                break
            elif jj + m + 1 > len(Nd):
                unstable = 1
                goto = 1
                break

            # 计算 Nd 累计和：sum(Nd[0:jj+m])

            N0 = sum(Nd[0:jj + m])


            # 转换 PnList 子项为 numpy 数组

            PnList_1_2 = np.array(PnList[jj + m - 1])

            
            # 分离 PnList1 与 PnList2

            PnList1 = PnList_1_2[:, 0].tolist()
            PnList2 = PnList_1_2[:, 1].tolist()

            
            # 计算 τ_t0_temp：调用 SlipPlane 对每个索引

            or_array = np.array([orList[N0 + PnList2[i]] for i in range(len(PnList2))])
            t_array = np.array([t_List[m][PnList1[i]] for i in range(len(PnList1))])
            τ_t0 = BasicParams.SlipPlane(or_array, t_array)                            

            
            # 提取 τ0 并计算 t0

            τ0 = τ_t0[0]  # 第一组数据
        # 检查传递给 SlipPlane 的晶粒数 N
            N_grains = or_array.shape[0]

            if N_grains == 1:
                # N=1 的情况: τ_t0[0] 是标量, τ_t0[1] 和 [2] 是向量
                τ0 = [τ_t0[0]]  # 修复: 将 0-D 标量包装在列表中，使其可迭代
                t0 = [τ_t0[0] * np.outer(τ_t0[1], τ_t0[2])]
            else:
                # N > 1 的情况: τ_t0[0] 已经是 1D 数组 (可迭代)
                τ0 = τ_t0[0]  # 保持不变 (已经是 1D 数组)
                t0 = [t[0] * np.outer(t[1], t[2]) for t in zip(*τ_t0)]


            
            # 分组操作及加权平均（使用 optimized_grouping）

            Pn_current = PnList[jj + m - 1]
            t_gs, τ_gs = optimized_grouping(Pn_current, RnAS[jj + m - 1], τ0, t0)

            
            # 计算 σ_f0 和 σ_fr[m+1]

            σ_f0 = [σ_f_List[N0 + p[1]] for p in PnList[jj + m - 1]]
            τ0_arr = np.array(τ0)       # shape (N, ...)
            σ_f0_arr = np.array(σ_f0)   # shape (N,)
            temp_array = τ0_arr / σ_f0_arr  # 逐元素除法，得到 shape (N, ...)
            σ_fr[m + 1] = 1 / (RnAS[jj + m - 1].dot(temp_array))

            
            # 赋值操作：复制 τ_gs, t_gs

            τ_List[m + 1] = τ_gs.copy()
            t_List[m + 1] = t_gs.copy()

            
            # 计算 cc

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
            cc = CC1(aii, σ_fr, rnList[0]) # calculate the length of the slip band in Stage I
            Δδ = 0
            for m in range(len(rnList)):

                
                # 判断是否满足提前终止条件
                if cc < rnList[jj + m - 1]:
                    Δδ = CTSD(aii, cc, m - 1, σ_fr, τ1, rnList[jj - 1:jj + m], DataImport)
                    break
                elif jj + m + 1 > len(Nd):
                    unstable = 1
                    goto = 1
                    break

                # 计算 Nd 累计和：sum(Nd[0:jj+m])

                N0 = sum(Nd[0:jj + m])


                # 转换 PnList 子项为 numpy 数组

                PnList_1_2 = np.array(PnList[jj + m - 1])

                
                # 分离 PnList1 与 PnList2

                PnList1 = PnList_1_2[:, 0].tolist()
                PnList2 = PnList_1_2[:, 1].tolist()

                
                # 计算 τ_t0_temp：调用 SlipPlane 对每个索引

                or_array = np.array([orList[N0 + PnList2[i]] for i in range(len(PnList2))])
                t_array = np.array([t_List[m][PnList1[i]] for i in range(len(PnList1))])
                τ_t0 = BasicParams.SlipPlane(or_array, t_array)                            

                
                # 提取 τ0 并计算 t0
                N_grains = or_array.shape[0]
                if N_grains == 1:
                    # N=1 的情况: τ_t0[0] 是标量, τ_t0[1] 和 [2] 是向量
                    τ0 = [τ_t0[0]]  # 修复: 将 0-D 标量包装在列表中，使其可迭代
                    t0 = [τ_t0[0] * np.outer(τ_t0[1], τ_t0[2])]
                else:
                    # N > 1 的情况: τ_t0[0] 已经是 1D 数组 (可迭代)
                    τ0 = τ_t0[0]  # 保持不变 (已经是 1D 数组)
                    t0 = [t[0] * np.outer(t[1], t[2]) for t in zip(*τ_t0)]


                
                # 分组操作及加权平均（使用 optimized_grouping）

                Pn_current = PnList[jj + m - 1]
                t_gs, τ_gs = optimized_grouping(Pn_current, RnAS[jj + m - 1], τ0, t0)

                
                # 计算 σ_f0 和 σ_fr[m+1]

                σ_f0 = [σ_f_List[N0 + p[1]] for p in PnList[jj + m - 1]]
                τ0_arr = np.array(τ0)       # shape (N, ...)
                σ_f0_arr = np.array(σ_f0)   # shape (N,)
                temp_array = τ0_arr / σ_f0_arr  # 逐元素除法，得到 shape (N, ...)
                σ_fr[m + 1] = 1 / (RnAS[jj + m - 1].dot(temp_array))


                # 赋值操作：复制 τ_gs, t_gs

                τ_List[m + 1] = τ_gs.copy()
                t_List[m + 1] = t_gs.copy()


                # 计算 cc

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

    # Stage I
    if i < BasicParams.eval_num_stage1:
        if i == 0:
            S_cyc0[i] = 0
            dNda = 0
        else:
            if S_Δδ0[i - 1] == 0 or S_Δδ0[i] == 0:
                #print("**************Abort current calculation1************")
                return S_cyc0, dNdaList

            dNda0 = 1 / (BasicParams.c_paris * S_Δδ0[i - 1]**BasicParams.n_paris)
            dNda1 = 1 / (BasicParams.c_paris * S_Δδ0[i]**BasicParams.n_paris)
            dNda = 0.5 * (dNda0 + dNda1)

            S_cyc0[i] = dNda * (ai[i] - ai[i - 1]) + S_cyc0[i - 1]
        NSuspend = NNf(S_cyc0[i] / nTemp)

        # To improve the calculation efficiency, the calculation is terminated when the number of cycles exceeds 10 times the minimum life
        if S_cyc0[i] > NSuspend * LifeMin(ai[i]):
            #print("**************Abort current calculation2************")
            Label = 1
            return S_cyc0, dNdaList , Label             

        if S_cyc0[i] > 10 * LifeMin(ai[i]):
            #print("**************Abort current calculation3************")
            Label = 1
            return S_cyc0, dNdaList, Label

        dNdaList[0] = [0, dNda]
        # print("dNdaList[0] : ",dNdaList[0])
    
    elif (i - BasicParams.eval_num_stage1 + 1) % 3 == 0:
        ai1, ai2 = ai[i - 1] - ai[i - 2], ai[i] - ai[i - 2]
        dNda0 = 1 / (BasicParams.c_paris * (S_Δδ0[i - 2]**BasicParams.n_paris - BasicParams.Δδ_th**BasicParams.n_paris))
        dNda1 = 1 / (BasicParams.c_paris * (S_Δδ0[i - 1]**BasicParams.n_paris - BasicParams.Δδ_th**BasicParams.n_paris)) - dNda0
        dNda2 = 1 / (BasicParams.c_paris * (S_Δδ0[i]**BasicParams.n_paris - BasicParams.Δδ_th**BasicParams.n_paris)) - dNda0

        if ai1 * ai2 * (ai1 - ai2) == 0:
            #print("**************Abort current calculation4************")
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
            #print("**************Abort current calculation5************")
            return S_cyc0, dNdaList, Label

        for ii in range(i - 2, i + 1):
            if S_cyc0[ii] > nTemp:
                Label = 1
                #print("**************Abort current calculation6************")
                return S_cyc0, dNdaList, Label
            NSuspend = NNf(S_cyc0[ii] / nTemp)

            if S_cyc0[ii] > NSuspend * LifeMin(ai[ii]):
                Label = 1
                #print("**************Abort current calculation7************")
                return S_cyc0, dNdaList, Label

            if S_cyc0[ii] > 10 * LifeMin(ai[ii]):
                Label = 1
                #print("**************Abort current calculation8************")
                return S_cyc0, dNdaList, Label

    return S_cyc0, dNdaList, Label




def calculate_closure_Newman(R, S_max, sigma_0, alpha=3):
    """
    Calculates crack opening stress ratio (S_op / S_max) using Newman's 
    analytical closure model (1984).
    
    Parameters:
    -----------
    R : float
        Stress ratio (S_min / S_max).
    S_max : float
        Maximum applied nominal stress.
    sigma_0 : float
        Flow stress (average of yield and ultimate tensile strength).
    alpha : float, optional
        Constraint factor. 
        alpha = 1.0 for plane stress, 3.0 for plane strain. 
        Default 2.5 is typical for small surface cracks.
        
    Returns:
    --------
    sop_ratio : float
        Ratio of opening stress to max stress (S_op / S_max).
    """
    # Ensure R is clamped to realistic limits for this empirical formula if needed,
    # but typically it works well for -1 < R < 1.
    
    # Normalize max stress
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
        # Fallback for extreme negative R, though formula usually holds ok down to -2
        sop_ratio = A0 - 2.0 * A1 

    # Opening stress cannot be less than minimum stress physically for simple loading
    # (Though mathematically it can be. We usually clamp it.)
    # Actually, if Sop < Smin, crack is fully open, U = 1/(1-R) theoretically? 
    # Standard practice: U = (Smax - Sop) / (Smax - Smin)
    # If Sop < Smin, effective range is full range Smax - Smin.
    
    real_sop_ratio = max(sop_ratio, R) # Ensure op stress isn't below min stress effectively
    
    return real_sop_ratio

def calculate_effective_range(S_max, S_min, sop_ratio):
    """
    Calculates the effective stress range ratio U = DeltaS_eff / DeltaS.
    """
    R = S_min / S_max
    # S_op = sop_ratio * S_max
    # DeltaS_eff = S_max - max(S_op, S_min)
    # DeltaS = S_max - S_min
    # U = DeltaS_eff / DeltaS
    # Simplifying:
    
    if sop_ratio >= R:
        U = (1.0 - sop_ratio) / (1.0 - R)
    else:
        U = 1.0 # Fully open
        
    return max(min(U, 1.0), 0.0) # Clamp between 0 and 1
# ==============================================================================
# 5. Life Evaluation (Analytical - FINALIZED)
# ==============================================================================

def CrackLifeCalc(r1, rt1, or1, BasicParams, DataImport, sigma_nom, nTemp, LifeMin):
    """
    FINAL Analytical version of CrackLifeCalc.
    MODIFIED: Accepts nTemp and LifeMin to restore early-exit optimization.
    Calls the full, original evalCycle function.
    """
    # 1. 生成微观结构
    ff00_wu_proxy = lambda a: get_crack_aspect_ratio_Wu(a, r1, rt1, BasicParams)
    σ_f_List, rnList, Nd, PnList, RnAS, orList = generateGrains(r1, rt1, or1, BasicParams, DataImport, ff00_analytical=ff00_wu_proxy)

    # 2. 获取评估点 (使用稀疏点)
    ai, nai = get_fatigue_eval_points(r1, rt1, rnList, BasicParams, DataImport)
    
    # 3. 初始化结果数组
    S_Δδ0 = np.zeros(nai)
    S_K0 = np.zeros(nai)
    S_U0 = np.zeros(nai)
    S_S_op0 = np.zeros(nai)
    S_c0 = np.zeros(nai)
    S_asp0 = np.zeros(nai)
    S_cyc0 = np.full(nai, 1e18) 
    
    # (确保 dNdaList 足够长以匹配 evalCycle 的需求)
    dNdaList_init = [[0,0] for _ in range(nai // 3 + 2)] 
    
    # 4. 主计算循环
    unstable = 0
    for i in range(nai):
        aii = ai[i]

        # A. 计算驱动力
        ar, K, U, DeltaSigma_eff, sigma_op = calc_driving_force(
            i, aii, r1, rt1, BasicParams, DataImport, sigma_nom
        )

        # B. 计算 CTSD
        cc, DeltaDelta, unstable_ctsd, goto = evalCTSD(i, aii, DeltaSigma_eff, 
                                                  σ_f_List, orList, rnList, Nd, PnList, RnAS, 
                                                  BasicParams, DataImport)
        
        if goto == 1 or unstable_ctsd == 1: 
            unstable = 1
            break
        
        # C. 存储中间结果
        S_c0[i] = cc
        S_Δδ0[i] = DeltaDelta
        S_K0[i] = K
        S_U0[i] = U
        S_S_op0[i] = sigma_op
        S_asp0[i] = ar

        # D. 检查门槛 (Stage II)
        if i >= BasicParams.eval_num_stage1 and DeltaDelta < BasicParams.Δδ_th:
             unstable = 1 # 止裂
             break

        # E. 积分寿命 (调用原始的 evalCycle)
        # nTemp 和 LifeMin 现在是动态传入的
        S_cyc0, dNdaList_init, Label = evalCycle(
            i, cc, rnList, ai, nTemp, 
            S_Δδ0, S_cyc0, dNdaList_init, LifeMin, BasicParams
        )

        if Label == 1: # evalCycle 触发了提前中止
            unstable = 1
            break
            
    if unstable == 1:
        # 确保未计算的点保持 1e18
        S_cyc0[i:] = 1e18 

    # --- 打包所有结果 ---
    results_dict = {
        "Crack depth (a)": ai,
        "Cycles (N)": S_cyc0,
        # "da/dN": (evalCycle 不返回这个, 省略)
        "K_max": S_K0,
        "CTSD (Δδ)": S_Δδ0,
        "Closure (U)": S_U0,
        "Opening Stress": S_S_op0,
        "Slip band (c)": S_c0,
        "Aspect Ratio (a/c)": S_asp0
    }
    
    micro_data = (σ_f_List, orList, Nd, PnList, RnAS, rnList, rt1, r1)
    return S_cyc0, results_dict, micro_data

# ==============================================================================
# 6. Main Execution Loop (Monte Carlo Simulation)
# ==============================================================================

def main_monte_carlo():
    print("Starting Analytical Monte Carlo Simulation...")
    
    BasicParams = BasicParameters()
    DataImport = MaterialDataImporter(BasicParams)

    for sigma_nom in BasicParams.sigma_nom_list:
        print(f"\n=== Processing Sigma_nom = {sigma_nom} MPa ===")
        
        # --- MODIFIED: 遍历 iteration_num 数组 ---
        # 这样您可以在 BasicParams 里灵活定义要跑哪些 Iteration (例如 [1,2,3,4] 或 [5,6,7,8])
        for iteration_val in BasicParams.iteration_num:
            print(f"\n>>> Starting Run: Iteration {iteration_val} for {sigma_nom} MPa <<<")

            best_snapshot = BestLifeSnapshot(BasicParams=BasicParams)
            min_life = 1e18
            update_count = 0

            # 优化参数初始化
            nTemp = 1e18
            LifeMin = lambda x: 1e18 

            # 3. 准备晶粒
            FList = makeFList(BasicParams, DataImport)
            total_generated_grains = len(FList)
            num_to_test = min(20, int(total_generated_grains * BasicParams.grain_size_lim))
            grains_to_test = FList[:num_to_test]
            total_grains = len(grains_to_test)
                
            print(f"  Calculating {total_grains} filtered grains...")

            # 4. 晶粒计算循环
            update_step_grain = max(1, total_grains // 100)
            if total_grains > 30: update_step_grain = 10

            for grain_idx, r0 in enumerate(grains_to_test):
                
                # 进度显示
                if (grain_idx + 1) % update_step_grain == 0:
                    progress_percent = ((grain_idx + 1) / total_grains) * 100
                    print(f"  ...Grain: {grain_idx + 1}/{total_grains} ({progress_percent:.1f}%) | Min Life: {min_life:.2e} | Updates: {update_count}", end='\r')                
                # 初始化随机变量
                aspg = DataImport.FirstGrainAspectCDFr(random.random())
                r1 = aspg * r0 * 0.5
                rt1 = r0 * 0.5 / aspg
                or1 = BasicParams.makeEulerAngles(1)[0] 

                try:
                    # 计算寿命
                    S_cyc, results_dict, micro_data_packed = CrackLifeCalc(
                        r1, rt1, or1, BasicParams, DataImport, sigma_nom, nTemp, LifeMin 
                    )
                    current_life = S_cyc[-1]
                    
                    # 更新最弱环
                    if current_life < min_life:
                        min_life = current_life
                        update_count += 1
                        best_snapshot.valid = True
                        
                        # 解包微观数据
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

            print() # End progress bar
            
            # 5. 保存结果
            if best_snapshot.valid:
                print(f"--- Finished Iteration {iteration_val}. Best life: {min_life:.2e} cycles. ---")
                
                # 保存 Fatigue Life
                final_data = best_snapshot.detailed_results.copy()
                max_len = len(final_data["Crack depth (a)"])
                rn_list = best_snapshot.rnList if isinstance(best_snapshot.rnList, list) else best_snapshot.rnList.tolist()
                rn_padded = rn_list + [np.nan] * (max_len - len(rn_list))
                final_data["Grain Boundaries"] = rn_padded[:max_len]
                df_life = pd.DataFrame(final_data)
                
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # --- MODIFIED: 文件名包含 LifeToXmm, Stress, R, Iteration ---
                filename_life = f"LifeTo{BasicParams.T}mm_{BasicParams.steel}_{sigma_nom}MPa_R{BasicParams.R_ratio}_Iter{iteration_val}_{timestamp}.csv"
                
                df_life.to_csv(filename_life, index=False)
                print(f"  Saved: {filename_life}")

                # 计算 Arrest Condition (传入 iteration_val)
                calculate_arrest_condition(best_snapshot, DataImport, sigma_nom, iteration_val)
            else:
                print(f"\n--- Warning: No valid cracks grew in Iteration {iteration_val}. ---")
# ==============================================================================
# 7. Arrest Condition Calculation (Analytical R-Curve) [UPDATED with Aspect Ratio]
# ==============================================================================

def calculate_arrest_condition(snapshot, DataImport, sigma_nom_max, iteration_val):
    """
    Calculates the R-Curve (Arrest Condition) for the best snapshot using
    PURE ANALYTICAL methods.
    """
    BasicParams = snapshot.BasicParams
    
    print(f"--- Calculating Analytical Arrest Condition (R-Curve) ---")
    print(f"    Max search stress: {sigma_nom_max} MPa")

    r1, rt1 = snapshot.r1, snapshot.rt1
    
    # 1. Use high-density points for smooth R-curve plotting
    ai, nai = get_arrest_eval_points(r1, BasicParams)
    
    # 2. Initialize result arrays
    S_ai = np.zeros(nai)
    S_sigma_arrest = np.zeros(nai)
    S_K_arrest = np.zeros(nai)
    S_CTSD_arrest = np.zeros(nai)
    S_UU_arrest = np.zeros(nai)
    S_asp_arrest = np.zeros(nai) # --- MODIFIED: 新增数组存储长宽比 a/c ---
    
    # 3. Solver Settings
    MAX_ITER = 30
    TOL = 1e-3 

    for i in range(nai):
        aii = ai[i]
        S_ai[i] = aii
        
        if aii < 1e-9: continue 

        # --- Bisection Solver for sigma_arrest ---
        lo = 0.0
        hi = sigma_nom_max * 1.5 
        sigma_arrest = hi 

        # Check growth at max load first
        ar_hi, K_hi, U_hi, DSigma_hi, _ = calc_driving_force(i, aii, r1, rt1, BasicParams, DataImport, hi)
        
        grows_at_hi = False
        if aii <= r1: # Stage I
             tau_hi, _, _ = BasicParams.SlipPlane(snapshot.orList[0], DSigma_hi)
             if tau_hi.item() > DataImport.σ_fF: grows_at_hi = True
        else: # Stage II
             _, DDelta_hi, uns_hi, goto_hi = evalCTSD(i, aii, DSigma_hi, snapshot.sigma_f_List, snapshot.orList, snapshot.rnList, snapshot.Nd, snapshot.PnList, snapshot.RnAS, BasicParams, DataImport)
             if not (uns_hi or goto_hi) and DDelta_hi > BasicParams.Δδ_th: grows_at_hi = True
             
        if not grows_at_hi:
             S_sigma_arrest[i] = hi 
             S_K_arrest[i] = K_hi
             S_UU_arrest[i] = U_hi
             S_asp_arrest[i] = ar_hi # --- MODIFIED: 记录不扩展时的长宽比 ---
             continue

        # Bisection Loop
        for _ in range(MAX_ITER):
            mid = (lo + hi) / 2.0
            # Calculate Driving Force at 'mid'
            ar_mid, K_mid, U_mid, DSigma_mid, _ = calc_driving_force(i, aii, r1, rt1, BasicParams, DataImport, mid)
            
            # Check Growth Condition
            is_growing = False
            if aii <= r1: # Stage I
                tau_mid, _, _ = BasicParams.SlipPlane(snapshot.orList[0], DSigma_mid)
                if tau_mid.item() > DataImport.σ_fF: is_growing = True
            else: # Stage II
                _, DDelta_mid, unstable, goto = evalCTSD(i, aii, DSigma_mid, snapshot.sigma_f_List, snapshot.orList, snapshot.rnList, snapshot.Nd, snapshot.PnList, snapshot.RnAS, BasicParams, DataImport)
                if unstable or goto or DDelta_mid > BasicParams.Δδ_th:
                    is_growing = True

            if is_growing: hi = mid
            else: lo = mid
            
            if (hi - lo) < TOL:
                sigma_arrest = lo 
                break
        
        # === 循环结束，用最终确定的 sigma_arrest 重新计算一次精确状态 ===
        # 这里返回的 ar_final 就是最终的长宽比 a/c
        ar_final, K_final, U_final, DSigma_final, _ = calc_driving_force(
            i, aii, r1, rt1, BasicParams, DataImport, sigma_arrest
        )
        
        if aii > r1:
             _, DDelta_final, _, _ = evalCTSD(i, aii, DSigma_final, snapshot.sigma_f_List, snapshot.orList, snapshot.rnList, snapshot.Nd, snapshot.PnList, snapshot.RnAS, BasicParams, DataImport)
             S_CTSD_arrest[i] = DDelta_final
        
        S_sigma_arrest[i] = sigma_arrest
        S_UU_arrest[i] = U_final
        S_asp_arrest[i] = ar_final # --- MODIFIED: 保存最终的长宽比 ---
        
        if i % 2 == 0: 
            print(f"  i={i:3d}, a={aii:.4f} mm, Sigma_arrest={sigma_arrest:.2f} MPa, a/c={ar_final:.3f}")

    # 4. Save Results to CSV
    max_len = max(len(snapshot.rnList), nai)
    def pad(lst): return list(lst) + [np.nan] * (max_len - len(lst))
    # calculate K
    S_sigma_max = [2.0 * s / (1.0 - BasicParams.R_ratio) for s in S_sigma_arrest]  # R=-1 时等于 s
    S_K_arrest_nominal = []
    for a, asp, smax in zip(S_ai, S_asp_arrest, S_sigma_max):
        # asp = a/c  ->  c = a/asp
        c = a / asp
        # 用你现有的几何+Raju/Newman计算器算 K（这里函数名按你的工程里已有的来）
        # 关键：把应力输入换成 smax（由 Sigma_arrest 得到的名义应力最大值）
        Kmax = AnalyticalKCalculator.calculate_K(a, c,smax)
        S_K_arrest_nominal.append(Kmax)
    # 用名义K替换原来要输出的K
    S_K_arrest = S_K_arrest_nominal
    #######################################
    data = {
        "Crack depth (a)": pad(S_ai),
        "Arrest Stress": pad(S_sigma_arrest),
        "Arrest K_max": pad(S_K_arrest),
        "Arrest Closure (U)": pad(S_UU_arrest),
        "Aspect Ratio (a/c)": pad(S_asp_arrest), # --- MODIFIED: 添加到输出 ---
        "Arrest CTSD": pad(S_CTSD_arrest),
        "Grain Boundaries": pad(snapshot.rnList)
    }
    
    df = pd.DataFrame(data)
    timestamp1 = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # --- MODIFIED: 文件名包含 Iteration 和 R_ratio ---
    filename = f"R_Curve_Analytical_{BasicParams.steel}_{sigma_nom_max}MPa_R{BasicParams.R_ratio}_Iter{iteration_val}_{timestamp1}.csv"
    df.to_csv(filename, index=False)
    print(f"  Analytical R-Curve data saved to {filename}")
if __name__ == "__main__":
    main_monte_carlo()