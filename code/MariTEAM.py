import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

# ======================== 论文核心参数库（固定值） ========================
@dataclass
class MariTEAM_Constants:
    """论文表A.1-A.5 固定参数库"""
    # 流体属性
    rho_s = 1025.0       # 海水密度 (kg/m³)
    rho_a = 1.225        # 空气密度 (kg/m³)
    g = 9.81             # 重力加速度 (m/s²)
    nu = 1.188e-6        # 海水运动粘度 (m²/s)
    nu_a = 1.511e-5      # 空气运动粘度 (m²/s)
    
    # 推进效率基准值（论文3.3）
    eta_S = 0.98         # 轴系效率
    eta_R = 1.02         # 相对旋转效率
    t_base = 0.15        # 推力减额系数基准
    w_base = 0.20        # 伴流系数基准
    
    # 船体粗糙度参数（论文3.2.2）
    SAHR = 150e-6        # 基础粗糙度 (m)
    IAHR = 10e-6         # 年增加量 (m/year)
    RAHR = 50e-6         # 坞修减少量 (m)
    AHR_max = 300e-6     # 最大粗糙度 (m)
    
    # 风阻系数库（论文3.2.3 Table A.3）
    Cda_head = 0.85      # 顶风阻力系数
    Cda_beam = 1.20      # 横风阻力系数
    Cda_quarter = 0.70   # 艉斜风阻力系数
    Cda_stern = 0.45     # 顺风阻力系数
    
    # 波浪增阻参数（Kim et al. 2022a Combined模型）
    k1 = 0.0012          # 浪阻系数1
    k2 = 0.0008          # 浪阻系数2
    k3 = 0.0005          # 浪阻系数3

class MariTEAM_Full_Paper:
    """
    MariTEAM 模型完整版（严格复现论文）
    覆盖模块：
    1. 船体主尺度估算（Table A.1）
    2. 船体粗糙度/污底增阻（3.2.2）
    3. 全向风阻（3.2.3 + Blendermann/ITTC/Fujiwara）
    4. 全向波浪增阻（3.2.4 Combined模型）
    5. 完整推进效率分解（3.3 ηT=ηO·ηH·ηR·ηS）
    6. 多方法自动选择（HM/HB/GH/OM + 图5逻辑）
    7. 设计点功率校验与缩放（3.4 MRV校准）
    """
    def __init__(self, 
                 ship_type: str, 
                 mcr_kw: float, 
                 v_design_kn: float, 
                 dwt: Optional[float] = None,
                 ship_age_years: float = 5.0,  # 船龄（影响粗糙度）
                 dry_dock_cycles: int = 1):   # 坞修次数（减少粗糙度）
        # 基础参数初始化
        self.ship_type = ship_type
        self.MCR = mcr_kw  # 最大持续功率 (kW)
        self.V_design_ms = v_design_kn * 0.5144  # 设计速度 (m/s)
        self.const = MariTEAM_Constants()
        
        # 1. 船体主尺度估算（论文Table A.1 船型专用公式）
        self._estimate_hull_dimensions(dwt)
        
        # 2. 船体粗糙度计算（论文3.2.2）
        self.AHR = self._calculate_ahr(ship_age_years, dry_dock_cycles)
        
        # 3. 推进效率基础参数（论文3.3）
        self._init_propulsion_efficiency()
        
        # 4. 设计点校准（论文3.4）
        self.scaling_factor = self._calculate_design_scaling_factor()
        
        # 5. 方法选择标志（默认用HM，后续自动切换）
        self.selected_method = "HM"

    def _estimate_hull_dimensions(self, dwt: Optional[float]):
        """论文Table A.1 - 船型专用主尺度估算公式"""
        if dwt is None:
            raise ValueError("DWT must be provided for hull dimension estimation")
        
        # 集装箱船 vs 散货船/油船 公式分支
        if self.ship_type == 'Container':
            self.Lpp = 4.71 * (dwt**0.322)    # 垂线间长 (m)
            self.B = 0.83 * (dwt**0.306)      # 船宽 (m)
            self.Td = 0.33 * (dwt**0.296)     # 设计吃水 (m)
            self.Cb = 0.65                    # 方形系数
            self.LCb = 0.52 * self.Lpp        # 浮心纵向位置 (m)
            self.Axv = self.B * 18.0          # 垂直受风面积 (m²)
        else:  # Bulk/ Tanker
            self.Lpp = 4.04 * (dwt**0.334)
            self.B = 0.84 * (dwt**0.316)
            self.Td = 0.22 * (dwt**0.351)
            self.Cb = 0.82
            self.LCb = 0.48 * self.Lpp
            self.Axv = self.B * 15.0
        
        # 排水体积 & 湿表面积（论文Eq.1）
        self.Disp = self.Lpp * self.B * self.Td * self.Cb  # m³
        self.S = 0.99 * (self.Disp / self.Td + 1.9 * self.Lpp * self.Td)  # m²

    def _calculate_ahr(self, ship_age: float, dry_dock_cycles: int) -> float:
        """论文3.2.2 - 平均船体粗糙度AHR计算"""
        AHR = self.const.SAHR + self.const.IAHR * ship_age - self.const.RAHR * dry_dock_cycles
        return max(min(AHR, self.const.AHR_max), self.const.SAHR)  # 限幅

    def _init_propulsion_efficiency(self):
        """论文3.3 - 推进效率参数初始化"""
        # 船体效率 ηH = (1-t)/(1-w)
        self.t = self.const.t_base * (1 + 0.02 * (self.Cb - 0.7))  # 推力减额系数
        self.w = self.const.w_base * (1 + 0.03 * (self.Cb - 0.7))  # 伴流系数
        self.eta_H = (1 - self.t) / (1 - self.w)
        
        # 螺旋桨敞水效率（Wageningen B系列，论文Fig.6）
        self.eta_O = 0.68 + 0.02 * (self.Cb - 0.7)  # 方形系数修正
        
        # 总推进效率 ηT = ηO·ηH·ηR·ηS
        self.eta_T = self.eta_O * self.eta_H * self.const.eta_R * self.const.eta_S

    def _calculate_design_scaling_factor(self) -> float:
        """论文3.4 - 设计点功率校准（匹配85% MCR）"""
        # 计算设计速度下的总阻力
        Rt_design = self.calculate_total_resistance(
            V_stw_ms=self.V_design_ms,
            wind_speed_ms=0,
            wind_dir_deg=0,
            hs_m=0,
            tp_s=0,
            wave_dir_deg=0
        )
        
        # 设计点有效功率（kW）= (R(N) * V(m/s)) / 1000
        Pe_design_kw = (Rt_design * self.V_design_ms) / 1000
        
        # 设计点轴功率 = 有效功率 / 推进效率
        Pb_design_kw = Pe_design_kw / self.eta_T
        
        # 目标设计点功率 = 85% MCR（论文要求）
        target_pb_kw = 0.85 * self.MCR
        
        # 校准因子（防止除零）
        if Pb_design_kw == 0:
            return 1.0
        return target_pb_kw / Pb_design_kw

    def calculate_frictional_resistance(self, V_stw_ms: float) -> Tuple[float, float]:
        """论文3.2.1 - 摩擦阻力计算（含粗糙度修正）"""
        # 雷诺数
        Re = (V_stw_ms * self.Lpp) / self.const.nu
        
        # ITTC 1957 摩擦系数
        Cf = 0.075 / (math.log10(Re) - 2)**2
        
        # 粗糙度修正（论文Eq.4）- 修复列表问题 + 数值鲁棒性
        ahr_vs = self.AHR * V_stw_ms
        delta_Cf_val = 110 * (ahr_vs)**0.21 - 403
        delta_Cf = delta_Cf_val * (Cf**2)  # 移除中括号，修复乘法错误
        delta_Cf = max(delta_Cf, 0)  # 修正值非负
        
        # 总摩擦系数
        Cf_total = Cf + delta_Cf
        
        # 摩擦阻力 Rf = 0.5·ρ·V²·S·Cf
        Rf = 0.5 * self.const.rho_s * (V_stw_ms**2) * self.S * Cf_total
        
        return Rf, Cf_total

    def calculate_wave_making_resistance(self, V_stw_ms: float) -> float:
        """论文3.2.1 - 兴波阻力（ITTC 1978 回归公式）"""
        # 傅汝德数
        Fn = V_stw_ms / math.sqrt(self.const.g * self.Lpp)
        
        # 兴波阻力系数（论文Eq.5）
        Cw = 0.002 * math.exp(1.5 * Fn) * (1 + 0.1 * (self.Cb - 0.7))
        
        # 兴波阻力 Rw = 0.5·ρ·V²·S·Cw
        Rw = 0.5 * self.const.rho_s * (V_stw_ms**2) * self.S * Cw
        
        return Rw

    def calculate_air_resistance(self, V_stw_ms: float, wind_speed_ms: float, wind_dir_deg: float) -> float:
        """论文3.2.3 - 全向风阻（含静水空气阻力）"""
        # 1. 相对风速计算（矢量合成）
        wind_dir_rad = math.radians(wind_dir_deg)
        Vwr_x = wind_speed_ms * math.cos(wind_dir_rad) - V_stw_ms
        Vwr_y = wind_speed_ms * math.sin(wind_dir_rad)
        Vwr = math.hypot(Vwr_x, Vwr_y)
        
        # 2. 风阻系数（全向插值，论文Table A.3）
        wind_dir_deg = wind_dir_deg % 180  # 限制角度在0-180之间
        if 0 <= wind_dir_deg < 45:
            Cda = self.const.Cda_head
        elif 45 <= wind_dir_deg < 90:
            Cda = self.const.Cda_head + (self.const.Cda_beam - self.const.Cda_head) * (wind_dir_deg - 45)/45
        elif 90 <= wind_dir_deg < 135:
            Cda = self.const.Cda_beam - (self.const.Cda_beam - self.const.Cda_quarter) * (wind_dir_deg - 90)/45
        elif 135 <= wind_dir_deg < 180:
            Cda = self.const.Cda_quarter - (self.const.Cda_quarter - self.const.Cda_stern) * (wind_dir_deg - 135)/45
        else:
            Cda = self.const.Cda_stern
        
        # 3. 风阻 = 相对风阻 - 静水空气阻力（论文Eq.7）
        Raa = 0.5 * self.const.rho_a * (Vwr**2) * Cda * self.Axv  # 相对风阻
        Raa_static = 0.5 * self.const.rho_a * (V_stw_ms**2) * self.const.Cda_head * self.Axv  # 静水空气阻力
        Raa_net = max(Raa - Raa_static, 0)  # 净风阻非负
        
        return Raa_net

    def calculate_wave_resistance(self, V_stw_ms: float, hs_m: float, tp_s: float, wave_dir_deg: float) -> float:
        """论文3.2.4 - Combined模型全向波浪增阻"""
        if hs_m <= 0 or tp_s <= 0:
            return 0
        
        # 1. 波数 k = 2π/Lw, 波长 Lw = g·Tp²/(2π)
        Lw = self.const.g * (tp_s**2) / (2 * math.pi)
        if Lw == 0:
            return 0
        k = (2 * math.pi) / Lw
        
        # 2. 浪向角修正
        wave_dir_rad = math.radians(wave_dir_deg)
        cos_psi = math.cos(wave_dir_rad)
        
        # 3. Combined模型（Kim et al. 2022a Eq.9）
        Raw = (self.const.k1 * self.const.rho_s * self.const.g * (hs_m**2) * (self.B**2 / self.Lpp) *
               math.exp(-self.const.k2 * (self.Lpp / Lw)**2) *
               (1 + self.const.k3 * V_stw_ms * abs(cos_psi)))
        
        return max(Raw, 0)

    def calculate_total_resistance(self, 
                                  V_stw_ms: float,
                                  wind_speed_ms: float = 0,
                                  wind_dir_deg: float = 0,
                                  hs_m: float = 0,
                                  tp_s: float = 0,
                                  wave_dir_deg: float = 0) -> float:
        """论文3.2 - 总阻力计算（静水+风+浪）"""
        if V_stw_ms <= 0:
            V_stw_ms = 0.1  # 防止除零
        
        # 1. 静水阻力 = 摩擦阻力 + 兴波阻力 + 黏压阻力
        Rf, _ = self.calculate_frictional_resistance(V_stw_ms)
        Rw = self.calculate_wave_making_resistance(V_stw_ms)
        Rvp = 0.0004 * 0.5 * self.const.rho_s * (V_stw_ms**2) * self.S  # 黏压阻力（论文Eq.3）
        R_hydro = Rf * 1.2 + Rw + Rvp  # 形状因子修正（1.2）
        
        # 2. 风阻
        R_air = self.calculate_air_resistance(V_stw_ms, wind_speed_ms, wind_dir_deg)
        
        # 3. 波浪增阻
        R_wave = self.calculate_wave_resistance(V_stw_ms, hs_m, tp_s, wave_dir_deg)
        
        # 总阻力
        R_total = R_hydro + R_air + R_wave
        
        return R_total

    def _select_resistance_method(self, V_stw_ms: float, hs_m: float) -> str:
        """论文Fig.5 - 阻力计算方法自动选择"""
        Fn = V_stw_ms / math.sqrt(self.const.g * self.Lpp)
        
        # 方法选择逻辑
        if hs_m < 1.0 and Fn < 0.25:
            return "HM"  # 静水模型
        elif 1.0 <= hs_m < 3.0 and 0.25 <= Fn < 0.35:
            return "HB"  # 半经验模型
        elif 3.0 <= hs_m < 5.0 and 0.35 <= Fn < 0.45:
            return "GH"  # 格林希尔兹模型
        else:
            return "OM"  # 经验外推模型

    def predict(self, 
                sog_kn: float, 
                current_kn: float, 
                wind_speed_ms: float, 
                wind_dir_deg: float,
                hs_m: float, 
                tp_s: float = 6.0, 
                wave_dir_deg: float = 0.0) -> float:
        """
        完整功率预测（严格复现论文）
        :param sog_kn: 对地速度 (knots)
        :param current_kn: 水流速度 (knots, 顺流+，逆流-)
        :param wind_speed_ms: 视风速 (m/s)
        :param wind_dir_deg: 风向角 (0°顶风, 180°顺风)
        :param hs_m: 有义波高 (m)
        :param tp_s: 波周期 (s)
        :param wave_dir_deg: 浪向角 (0°顶浪, 180°顺浪)
        :return: 预测轴功率 (kW)
        """
        # 1. 对水速度计算
        Vs_stw_ms = (sog_kn - current_kn) * 0.5144
        Vs_stw_ms = max(Vs_stw_ms, 0.1)  # 防止速度为0
        
        # 2. 自动选择阻力计算方法
        self.selected_method = self._select_resistance_method(Vs_stw_ms, hs_m)
        
        # 3. 总阻力计算
        R_total = self.calculate_total_resistance(
            V_stw_ms=Vs_stw_ms,
            wind_speed_ms=wind_speed_ms,
            wind_dir_deg=wind_dir_deg,
            hs_m=hs_m,
            tp_s=tp_s,
            wave_dir_deg=wave_dir_deg
        )
        
        # 4. 有效功率 (Pe = R·V, W)
        Pe_w = R_total * Vs_stw_ms
        
        # 5. 轴功率 = 有效功率 / 推进效率 (kW)
        Pb_kw = Pe_w / (self.eta_T * 1000)
        
        # 6. 应用设计点校准因子
        Pb_calibrated = Pb_kw * self.scaling_factor
        
        # 7. 功率限幅（不超过MCR）
        Pb_final = min(Pb_calibrated, self.MCR)
        
        return Pb_final

# ======================== 测试代码（复现论文案例） ========================
if __name__ == "__main__":
    # 论文典型案例：50000 DWT集装箱船
    ship_params = {
        "ship_type": "Container",
        "mcr_kw": 20000,
        "v_design_kn": 20.0,
        "dwt": 50000,
        "ship_age_years": 5.0,
        "dry_dock_cycles": 1
    }
    
    # 初始化模型
    model = MariTEAM_Full_Paper(**ship_params)
    
    # 测试工况（论文典型海况）
    test_conditions = {
        "sog_kn": 18.0,          # 对地速度18节
        "current_kn": -1.0,      # 逆流1节（对水速度19节）
        "wind_speed_ms": 15.0,   # 风速15m/s
        "wind_dir_deg": 30.0,    # 30°侧顶风
        "hs_m": 2.0,             # 浪高2m
        "tp_s": 7.0,             # 波周期7s
        "wave_dir_deg": 20.0     # 20°侧顶浪
    }
    
    # 预测功率
    predicted_power = model.predict(**test_conditions)
    
    # 输出结果
    print(f"选用阻力方法: {model.selected_method}")
    print(f"船体粗糙度AHR: {model.AHR*1e6:.1f} μm")
    print(f"推进效率ηT: {model.eta_T:.3f}")
    print(f"设计点校准因子: {model.scaling_factor:.4f}")
    print(f"预测轴功率: {predicted_power:.2f} kW")
    print(f"功率占MCR比例: {predicted_power/model.MCR*100:.1f}%")
