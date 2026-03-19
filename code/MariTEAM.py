import math

class MariTEAM_Full_Precision:
    def __init__(self, ship_type, mcr_kw, v_design_kn, dwt=None):
        self.ship_type = ship_type
        self.MCR = mcr_kw  # kW
        self.V_design_ms = v_design_kn * 0.5144
        self.rho_s = 1025.0
        self.rho_a = 1.225
        self.g = 9.81
        self.nu = 1.188e-6

        # 1. 参数估算 (Table A.1)
        self.Lpp = 4.71 * (dwt**0.322) if ship_type == 'Container' else 4.04 * (dwt**0.334)
        self.B = 0.83 * (dwt**0.306) if ship_type == 'Container' else 0.84 * (dwt**0.316)
        self.Td = 0.33 * (dwt**0.296) if ship_type == 'Container' else 0.22 * (dwt**0.351)
        self.Cb = 0.65 if ship_type == 'Container' else 0.82
        self.Disp = self.Lpp * self.B * self.Td * self.Cb
        self.S = 0.99 * (self.Disp / self.Td + 1.9 * self.Lpp * self.Td)
        
        # 2. 计算校准因子 (修正单位换算)
        self.L_F = self._calculate_scaling_factor()

    def calculate_Rt_HM(self, V_stw, T_current):
        """计算对水速度下的阻力 (单位: N)"""
        Re = (V_stw * self.Lpp) / self.nu
        Cf = 0.075 / (math.log10(Re) - 2)**2
        Rf = 0.5 * self.rho_s * (V_stw**2) * self.S * Cf
        
        # 完善兴波阻力回归 (简化比例以匹配论文数量级)
        Fn = V_stw / math.sqrt(self.g * self.Lpp)
        Rw = self.Disp * self.rho_s * self.g * 0.002 * math.exp(Fn * 1.5)
        
        Rt = Rf * 1.2 + Rw + (0.5 * self.rho_s * V_stw**2 * self.S * 0.0004)
        return Rt

    def _calculate_scaling_factor(self):
        """校准算法 (注意 kW 换算)"""
        Rt_design = self.calculate_Rt_HM(self.V_design_ms, self.Td)
        eta_t = 0.70
        # Pb (kW) = (R(N) * V(m/s)) / (eta * 1000)
        Pb_calc_kw = (Rt_design * self.V_design_ms) / (eta_t * 1000)
        
        target_pb = 0.85 * self.MCR # 设定设计点为 85% MCR
        return target_pb / Pb_calc_kw

    def predict(self, sog_kn, current_kn, wind_speed_ms, wind_dir_deg, hs_m):
        """
        :param sog_kn: 对地速度 (Knots)
        :param current_kn: 水流速度 (Knots, 顺流为正, 逆流为负)
        :param wind_speed_ms: 视风速 (m/s)
        :param wind_dir_deg: 风向角 (0度为顶风)
        """
        # 1. 转化为对水速度 (Vs_stw)
        Vs_stw = (sog_kn - current_kn) * 0.5144
        if Vs_stw <= 0: Vs_stw = 0.1 # 防止速度为0报错
        
        # 2. 静水阻力 (N)
        Rt = self.calculate_Rt_HM(Vs_stw, self.Td)
        
        # 3. 风阻 (考虑风向角)
        # 修正：Raa = 0.5 * rho_a * V_rel^2 * Cx * Area
        Cx = 0.8 * math.cos(math.radians(wind_dir_deg))
        Axv = self.B * 15.0 
        Raa = 0.5 * self.rho_a * (wind_speed_ms**2) * Cx * Axv
        
        # 4. 波浪增阻
        Raw = 2 * self.rho_s * self.g * (hs_m**2) * (self.B**2 / self.Lpp)
        
        # 5. 总功率计算 (单位: kW)
        eta_t = 0.70
        Pb_total_kw = ((Rt + max(0, Raa) + Raw) * Vs_stw) / (eta_t * 1000)
        
        # 应用校准因子并限幅
        final_pb = Pb_total_kw * self.L_F
        return min(final_pb, self.MCR)

# --- 运行测试 ---
if __name__ == "__main__":
    # 创建模型
    model = MariTEAM_Full_Precision('Container', 20000, 20.0, 50000)
    
    # 输入：对地速度 18节，逆流 1节 (即对水 19节)，视风速 15m/s (30度侧顶风)，浪高 2米
    res = model.predict(sog_kn=18, current_kn=-1, wind_speed_ms=15, wind_dir_deg=30, hs_m=2.0)
    
    print(f"修正后的校准因子 L_F: {model.L_F:.4f}")
    print(f"考虑水流和风偏角的预测功率: {res:.2f} kW")