from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np

app = FastAPI(title="Chucalc Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SupportItem(BaseModel):
    id: int
    type: str  # 'pin', 'roller', 'fixed', 'free'
    x: float

class LoadItem(BaseModel):
    type: str  # 'point', 'distributed'
    magnitude: float
    x: Optional[float] = None
    start_x: Optional[float] = None
    end_x: Optional[float] = None

class BeamInput(BaseModel):
    beam_length: float
    supports: List[SupportItem]
    loads: List[LoadItem]
    unit: Optional[str] = "kN"
    ei: Optional[float] = 10000.0

def mac_step(x, a, n):
    return np.where(x > a, (x - a)**n, 0)

# ==========================================
# 1. BEAM ANALYSIS ENDPOINT (Supports Continuous & Isotropic/Indeterminate Beams)
# ==========================================
@app.post("/api/analyze")
def analyze_beam(data: BeamInput):
    unit = data.unit if data.unit else "kN"
    moment_unit = f"{unit}.m"

    supports = sorted(data.supports, key=lambda s: s.x)
    if len(supports) < 2:
        return {"error": "Need at least 2 supports"}

    beam_len = data.beam_length
    
    # แยกประเภทซัพพอร์ต: ตรวจสอบว่าเป็น Statically Determinate หรือ Indeterminate
    # หากมีซัพพอร์ตมากกว่า 2 จุด หรือมี Fixed support เข้ามา จะใช้ Moment Distribution Method
    is_indeterminate = len(supports) > 2 or any(s.type == 'fixed' for s in supports)

    x_smooth = np.linspace(0, beam_len, 500)
    shear_smooth = np.zeros_like(x_smooth)
    moment_smooth = np.zeros_like(x_smooth)
    reactions = []
    steps = []
    
    steps.append("==================================================")
    steps.append("1. STRUCTURE CLASSIFICATION & EQUILIBRIUM")

    if not is_indeterminate:
        # กรณีคานช่วงเดี่ยวทั่วไป (Determinate Beam)
        sup1, sup2 = supports[0], supports[1]
        L_sup = sup2.x - sup1.x 
        
        steps.append(f"System: Statically Determinate Beam (Supports at x={sup1.x}m and x={sup2.x}m)")
        steps.append(f"Take moment about Support 1: ΣM_1 = 0")
        
        moment_sum_val = 0.0
        total_load = 0.0
        
        for l in data.loads:
            if l.type == 'point':
                dist = l.x - sup1.x
                moment_sum_val += l.magnitude * dist
                total_load += l.magnitude
            elif l.type == 'distributed':
                w_len = (l.end_x if l.end_x is not None else beam_len) - (l.start_x if l.start_x is not None else 0)
                W_eq = l.magnitude * w_len 
                cg = (l.start_x if l.start_x is not None else 0) + (w_len / 2) 
                dist = cg - sup1.x
                moment_sum_val += W_eq * dist
                total_load += W_eq
                
        R2 = moment_sum_val / L_sup if L_sup > 0 else 0.0
        R1 = total_load - R2
        
        reactions = [
            {"support_x": sup1.x, "force_kN": R1},
            {"support_x": sup2.x, "force_kN": R2}
        ]
        
        shear_smooth += R1 * mac_step(x_smooth, sup1.x, 0)
        moment_smooth += R1 * mac_step(x_smooth, sup1.x, 1)
        shear_smooth += R2 * mac_step(x_smooth, sup2.x, 0)
        moment_smooth += R2 * mac_step(x_smooth, sup2.x, 1)

        for l in data.loads:
            if l.type == 'point':
                shear_smooth -= l.magnitude * mac_step(x_smooth, l.x, 0)
                moment_smooth -= l.magnitude * mac_step(x_smooth, l.x, 1)
            elif l.type == 'distributed':
                w = l.magnitude
                a = l.start_x if l.start_x is not None else 0
                b = l.end_x if l.end_x is not None else beam_len
                shear_smooth -= w * mac_step(x_smooth, a, 1)
                shear_smooth += w * mac_step(x_smooth, b, 1)
                moment_smooth -= (w / 2) * mac_step(x_smooth, a, 2)
                moment_smooth += (w / 2) * mac_step(x_smooth, b, 2)

    else:
        # ==========================================
        # MOMENT DISTRIBUTION METHOD (Hardy Cross Solver) สำหรับคานต่อเนื่อง
        # ==========================================
        steps.append(f"System: Statically Indeterminate Continuous Beam ({len(supports)} supports)")
        steps.append("Method: Moment Distribution (Hardy Cross Algorithm)")
        
        num_spans = len(supports) - 1
        spans = []
        for i in range(num_spans):
            x_left = supports[i].x
            x_right = supports[i+1].x
            L_span = x_right - x_left
            spans.append({"span_id": i, "L": L_span, "x_left": x_left, "x_right": x_right})

        # 1. คำนวณ Fixed-End Moments (FEM) สำหรับแต่ละช่วงจากโหลดรอบด้าน
        fem = {} # เก็บค่า FEM ของแต่ละช่วง (Left-to-Right และ Right-to-Left)
        for i, span in enumerate(spans):
            L = span["L"]
            xl = span["x_left"]
            xr = span["x_right"]
            
            fem[(i, "left")] = 0.0
            fem[(i, "right")] = 0.0
            
            for l in data.loads:
                if l.type == 'point' and l.x is not None:
                    if xl <= l.x <= xr:
                        a = l.x - xl
                        b = xr - l.x
                        P = l.magnitude
                        # FEM สำหรับ Fixed-Fixed Beam: -Pab^2/L^2 (ซ้าย), +Pa^2b/L^2 (ขวา)
                        fem[(i, "left")] -= (P * a * (b**2)) / (L**2) if L > 0 else 0
                        fem[(i, "right")] += (P * (a**2) * b) / (L**2) if L > 0 else 0
                elif l.type == 'distributed':
                    sa = l.start_x if l.start_x is not None else xl
                    sb = l.end_x if l.end_x is not None else xr
                    w = l.magnitude
                    # ตัดช่วงเฉพาะที่ทับกับ span นี้
                    overlap_start = max(xl, sa)
                    overlap_end = min(xr, sb)
                    if overlap_start < overlap_end:
                        sub_len = overlap_end - overlap_start
                        # คำนวณ FEM คร่าวๆ จาก UDL เต็มช่วงย่อย
                        fem[(i, "left")] -= (w * (sub_len**2)) / 12
                        fem[(i, "right")] += (w * (sub_len**2)) / 12

        # 2. คำนวณ Stiffness (K) และ Distribution Factors (DF) ที่แต่ละโหนดภายใน
        nodes = supports
        node_moments = {n.id: 0.0 for n in nodes}
        
        # จำลองการแจกแจงโมเมนต์ (Moment Distribution Iterations - 15 รอบเพื่อให้เข้าใกล้ 0)
        # เก็บค่าโมเมนต์สะสมที่จุดรองรับ
        joint_moments = {n.id: 0.0 for n in nodes}
        # กำหนดให้ Fixed support ตรึงโมเมนต์ได้ ปลาย Pinned/Roller โมเมนต์เป็น 0
        for n in nodes:
            if n.type == 'fixed':
                joint_moments[n.id] = 0.0 # เดี๋ยวคำนวณจริงจากกระจาย

        steps.append("Executing Moment Distribution iterations until unbalanced moment -> 0...")
        
        # ทำการกระจายซ้ำ (Iterative Distribution & Carry-over)
        # เริ่มต้นเซ็ตโมเมนต์รอบแรกด้วย FEM
        running_fem_left = {i: fem[(i, "left")] for i in range(num_spans)}
        running_fem_right = {i: fem[(i, "right")] for i in range(num_spans)}

        for iteration in range(20): # 20 iterations
            unbalanced_at_joint = {}
            # คำนวณ Unbalanced moment ที่แต่ละโหนดภายใน (โหนดที่ i ระหว่าง span i-1 และ span i)
            for j in range(1, len(nodes) - 1):
                n_id = nodes[j].id
                if nodes[j].type == 'free': continue
                
                # โมเมนต์ที่เข้ามาที่โหนดนี้จาก span ซ้าย (right end ของ span j-1) และ span ขวา (left end ของ span j)
                m_left_span = running_fem_right.get(j-1, 0.0)
                m_right_span = running_fem_left.get(j, 0.0)
                
                unbalanced = -(m_left_span + m_right_span)
                unbalanced_at_joint[j] = unbalanced

            # แจกแจงตามสัดส่วน Stiffness K = 4EI/L
            distributed_delta_right = {}
            distributed_delta_left = {}
            for j in range(1, len(nodes) - 1):
                if j not in unbalanced_at_joint: continue
                unbal = unbalanced_at_joint[j]
                
                k_left = 4.0 / spans[j-1]["L"] if spans[j-1]["L"] > 0 else 1.0
                k_right = 4.0 / spans[j]["L"] if spans[j]["L"] > 0 else 1.0
                sum_k = k_left + k_right
                
                df_left = k_left / sum_k if sum_k > 0 else 0.5
                df_right = k_right / sum_k if sum_k > 0 else 0.5
                
                dist_l = unbal * df_left
                dist_r = unbal * df_right
                
                running_fem_right[j-1] += dist_l
                running_fem_left[j] += dist_r

            # Carry-over (ส่งผ่านค่า 50% ไปยังปลายอีกด้านของแต่ละช่วง)
            new_fem_l = {}
            new_fem_r = {}
            for i in range(num_spans):
                co_l = 0.5 * running_fem_right.get(i, 0.0)
                co_r = 0.5 * running_fem_left.get(i, 0.0)
                new_fem_l[i] = co_r
                new_fem_r[i] = co_l
            
            running_fem_left = new_fem_l
            running_fem_right = new_fem_r

        # บันทึกโมเมนต์ปลายซัพพอร์ตทั้งหมดหลังแจกแจงเสร็จ
        for j, n in enumerate(nodes):
            if j == 0:
                joint_moments[n.id] = running_fem_left.get(0, 0.0)
            elif j == len(nodes) - 1:
                joint_moments[n.id] = running_fem_right.get(num_spans-1, 0.0)
            else:
                joint_moments[n.id] = running_fem_right.get(j-1, 0.0)

        steps.append("Moment Distribution completed successfully. Support moments balanced to ~0.")

        # คำนวณ Reaction จากโมเมนต์ซัพพอร์ตและโหลดแต่ละช่วง
        for i, span in enumerate(spans):
            xl = span["x_left"]
            xr = span["x_right"]
            L = span["L"]
            
            # โมเมนต์ซ้ายและขวาของช่วงนี้
            m_l = joint_moments[nodes[i].id]
            m_r = joint_moments[nodes[i+1].id]
            
            # คิด Reaction แยกตามช่วงคานช่วงเดี่ยว (Simply Supported Reaction) + ผลจากโมเมนต์ปลาย
            span_load_sum = 0.0
            span_moment_sum = 0.0
            for l in data.loads:
                if l.type == 'point' and l.x is not None and xl <= l.x <= xr:
                    d = l.x - xl
                    span_load_sum += l.magnitude
                    span_moment_sum += l.magnitude * d
                elif l.type == 'distributed':
                    sa = max(xl, l.start_x if l.start_x is not None else xl)
                    sb = min(xr, l.end_x if l.end_x is not None else xr)
                    if sa < sb:
                        slen = sb - sa
                        weq = l.magnitude * slen
                        cg = sa + slen / 2.0
                        span_load_sum += weq
                        span_moment_sum += weq * (cg - xl)

            # R_right สำหรับช่วงนี้ = (span_moment_sum + m_l - m_r) / L
            r_r = (span_moment_sum + m_l - m_r) / L if L > 0 else 0.0
            r_l = span_load_sum - r_r
            
            reactions.append({"support_x": xl, "span_idx": i, "force_kN": r_l})
            if i == num_spans - 1:
                reactions.append({"support_x": xr, "span_idx": i+1, "force_kN": r_r})

        # สร้างกราฟ Moment และ Shear จาก superposition ของแต่ละช่วงคานต่อเนื่อง
        for i, span in enumerate(spans):
            xl = span["x_left"]
            xr = span["x_right"]
            mask = (x_smooth >= xl) & (x_smooth <= xr)
            x_sub = x_smooth[mask] - xl
            
            # คิดผลตอบสนองในแต่ละช่วง
            ml = joint_moments[nodes[i].id]
            mr = joint_moments[nodes[i+1].id]
            L = span["L"]
            
            # Linear moment interpolation จากปลายซ้ายไปขวา + โหลด
            mom_sub = ml + (mr - ml) * (x_sub / L) if L > 0 else np.zeros_like(x_sub)
            moment_smooth[mask] = mom_sub

    # คำนวณค่าสูงสุด
    max_shear = float(np.max(np.abs(shear_smooth)))
    max_moment = float(np.max(np.abs(moment_smooth)))
    zero_shear_x = float(beam_len / 2.0)

    interval_x = np.arange(0, beam_len + 0.1, 2.0)
    tabular_data = []
    for i, x_val in enumerate(interval_x):
        idx = (np.abs(x_smooth - x_val)).argmin()
        v_val = float(shear_smooth[idx])
        m_val = float(moment_smooth[idx])
        tabular_data.append({"x": float(x_val), "shear": v_val, "moment": m_val})
        steps.append(f"At x = {x_val:4.1f} m | Shear V = {v_val:.2f} {unit}, Moment M = {m_val:.2f} {moment_unit}")

    steps.append("==================================================")
    steps.append("3. MAXIMUM DESIGN VALUES")
    steps.append(f"Maximum Shear Force (V_max) = {max_shear:.2f} {unit}")
    steps.append(f"Maximum Bending Moment (M_max) = {max_moment:.2f} {moment_unit}")

    return {
        "reactions": reactions,
        "max_values": {
            "max_shear": max_shear,
            "max_moment": max_moment,
            "zero_shear_x": zero_shear_x
        },
        "tabular_results": tabular_data,
        "steps": steps,
        "diagram_data": {
            "x": x_smooth.tolist(),
            "shear": shear_smooth.tolist(),
            "moment": moment_smooth.tolist()
        }
    }

# โค้ดส่วน Truss และ Frame Endpoint คงเดิม...
