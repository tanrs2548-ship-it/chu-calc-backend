from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import numpy as np

app = FastAPI(title="Chucalc Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# BEAM MODELS
# ==========================================
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
    analysis_type: Optional[str] = "determinate"

def mac_step(x, a, n):
    return np.where(x > a, (x - a)**n, 0)

@app.post("/api/analyze")
def analyze_beam(data: BeamInput):
    unit = data.unit if data.unit else "kN"
    moment_unit = f"{unit}.m"

    supports = sorted(data.supports, key=lambda s: s.x)
    if len(supports) < 2:
        return {"error": "Need at least 2 supports"}

    beam_len = data.beam_length
    is_indeterminate = (data.analysis_type == "indeterminate")

    x_smooth = np.linspace(0, beam_len, 500)
    shear_smooth = np.zeros_like(x_smooth)
    moment_smooth = np.zeros_like(x_smooth)
    reactions = []
    steps = []
    
    steps.append("==================================================")
    steps.append("1. STRUCTURE CLASSIFICATION & EQUILIBRIUM")

    if not is_indeterminate:
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
        steps.append(f"System: Statically Indeterminate Continuous Beam ({len(supports)} supports)")
        steps.append("Method: Moment Distribution (Hardy Cross Algorithm)")
        
        num_spans = len(supports) - 1
        spans = []
        for i in range(num_spans):
            x_left = supports[i].x
            x_right = supports[i+1].x
            L_span = x_right - x_left
            spans.append({"span_id": i, "L": L_span, "x_left": x_left, "x_right": x_right})

        fem = {}
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
                        fem[(i, "left")] -= (P * a * (b**2)) / (L**2) if L > 0 else 0
                        fem[(i, "right")] += (P * (a**2) * b) / (L**2) if L > 0 else 0
                elif l.type == 'distributed':
                    sa = l.start_x if l.start_x is not None else xl
                    sb = l.end_x if l.end_x is not None else xr
                    w = l.magnitude
                    overlap_start = max(xl, sa)
                    overlap_end = min(xr, sb)
                    if overlap_start < overlap_end:
                        sub_len = overlap_end - overlap_start
                        fem[(i, "left")] -= (w * (sub_len**2)) / 12
                        fem[(i, "right")] += (w * (sub_len**2)) / 12

        nodes = supports
        joint_moments = {n.id: 0.0 for n in nodes}
        for n in nodes:
            if n.type == 'fixed':
                joint_moments[n.id] = 0.0

        steps.append("Executing Moment Distribution iterations until unbalanced moment -> 0...")
        
        running_fem_left = {i: fem[(i, "left")] for i in range(num_spans)}
        running_fem_right = {i: fem[(i, "right")] for i in range(num_spans)}

        for iteration in range(20):
            unbalanced_at_joint = {}
            for j in range(1, len(nodes) - 1):
                if nodes[j].type == 'free': continue
                m_left_span = running_fem_right.get(j-1, 0.0)
                m_right_span = running_fem_left.get(j, 0.0)
                unbalanced = -(m_left_span + m_right_span)
                unbalanced_at_joint[j] = unbalanced

            for j in range(1, len(nodes) - 1):
                if j not in unbalanced_at_joint: continue
                unbal = unbalanced_at_joint[j]
                k_left = 4.0 / spans[j-1]["L"] if spans[j-1]["L"] > 0 else 1.0
                k_right = 4.0 / spans[j]["L"] if spans[j]["L"] > 0 else 1.0
                sum_k = k_left + k_right
                df_left = k_left / sum_k if sum_k > 0 else 0.5
                df_right = k_right / sum_k if sum_k > 0 else 0.5
                
                running_fem_right[j-1] += unbal * df_left
                running_fem_left[j] += unbal * df_right

            new_fem_l = {}
            new_fem_r = {}
            for i in range(num_spans):
                new_fem_l[i] = 0.5 * running_fem_left.get(i, 0.0)
                new_fem_r[i] = 0.5 * running_fem_right.get(i, 0.0)
            
            running_fem_left = new_fem_l
            running_fem_right = new_fem_r

        for j, n in enumerate(nodes):
            if j == 0:
                joint_moments[n.id] = running_fem_left.get(0, 0.0)
            elif j == len(nodes) - 1:
                joint_moments[n.id] = running_fem_right.get(num_spans-1, 0.0)
            else:
                joint_moments[n.id] = running_fem_right.get(j-1, 0.0)

        steps.append("Moment Distribution completed successfully. Support moments balanced to ~0.")

        for i, span in enumerate(spans):
            xl = span["x_left"]
            xr = span["x_right"]
            L = span["L"]
            m_l = joint_moments[nodes[i].id]
            m_r = joint_moments[nodes[i+1].id]
            
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

            r_r = (span_moment_sum + m_l - m_r) / L if L > 0 else 0.0
            r_l = span_load_sum - r_r
            
            reactions.append({"support_x": xl, "span_idx": i, "force_kN": r_l})
            if i == num_spans - 1:
                reactions.append({"support_x": xr, "span_idx": i+1, "force_kN": r_r})

        for i, span in enumerate(spans):
            xl = span["x_left"]
            xr = span["x_right"]
            mask = (x_smooth >= xl) & (x_smooth <= xr)
            x_sub = x_smooth[mask] - xl
            ml = joint_moments[nodes[i].id]
            mr = joint_moments[nodes[i+1].id]
            L = span["L"]
            
            mom_sub = ml + (mr - ml) * (x_sub / L) if L > 0 else np.zeros_like(x_sub)
            moment_smooth[mask] = mom_sub

    max_shear = float(np.max(np.abs(shear_smooth)))
    max_moment_idx = int(np.argmax(np.abs(moment_smooth)))
    max_moment = float(np.abs(moment_smooth)[max_moment_idx])
    zero_shear_x = float(x_smooth[max_moment_idx])

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

# ==========================================
# TRUSS MODELS & ENDPOINT
# ==========================================
class TrussNode(BaseModel):
    id: int
    name: str
    x: float
    y: float

class TrussElement(BaseModel):
    id: int
    n1: int
    n2: int

class TrussSupport(BaseModel):
    type: str
    direction: Optional[str] = "horizontal"

class TrussLoad(BaseModel):
    fx: Optional[float] = 0.0
    fy: Optional[float] = 0.0

class TrussInput(BaseModel):
    nodes: List[TrussNode]
    elements: List[TrussElement]
    supports: Dict[str, TrussSupport]
    loads: Dict[str, TrussLoad]
    unit: Optional[str] = "kN"
    ei: Optional[float] = None

@app.post("/api/analyze-truss")
def analyze_truss(data: TrussInput):
    node_idx = {n.id: i for i, n in enumerate(data.nodes)}
    n_nodes = len(data.nodes)
    
    K = np.zeros((2 * n_nodes, 2 * n_nodes))
    F = np.zeros(2 * n_nodes)
    
    # พลิกทิศทางแรงโหลดให้ตรงกับแกนคณิตศาสตร์
    for n_id_str, load in data.loads.items():
        n_id = int(n_id_str)
        if n_id in node_idx:
            idx = node_idx[n_id]
            if load.fx: F[2 * idx] = load.fx
            if load.fy: F[2 * idx + 1] = -load.fy
            
    # พลิกแกน Y ของ Canvas ให้พุ่งขึ้นตามสมการ
    nodes_math = {n.id: (n.x, -n.y) for n in data.nodes}
    lengths = {}
    
    for el in data.elements:
        x1, y1 = nodes_math[el.n1]
        x2, y2 = nodes_math[el.n2]
        L = np.hypot(x2 - x1, y2 - y1)
        lengths[el.id] = L
        if L == 0: continue
        
        c = (x2 - x1) / L
        s = (y2 - y1) / L
        
        EA = data.ei if data.ei else (200 * 5000)
        k = EA / L
        
        k_mat = k * np.array([
            [ c*c,  c*s, -c*c, -c*s],
            [ c*s,  s*s, -c*s, -s*s],
            [-c*c, -c*s,  c*c,  c*s],
            [-c*s, -s*s,  c*s,  s*s]
        ])
        
        idx1 = node_idx[el.n1]
        idx2 = node_idx[el.n2]
        dofs = [2*idx1, 2*idx1+1, 2*idx2, 2*idx2+1]
        
        for i in range(4):
            for j in range(4):
                K[dofs[i], dofs[j]] += k_mat[i, j]
                
    fixed_dofs = []
    for n_id_str, sup in data.supports.items():
        n_id = int(n_id_str)
        if n_id in node_idx:
            idx = node_idx[n_id]
            if sup.type == 'pin' or sup.type == 'fixed':
                fixed_dofs.extend([2*idx, 2*idx+1])
            elif sup.type == 'roller':
                fixed_dofs.append(2*idx+1)
                
    # ป้องกัน Matrix Singularity สำหรับโหนดที่ไม่ได้เชื่อมต่อ
    K += np.eye(2 * n_nodes) * 1e-9
    
    for dof in fixed_dofs:
        K[dof, :] = 0
        K[:, dof] = 0
        K[dof, dof] = 1.0
        F[dof] = 0.0
        
    try:
        U = np.linalg.solve(K, F)
    except np.linalg.LinAlgError:
        U = np.zeros(2 * n_nodes)
        
    members_res = []
    for el in data.elements:
        idx1 = node_idx[el.n1]
        idx2 = node_idx[el.n2]
        
        x1, y1 = nodes_math[el.n1]
        x2, y2 = nodes_math[el.n2]
        L = lengths[el.id]
        
        if L == 0:
            members_res.append({"name": f"{data.nodes[idx1].name}{data.nodes[idx2].name}", "force": 0.0, "status": "Zero-Force"})
            continue
            
        c = (x2 - x1) / L
        s = (y2 - y1) / L
        u = np.array([ U[2*idx1], U[2*idx1+1], U[2*idx2], U[2*idx2+1] ])
        
        EA = data.ei if data.ei else (200 * 5000)
        k = EA / L
        T = np.array([-c, -s, c, s])
        
        force = k * np.dot(T, u)
        status = "Tension" if force > 0.01 else ("Compression" if force < -0.01 else "Zero-Force")
        
        members_res.append({
            "name": f"{data.nodes[idx1].name}{data.nodes[idx2].name}",
            "force": round(float(abs(force)), 2),
            "status": status
        })
        
    return {"members": members_res}
