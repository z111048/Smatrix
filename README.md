# Smatrix - 結構矩陣分析系統

<div align="center">

![Version](https://img.shields.io/badge/version-0.2.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.11+-yellow)
![React](https://img.shields.io/badge/react-18+-61DAFB)

**基於直接勁度法的 2D 結構分析 Web 應用程式**

[English](#english) | [繁體中文](#繁體中文)

</div>

---

## 繁體中文

### 功能特色

- 🏗️ **互動式結構建模**：點擊添加節點，拖曳連接形成梁元素
- 📊 **完整分析結果**：
  - 節點位移與轉角
  - 支承反力
  - 剪力圖 (SFD)
  - 彎矩圖 (BMD)
  - 變形曲線 (Hermite 插值)
- 🔧 **多種支承類型**：鉸支 (Pin)、滾支 (Roller)、固定端 (Fixed)
- ⚡ **多種載重類型**：節點集中力、節點彎矩、均佈載重 (UDL)
- 🔩 **2D 剛架/桁架分析** (v0.2.0 新增)：
  - 完整 6-DOF 元素（軸力 + 彎曲）
  - 斜向桿件座標轉換
  - 端部鉸接釋放（桁架行為）
  - 21 項自動化測試驗證

### 快速開始

#### 後端 (API Server)

```bash
cd backend
uv sync
uv run uvicorn app.main:app --reload --port 8000
```

#### 前端 (Web UI)

```bash
cd frontend
npm install
npm run dev
```

開啟瀏覽器訪問 `http://localhost:5173`

### 技術架構

| 層級 | 技術 |
|------|------|
| **計算核心** | Python + NumPy (矩陣運算) |
| **後端 API** | FastAPI + Pydantic |
| **前端 UI** | React 18 + TypeScript |
| **狀態管理** | Zustand |
| **畫布渲染** | React-Konva (Canvas) |
| **建置工具** | Vite |

### API 文檔

啟動後端後訪問 `http://localhost:8000/docs` 查看 Swagger UI。

#### POST /analyze

```json
{
  "nodes": [
    {"id": 1, "x": 0, "y": 0, "support": "pin"},
    {"id": 2, "x": 5, "y": 0},
    {"id": 3, "x": 10, "y": 0, "support": "roller"}
  ],
  "elements": [
    {"id": 1, "node_i": 1, "node_j": 2, "E": 200e9, "I": 1e-4},
    {"id": 2, "node_i": 2, "node_j": 3, "E": 200e9, "I": 1e-4}
  ],
  "point_loads": [
    {"node_id": 2, "Fy": -100000, "Mz": 0}
  ],
  "udls": [
    {"element_id": 1, "w": -10000}
  ]
}
```

### 工程理論

本系統採用 **直接勁度法 (Direct Stiffness Method)** 進行結構分析：

1. **元素勁度矩陣**：
   - 基本梁：4x4 彎曲勁度矩陣 (2 DOF/node: v, θ)
   - 剛架/桁架：6x6 完整勁度矩陣 (3 DOF/node: u, v, θ)
2. **座標轉換**：斜向桿件使用轉換矩陣 T 轉換至全域座標
3. **全域組裝**：依據節點編號組裝全域勁度矩陣
4. **邊界條件**：大數法 (Penalty Method) 處理支承約束
5. **端部釋放**：凝聚法處理鉸接端點（桁架行為）
6. **求解**：NumPy 線性代數求解 Kd = F
7. **後處理**：回代計算內力、繪製 SFD/BMD

---

## English

### Features

- 🏗️ **Interactive Modeling**: Click to add nodes, connect to create beams
- 📊 **Complete Analysis**:
  - Nodal displacements and rotations
  - Support reactions
  - Shear Force Diagram (SFD)
  - Bending Moment Diagram (BMD)
  - Deflection curve (Hermite interpolation)
- 🔧 **Support Types**: Pin, Roller, Fixed
- ⚡ **Load Types**: Point loads, Moments, UDL
- 🔩 **2D Frame/Truss Analysis** (v0.2.0):
  - Full 6-DOF elements (axial + bending)
  - Inclined member coordinate transformation
  - Moment releases for truss behavior
  - 21 automated test cases

### Quick Start

```bash
# Backend
cd backend && uv sync && uv run uvicorn app.main:app --reload

# Frontend
cd frontend && npm install && npm run dev
```

### License

MIT License - See [LICENSE](LICENSE) for details.

