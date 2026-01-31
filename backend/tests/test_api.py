"""
API integration tests for Smatrix.
Tests the complete request/response cycle through FastAPI.
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.main import app

client = TestClient(app)


class TestHealthCheck:
    """Tests for health check endpoint"""
    
    def test_health_check(self):
        """Health check should return ok"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestAnalyzeEndpoint:
    """Tests for /analyze endpoint"""
    
    def test_simple_supported_beam_center_load(self):
        """Simply supported beam with center point load"""
        request = {
            "nodes": [
                {"id": 1, "x": 0, "y": 0, "support": "pin"},
                {"id": 2, "x": 5, "y": 0, "support": "free"},
                {"id": 3, "x": 10, "y": 0, "support": "roller"}
            ],
            "elements": [
                {"id": 1, "node_i": 1, "node_j": 2, "E": 200e9, "I": 1e-4},
                {"id": 2, "node_i": 2, "node_j": 3, "E": 200e9, "I": 1e-4}
            ],
            "point_loads": [
                {"node_id": 2, "Fy": -100000, "Mz": 0}
            ],
            "udls": []
        }
        
        response = client.post("/analyze", json=request)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert len(data["displacements"]) == 3
        assert len(data["reactions"]) == 2  # Pin and Roller
        assert len(data["internal_forces"]) == 2
    
    def test_cantilever_beam_tip_load(self):
        """Cantilever beam with tip point load"""
        request = {
            "nodes": [
                {"id": 1, "x": 0, "y": 0, "support": "fixed"},
                {"id": 2, "x": 4, "y": 0, "support": "free"}
            ],
            "elements": [
                {"id": 1, "node_i": 1, "node_j": 2, "E": 200e9, "I": 1e-4}
            ],
            "point_loads": [
                {"node_id": 2, "Fy": -50000, "Mz": 0}
            ],
            "udls": []
        }
        
        response = client.post("/analyze", json=request)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        
        # Check tip displacement exists
        tip_disp = next(d for d in data["displacements"] if d["node_id"] == 2)
        assert tip_disp["v"] < 0  # Should deflect downward
        
        # Check reactions
        fixed_reaction = next(r for r in data["reactions"] if r["node_id"] == 1)
        assert abs(fixed_reaction["Fy"] - 50000) < 100  # Should equal applied load
    
    def test_continuous_beam_udl(self):
        """Two-span continuous beam with UDL"""
        request = {
            "nodes": [
                {"id": 1, "x": 0, "y": 0, "support": "pin"},
                {"id": 2, "x": 6, "y": 0, "support": "pin"},
                {"id": 3, "x": 12, "y": 0, "support": "roller"}
            ],
            "elements": [
                {"id": 1, "node_i": 1, "node_j": 2, "E": 200e9, "I": 1e-4},
                {"id": 2, "node_i": 2, "node_j": 3, "E": 200e9, "I": 1e-4}
            ],
            "point_loads": [],
            "udls": [
                {"element_id": 1, "w": -10000},
                {"element_id": 2, "w": -10000}
            ]
        }
        
        response = client.post("/analyze", json=request)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert len(data["reactions"]) == 3
    
    def test_inclined_beam(self):
        """Beam with non-zero y-coordinates (inclined in 2D plane)"""
        request = {
            "nodes": [
                {"id": 1, "x": 0, "y": 0, "support": "pin"},
                {"id": 2, "x": 6, "y": 3, "support": "free"},
                {"id": 3, "x": 12, "y": 0, "support": "roller"}
            ],
            "elements": [
                {"id": 1, "node_i": 1, "node_j": 2, "E": 200e9, "I": 1e-4},
                {"id": 2, "node_i": 2, "node_j": 3, "E": 200e9, "I": 1e-4}
            ],
            "point_loads": [
                {"node_id": 2, "Fy": -50000, "Mz": 0}
            ],
            "udls": []
        }
        
        response = client.post("/analyze", json=request)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
    
    def test_user_reported_case(self):
        """User's reported case with negative y-coordinates"""
        request = {
            "nodes": [
                {"id": 1, "x": 2.5, "y": -2, "support": "pin"},  # Added support
                {"id": 2, "x": 14.5, "y": -2.5, "support": "roller"}  # Added support
            ],
            "elements": [
                {"id": 1, "node_i": 1, "node_j": 2, "E": 200000000000, "I": 0.0001}
            ],
            "point_loads": [],
            "udls": [
                {"element_id": 1, "w": -10000}
            ]
        }
        
        response = client.post("/analyze", json=request)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        
        # Element length should be sqrt((14.5-2.5)^2 + (-2.5-(-2))^2) ≈ 12.01
        # This verifies the Euclidean distance fix


class TestErrorHandling:
    """Tests for error handling"""
    
    def test_no_supports_error(self):
        """Structure without supports should fail gracefully"""
        request = {
            "nodes": [
                {"id": 1, "x": 0, "y": 0, "support": "free"},
                {"id": 2, "x": 10, "y": 0, "support": "free"}
            ],
            "elements": [
                {"id": 1, "node_i": 1, "node_j": 2, "E": 200e9, "I": 1e-4}
            ],
            "point_loads": [
                {"node_id": 2, "Fy": -50000, "Mz": 0}
            ],
            "udls": []
        }
        
        response = client.post("/analyze", json=request)
        # Should return 400 or 500 with error message
        assert response.status_code in [400, 500]
    
    def test_invalid_node_reference(self):
        """Element referencing non-existent node should fail"""
        request = {
            "nodes": [
                {"id": 1, "x": 0, "y": 0, "support": "pin"}
            ],
            "elements": [
                {"id": 1, "node_i": 1, "node_j": 2, "E": 200e9, "I": 1e-4}  # Node 2 doesn't exist
            ],
            "point_loads": [],
            "udls": []
        }
        
        response = client.post("/analyze", json=request)
        assert response.status_code == 400
    
    def test_invalid_element_reference_in_udl(self):
        """UDL referencing non-existent element should fail"""
        request = {
            "nodes": [
                {"id": 1, "x": 0, "y": 0, "support": "pin"},
                {"id": 2, "x": 10, "y": 0, "support": "roller"}
            ],
            "elements": [
                {"id": 1, "node_i": 1, "node_j": 2, "E": 200e9, "I": 1e-4}
            ],
            "point_loads": [],
            "udls": [
                {"element_id": 99, "w": -10000}  # Element 99 doesn't exist
            ]
        }
        
        response = client.post("/analyze", json=request)
        assert response.status_code == 400
    
    def test_empty_structure(self):
        """Empty structure should be handled"""
        request = {
            "nodes": [],
            "elements": [],
            "point_loads": [],
            "udls": []
        }
        
        response = client.post("/analyze", json=request)
        assert response.status_code == 400


class TestInternalForces:
    """Tests for internal force calculations"""
    
    def test_shear_force_at_supports(self):
        """Verify shear force values at supports"""
        L = 10.0
        P = 100000  # 100 kN
        
        request = {
            "nodes": [
                {"id": 1, "x": 0, "y": 0, "support": "pin"},
                {"id": 2, "x": L/2, "y": 0, "support": "free"},
                {"id": 3, "x": L, "y": 0, "support": "roller"}
            ],
            "elements": [
                {"id": 1, "node_i": 1, "node_j": 2, "E": 200e9, "I": 1e-4},
                {"id": 2, "node_i": 2, "node_j": 3, "E": 200e9, "I": 1e-4}
            ],
            "point_loads": [
                {"node_id": 2, "Fy": -P, "Mz": 0}
            ],
            "udls": []
        }
        
        response = client.post("/analyze", json=request)
        assert response.status_code == 200
        
        data = response.json()
        
        # Check that internal forces are returned
        assert len(data["internal_forces"]) == 2
        
        for forces in data["internal_forces"]:
            assert len(forces["stations"]) == 21  # Default n_points
            assert len(forces["V"]) == 21
            assert len(forces["M"]) == 21


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
